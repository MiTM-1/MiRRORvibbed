"""Approved isolated Normal vs Turbo comparison. Never a production entrypoint."""
import base64, hashlib, json, os, shutil, subprocess, sys, time, uuid
from pathlib import Path
import requests

LORA = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
SHA = "5b9ab5ade15d0775676d01a907268a69a1468dc6033b3b0d3ded5502f3ebb84c"
def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def download(url, target, max_bytes):
    total = 0
    with requests.get(url, stream=True, timeout=(30, 120)) as r:
        r.raise_for_status()
        with target.open("xb") as f:
            for chunk in r.iter_content(4*1024*1024):
                total += len(chunk)
                if total > max_bytes:
                    raise RuntimeError("Download exceeded fixed size guard")
                f.write(chunk)
    return total

def main():
    if os.environ.get("MIRRORVIDGEN_APPROVED_TURBO_COMPARE") != "5USD":
        raise RuntimeError("Isolated comparison requires explicit approved budget flag")
    root = Path("/runpod-volume")
    if not root.is_mount():
        raise RuntimeError("Existing network model volume must be mounted at /runpod-volume")
    if shutil.disk_usage(root).free < 4*1024**3:
        raise RuntimeError("Insufficient free space; no storage expansion or download attempted")
    run_id = "turbo-compare-" + uuid.uuid4().hex[:12]
    output = root / "results" / "h3-turbo-tests" / run_id
    output.mkdir(parents=True, exist_ok=False)
    os.environ["MIRRORVIDGEN_H3_ADDONS_TEST"] = "true"
    import h3_baseline_handler as baseline
    import handler as candidate
    status = baseline.h3_ref2va_status()
    if not status.get("ready"):
        raise RuntimeError("Existing Ref2VA weights unavailable; base installer is never called")
    lora = root / "models" / "loras" / LORA
    lora.parent.mkdir(parents=True, exist_ok=True)
    if lora.exists():
        if digest(lora) != SHA:
            raise RuntimeError("Existing LoRA checksum mismatch; left untouched")
    else:
        staged = lora.with_name(LORA + "." + run_id + ".partial")
        download("https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/main/" + LORA, staged, 2100000000)
        if digest(staged) != SHA:
            raise RuntimeError("Downloaded LoRA checksum mismatch; staged file retained")
        # Hard-link promotion cannot overwrite a file another worker created.
        os.link(staged, lora)
        staged.unlink()
    reference = output / "reference.png"
    download("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/guitar-man.png", reference, 20000000)
    data = {
        "engine": "mirrorromax-h3", "mode": "h3_reference", "h3_workflow": "ref2va",
        "duration_seconds": 5, "aspect_ratio": "9:16", "quality": "480p", "seed": 42,
        "fps": 24,
        "prompt": "integrated_multimodal_description: [Shot 1] Preserve the man, face, clothing and red electric guitar from <Picture 1>. In one continuous realistic medium shot he gently strums the guitar, moving his hands naturally. Keep his identity, instrument and surroundings consistent. Static camera. overall_soundscape: Clean natural electric guitar notes synchronized to his strumming, quiet room ambience. non_diegetic_music: N/A.",
        "reference_images": [{"name": "reference.png", "data": base64.b64encode(reference.read_bytes()).decode()}]
    }
    (output / "inputs.json").write_text(json.dumps({k:v for k,v in data.items() if k != "reference_images"}, indent=2))
    report = {"run_id": run_id, "lora_sha256": SHA, "tests": [], "production_unchanged": True}
    started = time.monotonic()
    proc = subprocess.Popen(["/opt/venv/bin/python", "-u", "/comfyui/main.py", "--listen", "127.0.0.1", "--port", "8188"], cwd="/comfyui")
    try:
        baseline.wait_for_comfy()
        report["startup_seconds"] = time.monotonic() - started
        report["gpu"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True).strip()
        for speed in ("normal", "turbo_fast"):
            if time.monotonic() - started > 3800:
                raise RuntimeError("Budget guard: second test not started after long baseline")
            payload = dict(data, generation_speed=speed, action="preflight")
            ready = candidate.handler({"id": run_id + "-" + speed + "-check", "input": payload})
            if ready.get("ready") is not True:
                raise RuntimeError("Preflight failed: " + json.dumps(ready))
            payload.pop("action")
            t = time.monotonic()
            result = candidate.handler({"id": run_id + "-" + speed, "input": payload})
            elapsed = time.monotonic() - t
            if result.get("status") != "success":
                raise RuntimeError("Generation failed: " + json.dumps(result))
            item = result["outputs"][0]
            if item["type"] == "base64":
                blob = base64.b64decode(item["data"], validate=True)
            elif item["type"] == "network_path":
                blob = Path(item["data"]).read_bytes()
            else:
                raise RuntimeError("Unexpected result type")
            video = output / (speed + ".mp4")
            video.write_bytes(blob)
            probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_entries", "format=duration:stream=codec_type,codec_name,width,height,nb_read_frames", "-of", "json", str(video)], check=True, capture_output=True, text=True, timeout=120)
            info = json.loads(probe.stdout)
            if float(info["format"]["duration"]) < 4 or not any(s["codec_type"] == "video" for s in info["streams"]):
                raise RuntimeError("Invalid generated MP4")
            subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-f", "null", "-"], check=True, timeout=120)
            subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", "fps=1,scale=240:-1,tile=5x1", "-frames:v", "1", str(output / (speed + "-contact.png"))], check=True, timeout=120)
            report["tests"].append({"speed":speed,"wall_seconds":elapsed,"bytes":len(blob),"sha256":digest(video),"probe":info,"settings":result.get("settings")})
            (output / "report.json").write_text(json.dumps(report, indent=2))
            print("H3_COMPARE_TEST_DONE " + json.dumps(report["tests"][-1]), flush=True)
        report["comparison_seconds"] = time.monotonic() - started
        (output / "report.json").write_text(json.dumps(report, indent=2))
        print("H3_COMPARE_FINISHED " + str(output), flush=True)
    finally:
        proc.terminate()
        try: proc.wait(timeout=30)
        except subprocess.TimeoutExpired: proc.kill()
    print("STOP THE TEMPORARY POD NOW; compute can still bill while idle.", flush=True)

if __name__ == "__main__":
    main()
