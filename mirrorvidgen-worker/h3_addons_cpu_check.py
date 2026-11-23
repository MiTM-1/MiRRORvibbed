"""Candidate image verification. CPU only; no weights or generation calls."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.argv = [sys.argv[0], "--cpu"]
sys.path.insert(0, "/comfyui")

REQUIRED = ["MiniMaxH3ReferenceToVideo", "MiniMaxH3ImageToVideo",
            "MiniMaxH3SigmaShift", "LoraLoaderModelOnly", "UNETLoader",
            "CLIPLoader", "VAELoader", "VAEDecode", "VAEDecodeAudio",
            "SamplerCustomAdvanced", "CreateVideo", "SaveVideo",
            "MiniMaxH3MotionContext", "MiniMaxH3MotionContextTrim",
            "MiniMaxH3MotionContextSaveLatent", "MiniMaxH3MotionContextLoadLatent"]

def main():
    import torch
    # ComfyUI disables CLI parsing when embedded unless explicitly enabled.
    # The server process parses --cpu itself; this standalone layout probe must
    # enable parsing before importing ComfyUI model-management modules too.
    import comfy.options
    comfy.options.enable_args_parsing()
    import comfy.cli_args
    assert comfy.cli_args.args.cpu, "Embedded ComfyUI must be configured for CPU"
    assert not torch.cuda.is_available(), "CPU check must not access a GPU"
    assert os.environ.get("MIRRORVIDGEN_H3_ADDONS_TEST") == "false"
    dependencies = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    report = {"gpu_available": False, "generation_submitted": False,
              "dependency_check_exit": dependencies.returncode,
              "dependency_check": dependencies.stdout + dependencies.stderr,
              "torch_version": torch.__version__}
    log = open("/tmp/h3-addons-comfy-cpu.log", "w+")
    server = subprocess.Popen([sys.executable, "/comfyui/main.py", "--cpu", "--listen", "127.0.0.1",
                               "--port", "8188", "--disable-auto-launch"], cwd="/comfyui", stdout=log, stderr=log)
    try:
        deadline = time.monotonic() + 240
        info = None
        while time.monotonic() < deadline and server.poll() is None:
            try:
                with urllib.request.urlopen("http://127.0.0.1:8188/object_info", timeout=10) as response:
                    info = json.load(response)
                break
            except Exception:
                time.sleep(2)
        if info is None:
            log.seek(0)
            raise RuntimeError("CPU ComfyUI startup failed: " + log.read()[-16000:])
        report["nodes"] = {name: name in info for name in REQUIRED}
        assert all(report["nodes"].values()), "Required nodes missing: " + str(report["nodes"])
        with urllib.request.urlopen("http://127.0.0.1:8188/system_stats", timeout=10) as response:
            stats = json.load(response)
        report["system"] = stats.get("system")
        print(json.dumps(report, indent=2, default=str), flush=True)
        pack = Path("/comfyui/custom_nodes/ComfyUI-H3-Motion-Context")
        spec = importlib.util.spec_from_file_location("h3_motion_cpu", pack / "layout_contract.py")
        layout = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(layout)
        layout.ensure()
        assert layout.is_checked()
        report["real_comfy_layout_contract"] = "passed"
        report["turbo_shift_inputs"] = list(info["MiniMaxH3SigmaShift"].get("input", {}).get("required", {}))
        from h3_addons import node, validate_addon_schema
        probe_graph = {
            "context": node("MiniMaxH3MotionContext", conditioning=["conditioning", 0],
                vae=["vae", 0], latent=["latent", 0], context_length="22",
                audio_context_length=24, context_latent=["previous", 0]),
            "trim": node("MiniMaxH3MotionContextTrim", images=["decode", 0],
                audio=["audio", 0], trim_frames=["context", 1], fps=24.0, match_tail=True),
            "save": node("MiniMaxH3MotionContextSaveLatent", latent=["sample", 0],
                filename_prefix="h3_addons/cpu-contract/clip", clip_index=1),
            "load": node("MiniMaxH3MotionContextLoadLatent", latent_path="cpu-fixture.safetensors", clip_index=1),
            "lora": node("LoraLoaderModelOnly", model=["model", 0],
                lora_name="minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors", strength_model=1.0),
            "shift": node("MiniMaxH3SigmaShift", model=["lora", 0], shift_video=12.0, shift_audio=3.0),
        }
        validate_addon_schema(probe_graph, info)
        report["addon_input_schema_contract"] = "passed"
        print(json.dumps(report, indent=2, default=str), flush=True)
        # Existing base dependency conflicts must be compared separately rather
        # than fixed by changing the proven image's dependency versions.
        Path("/tmp/h3-addons-cpu-report.json").write_text(json.dumps(report, indent=2, default=str))
    finally:
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()
        log.close()

if __name__ == "__main__":
    main()
