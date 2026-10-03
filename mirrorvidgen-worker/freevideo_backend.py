"""MiniMax H3 / FreeVideo backend for the MiRRORvidgen RunPod worker.

FreeVideo code ships in the worker image. Its isolated runtime and model cache
live on the persistent RunPod network volume at FREEVIDEO_HOME, so adding H3
does not change or replace the existing LTX-2.5/ComfyUI pipeline.
"""
from __future__ import annotations

import base64
import json
import math
import mimetypes
import os
from pathlib import Path
import subprocess
import time

import requests


FREEVIDEO_HOME = Path(os.environ.get("FREEVIDEO_HOME", "/runpod-volume/freevideo-h3"))
_BUNDLED_FREEVIDEO_SOURCE = Path(os.environ.get("FREEVIDEO_SOURCE", "/opt/freevideo"))
_PERSISTENT_FREEVIDEO_SOURCE = FREEVIDEO_HOME / "source"
# Prefer the persistent, setup-matched checkout when present. The bundled copy
# remains a bootstrap fallback for a fresh volume and for the read-only plan.
FREEVIDEO_SOURCE = (
    _PERSISTENT_FREEVIDEO_SOURCE
    if (_PERSISTENT_FREEVIDEO_SOURCE / "freevideo").is_file()
    else _BUNDLED_FREEVIDEO_SOURCE
)
FREEVIDEO_BIN = FREEVIDEO_SOURCE / "freevideo"
FREEVIDEO_CONFIG = FREEVIDEO_HOME / "machine.json"
H3_RESULTS = Path(os.environ.get("MIRRORVIDGEN_H3_RESULTS", "/runpod-volume/results"))
H3_TIMEOUT = int(os.environ.get("MIRRORVIDGEN_H3_TIMEOUT", "3500"))
MAX_INLINE_VIDEO_BYTES = int(os.environ.get("MIRRORVIDGEN_MAX_INLINE_VIDEO_BYTES", "15000000"))

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv"}
_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}

QUALITY_AREA = {
    "smoke": 640 * 384,
    "480p": 864 * 480,
    "720p": 1344 * 768,
    "1080p": 1920 * 1088,
}


class H3Unavailable(RuntimeError):
    pass


def _machine_state():
    if not FREEVIDEO_CONFIG.is_file():
        return {}
    try:
        value = json.loads(FREEVIDEO_CONFIG.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def h3_status():
    machine = _machine_state()
    cache = Path(str(machine.get("cache") or "")) if machine.get("cache") else None
    source_ready = FREEVIDEO_BIN.is_file()
    configured = bool(machine)
    marked_ready = bool(machine.get("ready"))
    cache_ready = bool(cache and cache.exists())
    ready = source_ready and configured and marked_ready and cache_ready
    return {
        "engine": "freevideo_h3",
        "label": "MiniMax H3 · FreeVideo",
        "source_ready": source_ready,
        "configured": configured,
        "model_cache_ready": cache_ready,
        "ready": ready,
        "root": str(FREEVIDEO_HOME),
        "fps": 24,
        "supports": {
            "text_to_video": True,
            "image_to_video": True,
            "first_last_frame": True,
            "reference_image": True,
            "reference_video": True,
            "reference_audio": True,
            "native_audio": True,
            "seed": True,
            "custom_dimensions": True,
        },
    }


def h3_install_diagnostics():
    """Return a compact diagnostic from the newest retained FreeVideo setup run."""
    runs_root = FREEVIDEO_HOME / "setup-runs"
    if not runs_root.is_dir():
        return {
            "status": "h3_diagnostics",
            "found": False,
            "message": "No retained FreeVideo setup-runs directory was found.",
            "h3": h3_status(),
        }
    runs = sorted(
        (path for path in runs_root.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        return {
            "status": "h3_diagnostics",
            "found": False,
            "message": "No retained FreeVideo setup run was found.",
            "h3": h3_status(),
        }

    run = runs[0]
    status_path = run / "status.json"
    status = {}
    if status_path.is_file():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            status = {}

    failed_steps = []
    steps = status.get("steps") if isinstance(status, dict) else None
    if isinstance(steps, list):
        for row in steps:
            if not isinstance(row, dict) or row.get("status") != "failed":
                continue
            log_path = Path(str(row.get("log") or ""))
            tail = ""
            if log_path.is_file():
                try:
                    raw = log_path.read_text(encoding="utf-8", errors="replace")
                    tail = raw[-8000:]
                except OSError as error:
                    tail = "Could not read log: " + repr(error)
            failed_steps.append({
                "label": row.get("label"),
                "returncode": row.get("returncode"),
                "error": row.get("error"),
                "seconds": row.get("seconds"),
                "log": str(log_path),
                "log_tail": tail,
            })

    if not failed_steps:
        # Fall back to the newest log files so diagnostics still work if setup
        # stopped outside a normal child-command failure.
        logs = sorted(run.glob("*.log"), key=lambda path: path.stat().st_mtime, reverse=True)
        for log_path in logs[:3]:
            try:
                raw = log_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            failed_steps.append({
                "label": log_path.stem,
                "log": str(log_path),
                "log_tail": raw[-6000:],
            })

    return {
        "status": "h3_diagnostics",
        "found": True,
        "run": str(run),
        "resource_guard": status.get("resource_guard") if isinstance(status, dict) else None,
        "failed_steps": failed_steps[:4],
        "h3": h3_status(),
    }


def h3_install(accept_model_license=False):
    """Install/prepare FreeVideo H3 on the persistent network volume.

    This is intentionally gated behind an explicit request flag. The caller
    must pass accept_model_license=True; otherwise no model download starts.
    """
    if not accept_model_license:
        raise ValueError("H3 install requires input.accept_model_license=true")
    script = Path("/h3-setup.sh")
    if not script.is_file():
        raise H3Unavailable("H3 setup helper is missing from this worker image")
    result = subprocess.run(
        [str(script), "--accept-model-license"],
        env=_environment(),
        capture_output=True,
        text=True,
        timeout=int(os.environ.get("MIRRORVIDGEN_H3_SETUP_TIMEOUT", "7200")),
        check=False,
    )
    output = (result.stdout + "\n" + result.stderr).strip()
    if result.returncode:
        diagnostics = h3_install_diagnostics()
        detail = {
            "message": "H3 setup failed",
            "setup_log_tail": output[-8000:],
            "diagnostics": diagnostics,
        }
        raise H3Unavailable(json.dumps(detail, ensure_ascii=False))
    return {
        "status": "h3_installed",
        "h3": h3_status(),
        "log_tail": output[-12000:],
    }


def h3_setup_plan():
    """Run FreeVideo's read-only setup planner. This does not accept a licence."""
    if not FREEVIDEO_BIN.is_file():
        raise H3Unavailable("FreeVideo source is not installed in this worker image")
    FREEVIDEO_HOME.mkdir(parents=True, exist_ok=True)
    command = [
        str(FREEVIDEO_BIN), "--root", str(FREEVIDEO_HOME),
        "setup", "--plan", "--json", "--plain",
    ]
    result = subprocess.run(
        command,
        cwd=str(FREEVIDEO_SOURCE),
        env=_environment(),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if result.returncode:
        raise H3Unavailable((result.stderr or result.stdout or "FreeVideo setup planner failed")[-6000:])
    text = result.stdout.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}


def _environment():
    env = dict(os.environ)
    env["FREEVIDEO_HOME"] = str(FREEVIDEO_HOME)
    env["FREEVIDEO_SOURCE_DIR"] = str(FREEVIDEO_SOURCE)
    env.setdefault("HF_HOME", os.environ.get("FREEVIDEO_HF_HOME", "/runpod-volume/.cache/huggingface"))
    env.setdefault("PYTHONUNBUFFERED", "1")
    # FreeVideo records the physical GPU UUID used during setup. That is safe
    # on a fixed machine but not on RunPod Serverless, where a later request
    # may land on a different physical GPU. Supplying the current container's
    # logical device prevents FreeVideo from falling back to the stale UUID.
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    return env


def _snap32(value):
    return max(256, int(round(float(value) / 32.0)) * 32)


def dimensions(job_input):
    width = job_input.get("width")
    height = job_input.get("height")
    if width is not None or height is not None:
        if width is None or height is None:
            raise ValueError("H3 custom dimensions require both width and height")
        width, height = int(width), int(height)
        if width < 256 or height < 256 or width % 32 or height % 32:
            raise ValueError("H3 width and height must be at least 256 and divisible by 32")
        return width, height

    aspect = str(job_input.get("aspect_ratio") or "16:9")
    try:
        left, right = (int(part) for part in aspect.split(":", 1))
        ratio = left / right
    except (ValueError, ZeroDivisionError):
        raise ValueError("input.aspect_ratio must look like 16:9, 9:16 or 1:1")
    if ratio <= 0:
        raise ValueError("input.aspect_ratio must be positive")

    quality = str(job_input.get("quality") or "720p").lower()
    if quality not in QUALITY_AREA:
        raise ValueError("H3 quality must be one of: smoke, 480p, 720p, 1080p")
    area = QUALITY_AREA[quality]
    return _snap32(math.sqrt(area * ratio)), _snap32(math.sqrt(area / ratio))


def _decode_data(value):
    raw = str(value)
    if "," in raw and raw.lstrip().startswith("data:"):
        raw = raw.split(",", 1)[1]
    return base64.b64decode(raw, validate=True)


def h3_gpu_check():
    """Verify FreeVideo's installed Torch runtime can see this RunPod worker GPU."""
    machine = _machine_state()
    python = Path(str(machine.get("python") or ""))
    if not python.is_file():
        return {
            "status": "h3_gpu_check",
            "ready": False,
            "error": "FreeVideo Python runtime is missing",
            "h3": h3_status(),
        }
    code = (
        "import json, torch; "
        "ok=torch.cuda.is_available(); "
        "print(json.dumps({"
        "'cuda_available':ok,"
        "'device_count':torch.cuda.device_count(),"
        "'device_name':torch.cuda.get_device_name(0) if ok else None,"
        "'capability':list(torch.cuda.get_device_capability(0)) if ok else None,"
        "'torch_version':torch.__version__,"
        "'torch_cuda':torch.version.cuda"
        "}))"
    )
    env = _environment()
    result = subprocess.run(
        [str(python), "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    payload = {}
    if result.stdout.strip():
        try:
            payload = json.loads(result.stdout.strip().splitlines()[-1])
        except json.JSONDecodeError:
            payload = {"stdout": result.stdout[-4000:]}
    return {
        "status": "h3_gpu_check",
        "ready": bool(result.returncode == 0 and payload.get("cuda_available")),
        "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
        "saved_setup_gpu_uuid": machine.get("gpu_uuid"),
        "probe": payload,
        "stderr": result.stderr[-4000:],
        "returncode": result.returncode,
        "h3": h3_status(),
    }


def h3_generation_diagnostics():
    """Return compact diagnostics from the newest retained H3 generation."""
    if not H3_RESULTS.is_dir():
        return {"status": "h3_generation_diagnostics", "found": False, "message": "No H3 results directory found."}
    runs = sorted(
        (path for path in H3_RESULTS.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        return {"status": "h3_generation_diagnostics", "found": False, "message": "No retained H3 generation directory found."}
    run = runs[0]
    files = []
    candidates = sorted(
        (path for path in run.rglob("*") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates[:20]:
        row = {"path": str(path), "bytes": path.stat().st_size}
        if path.suffix.lower() in {".log", ".txt", ".json"} or path.name.endswith(".stderr") or path.name.endswith(".stdout"):
            try:
                row["tail"] = path.read_text(encoding="utf-8", errors="replace")[-8000:]
            except OSError as error:
                row["tail"] = "Could not read: " + repr(error)
        files.append(row)
    return {
        "status": "h3_generation_diagnostics",
        "found": True,
        "run": str(run),
        "files": files,
        "h3": h3_status(),
    }


def _validate_downloaded_asset(path):
    """Reject HTML/error pages that were saved with an image/video filename."""
    suffix = path.suffix.lower()
    head = path.read_bytes()[:32]
    if suffix in {".jpg", ".jpeg"}:
        if not head.startswith(b"\xff\xd8\xff"):
            preview = path.read_bytes()[:300].decode("utf-8", errors="replace")
            raise ValueError("Downloaded JPEG is not actually a JPEG. The URL may be a preview/sign-in page. First bytes: " + preview)
    elif suffix == ".png":
        if not head.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("Downloaded PNG is not actually a PNG")
    elif suffix == ".webp":
        if not (head.startswith(b"RIFF") and head[8:12] == b"WEBP"):
            raise ValueError("Downloaded WEBP is not actually a WEBP")
    elif suffix == ".mp4":
        if len(head) < 12 or head[4:8] != b"ftyp":
            raise ValueError("Downloaded MP4 is not actually an MP4")
    return path


def _write_item(item, folder):
    name = Path(str(item.get("name") or "asset")).name
    if not name:
        raise ValueError("H3 asset has no filename")
    target = folder / name
    if item.get("url"):
        response = requests.get(str(item["url"]), stream=True, timeout=180)
        response.raise_for_status()
        with target.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    else:
        encoded = item.get("data") or item.get("image")
        if not encoded:
            raise ValueError(f"H3 asset data is missing: {name}")
        target.write_bytes(_decode_data(encoded))
    if not target.is_file() or target.stat().st_size == 0:
        raise ValueError(f"H3 asset could not be materialised: {name}")
    return _validate_downloaded_asset(target)


def _materialize(job_input, folder):
    folder.mkdir(parents=True, exist_ok=False)
    rows = []
    seen = set()
    for collection in ("images", "assets"):
        for raw in job_input.get(collection) or []:
            if not isinstance(raw, dict):
                continue
            item = dict(raw)
            name = Path(str(item.get("name") or "asset")).name
            if not name or name in seen:
                continue
            seen.add(name)
            path = _write_item(item, folder)
            rows.append({
                "path": path,
                "name": name,
                "role": str(item.get("role") or "").strip().lower(),
                "type": str(item.get("type") or "").strip().lower(),
            })
    for key in ("source_video", "video"):
        raw = job_input.get(key)
        if isinstance(raw, dict):
            item = dict(raw)
            name = Path(str(item.get("name") or "source.mp4")).name
            if name and name not in seen:
                seen.add(name)
                path = _write_item(item, folder)
                rows.append({"path": path, "name": name, "role": key, "type": "video"})
    return rows


def _kind(path):
    ext = path.suffix.lower()
    if ext in _IMAGE_EXTS:
        return "image"
    if ext in _VIDEO_EXTS:
        return "video"
    if ext in _AUDIO_EXTS:
        return "audio"
    return None


def _by_role(rows, accepted):
    accepted = {value.lower() for value in accepted}
    for row in rows:
        if row["role"] in accepted:
            return row
    return None


def _media_request(job_input, rows, folder):
    mode = str(job_input.get("mode") or "text_to_video").strip().lower()
    first_modes = {"h3_i2v", "image_to_video", "first_frame_to_video"}
    fl_modes = {"h3_first_last", "first_last_frame_to_video", "first_and_last_frame_to_video"}
    ref_modes = {"h3_reference", "reference_to_video", "character_to_video", "reference_video"}

    images = [row for row in rows if _kind(row["path"]) == "image"]
    media = {"version": 1}

    if mode in fl_modes:
        first = _by_role(rows, {"start", "first", "first frame", "opening"}) or (images[0] if images else None)
        last = _by_role(rows, {"end", "last", "last frame", "ending"}) or (images[1] if len(images) > 1 else None)
        if not first or not last:
            raise ValueError("H3 first + last frame mode requires two images")
        media["first"] = str(first["path"])
        media["last"] = str(last["path"])
    elif mode in first_modes:
        first = _by_role(rows, {"start", "first", "first frame", "opening"}) or (images[0] if images else None)
        if not first:
            raise ValueError("H3 Image → Video requires a first-frame image")
        media["first"] = str(first["path"])
    elif mode in ref_modes:
        references = []
        for row in rows:
            kind = _kind(row["path"])
            if kind:
                references.append({"path": str(row["path"]), "kind": kind})
        if not references:
            raise ValueError("H3 Reference → Video requires at least one image, video or audio reference")
        media["references"] = references[:32]
    else:
        return None

    path = folder / "media.json"
    path.write_text(json.dumps(media, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _put_result(upload, blob):
    url = str(upload.get("url") or "")
    if not url:
        return None
    headers = dict(upload.get("headers") or {})
    headers.setdefault("Content-Type", "video/mp4")
    response = requests.put(url, data=blob, headers=headers, timeout=600)
    response.raise_for_status()
    return upload.get("public_url") or url.split("?", 1)[0]


def run_h3_job(job, job_input):
    state = h3_status()
    if not state["ready"]:
        raise H3Unavailable(
            "MiniMax H3 / FreeVideo is installed in the worker image but its persistent runtime/model cache "
            "is not ready yet. Run the FreeVideo setup on the attached network volume first."
        )

    prompt = str(job_input.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("input.prompt is required for H3")
    if len(prompt) > 12000:
        raise ValueError("input.prompt is too long (max 12000 characters)")

    duration = float(job_input.get("duration_seconds", 5))
    if duration < 1.625 or duration > 20:
        raise ValueError("H3 duration must be between 1.625 and 20 seconds")

    fps = int(job_input.get("fps", 24))
    if fps != 24:
        raise ValueError("H3 generates natively at 24 fps")

    seed = int(job_input.get("seed", 42))
    if seed < 0 or seed > 0x7FFFFFFFFFFFFFFF:
        raise ValueError("input.seed is outside the supported range")

    width, height = dimensions(job_input)
    job_id = str(job.get("id") or f"h3-{int(time.time())}")
    safe_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in job_id)
    work = Path("/tmp") / f"mirrorvidgen_h3_{safe_id}"
    rows = _materialize(job_input, work)
    media_path = _media_request(job_input, rows, work)

    prompt_path = work / "prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")

    output_dir = H3_RESULTS / safe_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "mirrorvidgen_h3.mp4"
    if output.exists():
        output.unlink()

    command = [
        str(FREEVIDEO_BIN), "--root", str(FREEVIDEO_HOME),
        "generate",
        "--prompt-file", str(prompt_path),
        "--out", str(output),
        "--width", str(width),
        "--height", str(height),
        "--seconds", str(duration),
        "--seed", str(seed),
    ]
    if media_path is not None:
        command += ["--media", str(media_path)]

    result = subprocess.run(
        command,
        cwd=str(FREEVIDEO_SOURCE),
        env=_environment(),
        capture_output=True,
        text=True,
        timeout=H3_TIMEOUT,
        check=False,
    )
    if result.returncode:
        stdout_path = output_dir / "mirrorvidgen_wrapper.stdout"
        stderr_path = output_dir / "mirrorvidgen_wrapper.stderr"
        stdout_path.write_text(result.stdout or "", encoding="utf-8")
        stderr_path.write_text(result.stderr or "", encoding="utf-8")
        detail = (result.stdout + "\n" + result.stderr).strip()[-12000:]
        raise RuntimeError("FreeVideo H3 generation failed: " + detail)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("FreeVideo H3 finished without producing an MP4")

    blob = output.read_bytes()
    upload = job_input.get("result_upload") or {}
    uploaded = _put_result(upload, blob) if upload.get("url") else None
    if uploaded:
        item = {"filename": output.name, "kind": "video", "type": "url", "data": uploaded, "bytes": len(blob)}
    elif len(blob) <= MAX_INLINE_VIDEO_BYTES:
        item = {
            "filename": output.name,
            "kind": "video",
            "type": "base64",
            "mime": "video/mp4",
            "data": base64.b64encode(blob).decode("utf-8"),
            "bytes": len(blob),
        }
    else:
        item = {
            "filename": output.name,
            "kind": "video",
            "type": "network_path",
            "data": str(output),
            "bytes": len(blob),
            "note": "Provide input.result_upload.url for a browser-accessible result URL.",
        }

    return {
        "status": "success",
        "engine": "freevideo_h3",
        "outputs": [item],
        "settings": {
            "model": "MiniMax H3 · FreeVideo",
            "engine": "VDN-H3 8-step / FreeVideo",
            "fps": 24,
            "requested_duration_seconds": duration,
            "width": width,
            "height": height,
            "aspect_ratio": str(job_input.get("aspect_ratio") or "16:9"),
            "quality": str(job_input.get("quality") or "720p"),
            "seed": seed,
            "mode": str(job_input.get("mode") or "text_to_video"),
            "native_audio": True,
        },
    }
