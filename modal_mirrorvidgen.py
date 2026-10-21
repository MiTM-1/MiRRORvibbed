"""Isolated Modal runtime for the existing MiRRORvidgen v136 worker.

This file deliberately does not alter or route production traffic.  It keeps
the worker's established input/output contract while replacing RunPod's
container lifecycle with Modal Functions and a persistent Modal Volume.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

import modal
import requests


APP_NAME = "mirrorvidgen-v136-staging"
MIGRATION_BUILD = "v136-modal-staging-20"
GPU_KIND = "RTX-PRO-6000"
VOLUME_NAME = "mirrorvidgen-v136-data"
VOLUME_ROOT = Path("/runpod-volume")
STATUS_ROOT = VOLUME_ROOT / "modal-jobs"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

WORKER_CONTEXT = Path(__file__).parent / "mirrorvidgen-worker"
worker_image = modal.Image.from_dockerfile(
    WORKER_CONTEXT / "Dockerfile.modal",
    context_dir=WORKER_CONTEXT,
)
# Keep the public control plane independent from the multi-gigabyte H3 image.
# Sites has a bounded outbound request window, so the gateway must cold-start
# quickly while the durable native/H3 workers continue to scale independently.
control_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "fastapi==0.115.6",
    "requests==2.32.5",
)


_comfy_process = None


def _requires_comfy(payload: dict) -> bool:
    action = str(payload.get("action") or "").strip().lower()
    if action in {
        "h3_install", "h3_runtime_audit", "h3_status", "h3_setup_plan",
        "h3_diagnostics", "h3_generation_diagnostics", "h3_gpu_check",
        "h3_ref2va_install", "h3_ref2va_install_plan", "h3_ref2va_status",
    }:
        return False
    engine = str(payload.get("engine") or payload.get("model_engine") or "").strip().lower()
    mode = str(payload.get("mode") or "").strip().lower()
    workflow = str(payload.get("workflow") or "").strip().lower()
    explicit = str(payload.get("h3_workflow") or payload.get("model_workflow") or "").strip().lower()
    ref_modes = {
        "h3_reference", "reference_to_video", "character_to_video",
        "reference_video", "reference_to_audio_video", "ref2va",
    }
    h3_engines = {
        "mirrorromax-h3", "mirrorromax_h3",
        "freevideo_h3", "h3", "minimax_h3", "minimax-h3",
    }
    if (
        action in {"h3_comfy_runtime_audit", "h3_ref2va_schema_audit", "h3_ref2va_preflight"}
        or workflow == "ref2va"
        or explicit == "ref2va"
        or mode in {"h3_reference", "ref2va"}
        or (engine in h3_engines and mode in ref_modes)
    ):
        return True
    if engine in {"freevideo_h3", "h3", "minimax_h3", "minimax-h3"} or mode.startswith("h3_"):
        return False
    return True


def _ensure_comfy() -> None:
    global _comfy_process
    try:
        if requests.get("http://127.0.0.1:8188/system_stats", timeout=2).ok:
            return
    except requests.RequestException:
        pass

    log_path = Path("/tmp/modal-comfyui.log")
    log_handle = log_path.open("ab")
    _comfy_process = subprocess.Popen(
        [
            "/opt/venv/bin/python", "-u", "/comfyui/main.py",
            "--disable-auto-launch", "--disable-metadata", "--listen",
            "--verbose", "INFO", "--log-stdout",
        ],
        cwd="/comfyui",
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    deadline = time.time() + 300
    while time.time() < deadline:
        if _comfy_process.poll() is not None:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-6000:]
            raise RuntimeError(f"ComfyUI exited during Modal startup: {tail}")
        try:
            if requests.get("http://127.0.0.1:8188/system_stats", timeout=3).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    tail = log_path.read_text(encoding="utf-8", errors="replace")[-6000:]
    raise RuntimeError(f"ComfyUI did not become ready on Modal: {tail}")


def _enable_modal_rss_guard() -> None:
    """Keep the pinned persistent FreeVideo source aligned with the Modal patch."""
    os.environ["FREEVIDEO_ALLOW_RSS_GUARD"] = "1"
    os.environ["FREEVIDEO_UV_LINK_MODE"] = "copy"
    changed = False
    patched = Path("/opt/freevideo/freevideo_engine/ram.py")
    persistent = VOLUME_ROOT / "freevideo-h3" / "source" / "freevideo_engine" / "ram.py"
    if patched.is_file() and persistent.is_file():
        patched_bytes = patched.read_bytes()
        if persistent.read_bytes() != patched_bytes:
            temp = persistent.with_suffix(".modal.tmp")
            temp.write_bytes(patched_bytes)
            temp.replace(persistent)
            changed = True

    uv_old = "UV_PYTHON_INSTALL_DIR=str(self.root / 'python'), UV_LINK_MODE='hardlink',"
    uv_new = ("UV_PYTHON_INSTALL_DIR=str(self.root / 'python'), "
              "UV_LINK_MODE=os.environ.get('FREEVIDEO_UV_LINK_MODE', 'hardlink'),")
    for bootstrap in (
        Path("/opt/freevideo/freevideo_engine/bootstrap.py"),
        VOLUME_ROOT / "freevideo-h3" / "source" / "freevideo_engine" / "bootstrap.py",
    ):
        if not bootstrap.is_file():
            continue
        source = bootstrap.read_text(encoding="utf-8")
        if uv_old in source:
            temp = bootstrap.with_suffix(".modal.tmp")
            temp.write_text(source.replace(uv_old, uv_new, 1), encoding="utf-8")
            temp.replace(bootstrap)
            changed = changed or str(bootstrap).startswith(str(VOLUME_ROOT))
        elif uv_new not in source:
            raise RuntimeError(f"Unexpected pinned FreeVideo bootstrap source: {bootstrap}")
    if changed:
        volume.commit()


def _load_handler():
    if "/" not in sys.path:
        sys.path.insert(0, "/")
    from handler import handler

    return handler


def _write_status(job_id: str, data: dict) -> None:
    STATUS_ROOT.mkdir(parents=True, exist_ok=True)
    target = STATUS_ROOT / f"{job_id}.json"
    existing = {}
    if target.is_file():
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing = {}
    temp = target.with_suffix(".tmp")
    temp.write_text(
        json.dumps({**existing, **data}, separators=(",", ":")),
        encoding="utf-8",
    )
    temp.replace(target)
    volume.commit()


def _callback_allowed(webhook: str) -> bool:
    if not webhook:
        return False
    allowed = urlparse(
        os.environ.get(
            "MIRRORVISION_SITE_ORIGIN",
            "https://mirrorvidgen-modal-preview.mitm1.chatgpt.site",
        )
    )
    candidate = urlparse(webhook)
    return (
        candidate.scheme == "https"
        and candidate.hostname == allowed.hostname
        and candidate.port == allowed.port
        and candidate.username is None
        and candidate.password is None
    )


def _private_site_bearer_allowed(site_access_bearer: str) -> bool:
    if len(site_access_bearer) < 20:
        return False
    origin = os.environ.get(
        "MIRRORVISION_SITE_ORIGIN",
        "https://mirrorvidgen-modal-preview.mitm1.chatgpt.site",
    ).rstrip("/")
    try:
        response = requests.get(
            f"{origin}/",
            headers={"OAI-Sites-Authorization": f"Bearer {site_access_bearer}"},
            timeout=15,
            allow_redirects=False,
        )
        return 200 <= response.status_code < 300
    except requests.RequestException:
        return False


def _notify_callback(webhook: str, state: dict) -> None:
    if not _callback_allowed(webhook):
        return
    try:
        headers = {}
        site_access_bearer = os.environ.get("MIRRORVISION_SITE_ACCESS_BEARER", "").strip()
        if site_access_bearer:
            headers["OAI-Sites-Authorization"] = f"Bearer {site_access_bearer}"
        requests.post(
            webhook,
            json={"id": state["id"], "status": state["status"]},
            headers=headers,
            timeout=20,
            allow_redirects=False,
        ).raise_for_status()
    except requests.RequestException as error:
        print(f"Modal callback delivery failed for {state['id']}: {error}", flush=True)


def _execute(job_id: str, payload: dict, runtime: str, webhook: str = "") -> dict:
    started = time.time()
    _write_status(job_id, {"id": job_id, "status": "IN_PROGRESS", "runtime": runtime})
    try:
        result = _load_handler()({"id": job_id, "input": payload})
        if isinstance(result, dict) and result.get("error"):
            raise RuntimeError(str(result["error"]))
        state = {
            "id": job_id,
            "status": "COMPLETED",
            "runtime": runtime,
            "executionTime": round(time.time() - started, 3),
            "output": result,
        }
        _write_status(job_id, state)
        _notify_callback(webhook, state)
        return state
    except Exception as error:
        state = {
            "id": job_id,
            "status": "FAILED",
            "runtime": runtime,
            "executionTime": round(time.time() - started, 3),
            "error": str(error),
        }
        _write_status(job_id, state)
        _notify_callback(webhook, state)
        raise


@app.function(
    image=worker_image,
    volumes={str(VOLUME_ROOT): volume},
    cpu=4,
    memory=8192,
    timeout=3600,
    min_containers=0,
    max_containers=1,
    scaledown_window=2,
)
def native_job(
    job_id: str,
    payload: dict,
    webhook: str = "",
    site_access_bearer: str = "",
) -> dict:
    if str(payload.get("action") or "").lower() not in {"mirrorvision_render", "mirrorvision_health"}:
        raise ValueError("The CPU renderer only accepts MiRRORvision native actions")
    previous_bearer = os.environ.get("MIRRORVISION_SITE_ACCESS_BEARER")
    if site_access_bearer:
        os.environ["MIRRORVISION_SITE_ACCESS_BEARER"] = site_access_bearer
    try:
        return _execute(job_id, payload, "native-cpu", webhook)
    finally:
        if previous_bearer is None:
            os.environ.pop("MIRRORVISION_SITE_ACCESS_BEARER", None)
        else:
            os.environ["MIRRORVISION_SITE_ACCESS_BEARER"] = previous_bearer


@app.function(
    image=worker_image,
    gpu=GPU_KIND,
    volumes={str(VOLUME_ROOT): volume},
    cpu=8,
    memory=32768,
    timeout=7200,
    min_containers=0,
    max_containers=1,
    scaledown_window=2,
)
def h3_job(job_id: str, payload: dict, webhook: str = "") -> dict:
    _enable_modal_rss_guard()
    if _requires_comfy(payload):
        _ensure_comfy()
    return _execute(job_id, payload, "h3-gpu", webhook)


@app.function(
    image=worker_image,
    gpu=GPU_KIND,
    volumes={str(VOLUME_ROOT): volume},
    cpu=4,
    memory=32768,
    timeout=2100,
    min_containers=0,
    max_containers=1,
    scaledown_window=2,
)
def h3_bootstrap_job(job_id: str, payload: dict) -> dict:
    _enable_modal_rss_guard()
    if str(payload.get("action") or "").strip().lower() != "h3_install":
        raise ValueError("The bounded bootstrap function only accepts h3_install")
    return _execute(job_id, payload, "h3-bootstrap-gpu")


@app.function(
    image=worker_image,
    volumes={str(VOLUME_ROOT): volume},
    cpu=2,
    memory=4096,
    timeout=300,
    min_containers=0,
    max_containers=1,
    scaledown_window=2,
)
def maintenance_job(job_id: str, payload: dict) -> dict:
    action = str(payload.get("action") or "").strip().lower()
    if action not in {
        "capabilities", "mirrorvision_health", "h3_runtime_audit", "h3_ref2va_status"
    }:
        raise ValueError("The CPU maintenance function only accepts read-only audit actions")
    return _execute(job_id, payload, "maintenance-cpu")


@app.function(
    image=worker_image,
    volumes={str(VOLUME_ROOT): volume},
    cpu=2,
    memory=4096,
    timeout=300,
    min_containers=0,
    max_containers=1,
    scaledown_window=2,
)
def h3_output_audit() -> dict:
    candidates = sorted(
        (VOLUME_ROOT / "results").glob("modal-h3-smoke-*/mirrorvidgen_h3.mp4"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return {"status": "h3_output_audit", "found": False}
    target = candidates[0]
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration,size:stream=index,codec_type,codec_name,width,height,avg_frame_rate,channels",
            "-of", "json", str(target),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "status": "h3_output_audit",
        "found": True,
        "path": str(target),
        "bytes": target.stat().st_size,
        "sha256": digest.hexdigest(),
        "ffprobe_returncode": probe.returncode,
        "ffprobe": json.loads(probe.stdout) if probe.returncode == 0 else None,
        "ffprobe_error": probe.stderr[-2000:] if probe.returncode else None,
    }


@app.function(
    image=worker_image,
    volumes={str(VOLUME_ROOT): volume},
    cpu=2,
    memory=4096,
    timeout=300,
    min_containers=0,
    max_containers=1,
    scaledown_window=2,
)
def volume_fs_probe() -> dict:
    root = VOLUME_ROOT / "modal-fs-probe"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    result = {}
    try:
        result["runtime_python"] = {
            "executable": sys.executable,
            "version": sys.version,
            "configured_freevideo_python": os.environ.get("FREEVIDEO_PYTHON"),
        }

        script = root / "hello.sh"
        script.write_text("#!/bin/sh\necho modal-volume-exec\n", encoding="utf-8")
        script.chmod(0o755)
        result["execute"] = subprocess.run(
            [str(script)], capture_output=True, text=True, timeout=10
        ).stdout.strip()

        hardlink = root / "hello-hardlink.sh"
        try:
            os.link(script, hardlink)
            result["hardlink"] = {
                "supported": hardlink.stat().st_ino == script.stat().st_ino,
            }
        except OSError as error:
            result["hardlink"] = {
                "supported": False,
                "errno": error.errno,
                "error": str(error),
            }

        symlink = root / "hello-symlink.sh"
        try:
            symlink.symlink_to(script.name)
            result["symlink"] = {
                "supported": symlink.resolve() == script.resolve(),
            }
        except OSError as error:
            result["symlink"] = {
                "supported": False,
                "errno": error.errno,
                "error": str(error),
            }

        venv = root / "venv"
        try:
            created = subprocess.run(
                [sys.executable, "-m", "venv", "--copies", str(venv)],
                capture_output=True, text=True, timeout=120,
            )
            result["stdlib_venv"] = {
                "returncode": created.returncode,
                "stdout": created.stdout[-2000:],
                "stderr": created.stderr[-4000:],
                "python": (venv / "bin" / "python").is_file(),
            }
        except Exception as error:
            result["stdlib_venv"] = {
                "returncode": None,
                "error": repr(error),
                "python": (venv / "bin" / "python").is_file(),
            }

        if (venv / "bin" / "python").is_file():
            pip_result = subprocess.run(
                [
                    str(venv / "bin" / "python"), "-m", "pip", "install",
                    "--disable-pip-version-check", "packaging==24.2",
                ],
                capture_output=True, text=True, timeout=180,
            )
            pip_check = subprocess.run(
                [
                    str(venv / "bin" / "python"), "-c",
                    "import packaging; print(packaging.__version__)",
                ],
                capture_output=True, text=True, timeout=30,
            )
            result["pip_into_stdlib_venv"] = {
                "returncode": pip_result.returncode,
                "stdout": pip_result.stdout[-4000:],
                "stderr": pip_result.stderr[-8000:],
                "installed_version": pip_check.stdout.strip(),
                "import_returncode": pip_check.returncode,
            }

        uv_candidates = sorted((VOLUME_ROOT / "freevideo-h3").glob("**/uv"))
        result["uv_candidates"] = [str(path) for path in uv_candidates[:10]]
        if uv_candidates:
            env = dict(
                os.environ,
                UV_PYTHON_INSTALL_DIR=str(root / "uv-python"),
                UV_CACHE_DIR=str(root / "uv-cache"),
                UV_LINK_MODE="copy",
                UV_NO_CONFIG="1",
                RUST_BACKTRACE="1",
            )
            uv_pip_result = subprocess.run(
                [
                    str(uv_candidates[0]), "pip", "install", "--python",
                    str(venv / "bin" / "python"), "--link-mode", "copy",
                    "packaging==24.2",
                ],
                env=env, capture_output=True, text=True, timeout=180,
            )
            packaging_check = subprocess.run(
                [
                    str(venv / "bin" / "python"), "-c",
                    "import packaging; print(packaging.__version__)",
                ],
                capture_output=True, text=True, timeout=30,
            )
            result["uv_pip_into_stdlib_venv"] = {
                "returncode": uv_pip_result.returncode,
                "stdout": uv_pip_result.stdout[-4000:],
                "stderr": uv_pip_result.stderr[-8000:],
                "installed_version": packaging_check.stdout.strip(),
                "import_returncode": packaging_check.returncode,
            }

            uv_venv = root / "uv-venv"
            uv_venv_result = subprocess.run(
                [
                    str(uv_candidates[0]), "venv", "--seed", "--python",
                    os.environ.get("FREEVIDEO_PYTHON", sys.executable), str(uv_venv),
                ],
                env=env, capture_output=True, text=True, timeout=180,
            )
            result["uv_venv_from_image_python"] = {
                "returncode": uv_venv_result.returncode,
                "stdout": uv_venv_result.stdout[-4000:],
                "stderr": uv_venv_result.stderr[-8000:],
                "python": (uv_venv / "bin" / "python").is_file(),
            }

            uv_result = subprocess.run(
                [
                    str(uv_candidates[0]), "--verbose", "python", "install",
                    "--no-bin", "--no-registry", "3.12.3",
                ],
                env=env, capture_output=True, text=True, timeout=180,
            )
            result["uv_python_install"] = {
                "returncode": uv_result.returncode,
                "stdout": uv_result.stdout[-4000:],
                "stderr": uv_result.stderr[-8000:],
            }
        return result
    finally:
        shutil.rmtree(root, ignore_errors=True)


@app.function(
    image=worker_image,
    gpu=GPU_KIND,
    min_containers=0,
    max_containers=1,
    scaledown_window=2,
    timeout=120,
)
def gpu_identity() -> dict:
    row = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    name, memory_mib, driver = [part.strip() for part in row.split(",", 2)]
    result = {
        "gpu_name": name,
        "memory_total_mib": int(memory_mib),
        "driver_version": driver,
    }
    if "RTX PRO 6000" not in name.upper() or result["memory_total_mib"] < 90000:
        raise RuntimeError(f"Unsupported GPU allocation: {result}")
    return result


@app.function(image=control_image, volumes={str(VOLUME_ROOT): volume}, timeout=300)
def submit(payload: dict, webhook: str = "", site_access_bearer: str = "") -> dict:
    action = str(payload.get("action") or "").strip().lower()
    job_id = f"modal-{uuid.uuid4().hex}"
    _write_status(job_id, {"id": job_id, "status": "IN_QUEUE"})
    if action == "mirrorvision_render":
        call = native_job.spawn(job_id, payload, webhook, site_access_bearer)
    elif action == "capabilities":
        call = maintenance_job.spawn(job_id, payload)
    else:
        call = h3_job.spawn(job_id, payload, webhook)
    state = {
        "id": job_id,
        "status": "IN_QUEUE",
        "functionCallId": call.object_id,
    }
    _write_status(job_id, state)
    return state


@app.function(image=control_image, volumes={str(VOLUME_ROOT): volume}, timeout=60)
def status(job_id: str) -> dict:
    volume.reload()
    path = STATUS_ROOT / f"{job_id}.json"
    if not path.is_file():
        return {"id": job_id, "status": "NOT_FOUND"}
    return json.loads(path.read_text(encoding="utf-8"))


@app.function(image=control_image, volumes={str(VOLUME_ROOT): volume}, timeout=60)
def cancel(job_id: str) -> dict:
    current = status.remote(job_id)
    call_id = str(current.get("functionCallId") or "")
    if call_id and current.get("status") not in {"COMPLETED", "FAILED", "CANCELLED"}:
        try:
            modal.FunctionCall.from_id(call_id).cancel()
        except Exception as error:
            print(f"Modal cancellation warning for {job_id}: {error}", flush=True)
    state = {"id": job_id, "status": "CANCELLED"}
    _write_status(job_id, state)
    return state


@app.function(
    image=control_image,
    cpu=1,
    memory=1024,
    timeout=300,
    min_containers=0,
    max_containers=1,
    scaledown_window=2,
)
@modal.fastapi_endpoint(method="POST")
def api(request: dict) -> dict:
    from fastapi import HTTPException

    site_access_bearer = str(request.get("siteAccessBearer") or "").strip()
    if not _private_site_bearer_allowed(site_access_bearer):
        raise HTTPException(status_code=401, detail="Private Site authentication failed.")

    operation = str(request.get("op") or "").strip().lower()
    if operation == "health":
        return {
            "status": "healthy",
            "build": MIGRATION_BUILD,
            "provider": "modal",
            "gpu": GPU_KIND,
            "gpuSubmissionEnabled": False,
            "workers": {
                "idle": 0, "initializing": 0, "ready": 0,
                "running": 0, "throttled": 0, "unhealthy": 0,
            },
            "jobs": {"inQueue": 0, "inProgress": 0, "completed": 0, "failed": 0},
        }
    if operation == "run":
        payload = request.get("input")
        if not isinstance(payload, dict):
            raise HTTPException(status_code=422, detail="A worker input object is required.")
        if str(payload.get("action") or "").strip().lower() not in {"mirrorvision_render", "capabilities"}:
            raise HTTPException(
                status_code=423,
                detail="GPU submissions are disabled on the isolated Modal website preview.",
            )
        webhook = str(request.get("webhook") or "")
        if webhook and not _callback_allowed(webhook):
            raise HTTPException(status_code=422, detail="The callback origin is not allowed.")
        return submit.remote(payload, webhook, site_access_bearer)
    if operation == "status":
        job_id = str(request.get("id") or "")
        return status.remote(job_id)
    if operation == "cancel":
        job_id = str(request.get("id") or "")
        return cancel.remote(job_id)
    if operation == "runsync":
        payload = request.get("input")
        if not isinstance(payload, dict):
            raise HTTPException(status_code=422, detail="A worker input object is required.")
        action = str(payload.get("action") or "").strip().lower()
        if action not in {
            "capabilities", "mirrorvision_health", "h3_runtime_audit", "h3_ref2va_status"
        }:
            raise HTTPException(
                status_code=423,
                detail="This preview permits read-only checks and native CPU rendering only.",
            )
        return maintenance_job.remote(f"modal-sync-{uuid.uuid4().hex}", payload)
    raise HTTPException(status_code=404, detail="Unsupported Modal staging operation.")


@app.local_entrypoint()
def main(command: str = "identity"):
    if command == "identity":
        print(json.dumps(gpu_identity.remote(), sort_keys=True))
        return
    if command == "native-health":
        result = native_job.remote(
            "modal-native-health",
            {"action": "mirrorvision_health"},
        )
        print(json.dumps(result, sort_keys=True))
        return
    if command == "h3-audit":
        result = {
            "runtime": maintenance_job.remote(
                "modal-h3-runtime-audit",
                {"action": "h3_runtime_audit"},
            ),
            "ref2va": maintenance_job.remote(
                "modal-h3-ref2va-status",
                {"action": "h3_ref2va_status"},
            ),
        }
        print(json.dumps(result, sort_keys=True))
        return
    if command == "h3-plan":
        result = h3_job.remote(
            "modal-h3-setup-plan",
            {"action": "h3_setup_plan"},
        )
        print(json.dumps(result, sort_keys=True))
        return
    if command == "fs-probe":
        print(json.dumps(volume_fs_probe.remote(), sort_keys=True))
        return
    if command == "h3-bootstrap":
        result = h3_bootstrap_job.remote(
            "modal-h3-bootstrap",
            {"action": "h3_install", "accept_model_license": True},
        )
        print(json.dumps(result, sort_keys=True))
        return
    if command == "h3-output-audit":
        print(json.dumps(h3_output_audit.remote(), sort_keys=True))
        return
    if command == "h3-smoke":
        from PIL import Image, ImageDraw

        image = Image.new("RGB", (256, 256), "#13040b")
        draw = ImageDraw.Draw(image)
        draw.ellipse((52, 52, 204, 204), fill="#e5195b", outline="#ffd18a", width=8)
        draw.rectangle((40, 192, 216, 216), fill="#ffe3a8")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        job_id = "modal-h3-smoke-" + uuid.uuid4().hex[:12]
        result = h3_job.remote(
            job_id,
            {
                "engine": "freevideo_h3",
                "mode": "h3_i2v",
                "prompt": (
                    "Preserve the supplied abstract artwork exactly. Animate the red circular "
                    "light with one gentle cinematic pulse and subtle camera depth. Do not add "
                    "people, animals, lettering, logos, or unrelated objects."
                ),
                "duration_seconds": 1.625,
                "fps": 24,
                "width": 256,
                "height": 256,
                "quality": "smoke",
                "seed": 136,
                "images": [{"name": "modal-smoke.png", "role": "first", "data": encoded}],
            },
        )
        output = result.get("output") if isinstance(result, dict) else None
        if isinstance(output, dict):
            for item in output.get("outputs") or []:
                if isinstance(item, dict) and isinstance(item.get("data"), str):
                    item["data"] = "<omitted from CI log>"
        print(json.dumps(result, sort_keys=True))
        return
    raise ValueError(
        "command must be identity, native-health, h3-audit, h3-plan, fs-probe, "
        "h3-bootstrap, h3-output-audit or h3-smoke"
    )

