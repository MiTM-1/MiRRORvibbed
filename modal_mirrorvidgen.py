"""Isolated Modal runtime for the existing MiRRORvidgen v136 worker.

This file deliberately does not alter or route production traffic.  It keeps
the worker's established input/output contract while replacing RunPod's
container lifecycle with Modal Functions and a persistent Modal Volume.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import modal
import requests


APP_NAME = "mirrorvidgen-v136-staging"
MIGRATION_BUILD = "v136-modal-staging-2"
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


def _load_handler():
    if "/" not in sys.path:
        sys.path.insert(0, "/")
    from handler import handler

    return handler


def _write_status(job_id: str, data: dict) -> None:
    STATUS_ROOT.mkdir(parents=True, exist_ok=True)
    target = STATUS_ROOT / f"{job_id}.json"
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    temp.replace(target)
    volume.commit()


def _execute(job_id: str, payload: dict, runtime: str) -> dict:
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
def native_job(job_id: str, payload: dict) -> dict:
    if str(payload.get("action") or "").lower() not in {"mirrorvision_render", "mirrorvision_health"}:
        raise ValueError("The CPU renderer only accepts MiRRORvision native actions")
    return _execute(job_id, payload, "native-cpu")


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
def h3_job(job_id: str, payload: dict) -> dict:
    if _requires_comfy(payload):
        _ensure_comfy()
    return _execute(job_id, payload, "h3-gpu")


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
    if action not in {"h3_runtime_audit", "h3_ref2va_status"}:
        raise ValueError("The CPU maintenance function only accepts read-only audit actions")
    return _execute(job_id, payload, "maintenance-cpu")


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


@app.function(image=worker_image, volumes={str(VOLUME_ROOT): volume}, timeout=300)
def submit(payload: dict) -> dict:
    action = str(payload.get("action") or "").strip().lower()
    job_id = f"modal-{uuid.uuid4().hex}"
    _write_status(job_id, {"id": job_id, "status": "IN_QUEUE"})
    if action == "mirrorvision_render":
        call = native_job.spawn(job_id, payload)
    else:
        call = h3_job.spawn(job_id, payload)
    state = {
        "id": job_id,
        "status": "IN_QUEUE",
        "functionCallId": call.object_id,
    }
    _write_status(job_id, state)
    return state


@app.function(image=worker_image, volumes={str(VOLUME_ROOT): volume}, timeout=60)
def status(job_id: str) -> dict:
    volume.reload()
    path = STATUS_ROOT / f"{job_id}.json"
    if not path.is_file():
        return {"id": job_id, "status": "NOT_FOUND"}
    return json.loads(path.read_text(encoding="utf-8"))


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
    if command == "h3-bootstrap":
        result = h3_bootstrap_job.remote(
            "modal-h3-bootstrap",
            {"action": "h3_install", "accept_model_license": True},
        )
        print(json.dumps(result, sort_keys=True))
        return
    raise ValueError("command must be identity, native-health, h3-audit, h3-plan or h3-bootstrap")

