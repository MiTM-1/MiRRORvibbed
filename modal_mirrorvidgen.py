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


APP_NAME = "mirrorvidgen-v136-staging"
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
    return _execute(job_id, payload, "h3-gpu")


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
    if command == "h3-bootstrap":
        result = h3_job.remote(
            "modal-h3-bootstrap",
            {"action": "h3_install", "accept_model_license": True},
        )
        print(json.dumps(result, sort_keys=True))
        return
    raise ValueError("command must be identity, native-health, h3-audit or h3-bootstrap")

