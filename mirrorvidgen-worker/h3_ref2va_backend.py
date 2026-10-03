"""Official/ComfyUI MiniMax H3 Ref2VA model management for MiRRORvidgen.

This module intentionally lives beside, not inside, the existing FreeVideo/VDN
FL2VA backend.  It only manages the persistent ComfyUI model files required by
the official ComfyUI MiniMax H3 Ref2VA workflow and never deletes or rewrites
the FreeVideo installation.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import time

import requests


MODEL_ROOT = Path(os.environ.get("MIRRORVIDGEN_MODEL_ROOT", "/runpod-volume/models"))

# These are the model files referenced by Comfy-Org's current MiniMax H3
# reference-to-video workflow template.  The Turbo LoRA is optional and is
# deliberately excluded from the core install until the base Ref2VA path has
# been proven on this worker.
REF2VA_MODELS = {
    "video_vae": {
        "required": True,
        "folder": "vae",
        "name": "minimax_h3_video_vae_int8_convrot.safetensors",
        "url": "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_int8_convrot.safetensors",
        "min_bytes": 2_000_000_000,
    },
    "audio_vae": {
        "required": True,
        "folder": "vae",
        "name": "minimax_h3_audio_vae_fp32.safetensors",
        "url": "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors",
        "min_bytes": 400_000_000,
    },
    "ref2va_model": {
        "required": True,
        "folder": "diffusion_models",
        "name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        "url": "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        "min_bytes": 15_000_000_000,
    },
    "text_encoder": {
        "required": True,
        "folder": "text_encoders",
        "name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "url": "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "min_bytes": 10_000_000_000,
    },
    "turbo_lora": {
        "required": False,
        "folder": "loras",
        "name": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
        "url": "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
        "min_bytes": 1_000_000_000,
    },
}


def _target(spec):
    return MODEL_ROOT / spec["folder"] / spec["name"]


def _present(spec):
    path = _target(spec)
    try:
        size = path.stat().st_size
    except OSError:
        return False, None
    return bool(path.is_file() and size >= int(spec["min_bytes"])), size


def h3_ref2va_status():
    models = {}
    required_ready = True
    for key, spec in REF2VA_MODELS.items():
        present, size = _present(spec)
        models[key] = {
            "required": bool(spec["required"]),
            "present": present,
            "bytes": size,
            "path": str(_target(spec)),
            "name": spec["name"],
        }
        if spec["required"] and not present:
            required_ready = False
    return {
        "engine": "comfyui_minimax_h3_ref2va",
        "label": "MiRRORmax H3 Ref2VA",
        "models_ready": required_ready,
        "ready": required_ready,
        "model_root": str(MODEL_ROOT),
        "models": models,
    }


def _remote_size(url):
    try:
        response = requests.head(url, allow_redirects=True, timeout=(20, 60))
        response.raise_for_status()
        value = int(response.headers.get("Content-Length") or 0)
        return value if value > 0 else None
    except Exception:
        return None


def h3_ref2va_install_plan(include_turbo=False):
    selected = [
        (key, spec) for key, spec in REF2VA_MODELS.items()
        if spec["required"] or (include_turbo and key == "turbo_lora")
    ]
    rows = []
    required_bytes = 0
    for key, spec in selected:
        present, size = _present(spec)
        remote_size = None if present else _remote_size(spec["url"])
        estimate = remote_size or int(spec["min_bytes"] * 1.20)
        if not present:
            required_bytes += estimate
        rows.append({
            "key": key,
            "name": spec["name"],
            "path": str(_target(spec)),
            "present": present,
            "existing_bytes": size,
            "remote_bytes": remote_size,
            "estimated_download_bytes": 0 if present else estimate,
            "required": bool(spec["required"]),
        })

    usage = shutil.disk_usage(MODEL_ROOT.parent if MODEL_ROOT.parent.exists() else Path("/runpod-volume"))
    return {
        "status": "h3_ref2va_install_plan",
        "read_only": True,
        "include_turbo": include_turbo,
        "models": rows,
        "estimated_missing_bytes": required_bytes,
        "free_bytes": usage.free,
        "enough_space": usage.free > required_bytes + 10_000_000_000,
        "model_root": str(MODEL_ROOT),
    }


def _download_atomic(url, target, min_bytes):
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    existing = part.stat().st_size if part.is_file() else 0
    headers = {}
    mode = "wb"
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"

    response = requests.get(
        url,
        stream=True,
        allow_redirects=True,
        headers=headers,
        timeout=(30, 300),
    )
    if existing and response.status_code != 206:
        existing = 0
        mode = "wb"
    response.raise_for_status()

    with part.open(mode) as handle:
        for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
            if chunk:
                handle.write(chunk)
        handle.flush()
        os.fsync(handle.fileno())

    size = part.stat().st_size
    if size < int(min_bytes):
        raise RuntimeError(
            f"Downloaded file is unexpectedly small: {target.name} ({size} bytes)"
        )
    os.replace(part, target)
    return size


def h3_ref2va_install(*, accept_model_license=False, include_turbo=False):
    if not accept_model_license:
        raise ValueError("Ref2VA install requires input.accept_model_license=true")

    plan = h3_ref2va_install_plan(include_turbo=include_turbo)
    if not plan["enough_space"]:
        raise RuntimeError(
            "Not enough persistent storage for the missing Ref2VA model files"
        )

    selected = [
        (key, spec) for key, spec in REF2VA_MODELS.items()
        if spec["required"] or (include_turbo and key == "turbo_lora")
    ]
    started = time.time()
    completed = []
    for key, spec in selected:
        present, size = _present(spec)
        target = _target(spec)
        if present:
            completed.append({
                "key": key,
                "name": spec["name"],
                "status": "already_present",
                "bytes": size,
                "path": str(target),
            })
            continue

        size = _download_atomic(spec["url"], target, spec["min_bytes"])
        completed.append({
            "key": key,
            "name": spec["name"],
            "status": "downloaded",
            "bytes": size,
            "path": str(target),
        })

    status = h3_ref2va_status()
    if not status["models_ready"]:
        raise RuntimeError("Ref2VA model installation finished but required files are still incomplete")

    return {
        "status": "h3_ref2va_installed",
        "seconds": round(time.time() - started, 3),
        "include_turbo": include_turbo,
        "files": completed,
        "ref2va": status,
    }
