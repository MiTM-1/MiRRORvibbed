"""Candidate-only wrapper. Normal requests call the original handler unchanged.

No add-on paid generation is enabled by default. No installer is called here.
The lock scopes the baseline builder substitution across preflight/generation.
"""
from __future__ import annotations
import base64
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading

import requests
import runpod
import h3_baseline_handler as baseline
from h3_addons import AddonUnavailable, Options, adapt_ref2va, validate_addon_schema
from h3_context_store import ContextStore

_LOCK = threading.RLock()


def _decoded_frames(result):
    output = result["outputs"][0]
    if output.get("type") == "base64":
        blob = base64.b64decode(output["data"], validate=True)
    elif output.get("type") == "network_path":
        blob = Path(output["data"]).read_bytes()
    else:
        # An upload may have succeeded already. Keep it available even if this
        # separate verification cannot fetch it; caller reports context error.
        response = requests.get(output["data"], timeout=(30, 120))
        response.raise_for_status()
        blob = response.content
    with tempfile.TemporaryDirectory(prefix="h3-addon-probe-") as folder:
        video = Path(folder) / "result.mp4"
        video.write_bytes(blob)
        probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames", "-of", "json", str(video)],
            check=True, capture_output=True, text=True, timeout=120)
        return int(json.loads(probe.stdout)["streams"][0]["nb_read_frames"])


def handler(job):
    data = job.get("input") or {}
    try:
        options = Options.from_input(data)
        if not options.requested:
            with _LOCK:
                return baseline.handler(job)
        if not baseline.is_ref2va_request(data):
            raise AddonUnavailable("ComfyUI add-ons are not validated for existing FreeVideo FL2VA")
        preflight = data.get("action") in {"preflight", "h3_ref2va_preflight"}
        if not preflight and os.environ.get("MIRRORVIDGEN_H3_ADDONS_TEST") != "true":
            raise AddonUnavailable("Paid add-on generation is disabled pending approved compatibility tests")
        if options.motion and options.speed != "normal" and os.environ.get("MIRRORVIDGEN_H3_ADDONS_COMBINED_TEST") != "true":
            raise AddonUnavailable("Test Motion Context and Turbo independently before combined tests")
        store = ContextStore()
        source_manifest, latent = None, None
        if options.source_job_id:
            if options.source_job_id == job.get("id"):
                raise AddonUnavailable("A continuation must use a new job ID")
            source_manifest, latent = store.stage(options.source_job_id, job["id"])
        settings_holder = {}
        with _LOCK:
            original = baseline.build_ref2va_workflow
            def build(job_input, prepared):
                graph, settings = original(job_input, prepared)
                graph, settings = adapt_ref2va(graph, settings, options, job_id=job["id"],
                    source_manifest=source_manifest, staged_latent=latent)
                response = requests.get(f"http://{baseline.COMFY_HOST}/object_info", timeout=60)
                response.raise_for_status()
                validate_addon_schema(graph, response.json())
                settings_holder.update(settings)
                return graph, settings
            baseline.build_ref2va_workflow = build
            try:
                result = baseline.run_h3_ref2va_job(job, data, preflight_only=preflight)
            finally:
                baseline.build_ref2va_workflow = original
                if latent:
                    # Only this request's staged copy; durable source is kept.
                    (store.output / latent).unlink(missing_ok=True)
        if result.get("status") == "success":
            dna = result.get("settings", {}).get("generation_dna", {})
            dna["addons"] = settings_holder["addons"]
            if options.motion:
                try:
                    result["context_checkpoint"] = store.commit(job["id"], settings_holder, _decoded_frames(result))
                except Exception as error:
                    # Preserve the completed video; do not regenerate or mask it.
                    result["context_checkpoint_error"] = type(error).__name__
                    result["context_ready"] = False
                else:
                    result["context_ready"] = True
        return result
    except Exception as error:
        return {"error": str(error), "error_type": "addon_unavailable"}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
