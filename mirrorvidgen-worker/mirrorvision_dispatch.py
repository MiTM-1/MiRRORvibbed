"""Additive MiRRORvision dispatcher.

The inherited H3 add-on handler remains the authority for every existing action.
Only the new, explicit mirrorvision_render action is routed to the CPU renderer.
"""
from __future__ import annotations

import shutil
import subprocess

import runpod
import h3_addon_handler_existing as inherited
from mirrorvision_renderer import handle as render

VERSION = 1


def _healthy():
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def _capabilities(result):
    if not isinstance(result, dict):
        return result
    capabilities = result.setdefault("capabilities", {})
    capabilities["mirrorvision"] = {
        "rendererVersion": VERSION,
        "native2d": True,
        "advanced3d": False,
        "serverAssembly": True,
        "exactTypography": True,
        "selectedAudioAnalysis": True,
        "previousFrameConditioning": False,
    }
    return result


def handler(job):
    data = job.get("input") or {}
    try:
        action = str(data.get("action") or "").strip().lower()
        if action == "mirrorvision_render":
            if not _healthy():
                raise RuntimeError("MiRRORvision renderer dependencies are unavailable")
            return render(data)
        result = inherited.handler(job)
        if action == "capabilities" and _healthy():
            return _capabilities(result)
        return result
    except Exception as error:
        return {"error": str(error), "error_type": "mirrorvision_render_failed"}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
