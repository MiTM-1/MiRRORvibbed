"""Additive MiRRORvision dispatcher.

The inherited H3 add-on handler remains the authority for every existing action.
Only the new, explicit mirrorvision_render action is routed to the CPU renderer.
"""
from __future__ import annotations

import base64
import json
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import runpod
import h3_addon_handler_existing as inherited
from mirrorvision_renderer import handle as render, extract_boundary

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
        "previousFrameConditioning": True,
        "previousFrameMechanism": "decoded-final-frame-reference",
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
        if action == "generate" and data.get("workflow") == "ref2va" and data.get("boundary_upload"):
            try:
                _upload_boundary(result, data["boundary_upload"])
                result["continuity_frame_stored"] = True
            except Exception:
                # Keep the paid H3 result recoverable; orchestration can retry extraction
                # without regenerating the expensive scene.
                result["continuity_frame_stored"] = False
                result["continuity_error"] = "Final frame upload failed; completed video is preserved"
        if action == "capabilities" and _healthy():
            return _capabilities(result)
        return result
    except Exception as error:
        return {"error": str(error), "error_type": "mirrorvision_render_failed"}


def _output_bytes(result):
    outputs = result.get("outputs") if isinstance(result, dict) else None
    first = outputs[0] if isinstance(outputs, list) and outputs else None
    if not isinstance(first, dict):
        raise RuntimeError("H3 result has no video for continuity extraction")
    kind = str(first.get("type") or "").lower(); value = first.get("data")
    if not isinstance(value, str) or not value:
        raise RuntimeError("H3 result video is unavailable for continuity extraction")
    if kind == "base64":
        return base64.b64decode(value.split(",",1)[-1], validate=True)
    if kind == "network_path":
        return Path(value).read_bytes()
    if kind == "url":
        with urllib.request.urlopen(urllib.request.Request(value, headers={"User-Agent":"MiRRORvidgen-Worker/1.0"}), timeout=180) as response:
            return response.read(200*1024*1024+1)
    raise RuntimeError("Unsupported H3 result transport for continuity extraction")


def _upload_boundary(result, target):
    from urllib.parse import urlparse
    allowed = __import__('os').environ.get('MIRRORVISION_SITE_ORIGIN','https://mirrorvidgen.mitm1.chatgpt.site')
    parsed=urlparse(str(target))
    if f"{parsed.scheme}://{parsed.netloc}" != allowed or not parsed.path.endswith('/boundary'):
        raise ValueError('Untrusted continuity callback URL')
    video=_output_bytes(result)
    if len(video)>200*1024*1024: raise ValueError('H3 result is too large for continuity extraction')
    with tempfile.TemporaryDirectory() as temp:
        source=Path(temp)/'section.mp4'; frame=Path(temp)/'boundary.png';source.write_bytes(video)
        extract_boundary(str(source), str(frame))
        body=frame.read_bytes()
    request=urllib.request.Request(str(target),data=body,headers={'User-Agent':'MiRRORvidgen-Worker/1.0','Content-Type':'image/png'},method='POST')
    with urllib.request.urlopen(request,timeout=90) as response:
        if response.status!=200: raise RuntimeError('Continuity frame storage rejected upload')


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
