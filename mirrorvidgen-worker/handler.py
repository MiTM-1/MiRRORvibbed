import base64
import hashlib
import json
import mimetypes
import os
import subprocess
import time
import traceback
from pathlib import Path

import requests

from ltx25_workflow import (
    CapabilityUnavailable,
    build_ltx25_continuation,
    build_ltx25_director,
    build_ltx25_first_last,
    build_ltx25_i2v,
    build_ltx25_ingredients,
    build_ltx25_motion_track,
    build_ltx25_t2v,
    build_ltx25_union_control,
    validate_workflow_inputs,
    worker_capabilities,
)
import runpod

COMFY_HOST = os.environ.get("COMFY_HOST", "127.0.0.1:8188")
POLL_INTERVAL = float(os.environ.get("MIRRORVIDGEN_POLL_INTERVAL", "1"))
POLL_TIMEOUT = int(os.environ.get("MIRRORVIDGEN_POLL_TIMEOUT", "3500"))
MAX_INLINE_VIDEO_BYTES = int(os.environ.get("MIRRORVIDGEN_MAX_INLINE_VIDEO_BYTES", "15000000"))
NETWORK_RESULTS = Path("/runpod-volume/results")


def wait_for_comfy():
    deadline = time.time() + 300
    last_error = None
    while time.time() < deadline:
        try:
            r = requests.get(f"http://{COMFY_HOST}/system_stats", timeout=5)
            if r.ok:
                return True
        except Exception as e:
            last_error = e
        time.sleep(1)
    raise RuntimeError(f"ComfyUI did not become ready: {last_error}")


def upload_image(name, encoded):
    if "," in encoded:
        encoded = encoded.split(",", 1)[1]
    blob = base64.b64decode(encoded)
    files = {"image": (name, blob, "application/octet-stream")}
    data = {"overwrite": "true"}
    r = requests.post(f"http://{COMFY_HOST}/upload/image", files=files, data=data, timeout=120)
    r.raise_for_status()


def download_asset(name, url):
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    target = Path("/comfyui/input") / Path(name).name
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


def write_encoded_asset(name, encoded):
    """Materialise an image/video data URL in ComfyUI's input directory."""
    if "," in encoded:
        encoded = encoded.split(",", 1)[1]
    blob = base64.b64decode(encoded, validate=True)
    target = Path("/comfyui/input") / Path(name).name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(blob)


def materialize_assets(job_input):
    """Write browser-provided references/videos into ComfyUI input safely."""
    for image in job_input.get("images") or []:
        encoded = image.get("image")
        if encoded:
            upload_image(image["name"], encoded)
        elif image.get("url"):
            download_asset(image["name"], image["url"])
        else:
            raise ValueError("Reference image data is missing")

    for asset in job_input.get("assets") or []:
        if asset.get("url") and asset.get("name"):
            download_asset(asset["name"], asset["url"])
        elif asset.get("data") and asset.get("name"):
            write_encoded_asset(asset["name"], asset["data"])
        elif asset.get("image") and asset.get("name"):
            write_encoded_asset(asset["name"], asset["image"])

    # Continuation payloads may carry the source as a dedicated object rather
    # than inside ``assets``.  Keep the same safe basename handling.
    for key in ("source_video", "video"):
        item = job_input.get(key)
        if not isinstance(item, dict) or not item.get("name"):
            continue
        name = Path(str(item["name"])).name
        if item.get("url"):
            download_asset(name, item["url"])
        elif item.get("data"):
            write_encoded_asset(name, item["data"])


def materialize_source_only(job_input, source_name):
    """Materialise only a continuation source for no-GPU preflight."""
    target = Path("/comfyui/input") / Path(source_name).name
    if target.is_file() and target.stat().st_size > 0:
        return target
    candidates = []
    for key in ("source_video", "video"):
        item = job_input.get(key)
        if isinstance(item, dict):
            candidates.append(item)
    candidates.extend(item for item in (job_input.get("assets") or []) if isinstance(item, dict))
    for item in candidates:
        if Path(str(item.get("name") or "")).name != target.name:
            continue
        if item.get("url"):
            download_asset(target.name, item["url"])
            break
        if item.get("data"):
            write_encoded_asset(target.name, item["data"])
            break
    if not target.is_file() or target.stat().st_size == 0:
        raise ValueError("The completed source MP4 is not accessible on the worker")
    return target


def validate_source_video(path):
    """Return basic source metadata without invoking ComfyUI or RunPod."""
    try:
        duration = _probe_duration(path)
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height,r_frame_rate", "-of", "json", str(path)],
            check=True, capture_output=True, text=True, timeout=120,
        )
        streams = json.loads(probe.stdout).get("streams") or []
        if not streams or duration <= 0:
            raise ValueError
        stream = streams[0]
        return {
            "duration_seconds": duration,
            "width": int(stream.get("width") or 0),
            "height": int(stream.get("height") or 0),
            "frame_rate": str(stream.get("r_frame_rate") or ""),
        }
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("The completed source MP4 metadata could not be read") from error


def normalize_control_video(name, settings, job_id):
    """Prepare a control video at the selected native LTX size/frame count.

    Union Control derives its latent dimensions and frame count from the
    guide video.  Normalising the already-downloaded guide before ComfyUI
    sees it keeps the user's selected aspect, duration, and 24 fps contract
    authoritative without falling back to a prompt-only workflow.
    """
    source = Path("/comfyui/input") / Path(name).name
    if not source.is_file():
        raise ValueError("Union Control guide video could not be materialised")
    width = int(settings.get("stage1_width") or 0)
    height = int(settings.get("stage1_height") or 0)
    frames = int(settings.get("frames") or 0)
    fps = int(settings.get("fps") or 24)
    if not width or not height or not frames or fps != 24:
        raise ValueError("Union Control guide dimensions are invalid")
    job_token = "".join(ch if str(ch).isalnum() or ch in {"-", "_"} else "_" for ch in str(job_id))
    target_name = f"mirrorvidgen_control_{Path(name).stem}_{job_token}.mp4"
    target = Path("/comfyui/input") / target_name
    # A clone pad makes a short guide long enough for the requested clip;
    # -frames:v then guarantees the valid LTX frame count exactly.
    vf = (
        f"fps={fps},scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,"
        f"tpad=stop_mode=clone:stop_duration={max(1, frames / fps):.3f}"
    )
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source), "-vf", vf, "-frames:v", str(frames),
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(target),
    ]
    try:
        subprocess.run(command, check=True, timeout=900, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise CapabilityUnavailable("Union Control requires ffmpeg on the worker") from error
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or "").strip()[-1200:]
        raise ValueError(f"Union Control guide video could not be prepared: {detail}") from error
    if not target.is_file() or target.stat().st_size == 0:
        raise ValueError("Union Control guide preparation produced no video")
    return target_name


def normalize_continuation_tail(name, settings, job_id):
    """Extract a real ending context for the official LTX V2V graph."""
    source = Path("/comfyui/input") / Path(name).name
    if not source.is_file():
        raise ValueError("The completed source MP4 could not be materialised")
    width = int(settings.get("stage1_width") or 0)
    height = int(settings.get("stage1_height") or 0)
    frames = int(settings.get("context_frames") or 0)
    fps = int(settings.get("fps") or 24)
    context_seconds = float(settings.get("context_duration_seconds") or 0)
    if not width or not height or not frames or fps != 24 or context_seconds <= 0:
        raise ValueError("Continuation source context is invalid")
    job_token = "".join(ch if str(ch).isalnum() or ch in {"-", "_"} else "_" for ch in str(job_id))
    target_name = f"mirrorvidgen_tail_{Path(name).stem}_{job_token}.mp4"
    target = Path("/comfyui/input") / target_name
    vf = (
        f"fps={fps},scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,"
        f"tpad=stop_mode=clone:stop_duration={max(1, context_seconds):.3f}"
    )
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-sseof", f"-{max(1, context_seconds):.3f}", "-i", str(source),
        "-vf", vf, "-frames:v", str(frames), "-map", "0:v:0", "-map", "0:a?",
        "-c:v", "libx264", "-c:a", "aac", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(target),
    ]
    try:
        subprocess.run(command, check=True, timeout=900, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise CapabilityUnavailable("Video continuation requires ffmpeg on the worker") from error
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or "").strip()[-1200:]
        raise ValueError(f"Continuation source context could not be prepared: {detail}") from error
    if not target.is_file() or target.stat().st_size == 0:
        raise ValueError("Continuation source preparation produced no video")
    return target_name


def _probe_duration(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True, capture_output=True, text=True, timeout=120,
    )
    return float(result.stdout.strip())


def _has_audio(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=index", "-of", "csv=p=0", str(path)],
        check=False, capture_output=True, text=True, timeout=120,
    )
    return bool(result.stdout.strip())


def stitch_continuation(source_name, segment_blob, settings, job_id):
    """Join the preserved source with the newly generated segment.

    The source remains untouched.  The returned file is a new self-contained
    result; identical source bytes or a non-longer output are rejected.
    """
    source = Path("/comfyui/input") / Path(source_name).name
    if not source.is_file():
        raise ValueError("Continuation source video is unavailable")
    segment_token = "".join(ch if str(ch).isalnum() or ch in {"-", "_"} else "_" for ch in str(job_id))
    segment = Path("/comfyui/input") / f"mirrorvidgen_segment_{segment_token}.mp4"
    segment.write_bytes(segment_blob)
    if hashlib.sha256(source.read_bytes()).digest() == hashlib.sha256(segment_blob).digest():
        raise RuntimeError("Continuation failed validation: generated segment is byte-identical to the source")
    target = Path("/comfyui/input") / f"mirrorvidgen_continued_{segment_token}.mp4"
    list_file = Path("/tmp") / f"mirrorvidgen_concat_{segment_token}.txt"
    list_file.write_text(
        "file '" + str(source).replace("'", "'\\''") + "'\n" +
        "file '" + str(segment).replace("'", "'\\''") + "'\n",
        encoding="utf-8",
    )
    with_audio = _has_audio(source) and _has_audio(segment)
    if with_audio:
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat",
            "-safe", "0", "-i", str(list_file), "-c:v", "libx264", "-preset",
            "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac",
            "-movflags", "+faststart", str(target),
        ]
    else:
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat",
            "-safe", "0", "-i", str(list_file), "-c:v", "libx264", "-preset",
            "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-an",
            "-movflags", "+faststart", str(target),
        ]
    try:
        subprocess.run(command, check=True, timeout=1200, capture_output=True, text=True)
        source_duration = _probe_duration(source)
        combined_duration = _probe_duration(target)
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError) as error:
        detail = getattr(error, "stderr", "") or str(error)
        raise RuntimeError(f"Continuation stitching failed: {str(detail)[-1200:]}") from error
    if combined_duration <= source_duration + 0.25:
        raise RuntimeError("Continuation failed validation: final video is not longer than the source")
    settings["combined_duration_seconds"] = combined_duration
    settings["audio_stitched"] = with_audio
    return target.name, target.read_bytes()


def queue_workflow(workflow):
    r = requests.post(
        f"http://{COMFY_HOST}/prompt",
        json={"prompt": workflow},
        timeout=60,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"ComfyUI workflow validation failed ({r.status_code}): {r.text[:8000]}")
    data = r.json()
    prompt_id = data.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"ComfyUI returned no prompt_id: {data}")
    return prompt_id


def wait_for_history(prompt_id):
    deadline = time.time() + POLL_TIMEOUT
    while time.time() < deadline:
        r = requests.get(f"http://{COMFY_HOST}/history/{prompt_id}", timeout=30)
        r.raise_for_status()
        data = r.json()
        if prompt_id in data:
            entry = data[prompt_id]
            status = entry.get("status", {})
            if status.get("completed"):
                return entry
            msgs = status.get("messages", [])
            for msg in msgs:
                if isinstance(msg, (list, tuple)) and msg and msg[0] == "execution_error":
                    raise RuntimeError(json.dumps(msg[1] if len(msg) > 1 else msg))
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(f"Generation exceeded {POLL_TIMEOUT} seconds")


def iter_output_files(value):
    if isinstance(value, dict):
        if "filename" in value:
            yield value
        for v in value.values():
            yield from iter_output_files(v)
    elif isinstance(value, list):
        for v in value:
            yield from iter_output_files(v)


def fetch_output_file(info):
    filename = info.get("filename")
    if not filename:
        return None, None
    file_type = info.get("type", "output")
    if file_type == "temp":
        return None, None
    params = {
        "filename": filename,
        "subfolder": info.get("subfolder", ""),
        "type": file_type,
    }
    r = requests.get(f"http://{COMFY_HOST}/view", params=params, timeout=600)
    r.raise_for_status()
    return filename, r.content


def first_video_from_history(history):
    """Return the first non-temporary video emitted by a Comfy history entry."""
    for node_output in (history.get("outputs") or {}).values():
        for info in iter_output_files(node_output):
            filename = str(info.get("filename") or "")
            if Path(filename).suffix.lower() not in {".mp4", ".webm", ".mov", ".mkv", ".gif"}:
                continue
            filename, blob = fetch_output_file(info)
            if filename and blob is not None:
                return filename, blob
    raise RuntimeError("Director workflow produced no video output")


def put_result(upload, blob, content_type):
    url = upload.get("url")
    if not url:
        return None
    headers = dict(upload.get("headers") or {})
    headers.setdefault("Content-Type", content_type)
    r = requests.put(url, data=blob, headers=headers, timeout=600)
    r.raise_for_status()
    return upload.get("public_url") or url.split("?", 1)[0]


def persist_large_video(job_id, filename, blob):
    folder = NETWORK_RESULTS / job_id
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / Path(filename).name
    target.write_bytes(blob)
    return str(target)


def run_director(job, job_input):
    """Execute one ordered Director generation inside one RunPod job.

    The first segment uses the selected real workflow. Every later segment
    uses the official V2V temporal graph against the previous result tail;
    this is a chain, never a set of independent T2V clips.
    """
    wait_for_comfy()
    materialize_assets(job_input)
    first_workflow, director_settings = build_ltx25_director(job_input)
    # Validate the first graph before occupying the single worker with a
    # paid Director chain. Later segments are checked before their queue call.
    validate_workflow_inputs(first_workflow, label="Director first segment")
    first_prompt_id = queue_workflow(first_workflow)
    first_history = wait_for_history(first_prompt_id)
    _, first_blob = first_video_from_history(first_history)
    job_token = "".join(ch if str(ch).isalnum() or ch in {"-", "_"} else "_" for ch in str(job.get("id", "director")))
    current_name = f"mirrorvidgen_director_{job_token}_segment0.mp4"
    current_path = Path("/comfyui/input") / current_name
    current_path.write_bytes(first_blob)
    current_blob = first_blob
    prompt_ids = [first_prompt_id]
    segments = director_settings.get("director_segments") or []

    for index, segment in enumerate(segments[1:], start=1):
        continuation_input = dict(job_input)
        continuation_input.update({
            "mode": "continue_video",
            "source_video_name": current_name,
            "added_duration_seconds": segment["duration_seconds"],
            "follow_on_prompt": segment.get("prompt") or "",
            "continuation_context_seconds": float(job_input.get("continuation_context_seconds", 2)),
        })
        continuation_workflow, continuation_settings = build_ltx25_continuation(continuation_input)
        validate_workflow_inputs(
            continuation_workflow,
            label=f"Director segment {index + 1}",
        )
        tail_name = normalize_continuation_tail(current_name, continuation_settings, f"{job_token}_{index}")
        for _, node in continuation_workflow.items():
            if node.get("class_type") == "LoadVideo":
                node.setdefault("inputs", {})["video"] = tail_name
        prompt_id = queue_workflow(continuation_workflow)
        prompt_ids.append(prompt_id)
        history = wait_for_history(prompt_id)
        _, segment_blob = first_video_from_history(history)
        current_name, current_blob = stitch_continuation(
            current_name,
            segment_blob,
            continuation_settings,
            f"{job_token}_{index}",
        )
        director_settings.setdefault("segments", []).append({
            "index": index,
            "prompt_id": prompt_id,
            "added_duration_seconds": continuation_settings.get("actual_added_duration_seconds"),
            "context_frames": continuation_settings.get("context_frames"),
        })

    result_upload = job_input.get("result_upload") or {}
    filename = current_name
    content_type = "video/mp4"
    uploaded = put_result(result_upload, current_blob, content_type) if result_upload.get("url") else None
    if uploaded:
        output = {"filename": filename, "kind": "video", "type": "url", "data": uploaded, "bytes": len(current_blob)}
    elif len(current_blob) <= MAX_INLINE_VIDEO_BYTES:
        output = {
            "filename": filename,
            "kind": "video",
            "type": "base64",
            "mime": content_type,
            "data": base64.b64encode(current_blob).decode("utf-8"),
            "bytes": len(current_blob),
        }
    else:
        path = persist_large_video(job.get("id", "director"), filename, current_blob)
        output = {
            "filename": filename,
            "kind": "video",
            "type": "network_path",
            "data": path,
            "bytes": len(current_blob),
            "note": "Provide input.result_upload.url for a browser-accessible result URL.",
        }
    director_settings["final_filename"] = filename
    director_settings["director_prompt_ids"] = prompt_ids
    director_settings["final_actual_duration_seconds"] = _probe_duration(Path("/comfyui/input") / current_name)
    return {
        "status": "success",
        "prompt_id": first_prompt_id,
        "prompt_ids": prompt_ids,
        "outputs": [output],
        "settings": director_settings,
    }


def handler(job):
    try:
        job_input = dict(job.get("input") or {})

        normalized_images = []
        for image in job_input.get("images") or []:
            if not isinstance(image, dict):
                continue
            item = dict(image)
            raw_name = str(item.get("name") or "reference.png")
            item["name"] = Path(raw_name).name or "reference.png"
            normalized_images.append(item)
        job_input["images"] = normalized_images

        normalized_assets = []
        for asset in job_input.get("assets") or []:
            if not isinstance(asset, dict):
                continue
            item = dict(asset)
            raw_name = str(item.get("name") or "asset")
            item["name"] = Path(raw_name).name or "asset"
            normalized_assets.append(item)
        job_input["assets"] = normalized_assets

        action = str(job_input.get("action") or "").strip().lower()
        if action == "capabilities":
            return {"status": "capabilities", "capabilities": worker_capabilities()}

        workflow = job_input.get("workflow")
        generated_settings = None
        mode = str(job_input.get("mode") or "").strip().lower()

        if (
            not isinstance(workflow, dict)
            and action != "preflight"
            and mode in {"director_video", "director_30", "director_30s", "director"}
        ):
            try:
                return run_director(job, job_input)
            except CapabilityUnavailable as error:
                return {"error": str(error), "error_type": "capability_unavailable"}
            except ValueError as error:
                return {"error": str(error), "error_type": "invalid_input"}

        if not isinstance(workflow, dict):
            if action == "preflight":
                # Build and validate the real workflow, but deliberately do
                # not start ComfyUI or queue a paid generation.
                try:
                    workflow, generated_settings = _build_mode(mode, job_input)
                    validate_workflow_inputs(workflow)
                    if mode in {"continue_video", "extend_video"}:
                        source_name = generated_settings.get("source_video_name")
                        source_path = materialize_source_only(job_input, source_name)
                        source_meta = validate_source_video(source_path)
                        generated_settings["source_preflight"] = source_meta
                        # Build the actual ending context too. This is still
                        # ffmpeg-only and proves the temporal input can be
                        # prepared before any paid queue call.
                        generated_settings["preflight_tail_video_name"] = normalize_continuation_tail(
                            source_name, generated_settings, job.get("id", "preflight")
                        )
                    return {
                        "status": "preflight",
                        "ready": True,
                        "mode": mode,
                        "settings": generated_settings,
                        "node_count": len(workflow),
                    }
                except Exception as error:
                    return {
                        "status": "preflight",
                        "ready": False,
                        "mode": mode,
                        "message": str(error),
                        "error_type": type(error).__name__,
                    }
            try:
                workflow, generated_settings = _build_mode(mode, job_input)
                validate_workflow_inputs(workflow)
            except CapabilityUnavailable as error:
                return {"error": str(error), "error_type": "capability_unavailable"}
            except ValueError as error:
                return {"error": str(error), "error_type": "invalid_input"}

        # A caller-supplied prompt map (kept for backwards compatibility)
        # receives the same fail-closed validation as server-built graphs.
        validate_workflow_inputs(workflow)
        wait_for_comfy()

        materialize_assets(job_input)

        # Union Control's official graph takes its size and frame count from
        # the guide video.  Prepare that guide once, after materialisation,
        # so Pose/Depth/Canny use the selected native aspect and duration.
        if generated_settings and generated_settings.get("mode") in {
            "pose_to_video", "depth_to_video", "canny_to_video"
        }:
            guide_name = generated_settings.get("guide_video_name")
            prepared_name = normalize_control_video(guide_name, generated_settings, job.get("id", "job"))
            for _, node in workflow.items():
                if node.get("class_type") == "LoadVideo":
                    node.setdefault("inputs", {})["video"] = prepared_name
            generated_settings["prepared_guide_video_name"] = prepared_name

        if generated_settings and generated_settings.get("mode") in {
            "continue_video", "extend_video"
        }:
            source_name = generated_settings.get("source_video_name")
            tail_name = normalize_continuation_tail(source_name, generated_settings, job.get("id", "job"))
            for _, node in workflow.items():
                if node.get("class_type") == "LoadVideo":
                    node.setdefault("inputs", {})["video"] = tail_name
            generated_settings["prepared_tail_video_name"] = tail_name

        prompt_id = queue_workflow(workflow)
        history = wait_for_history(prompt_id)
        outputs = history.get("outputs") or {}

        results = []
        seen = set()
        result_upload = job_input.get("result_upload") or {}

        for node_output in outputs.values():
            for info in iter_output_files(node_output):
                key = (info.get("subfolder", ""), info.get("filename", ""), info.get("type", "output"))
                if key in seen:
                    continue
                seen.add(key)

                filename, blob = fetch_output_file(info)
                if not filename or blob is None:
                    continue

                ext = Path(filename).suffix.lower()
                content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
                is_video = ext in {".mp4", ".webm", ".mov", ".mkv", ".gif"}

                if (
                    is_video
                    and generated_settings
                    and generated_settings.get("mode") in {"continue_video", "extend_video"}
                    and generated_settings.get("source_video_name")
                ):
                    filename, blob = stitch_continuation(
                        generated_settings["source_video_name"],
                        blob,
                        generated_settings,
                        job.get("id", "job"),
                    )
                    ext = ".mp4"
                    content_type = "video/mp4"

                if is_video:
                    uploaded = None
                    if result_upload.get("url"):
                        uploaded = put_result(result_upload, blob, content_type)
                    if uploaded:
                        results.append({
                            "filename": filename,
                            "kind": "video",
                            "type": "url",
                            "data": uploaded,
                            "bytes": len(blob),
                        })
                    elif len(blob) <= MAX_INLINE_VIDEO_BYTES:
                        results.append({
                            "filename": filename,
                            "kind": "video",
                            "type": "base64",
                            "mime": content_type,
                            "data": base64.b64encode(blob).decode("utf-8"),
                            "bytes": len(blob),
                        })
                    else:
                        path = persist_large_video(job["id"], filename, blob)
                        results.append({
                            "filename": filename,
                            "kind": "video",
                            "type": "network_path",
                            "data": path,
                            "bytes": len(blob),
                            "note": "Provide input.result_upload.url for a browser-accessible result URL.",
                        })
                else:
                    results.append({
                        "filename": filename,
                        "kind": "image",
                        "type": "base64",
                        "mime": content_type,
                        "data": base64.b64encode(blob).decode("utf-8"),
                        "bytes": len(blob),
                    })

        response = {
            "status": "success",
            "prompt_id": prompt_id,
            "outputs": results,
        }
        if generated_settings is not None:
            response["settings"] = generated_settings
        return response

    except Exception as e:
        traceback.print_exc()
        return {
            "error": str(e),
            "error_type": type(e).__name__,
        }


def _build_mode(mode, job_input):
    """Map every paid mode to a real builder; never silently use T2V."""
    if mode in {"ltx25_t2v", "mirror_motion_1", "text_to_video"}:
        return build_ltx25_t2v(job_input)
    if mode in {"ltx25_i2v", "image_to_video", "first_frame_to_video"}:
        return build_ltx25_i2v(job_input)
    if mode in {"reference_to_video", "ingredients_to_video", "character_to_video"}:
        return build_ltx25_ingredients(job_input)
    if mode in {"first_last_frame_to_video", "first_and_last_frame_to_video"}:
        return build_ltx25_first_last(job_input)
    if mode in {"motion_track_to_video", "motion_track"}:
        return build_ltx25_motion_track(job_input)
    if mode in {"pose_to_video", "pose"}:
        return build_ltx25_union_control(job_input, "pose")
    if mode in {"depth_to_video", "depth"}:
        return build_ltx25_union_control(job_input, "depth")
    if mode in {"canny_to_video", "edge_to_video", "canny"}:
        return build_ltx25_union_control(job_input, "canny")
    if mode in {"continue_video", "extend_video"}:
        return build_ltx25_continuation(job_input)
    if mode in {"director_video", "director_30", "director_30s", "director"}:
        return build_ltx25_director(job_input)
    raise CapabilityUnavailable(
        "No connected worker workflow exists for this mode. Unsupported modes never fall back to Text → Video."
    )


if __name__ == "__main__":
    print("MiRRORvidgen RunPod worker starting")
    runpod.serverless.start({"handler": handler})
