"""Official MiniMax H3 Ref2VA generation path for MiRRORvidgen.

This module is deliberately isolated from the existing FreeVideo/VDN FL2VA
path and from LTX-2.5. It validates and materialises Ref2VA references, builds
an explicit ComfyUI prompt graph, and returns only local filenames/metadata.
No signed URLs or secrets are ever copied into generation DNA.
"""
from __future__ import annotations

import base64
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

import requests

from h3_ref2va_backend import h3_ref2va_status


COMFY_INPUT = Path(os.environ.get("MIRRORVIDGEN_COMFY_INPUT", "/comfyui/input"))
TMP_ROOT = Path(os.environ.get("MIRRORVIDGEN_TMP_ROOT", "/tmp"))

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus"}

MAX_IMAGE_BYTES = int(os.environ.get("MIRRORVIDGEN_REF_IMAGE_MAX_BYTES", "50000000"))
MAX_VIDEO_BYTES = int(os.environ.get("MIRRORVIDGEN_REF_VIDEO_MAX_BYTES", "750000000"))
MAX_AUDIO_BYTES = int(os.environ.get("MIRRORVIDGEN_REF_AUDIO_MAX_BYTES", "150000000"))

REF2VA_MODEL = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
TEXT_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_int8_convrot.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"

REF_MODES = {
    "h3_reference",
    "reference_to_video",
    "character_to_video",
    "reference_video",
    "reference_to_audio_video",
    "ref2va",
}

SUPPORTED_ASPECTS = {
    "16:9": (1344, 768),
    "9:16": (768, 1344),
    "1:1": (768, 768),
    "4:3": (1024, 768),
    "3:4": (768, 1024),
    "3:2": (1152, 768),
    "2:3": (768, 1152),
    "21:9": (1536, 672),
}
QUALITY_SCALE = {
    "smoke": 0.50,
    "480p": 0.625,
    "720p": 0.9375,
    "768p": 1.0,
    "1080p": 1.0,  # H3 Base is a 768-short-edge model; never fake 1080p here.
}


class Ref2VAUnavailable(RuntimeError):
    pass


def is_ref2va_request(job_input: dict[str, Any]) -> bool:
    mode = str(job_input.get("mode") or "").strip().lower()
    workflow = job_input.get("workflow")
    workflow_name = str(workflow).strip().lower() if isinstance(workflow, str) else ""
    engine = str(job_input.get("engine") or job_input.get("model_engine") or "").strip().lower()
    explicit = str(job_input.get("h3_workflow") or job_input.get("model_workflow") or "").strip().lower()
    if workflow_name == "ref2va" or explicit == "ref2va" or mode in {"h3_reference", "ref2va"}:
        return True
    h3_engines = {
        "mirrorromax-h3", "mirrorromax_h3",
        "freevideo_h3", "h3", "minimax_h3", "minimax-h3",
    }
    return engine in h3_engines and mode in REF_MODES


def _safe_token(value: Any) -> str:
    text = "".join(ch if str(ch).isalnum() or ch in {"-", "_"} else "_" for ch in str(value or "job"))
    return text[:80] or "job"


def _basename(value: Any, fallback: str) -> str:
    name = Path(str(value or "")).name
    return name or fallback


def _kind_from_item(item: dict[str, Any], fallback: str | None = None) -> str | None:
    explicit = str(
        item.get("reference_type")
        or item.get("media_type")
        or item.get("kind")
        or item.get("type")
        or ""
    ).strip().lower()
    aliases = {
        "picture": "image",
        "photo": "image",
        "image": "image",
        "video": "video",
        "movie": "video",
        "audio": "audio",
        "sound": "audio",
    }
    if explicit in aliases:
        return aliases[explicit]

    mime = str(item.get("mime") or item.get("content_type") or "").lower()
    if mime.startswith("image/"):
        return "image"
    if mime.startswith("video/"):
        return "video"
    if mime.startswith("audio/"):
        return "audio"

    suffix = Path(str(item.get("name") or "")).suffix.lower()
    if suffix in IMAGE_EXTS:
        return "image"
    if suffix in VIDEO_EXTS:
        return "video"
    if suffix in AUDIO_EXTS:
        return "audio"
    return fallback


def _has_payload(item: dict[str, Any]) -> bool:
    return bool(item.get("url") or item.get("data") or item.get("image") or item.get("audio"))


def collect_ref2va_items(job_input: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Collect refs in stable per-modality order without leaking URLs."""
    result: dict[str, list[dict[str, Any]]] = {"image": [], "video": [], "audio": []}
    seen: set[tuple[str, str, str]] = set()

    def add(raw: Any, fallback: str | None = None) -> None:
        if not isinstance(raw, dict) or not _has_payload(raw):
            return
        kind = _kind_from_item(raw, fallback)
        if kind not in result:
            return
        name = _basename(raw.get("name"), f"reference.{ {'image':'png','video':'mp4','audio':'wav'}[kind] }")
        identity = (
            kind,
            name,
            str(raw.get("url") or raw.get("data") or raw.get("image") or raw.get("audio") or "")[:96],
        )
        if identity in seen:
            return
        seen.add(identity)
        item = dict(raw)
        item["name"] = name
        result[kind].append(item)

    for key, kind in (
        ("reference_images", "image"),
        ("reference_videos", "video"),
        ("reference_audio", "audio"),
        ("reference_audios", "audio"),
    ):
        for item in job_input.get(key) or []:
            add(item, kind)

    for item in job_input.get("images") or []:
        add(item, "image")
    for item in job_input.get("assets") or []:
        add(item, None)

    for key, kind in (
        ("reference_image", "image"),
        ("reference_video", "video"),
        ("reference_audio_file", "audio"),
    ):
        item = job_input.get(key)
        if isinstance(item, dict):
            add(item, kind)

    image_count = len(result["image"])
    video_count = len(result["video"])
    audio_count = len(result["audio"])
    mixed_count = image_count + video_count + audio_count
    if image_count > 9:
        raise ValueError(f"Ref2VA supports at most 9 reference images; received {image_count}")
    if video_count > 3:
        raise ValueError(f"Ref2VA supports at most 3 reference videos; received {video_count}")
    if audio_count > 3:
        raise ValueError(f"Ref2VA supports at most 3 standalone reference audio files; received {audio_count}")
    if mixed_count > 12:
        raise ValueError(f"Ref2VA supports at most 12 mixed reference files; received {mixed_count}")
    if mixed_count == 0:
        raise ValueError("Ref2VA requires at least one reference image, video, or audio file")
    return result


def _decode_base64(value: str) -> bytes:
    raw = str(value)
    if "," in raw and raw.lstrip().startswith("data:"):
        raw = raw.split(",", 1)[1]
    try:
        return base64.b64decode(raw, validate=True)
    except Exception as error:
        raise ValueError("Reference contains invalid base64 data") from error


def _write_item(item: dict[str, Any], target: Path, max_bytes: int) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    if item.get("url"):
        response = requests.get(
            str(item["url"]),
            stream=True,
            allow_redirects=True,
            timeout=(30, 300),
        )
        response.raise_for_status()
        announced = int(response.headers.get("Content-Length") or 0)
        if announced and announced > max_bytes:
            raise ValueError(f"Reference file is too large ({announced} bytes)")
        total = 0
        with target.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(f"Reference file exceeds the {max_bytes} byte limit")
                handle.write(chunk)
        return total

    value = item.get("data")
    if value is None:
        value = item.get("image")
    if value is None:
        value = item.get("audio")
    if value is None:
        raise ValueError("Reference file data is missing")
    blob = _decode_base64(str(value))
    if len(blob) > max_bytes:
        raise ValueError(f"Reference file exceeds the {max_bytes} byte limit")
    target.write_bytes(blob)
    return len(blob)


def _run(command: list[str], timeout: int = 180) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as error:
        raise Ref2VAUnavailable("Ref2VA media validation requires ffmpeg/ffprobe on the worker") from error
    except subprocess.TimeoutExpired as error:
        raise ValueError("Reference media decode timed out") from error
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "").strip()[-1600:]
        raise ValueError(f"Reference media could not be decoded: {detail or 'ffmpeg/ffprobe failed'}") from error


def _probe(path: Path) -> dict[str, Any]:
    result = _run([
        "ffprobe", "-v", "error",
        "-show_entries",
        "format=duration,format_name:stream=index,codec_type,codec_name,width,height,r_frame_rate,sample_rate,channels",
        "-of", "json", str(path),
    ])
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("Reference media metadata is invalid") from error
    if not isinstance(value, dict):
        raise ValueError("Reference media metadata is invalid")
    return value


def _duration(probe: dict[str, Any]) -> float:
    try:
        return float((probe.get("format") or {}).get("duration") or 0)
    except (TypeError, ValueError):
        return 0.0


def _video_stream(probe: dict[str, Any]) -> dict[str, Any] | None:
    return next((s for s in probe.get("streams") or [] if s.get("codec_type") == "video"), None)


def _audio_stream(probe: dict[str, Any]) -> dict[str, Any] | None:
    return next((s for s in probe.get("streams") or [] if s.get("codec_type") == "audio"), None)


def _full_decode(path: Path, kind: str) -> None:
    selector = "0:v:0" if kind in {"image", "video"} else "0:a:0"
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path), "-map", selector]
    if kind == "image":
        command += ["-frames:v", "1"]
    command += ["-f", "null", "-"]
    _run(command, timeout=240)


def _fps_value(text: Any) -> float:
    try:
        return float(Fraction(str(text or "0")))
    except Exception:
        return 0.0


def _prepare_image(source: Path, target: Path, label: str) -> dict[str, Any]:
    probe = _probe(source)
    stream = _video_stream(probe)
    if stream is None or int(stream.get("width") or 0) <= 0 or int(stream.get("height") or 0) <= 0:
        raise ValueError(f"{label} is not a decodable image")
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    # ComfyUI's current LoadImage path decodes still images through PyAV.
    # Extremely tiny images such as 1x1 can pass ffmpeg probing yet fail later
    # when PyAV configures its filter graph (EINVAL / errno 22). Fail closed
    # during preflight so an invalid reference can never reach the paid queue.
    if width < 32 or height < 32:
        raise ValueError(
            f"{label} is too small for ComfyUI/H3 reference decoding; "
            f"minimum is 32x32, received {width}x{height}"
        )
    _full_decode(source, "image")
    shutil.copy2(source, target)
    return {
        "name": target.name,
        "width": width,
        "height": height,
        "codec": str(stream.get("codec_name") or ""),
    }


def _prepare_video(source: Path, target: Path, label: str) -> dict[str, Any]:
    probe = _probe(source)
    stream = _video_stream(probe)
    if stream is None:
        raise ValueError(f"{label} contains no video stream")
    duration = _duration(probe)
    if duration < 2.0 or duration > 15.0:
        raise ValueError(f"{label} duration must be 2–15 seconds; received {duration:.3f}s")
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    if width <= 0 or height <= 0:
        raise ValueError(f"{label} has invalid video dimensions")
    _full_decode(source, "video")

    fps = _fps_value(stream.get("r_frame_rate"))
    has_audio = _audio_stream(probe) is not None
    # MiniMaxH3ReferenceToVideo expects video frame tensors at 24 fps.
    # Preserve the entire clip and soundtrack; this is never a single-frame extraction.
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source),
        "-map", "0:v:0", "-map", "0:a?",
        "-vf", "fps=24",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ar", "32000", "-ac", "2",
        "-movflags", "+faststart",
        str(target),
    ]
    _run(command, timeout=600)
    normalized = _probe(target)
    normalized_stream = _video_stream(normalized)
    normalized_duration = _duration(normalized)
    if normalized_stream is None or normalized_duration <= 0:
        raise ValueError(f"{label} failed 24 fps normalization")
    _full_decode(target, "video")
    return {
        "name": target.name,
        "duration_seconds": round(normalized_duration, 3),
        "source_fps": round(fps, 4) if fps else None,
        "fps": 24,
        "width": int(normalized_stream.get("width") or width),
        "height": int(normalized_stream.get("height") or height),
        "codec": str(normalized_stream.get("codec_name") or ""),
        "container": str((normalized.get("format") or {}).get("format_name") or ""),
        "has_audio": bool(has_audio),
    }


def _prepare_audio(source: Path, target: Path, label: str) -> dict[str, Any]:
    probe = _probe(source)
    stream = _audio_stream(probe)
    if stream is None:
        raise ValueError(f"{label} contains no audio stream")
    duration = _duration(probe)
    if duration < 2.0 or duration > 15.0:
        raise ValueError(f"{label} duration must be 2–15 seconds; received {duration:.3f}s")
    _full_decode(source, "audio")
    shutil.copy2(source, target)
    return {
        "name": target.name,
        "duration_seconds": round(duration, 3),
        "codec": str(stream.get("codec_name") or ""),
        "sample_rate": int(stream.get("sample_rate") or 0),
        "channels": int(stream.get("channels") or 0),
    }


def cleanup_ref2va_paths(prepared: dict[str, Any] | None) -> None:
    if not prepared:
        return
    for raw in prepared.get("cleanup_files") or []:
        try:
            Path(raw).unlink(missing_ok=True)
        except OSError:
            pass
    temp_dir = prepared.get("temp_dir")
    if temp_dir:
        shutil.rmtree(temp_dir, ignore_errors=True)


def prepare_ref2va_references(job_id: Any, job_input: dict[str, Any]) -> dict[str, Any]:
    items = collect_ref2va_items(job_input)
    token = _safe_token(job_id)
    temp_dir = TMP_ROOT / f"mirrorvidgen_ref2va_{token}"
    shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    COMFY_INPUT.mkdir(parents=True, exist_ok=True)
    cleanup_files: list[str] = []
    prepared: dict[str, Any] = {
        "temp_dir": str(temp_dir),
        "cleanup_files": cleanup_files,
        "images": [],
        "videos": [],
        "audios": [],
    }

    try:
        for kind, plural, max_bytes in (
            ("image", "images", MAX_IMAGE_BYTES),
            ("video", "videos", MAX_VIDEO_BYTES),
            ("audio", "audios", MAX_AUDIO_BYTES),
        ):
            for index, item in enumerate(items[kind], start=1):
                label = f"reference {kind} {index}"
                original_name = _basename(item.get("name"), f"{kind}_{index}")
                suffix = Path(original_name).suffix.lower()
                if not suffix:
                    suffix = {"image": ".png", "video": ".mp4", "audio": ".wav"}[kind]
                source = temp_dir / f"{kind}_{index}_source{suffix}"
                _write_item(item, source, max_bytes)

                if kind == "video":
                    target = COMFY_INPUT / f"mv_ref2va_{token}_video_{index}.mp4"
                    meta = _prepare_video(source, target, label)
                elif kind == "image":
                    target = COMFY_INPUT / f"mv_ref2va_{token}_image_{index}{suffix}"
                    meta = _prepare_image(source, target, label)
                else:
                    target = COMFY_INPUT / f"mv_ref2va_{token}_audio_{index}{suffix}"
                    meta = _prepare_audio(source, target, label)

                cleanup_files.append(str(target))
                meta["index"] = index
                meta["original_name"] = original_name
                prepared[plural].append(meta)

        total_video = sum(float(item["duration_seconds"]) for item in prepared["videos"])
        if total_video > 15.05:
            raise ValueError(
                f"Total reference video duration must not exceed 15 seconds; received {total_video:.3f}s"
            )
        total_audio = sum(float(item["duration_seconds"]) for item in prepared["audios"])
        if total_audio > 15.05:
            raise ValueError(
                f"Total standalone reference audio duration must not exceed 15 seconds; received {total_audio:.3f}s"
            )

        prepared["counts"] = {
            "images": len(prepared["images"]),
            "videos": len(prepared["videos"]),
            "audio": len(prepared["audios"]),
            "mixed_files": len(prepared["images"]) + len(prepared["videos"]) + len(prepared["audios"]),
        }
        prepared["total_video_duration_seconds"] = round(total_video, 3)
        prepared["total_audio_duration_seconds"] = round(total_audio, 3)
        return prepared
    except Exception:
        cleanup_ref2va_paths(prepared)
        raise


def _snap32(value: float) -> int:
    return max(32, int(round(value / 32.0)) * 32)


def _dimensions(job_input: dict[str, Any]) -> tuple[int, int, str, str]:
    aspect = str(job_input.get("aspect_ratio") or "16:9")
    if aspect not in SUPPORTED_ASPECTS:
        raise ValueError("input.aspect_ratio is unsupported for MiRRORmax H3 Ref2VA")

    explicit_w = job_input.get("width")
    explicit_h = job_input.get("height")
    if explicit_w is not None or explicit_h is not None:
        if explicit_w is None or explicit_h is None:
            raise ValueError("Ref2VA requires both input.width and input.height when either is supplied")
        width, height = int(explicit_w), int(explicit_h)
        if width < 256 or height < 256 or width > 2048 or height > 2048:
            raise ValueError("Ref2VA width/height must each be between 256 and 2048")
        if width % 32 or height % 32:
            raise ValueError("Ref2VA width and height must be divisible by 32")
        if width * height > 1_050_000:
            raise ValueError("Ref2VA base generation is limited to roughly 1.05 megapixels")
        return width, height, aspect, "custom"

    quality = str(job_input.get("quality") or "768p").lower()
    if quality not in QUALITY_SCALE:
        raise ValueError("input.quality must be one of: smoke, 480p, 720p, 768p, 1080p")
    base_w, base_h = SUPPORTED_ASPECTS[aspect]
    scale = QUALITY_SCALE[quality]
    width = _snap32(base_w * scale)
    height = _snap32(base_h * scale)
    return width, height, aspect, quality


def _frame_length(duration_seconds: float) -> int:
    frames = max(5, int(round(duration_seconds * 24)))
    return frames + (5 - (frames % 17) + 17) % 17


def generation_values(job_input: dict[str, Any]) -> dict[str, Any]:
    prompt = str(job_input.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("input.prompt is required for MiRRORmax H3 Ref2VA")
    if len(prompt) > 12000:
        raise ValueError("input.prompt is too long (max 12000 characters)")

    fps = int(job_input.get("fps", 24))
    if fps != 24:
        raise ValueError("MiRRORmax H3 native generation is 24 fps")

    duration = float(job_input.get("duration_seconds", job_input.get("duration", 5)))
    if duration < 4 or duration > 15:
        raise ValueError("MiRRORmax H3 Ref2VA duration must be between 4 and 15 seconds")

    width, height, aspect, quality = _dimensions(job_input)
    seed = int(job_input.get("seed", 42))
    if seed < 0 or seed > 0x7FFFFFFFFFFFFFFF:
        raise ValueError("input.seed is outside the supported range")

    scheduler = str(job_input.get("scheduler") or "beta")
    if scheduler not in {"simple", "beta", "normal"}:
        raise ValueError("Ref2VA scheduler must be simple, beta, or normal")
    steps = int(job_input.get("steps", 20))
    if steps < 1 or steps > 100:
        raise ValueError("Ref2VA steps must be between 1 and 100")

    length = _frame_length(duration)
    return {
        "prompt": prompt,
        "fps": fps,
        "requested_duration_seconds": duration,
        "length": length,
        "actual_duration_seconds": round(length / 24.0, 3),
        "width": width,
        "height": height,
        "aspect_ratio": aspect,
        "quality": quality,
        "seed": seed,
        "scheduler": scheduler,
        "steps": steps,
        "ref_image_size": str(job_input.get("ref_image_size") or "match"),
    }


def _node(class_type: str, inputs: dict[str, Any], title: str | None = None) -> dict[str, Any]:
    return {
        "class_type": class_type,
        "inputs": inputs,
        "_meta": {"title": title or class_type},
    }


def build_ref2va_workflow(job_input: dict[str, Any], prepared: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    status = h3_ref2va_status()
    if not status.get("ready"):
        raise Ref2VAUnavailable("MiRRORmax H3 Ref2VA core model files are not ready")

    values = generation_values(job_input)
    workflow: dict[str, Any] = {}

    workflow["h3_model"] = _node("UNETLoader", {
        "unet_name": REF2VA_MODEL,
        "weight_dtype": "default",
    }, "H3 Ref2VA Model")
    workflow["h3_clip"] = _node("CLIPLoader", {
        "clip_name": TEXT_ENCODER,
        "type": "minimax",
        "device": "default",
    }, "H3 Qwen3-VL")
    workflow["h3_video_vae"] = _node("VAELoader", {"vae_name": VIDEO_VAE}, "H3 Video VAE")
    workflow["h3_audio_vae"] = _node("VAELoader", {"vae_name": AUDIO_VAE}, "H3 Audio VAE")

    ref_inputs: dict[str, Any] = {
        "clip": ["h3_clip", 0],
        "vae": ["h3_video_vae", 0],
        "audio_vae": ["h3_audio_vae", 0],
        "prompt": values["prompt"],
        "width": values["width"],
        "height": values["height"],
        "length": values["length"],
        "ref_image_size": values["ref_image_size"],
    }

    for i, item in enumerate(prepared.get("images") or []):
        node_id = f"h3_ref_image_{i + 1}"
        workflow[node_id] = _node("LoadImage", {"image": item["name"]}, f"Picture {i + 1}")
        ref_inputs[f"ref_images.ref_image_{i}"] = [node_id, 0]

    for i, item in enumerate(prepared.get("videos") or []):
        load_id = f"h3_ref_video_{i + 1}"
        split_id = f"h3_ref_video_components_{i + 1}"
        workflow[load_id] = _node("LoadVideo", {"file": item["name"]}, f"Video {i + 1}")
        workflow[split_id] = _node("GetVideoComponents", {"video": [load_id, 0]}, f"Video {i + 1} Components")
        # The entire 24 fps frame sequence is connected. This is true Ref2VA
        # video conditioning, never a single extracted frame.
        ref_inputs[f"ref_videos.ref_video_{i}"] = [split_id, 0]
        if item.get("has_audio"):
            ref_inputs[f"ref_video_audios.ref_video_audio_{i}"] = [split_id, 1]

    for i, item in enumerate(prepared.get("audios") or []):
        node_id = f"h3_ref_audio_{i + 1}"
        workflow[node_id] = _node("LoadAudio", {"audio": item["name"]}, f"Audio {i + 1}")
        ref_inputs[f"ref_audios.ref_audio_{i}"] = [node_id, 0]

    workflow["h3_ref2va"] = _node("MiniMaxH3ReferenceToVideo", ref_inputs, "MiRRORmax H3 Ref2VA")
    workflow["h3_noise"] = _node("RandomNoise", {"noise_seed": values["seed"]}, "H3 Noise")
    workflow["h3_sampler_select"] = _node("KSamplerSelect", {"sampler_name": "res_multistep"}, "H3 Sampler")
    workflow["h3_scheduler"] = _node("BasicScheduler", {
        "model": ["h3_model", 0],
        "scheduler": values["scheduler"],
        "steps": values["steps"],
        "denoise": 1.0,
    }, "H3 Scheduler")
    workflow["h3_guider"] = _node("BasicGuider", {
        "model": ["h3_model", 0],
        "conditioning": ["h3_ref2va", 0],
    }, "H3 Guider")
    workflow["h3_sample"] = _node("SamplerCustomAdvanced", {
        "noise": ["h3_noise", 0],
        "guider": ["h3_guider", 0],
        "sampler": ["h3_sampler_select", 0],
        "sigmas": ["h3_scheduler", 0],
        "latent_image": ["h3_ref2va", 1],
    }, "H3 Sample")
    workflow["h3_decode_video"] = _node("VAEDecode", {
        "samples": ["h3_sample", 0],
        "vae": ["h3_video_vae", 0],
    }, "H3 Decode Video")
    workflow["h3_decode_audio"] = _node("VAEDecodeAudio", {
        "samples": ["h3_sample", 0],
        "vae": ["h3_audio_vae", 0],
    }, "H3 Decode Audio")
    workflow["h3_create_video"] = _node("CreateVideo", {
        "images": ["h3_decode_video", 0],
        "audio": ["h3_decode_audio", 0],
        "fps": 24.0,
        "bit_depth": 8,
        "color_space": "sRGB",
    }, "H3 Create Video")
    workflow["h3_save_video"] = _node("SaveVideo", {
        "video": ["h3_create_video", 0],
        "filename_prefix": "mirrorvidgen_h3_ref2va",
        "format": "mp4",
        "codec": "h264",
    }, "H3 Save Video")

    settings = {
        "model": "MiRRORmax H3",
        "engine": "comfyui_minimax_h3_ref2va",
        "workflow": "ref2va",
        "native_fps": 24,
        "requested_duration_seconds": values["requested_duration_seconds"],
        "actual_duration_seconds": values["actual_duration_seconds"],
        "frames": values["length"],
        "width": values["width"],
        "height": values["height"],
        "aspect_ratio": values["aspect_ratio"],
        "quality": values["quality"],
        "seed": values["seed"],
        "scheduler": values["scheduler"],
        "steps": values["steps"],
        "reference_counts": dict(prepared.get("counts") or {}),
        "total_reference_video_seconds": prepared.get("total_video_duration_seconds", 0),
        "total_reference_audio_seconds": prepared.get("total_audio_duration_seconds", 0),
        "reference_tags": {
            "pictures": [f"<Picture {i}>" for i in range(1, len(prepared.get("images") or []) + 1)],
            "videos": [f"<Video {i}>" for i in range(1, len(prepared.get("videos") or []) + 1)],
            "audio": [f"<Audio {i}>" for i in range(1, len(prepared.get("audios") or []) + 1)],
        },
        "model_files": {
            "diffusion": REF2VA_MODEL,
            "text_encoder": TEXT_ENCODER,
            "video_vae": VIDEO_VAE,
            "audio_vae": AUDIO_VAE,
        },
        "worker_path": "h3-ref2va-comfy-v1",
    }
    return workflow, settings


REQUIRED_NODE_INPUTS = {
    "UNETLoader": ("unet_name", "weight_dtype"),
    "CLIPLoader": ("clip_name", "type"),
    "VAELoader": ("vae_name",),
    "LoadImage": ("image",),
    "LoadVideo": ("file",),
    "GetVideoComponents": ("video",),
    "LoadAudio": ("audio",),
    "MiniMaxH3ReferenceToVideo": ("clip", "vae", "audio_vae", "prompt", "width", "height", "length", "ref_image_size"),
    "RandomNoise": ("noise_seed",),
    "KSamplerSelect": ("sampler_name",),
    "BasicScheduler": ("model", "scheduler", "steps", "denoise"),
    "BasicGuider": ("model", "conditioning"),
    "SamplerCustomAdvanced": ("noise", "guider", "sampler", "sigmas", "latent_image"),
    "VAEDecode": ("samples", "vae"),
    "VAEDecodeAudio": ("samples", "vae"),
    "CreateVideo": ("images", "fps"),
    "SaveVideo": ("video", "filename_prefix", "format"),
}


def validate_ref2va_workflow(workflow: dict[str, Any], object_info: dict[str, Any] | None = None) -> bool:
    missing: list[str] = []
    missing_nodes: list[str] = []
    for node_id, node in workflow.items():
        class_type = str(node.get("class_type") or "")
        if object_info is not None and class_type not in object_info:
            missing_nodes.append(class_type)
        inputs = node.get("inputs") or {}
        for name in REQUIRED_NODE_INPUTS.get(class_type, ()):
            value = inputs.get(name)
            if value is None or value == "" or value == []:
                missing.append(f"{node_id} {class_type}.{name}")
    if missing_nodes:
        unique = ", ".join(sorted(set(missing_nodes)))
        raise Ref2VAUnavailable(f"Required Ref2VA ComfyUI nodes are unavailable: {unique}")
    if missing:
        raise ValueError("Ref2VA workflow is missing required inputs: " + ", ".join(missing[:12]))
    return True


def public_prepared_summary(prepared: dict[str, Any]) -> dict[str, Any]:
    """Strip local temp paths while retaining useful validation metadata."""
    return {
        "counts": dict(prepared.get("counts") or {}),
        "total_video_duration_seconds": prepared.get("total_video_duration_seconds", 0),
        "total_audio_duration_seconds": prepared.get("total_audio_duration_seconds", 0),
        "images": [
            {k: v for k, v in item.items() if k not in {"name", "original_name"}}
            for item in prepared.get("images") or []
        ],
        "videos": [
            {k: v for k, v in item.items() if k not in {"name", "original_name"}}
            for item in prepared.get("videos") or []
        ],
        "audios": [
            {k: v for k, v in item.items() if k not in {"name", "original_name"}}
            for item in prepared.get("audios") or []
        ],
    }
