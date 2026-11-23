"""Experimental H3 graph adapter; no downloads, queueing or default changes.

Production builders are called first and their graphs are copied. Only the
candidate handler imports this module. Runtime validation is still required.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import re

MOTION_REVISION = "5335715abe54c1a9bfbe3494da29aae3e8635ce3"
TURBO_REVISION = "02e26d591f7a04d5d1a074c9566d5dd4f22f6225"
REF2VA_TURBO = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
PRODUCTION_IMAGE = "ghcr.io/mitm-1/mirrorvidgen-worker:7b21b2c015b21a512498cdcaa357da4720208cad"


class AddonUnavailable(ValueError):
    pass


def safe_job_id(value):
    text = str(value or "")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", text):
        raise AddonUnavailable("A safe, unique job ID is required for context storage")
    return text


@dataclass(frozen=True)
class Options:
    speed: str = "normal"
    motion: bool = False
    source_job_id: str | None = None

    @classmethod
    def from_input(cls, data):
        speed = data.get("generation_speed", "normal")
        if speed not in {"normal", "turbo_balanced", "turbo_fast"}:
            raise AddonUnavailable("Unsupported generation speed")
        context = data.get("motion_context", {})
        if not isinstance(context, dict):
            raise AddonUnavailable("motion_context must be an object")
        enabled = context.get("enabled", False)
        if not isinstance(enabled, bool):
            raise AddonUnavailable("Motion enabled must be a boolean")
        source = context.get("source_job_id")
        if source is not None:
            source = safe_job_id(source)
        if source and not enabled:
            raise AddonUnavailable("A motion source requires Motion Continuation")
        if enabled and context.get("preserve_audio", True) is not True:
            raise AddonUnavailable("Latent mode currently preserves both streams; visual-only mode is not validated")
        if enabled and (context.get("context_frames", 22) != 22 or context.get("audio_context_frames", 24) != 24):
            raise AddonUnavailable("Initial candidate supports only 22 visual / 24 audio context frames")
        return cls(speed, enabled, source)

    @property
    def requested(self):
        return self.speed != "normal" or self.motion


def duration_plan(sampled_frames, *, continuation=False):
    if not isinstance(sampled_frames, int) or sampled_frames < 5 or (sampled_frames - 5) % 17:
        raise AddonUnavailable("H3 frame count must be 17*n+5")
    # Baseline 15s rounds up to 362 frames. Never extend its sampled ceiling.
    if sampled_frames > 362:
        raise AddonUnavailable("Requested output exceeds the existing H3 sampled-frame limit")
    trim = 22 if continuation else 0
    if sampled_frames <= trim:
        raise AddonUnavailable("Context consumes the whole sampled clip")
    return {"sampled_frames": sampled_frames, "trimmed_frames": trim,
            "delivered_frames": sampled_frames - trim,
            "sampled_duration_seconds": sampled_frames / 24,
            "delivered_duration_seconds": (sampled_frames - trim) / 24,
            "overlap_seconds": trim / 24}


def node(kind, **inputs):
    return {"class_type": kind, "inputs": inputs}


def adapt_ref2va(graph, settings, options, *, job_id, source_manifest=None,
                 staged_latent=None, model_root=Path("/runpod-volume/models")):
    if not options.requested:
        return graph, settings
    if settings.get("workflow") != "ref2va":
        raise AddonUnavailable("Existing FreeVideo FL2VA is not a ComfyUI add-on path")
    if options.speed == "turbo_balanced":
        raise AddonUnavailable("No verified 8-step Ref2VA Turbo configuration is available")
    result, metadata = deepcopy(graph), deepcopy(settings)
    if options.speed == "turbo_fast":
        lora = model_root / "loras" / REF2VA_TURBO
        if not lora.is_file() or lora.stat().st_size < 1_000_000_000:
            raise AddonUnavailable("Compatible Ref2VA Turbo LoRA is missing; no download was started")
        result["h3_turbo_lora"] = node("LoraLoaderModelOnly", model=["h3_model", 0],
                                       lora_name=REF2VA_TURBO, strength_model=1.0)
        result["h3_turbo_shift"] = node("MiniMaxH3SigmaShift", model=["h3_turbo_lora", 0],
                                        shift_video=12.0, shift_audio=3.0)
        result["h3_guider"]["inputs"]["model"] = ["h3_turbo_shift", 0]
        # Match the official reference graph: scheduler uses the base model.
        result["h3_scheduler"]["inputs"].update(scheduler="simple", steps=4, denoise=1.0)
        result["h3_sampler_select"]["inputs"]["sampler_name"] = "euler"
        metadata.update(steps=4, scheduler="simple", sampler="euler")

    plan = duration_plan(int(settings["frames"]), continuation=bool(options.motion and options.source_job_id))
    if options.motion:
        identity = safe_job_id(job_id)
        result["h3_context_save"] = node("MiniMaxH3MotionContextSaveLatent",
            latent=["h3_sample", 0], filename_prefix=f"h3_addons/{identity}/clip", clip_index=1)
        if options.source_job_id:
            if not source_manifest or not staged_latent:
                raise AddonUnavailable("Saved paired latent context is required; an MP4 alone is not a saved latent")
            expected = {"width": settings["width"], "height": settings["height"],
                        "fps": 24, "workflow": "ref2va", "motion_revision": MOTION_REVISION}
            if any(source_manifest.get(k) != v for k, v in expected.items()):
                raise AddonUnavailable("Context resolution, workflow or latent revision does not match")
            if source_manifest.get("job_id") != options.source_job_id:
                raise AddonUnavailable("Context identity does not match the selected source")
            result["h3_context_load"] = node("MiniMaxH3MotionContextLoadLatent",
                latent_path=staged_latent, clip_index=1)
            result["h3_motion_context"] = node("MiniMaxH3MotionContext",
                conditioning=["h3_ref2va", 0], vae=["h3_video_vae", 0],
                latent=["h3_ref2va", 1], context_length="22", audio_context_length=24,
                context_latent=["h3_context_load", 0])
            result["h3_guider"]["inputs"]["conditioning"] = ["h3_motion_context", 0]
            result["h3_context_trim"] = node("MiniMaxH3MotionContextTrim",
                images=["h3_decode_video", 0], audio=["h3_decode_audio", 0],
                trim_frames=["h3_motion_context", 1], fps=24.0, match_tail=True)
            result["h3_create_video"]["inputs"].update(images=["h3_context_trim", 0], audio=["h3_context_trim", 1])
    metadata["addons"] = {"experimental": True, "generation_speed": options.speed,
        "motion_context": options.motion, "source_job_id": options.source_job_id,
        "motion_revision": MOTION_REVISION if options.motion else None,
        "turbo_revision": TURBO_REVISION if options.speed != "normal" else None,
        "context_frames": 22 if options.source_job_id else 0,
        "audio_context_frames": 24 if options.source_job_id else 0, **plan,
        "music_master_policy": "retain existing master soundtrack and exact timeline"}
    metadata["actual_duration_seconds"] = plan["delivered_duration_seconds"]
    return result, metadata


def validate_addon_schema(graph, object_info):
    """Check installed node input names before any GPU prompt is queued."""
    for item in graph.values():
        kind = item["class_type"]
        schema = object_info.get(kind)
        if not schema:
            raise AddonUnavailable(f"Required candidate node is unavailable: {kind}")
        if kind.startswith("MiniMaxH3Motion") or kind in {"LoraLoaderModelOnly", "MiniMaxH3SigmaShift"}:
            inputs = schema.get("input", {})
            required = inputs.get("required", {})
            allowed = set(required) | set(inputs.get("optional", {}))
            provided = item["inputs"]
            if set(required) - set(provided) or set(provided) - allowed:
                raise AddonUnavailable(f"Installed candidate node schema differs: {kind}")
