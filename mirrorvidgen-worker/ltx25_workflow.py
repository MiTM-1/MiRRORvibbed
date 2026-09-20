import copy
import json
import os
from pathlib import Path

from workflow_compiler import (
    compile_workflow,
    disable_prompt_enhancer,
    iter_nodes,
)

TEMPLATE_PATH = Path("/ltx25-t2v-template.json")

DEFAULT_NEGATIVE_PROMPT = (
    "cartoon, illustration, video game, low quality, distorted, text, watermark"
)

# Stage-1 dimensions. The latent spatial upscaler produces 2x output.
# Every stage-1 dimension is divisible by 32, as required by LTX-2.5.
QUALITY_STAGE1_AREA = {
    "smoke": 320 * 192,
    "480p": 448 * 256,
    "720p": 640 * 352,
    "1080p": 960 * 544,
}
SUPPORTED_ASPECTS = {"1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9", "21:9"}


def _snap32(value):
    return max(32, int(round(value / 32.0)) * 32)


def _stage1_dimensions(quality, aspect_ratio):
    if quality not in QUALITY_STAGE1_AREA:
        raise ValueError("input.quality must be one of: smoke, 480p, 720p, 1080p")
    if aspect_ratio not in SUPPORTED_ASPECTS:
        raise ValueError("input.aspect_ratio is unsupported")
    left, right = (int(part) for part in aspect_ratio.split(":"))
    ratio = left / right
    area = QUALITY_STAGE1_AREA[quality]
    width = _snap32((area * ratio) ** 0.5)
    height = _snap32((area / ratio) ** 0.5)
    return width, height


def _snap_frames(seconds, fps):
    requested = max(1, int(round(seconds * fps)))
    if requested % 8 == 1:
        return requested
    return ((requested - 1 + 7) // 8) * 8 + 1

def _primary_reference_name(job_input):
    explicit = str(job_input.get("reference_image_name") or "").strip()
    if explicit:
        return Path(explicit).name

    for image in job_input.get("images") or []:
        if isinstance(image, dict):
            name = str(image.get("name") or "").strip()
            if name and image.get("image"):
                return Path(name).name

    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
    for asset in job_input.get("assets") or []:
        if isinstance(asset, dict):
            name = str(asset.get("name") or "").strip()
            if name and asset.get("url") and Path(name).suffix.lower() in image_exts:
                return Path(name).name

    return None


def _inject_i2v_reference(workflow, image_name, strength):
    workflow["450"] = {
        "inputs": {"image": image_name},
        "class_type": "LoadImage",
        "_meta": {"title": "Load Reference Image"},
    }
    workflow["451"] = {
        "inputs": {
            "vae": ["440", 0],
            "image": ["450", 0],
            "latent": ["434", 0],
            "strength": strength,
            "bypass": False,
        },
        "class_type": "LTXVImgToVideoConditionOnly",
        "_meta": {"title": "Reference Conditioning — Stage 1"},
    }
    workflow["452"] = {
        "inputs": {
            "vae": ["440", 0],
            "image": ["450", 0],
            "latent": ["416", 0],
            "strength": strength,
            "bypass": False,
        },
        "class_type": "LTXVImgToVideoConditionOnly",
        "_meta": {"title": "Reference Conditioning — Stage 2"},
    }
    workflow["431"]["inputs"]["video_latent"] = ["451", 0]
    workflow["414"]["inputs"]["video_latent"] = ["452", 0]



def build_ltx25_t2v(job_input, require_reference=False):
    prompt = str(job_input.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("input.prompt is required for ltx25_t2v")
    if len(prompt) > 12000:
        raise ValueError("input.prompt is too long (max 12000 characters)")

    negative_prompt = str(
        job_input.get("negative_prompt") or DEFAULT_NEGATIVE_PROMPT
    ).strip()
    if len(negative_prompt) > 4000:
        raise ValueError("input.negative_prompt is too long (max 4000 characters)")

    fps = int(job_input.get("fps", 24))
    if fps != 24:
        raise ValueError("MiRROR Motion 1 currently supports 24 fps only")

    duration_seconds = float(job_input.get("duration_seconds", 2))
    if duration_seconds < 1 or duration_seconds > 20:
        raise ValueError("input.duration_seconds must be between 1 and 20")

    aspect_ratio = str(job_input.get("aspect_ratio", "16:9"))
    quality = str(job_input.get("quality", "smoke"))

    seed = int(job_input.get("seed", 42))
    if seed < 0 or seed > 0x7FFFFFFFFFFFFFFF:
        raise ValueError("input.seed is outside the supported range")

    frames = _snap_frames(duration_seconds, fps)
    width, height = _stage1_dimensions(quality, aspect_ratio)

    reference_image_name = _primary_reference_name(job_input)
    if require_reference and not reference_image_name:
        raise ValueError("Image → Video requires one uploaded conditioning image")
    reference_strength = float(job_input.get("reference_strength", 0.85))
    if reference_strength < 0.0 or reference_strength > 1.0:
        raise ValueError("input.reference_strength must be between 0 and 1")

    template = json.loads(TEMPLATE_PATH.read_text())
    workflow = copy.deepcopy(template["input"]["workflow"])

    if reference_image_name:
        _inject_i2v_reference(workflow, reference_image_name, reference_strength)

    workflow["432"]["inputs"]["text"] = prompt
    workflow["433"]["inputs"]["text"] = negative_prompt
    workflow["429"]["inputs"]["noise_seed"] = seed
    workflow["421"]["inputs"]["noise_seed"] = (seed + 1) & 0x7FFFFFFFFFFFFFFF

    workflow["430"]["inputs"]["frame_rate"] = fps
    workflow["434"]["inputs"].update({
        "width": width,
        "height": height,
        "length": frames,
        "batch_size": 1,
    })
    workflow["435"]["inputs"].update({
        "frames_number": frames,
        "frame_rate": fps,
        "batch_size": 1,
    })
    workflow["413"]["inputs"]["fps"] = fps
    workflow["999"]["inputs"]["filename_prefix"] = "mirrorvidgen_ltx25"

    settings = {
        "model": "MiRROR Motion 1",
        "engine": "LTX-2.5 distilled INT8 ConvRot",
        "prompt_enhancer": False,
        "fps": fps,
        "requested_duration_seconds": duration_seconds,
        "actual_duration_seconds": (frames - 1) / fps,
        "frames": frames,
        "aspect_ratio": aspect_ratio,
        "quality": quality,
        "stage1_width": width,
        "stage1_height": height,
        "output_width": width * 2,
        "output_height": height * 2,
        "seed": seed,
        "reference_conditioning": bool(reference_image_name),
        "reference_mode": "i2v_first_frame" if reference_image_name else "none",
        "reference_image_name": reference_image_name,
        "reference_strength": reference_strength if reference_image_name else None,
    }
    return workflow, settings


def build_ltx25_i2v(job_input):
    """Explicit first-frame Image → Video route. Never falls back to T2V."""
    return build_ltx25_t2v(job_input, require_reference=True)


# ---------------------------------------------------------------------------
# Official LTX-2.5 control workflows
# ---------------------------------------------------------------------------

WORKFLOW_DIRS = tuple(
    Path(item)
    for item in (
        os.environ.get("MIRRORVIDGEN_WORKFLOW_DIR", ""),
        "/workspace/mirrorvidgen/workflows",
        "/runpod-volume/mirrorvidgen/workflows",
        "/comfyui/workflows",
        "/workflows",
    )
    if item
)

WORKFLOW_FILES = {
    "ingredients": "LTX-2.5_ICLoRA_Ingredients_Single_Stage_Distilled.json",
    "motion_track": "LTX-2.5_ICLoRA_Motion_Track_Distilled.json",
    "union_control": "LTX-2.5_ICLoRA_Union_Control_Distilled.json",
    "first_last": "video_ltx2_5_flf2v.json",
    "continuation": "LTX-2.5_V2V_ICLoRA_Single_Stage_Distilled.json",
}

MODEL_ROOTS = (
    Path("/runpod-volume/models"),
    Path("/workspace/models"),
    Path("/comfyui/models"),
)

# The official 2.5 example graphs still use the 2.3 IC-LoRA weights for
# Ingredients and V2V deblur. Some earlier volume installs used the 2.5
# filename aliases, so accept either exact installed filename and always pass
# the one that actually exists to ComfyUI. This is a real file check, never a
# capability override.
INGREDIENTS_LORA_FILENAMES = (
    "ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors",
    "ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors",
)
DEBLUR_LORA_FILENAMES = (
    "ltx-2.3-22b-ic-lora-deblur-0.9.safetensors",
    "ltx-2.5-22b-ic-lora-deblur-0.9.safetensors",
)


class CapabilityUnavailable(RuntimeError):
    """A deliberate preflight lock, never a fallback to another workflow."""


def _workflow_path(kind):
    filename = WORKFLOW_FILES[kind]
    for directory in WORKFLOW_DIRS:
        candidate = directory / filename
        if candidate.is_file() and candidate.stat().st_size > 100:
            return candidate
    raise CapabilityUnavailable(
        f"The official {kind.replace('_', ' ')} workflow is not installed on this worker."
    )


def _model_present(filename, subdirectories=("", "loras", "vae", "text_encoders", "diffusion_models", "model_patches")):
    for root in MODEL_ROOTS:
        for subdirectory in subdirectories:
            if (root / subdirectory / filename).is_file():
                return True
    # ComfyUI can also expose a model through an alternate extra_model_paths
    # entry; a recursive filename check keeps this preflight conservative.
    if os.environ.get("MIRRORVIDGEN_SKIP_MODEL_PREFLIGHT") == "1":
        return True
    return any(root.exists() and any(root.rglob(filename)) for root in MODEL_ROOTS)


def _first_present(filenames):
    """Return the first installed filename from an explicit compatibility set."""
    for filename in filenames:
        if _model_present(filename):
            return filename
    return None


def _require_any_model(filenames, label):
    selected = _first_present(filenames)
    if selected is None and os.environ.get("MIRRORVIDGEN_SKIP_MODEL_PREFLIGHT") != "1":
        raise CapabilityUnavailable(
            f"Required {label} model files are missing: " + ", ".join(filenames)
        )
    return selected or filenames[0]


def _require_models(filenames):
    missing = [name for name in filenames if not _model_present(name)]
    if missing and os.environ.get("MIRRORVIDGEN_SKIP_MODEL_PREFLIGHT") != "1":
        raise CapabilityUnavailable(
            "Required LTX model files are missing: " + ", ".join(missing)
        )


def _prompt_values(job_input):
    prompt = str(job_input.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("input.prompt is required for the selected LTX workflow")
    if len(prompt) > 12000:
        raise ValueError("input.prompt is too long (max 12000 characters)")
    negative = str(job_input.get("negative_prompt") or DEFAULT_NEGATIVE_PROMPT).strip()
    if len(negative) > 4000:
        raise ValueError("input.negative_prompt is too long (max 4000 characters)")
    return prompt, negative


def _generation_values(job_input, max_seconds=20):
    fps = int(job_input.get("fps", 24))
    if fps != 24:
        raise ValueError("MiRROR Motion 1 currently supports 24 fps only")
    duration = float(job_input.get("duration_seconds", 2))
    if duration < 1 or duration > max_seconds:
        raise ValueError(f"input.duration_seconds must be between 1 and {max_seconds}")
    aspect = str(job_input.get("aspect_ratio", "16:9"))
    quality = str(job_input.get("quality", "480p"))
    frames = _snap_frames(duration, fps)
    width, height = _stage1_dimensions(quality, aspect)
    seed = int(job_input.get("seed", 42))
    if seed < 0 or seed > 0x7FFFFFFFFFFFFFFF:
        raise ValueError("input.seed is outside the supported range")
    return fps, duration, frames, aspect, quality, width, height, seed


def _asset_name(item):
    if not isinstance(item, dict):
        return None
    name = str(item.get("name") or "").strip()
    return Path(name).name if name else None


def _images(job_input, role=None):
    found = []
    for item in job_input.get("images") or []:
        if not isinstance(item, dict):
            continue
        if role and str(item.get("role") or "").lower() != role.lower():
            continue
        name = _asset_name(item)
        if name and (item.get("image") or item.get("url")):
            found.append(name)
    for item in job_input.get("assets") or []:
        if not isinstance(item, dict):
            continue
        if role and str(item.get("role") or "").lower() != role.lower():
            continue
        name = _asset_name(item)
        if name and Path(name).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}:
            if item.get("url") or item.get("data") or item.get("image"):
                found.append(name)
    return list(dict.fromkeys(found))


def _videos(job_input, role=None):
    found = []
    for item in job_input.get("assets") or []:
        if not isinstance(item, dict):
            continue
        if role and str(item.get("role") or "").lower() != role.lower():
            continue
        name = _asset_name(item)
        if name and Path(name).suffix.lower() in {".mp4", ".mov", ".webm", ".mkv"} and (item.get("url") or item.get("data")):
            found.append(name)
    for item in job_input.get("images") or []:
        if not isinstance(item, dict) or str(item.get("type") or "").lower() != "video":
            continue
        if role and str(item.get("role") or "").lower() != role.lower():
            continue
        name = _asset_name(item)
        if name:
            found.append(name)
    return list(dict.fromkeys(found))


def _set_prompt_nodes(workflow, prompt, negative):
    positive = []
    negative_nodes = []
    for node_id, node in iter_nodes(workflow, "CLIPTextEncode"):
        title = str((node.get("_meta") or {}).get("title") or "").lower()
        if "negative" in title or node_id.endswith(":217"):
            negative_nodes.append(node)
        else:
            positive.append(node)
    for node in positive:
        node.setdefault("inputs", {})["text"] = prompt
    for node in negative_nodes:
        node.setdefault("inputs", {})["text"] = negative
    if not positive:
        raise CapabilityUnavailable("The official workflow has no positive prompt encoder")


def _set_common_generation(workflow, fps, frames, width, height, seed, prefix):
    for _, node in iter_nodes(workflow, "LTXVConditioning"):
        node.setdefault("inputs", {})["frame_rate"] = fps
    for _, node in iter_nodes(workflow, "CreateVideo"):
        node.setdefault("inputs", {})["fps"] = fps
    for _, node in iter_nodes(workflow, "EmptyLTXVLatentVideo"):
        node.setdefault("inputs", {}).update({"width": width, "height": height, "length": frames})
    for _, node in iter_nodes(workflow, "LTXVEmptyLatentAudio"):
        node.setdefault("inputs", {}).update({"frames_number": frames, "frame_rate": fps})
    for _, node in iter_nodes(workflow, "RandomNoise"):
        node.setdefault("inputs", {})["noise_seed"] = seed
    for _, node in iter_nodes(workflow, "SaveVideo"):
        node.setdefault("inputs", {})["filename_prefix"] = prefix


def _set_duration_expressions(workflow, fps, duration):
    """Update the official graph's valid LTX frame-grid expression.

    The IC-LoRA examples derive ``1 + floor(fps * seconds / 8) * 8`` from
    two primitive inputs.  Leaving either primitive at the example value
    silently produces the wrong duration even if the latent node itself is
    patched, so update every compiled duration expression explicitly.
    """
    updated = 0
    for _, node in iter_nodes(workflow, "ComfyMathExpression"):
        # The expression is represented as a widget in the official graph,
        # therefore it is not present in the API inputs after compilation.
        # The stable input names are the only values the node executes with.
        inputs = node.setdefault("inputs", {})
        # The duration expression has both primitive inputs.  The motion
        # graph also contains a separate one-input ``a*32`` expression for
        # latent alignment; never overwrite that spatial multiplier.
        if "values.a" in inputs and "values.b" in inputs:
            inputs["values.a"] = fps
            inputs["values.b"] = duration
            updated += 1
    return updated


def _set_ic_lora_guide(workflow, *, reference_only=False):
    """Set the real IC-LoRA guide controls, preserving the role semantics.

    Ingredients references are placed outside the generated frame range
    (``frame_idx=-1``), so they steer identity without becoming frame one.
    Motion/Union guides use frame zero when an opening image is supplied.
    """
    for _, node in iter_nodes(workflow, "LTXAddVideoICLoRAGuide"):
        inputs = node.setdefault("inputs", {})
        inputs.update({
            "frame_idx": -1 if reference_only else 0,
            "strength": 1.0,
            "crop": "disabled",
            "use_tiled_encode": False,
            "tile_size": 256,
            "tile_overlap": 64,
        })


def _set_resize_dimensions(workflow, width, height, node_suffixes):
    """Make image/control guides match the selected native latent size."""
    for node_id, node in iter_nodes(workflow, "ResizeImageMaskNode"):
        if not any(node_id.endswith(suffix) for suffix in node_suffixes):
            continue
        inputs = node.setdefault("inputs", {})
        # ComfyUI's DynamicCombo serialises the active branch as dotted
        # fields in the API prompt.  Removing the old ``multiple`` branch is
        # important: otherwise the linked guide-video size silently wins.
        inputs.pop("resize_type.multiple", None)
        inputs["resize_type.width"] = width
        inputs["resize_type.height"] = height
        inputs["resize_type.crop"] = "disabled"
        inputs["scale_method"] = "lanczos"


def _set_models(workflow, lora=None):
    for _, node in iter_nodes(workflow, "UNETLoader"):
        node.setdefault("inputs", {})["unet_name"] = "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
    vaes = list(iter_nodes(workflow, "VAELoader"))
    for node_id, node in vaes:
        title = str((node.get("_meta") or {}).get("title") or "").lower()
        if node_id.endswith(":227") or "audio" in title:
            node.setdefault("inputs", {})["vae_name"] = "ltx-2.5-audio-vae-bf16.safetensors"
        else:
            node.setdefault("inputs", {})["vae_name"] = "ltx-2.5-video-vae-bf16.safetensors"
    for _, node in iter_nodes(workflow, "CLIPLoader"):
        clip_name = str((node.get("inputs") or {}).get("clip_name") or "")
        if "e2b" in clip_name.lower():
            node.setdefault("inputs", {})["clip_name"] = "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
        else:
            node.setdefault("inputs", {})["clip_name"] = "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
    for _, node in iter_nodes(workflow, "LTXICLoRALoaderModelOnly"):
        if lora:
            node.setdefault("inputs", {})["lora_name"] = lora


def _prepare_official(kind):
    workflow = compile_workflow(_workflow_path(kind))
    disable_prompt_enhancer(workflow)
    # The official Union/FLF graphs include a separate enhancer CLIP loader.
    # It is not needed once the enhancer branches are removed and must never
    # be allowed to load the optional enhancer model.
    for node_id, node in list(workflow.items()):
        if node.get("class_type") == "CLIPLoader" and (
            "e2b" in str((node.get("inputs") or {}).get("clip_name") or "").lower()
            or node_id.endswith(":251:246")
        ):
            workflow.pop(node_id, None)
    return workflow


def build_ltx25_ingredients(job_input):
    """Official Ingredients IC-LoRA identity/reference workflow."""
    prompt, negative = _prompt_values(job_input)
    fps, duration, frames, aspect, quality, width, height, seed = _generation_values(job_input)
    references = _images(job_input)
    if not references:
        raise ValueError("Ingredients / Reference → Video requires an uploaded image reference")
    if len(references) > 1 and not job_input.get("allow_multiple_references"):
        raise ValueError("The official Ingredients workflow accepts one reference sheet per generation")
    ingredients_lora = _require_any_model(INGREDIENTS_LORA_FILENAMES, "Ingredients / identity")
    _require_models([
        "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
        "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
        "ltx-2.5-video-vae-bf16.safetensors",
        "ltx-2.5-audio-vae-bf16.safetensors",
    ])
    workflow = _prepare_official("ingredients")
    _set_prompt_nodes(workflow, prompt, negative)
    _set_common_generation(workflow, fps, frames, width, height, seed, "mirrorvidgen_ingredients")
    _set_duration_expressions(workflow, fps, duration)
    _set_models(workflow, ingredients_lora)
    for _, node in iter_nodes(workflow, "LoadImage"):
        node.setdefault("inputs", {})["image"] = references[0]
    _set_resize_dimensions(workflow, width, height, (":5014:4990",))
    # Ingredients uses the reference as a repeated IC-LoRA guide, not frame 1.
    _set_ic_lora_guide(workflow, reference_only=True)
    return workflow, {
        "mode": "ingredients_to_video",
        "reference_conditioning": "ingredients_ic_lora",
        "prompt_enhancer": False,
        "fps": fps,
        "requested_duration_seconds": duration,
        "actual_duration_seconds": (frames - 1) / fps,
        "frames": frames,
        "aspect_ratio": aspect,
        "quality": quality,
        "stage1_width": width,
        "stage1_height": height,
        "output_width": width * 2,
        "output_height": height * 2,
        "seed": seed,
        "reference_image_name": references[0],
        "ingredients_lora": ingredients_lora,
    }


def build_ltx25_motion_track(job_input):
    """Official Motion Track IC-LoRA workflow with real sparse tracks."""
    prompt, negative = _prompt_values(job_input)
    fps, duration, frames, aspect, quality, width, height, seed = _generation_values(job_input)
    references = _images(job_input)
    tracks = job_input.get("motion_tracks")
    if not references:
        raise ValueError("Motion Track requires an input image")
    if not isinstance(tracks, str) or not tracks.strip() or tracks.strip() == "[]":
        raise ValueError("Motion Track requires the user's sparse track guide")
    try:
        parsed_tracks = json.loads(tracks)
        if not isinstance(parsed_tracks, list) or not parsed_tracks:
            raise ValueError
    except (TypeError, ValueError, json.JSONDecodeError):
        raise ValueError("input.motion_tracks must be a JSON array of sparse tracks")
    _require_models([
        "ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors",
        "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
        "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
        "ltx-2.5-video-vae-bf16.safetensors",
        "ltx-2.5-audio-vae-bf16.safetensors",
    ])
    workflow = _prepare_official("motion_track")
    _set_prompt_nodes(workflow, prompt, negative)
    _set_common_generation(workflow, fps, frames, width, height, seed, "mirrorvidgen_motion_track")
    _set_duration_expressions(workflow, fps, duration)
    _set_models(workflow, "ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors")
    for _, node in iter_nodes(workflow, "LoadImage"):
        node.setdefault("inputs", {})["image"] = references[0]
    _set_resize_dimensions(workflow, width, height, (":5014:4990", ":5014:5562"))
    _set_ic_lora_guide(workflow, reference_only=False)
    # The official Motion Track graph conditions on the supplied still as its
    # opening frame.  Do not bypass that real image-conditioning node.
    for _, node in iter_nodes(workflow, "LTXVImgToVideoInplace"):
        node.setdefault("inputs", {})["bypass"] = False
    for _, node in iter_nodes(workflow, "LTXVSparseTrackEditor"):
        node.setdefault("inputs", {}).update({"points_store": "[]", "coordinates": tracks, "points_to_sample": frames})
    return workflow, {
        "mode": "motion_track_to_video",
        "reference_conditioning": "motion_track_ic_lora",
        "prompt_enhancer": False,
        "fps": fps,
        "requested_duration_seconds": duration,
        "actual_duration_seconds": (frames - 1) / fps,
        "frames": frames,
        "aspect_ratio": aspect,
        "quality": quality,
        "stage1_width": width,
        "stage1_height": height,
        "output_width": width * 2,
        "output_height": height * 2,
        "seed": seed,
        "motion_tracks": tracks,
    }


def build_ltx25_union_control(job_input, control_mode):
    """Official Union Control workflow for Pose, Depth, or Canny."""
    control_mode = str(control_mode or "").lower()
    if control_mode not in {"pose", "depth", "canny"}:
        raise ValueError("input.control_mode must be pose, depth, or canny")
    prompt, negative = _prompt_values(job_input)
    fps, duration, frames, aspect, quality, width, height, seed = _generation_values(job_input)
    guides = _videos(job_input) or _videos(job_input, "Motion")
    if not guides:
        raise ValueError("Union Control requires an uploaded guide video")
    _require_models([
        "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
        "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
        "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
        "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
        "ltx-2.5-video-vae-bf16.safetensors",
        "ltx-2.5-audio-vae-bf16.safetensors",
    ])
    if control_mode == "depth" and not _model_present("video_depth_anything_vits.pth"):
        raise CapabilityUnavailable("Depth Control requires the VideoDepthAnything model")
    if control_mode == "pose":
        _require_models(["yolox_l.onnx", "dw-ll_ucoco_384_bs5.torchscript.pt"])
    workflow = _prepare_official("union_control")
    _set_prompt_nodes(workflow, prompt, negative)
    _set_common_generation(workflow, fps, frames, width, height, seed, f"mirrorvidgen_{control_mode}")
    _set_models(workflow, "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors")
    _set_ic_lora_guide(workflow, reference_only=False)
    for _, node in iter_nodes(workflow, "LoadVideo"):
        node.setdefault("inputs", {})["video"] = guides[0]
    # The official guide graph wires Depth by default. Switch the actual
    # conditioning input, never merely a UI label, for Pose/Canny.
    guide_resize = next((node for node_id, node in workflow.items() if node_id.endswith(":5548:5028")), None)
    if guide_resize is not None:
        guide_resize.setdefault("inputs", {})["input"] = {
            "pose": ["r:5548:4986", 0],
            "canny": ["r:5548:4991", 0],
            "depth": ["r:5548:5062", 0],
        }[control_mode]
    _set_resize_dimensions(workflow, width, height, (":5548:5028",))
    # Do not force a still start frame for a pure control-video workflow.
    for _, node in iter_nodes(workflow, "LTXVImgToVideoInplace"):
        node.setdefault("inputs", {})["bypass"] = not bool(_images(job_input, "Start"))
    for _, node in iter_nodes(workflow, "LoadImage"):
        starts = _images(job_input, "Start")
        node.setdefault("inputs", {})["image"] = starts[0] if starts else ("" if node.get("inputs", {}).get("image") == "image" else node.get("inputs", {}).get("image", ""))
    return workflow, {
        "mode": f"{control_mode}_to_video",
        "reference_conditioning": f"union_control_{control_mode}",
        "prompt_enhancer": False,
        "fps": fps,
        "requested_duration_seconds": duration,
        "actual_duration_seconds": (frames - 1) / fps,
        "frames": frames,
        "aspect_ratio": aspect,
        "quality": quality,
        "stage1_width": width,
        "stage1_height": height,
        "output_width": width * 2,
        "output_height": height * 2,
        "seed": seed,
        "control_mode": control_mode,
        "guide_video_name": guides[0],
    }


def build_ltx25_first_last(job_input):
    """Official First & Last Frame workflow; both images are real guides."""
    prompt, negative = _prompt_values(job_input)
    fps, duration, frames, aspect, quality, width, height, seed = _generation_values(job_input)
    starts = _images(job_input, "Start")
    ends = _images(job_input, "End")
    if not starts or not ends:
        raise ValueError("First + Last requires one START image and one END image")
    _require_models([
        "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
        "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
        "ltx-2.5-video-vae-bf16.safetensors",
        "ltx-2.5-audio-vae-bf16.safetensors",
    ])
    workflow = _prepare_official("first_last")
    _set_prompt_nodes(workflow, prompt, negative)
    _set_common_generation(workflow, fps, frames, width, height, seed, "mirrorvidgen_first_last")
    _set_models(workflow)
    loaders = [(node_id, node) for node_id, node in iter_nodes(workflow, "LoadImage")]
    for node_id, node in loaders:
        node.setdefault("inputs", {})["image"] = starts[0] if node_id.endswith(":31") else ends[0]
    # The official graph uses these stable node IDs for the two latent guides.
    for suffix, image_name in ((":251:213", starts[0]), (":251:214", ends[0])):
        node = next((n for node_id, n in workflow.items() if node_id.endswith(suffix)), None)
        if node:
            node.setdefault("inputs", {}).update({"resize_type.width": width, "resize_type.height": height})
    for node_id, node in workflow.items():
        if node_id.endswith(":251:226") and node.get("class_type") == "ComfyMathExpression":
            node.setdefault("inputs", {}).update({"values.a": duration, "values.b": fps})
        if node_id.endswith(":251:212") and node.get("class_type") == "ComfyMathExpression":
            node.setdefault("inputs", {})["values.a"] = fps
        if node_id.endswith(":251:230"):
            node.setdefault("inputs", {})["unet_name"] = "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
        if node_id.endswith(":251:229"):
            node.setdefault("inputs", {})["vae_name"] = "ltx-2.5-video-vae-bf16.safetensors"
        if node_id.endswith(":251:227"):
            node.setdefault("inputs", {})["vae_name"] = "ltx-2.5-audio-vae-bf16.safetensors"
        if node_id.endswith(":251:228"):
            node.setdefault("inputs", {})["clip_name"] = "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
    return workflow, {
        "mode": "first_last_frame_to_video",
        "reference_conditioning": "first_and_last_frame",
        "prompt_enhancer": False,
        "fps": fps,
        "requested_duration_seconds": duration,
        "actual_duration_seconds": (frames - 1) / fps,
        "frames": frames,
        "aspect_ratio": aspect,
        "quality": quality,
        "stage1_width": width,
        "stage1_height": height,
        "output_width": width * 2,
        "output_height": height * 2,
        "seed": seed,
        "start_image_name": starts[0],
        "end_image_name": ends[0],
    }


def worker_capabilities():
    """Return conservative capability state from installed files only."""
    def workflow_ready(kind):
        try:
            _workflow_path(kind)
            return True
        except CapabilityUnavailable:
            return False

    core = [
        "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
        "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
        "ltx-2.5-video-vae-bf16.safetensors",
        "ltx-2.5-audio-vae-bf16.safetensors",
    ]
    union_lora = "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"
    union_ready = workflow_ready("union_control") and all(_model_present(name) for name in core + [union_lora])
    ingredients_ready = (
        workflow_ready("ingredients")
        and all(_model_present(name) for name in core)
        and _first_present(INGREDIENTS_LORA_FILENAMES) is not None
    )
    deblur_ready = (
        workflow_ready("continuation")
        and all(_model_present(name) for name in core)
        and _first_present(DEBLUR_LORA_FILENAMES) is not None
    )
    return {
        "text_to_video": True,
        "image_to_video": True,
        "first_frame_to_video": True,
        "ingredients": ingredients_ready,
        "first_last": workflow_ready("first_last") and all(_model_present(name) for name in core),
        "motion_track": workflow_ready("motion_track") and all(_model_present(name) for name in core + ["ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors"]),
        "pose": union_ready and all(_model_present(name) for name in ["yolox_l.onnx", "dw-ll_ucoco_384_bs5.torchscript.pt"]),
        "canny": union_ready,
        "depth": union_ready and _model_present("video_depth_anything_vits.pth"),
        # Director uses the same real V2V continuation graph between ordered
        # segments; the handler owns the chain and stitches one result.
        "director": deblur_ready,
        "director30": deblur_ready,
        "continue": deblur_ready,
        "extend": deblur_ready,
    }


def _director_segments(job_input):
    requested = float(job_input.get("duration_seconds", 30))
    raw = job_input.get("director_segments")
    if raw is None:
        if requested == 30:
            return [
                {"duration_seconds": 10.0, "prompt": str(job_input.get("prompt") or "").strip()},
                {"duration_seconds": 10.0, "prompt": str(job_input.get("director_follow_on_1") or "").strip()},
                {"duration_seconds": 10.0, "prompt": str(job_input.get("director_follow_on_2") or "").strip()},
            ]
        return [{"duration_seconds": requested, "prompt": str(job_input.get("prompt") or "").strip()}]
    if not isinstance(raw, list) or not raw:
        raise ValueError("director_segments must be a non-empty ordered list")
    segments = []
    for index, item in enumerate(raw):
        if isinstance(item, dict):
            duration = float(item.get("duration_seconds", item.get("duration", 0)))
            prompt = str(item.get("prompt") or item.get("follow_on_prompt") or "").strip()
        else:
            duration = float(item)
            prompt = ""
        if duration < 1 or duration > 20:
            raise ValueError(f"Director segment {index + 1} must be between 1 and 20 seconds")
        if not prompt and index == 0:
            prompt = str(job_input.get("prompt") or "").strip()
        if index == 0 and not prompt:
            raise ValueError(f"Director segment {index + 1} needs an ordered prompt")
        segments.append({"duration_seconds": duration, "prompt": prompt})
    if requested == 30 and abs(sum(item["duration_seconds"] for item in segments) - 30) > 0.01:
        raise ValueError("30s Director segments must total exactly 30 seconds")
    return segments


def build_ltx25_director(job_input):
    """Preflight/build the first real segment of a chained Director job."""
    segments = _director_segments(job_input)
    _require_any_model(DEBLUR_LORA_FILENAMES, "V2V continuation")
    _require_models([
        "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
        "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
        "ltx-2.5-video-vae-bf16.safetensors",
        "ltx-2.5-audio-vae-bf16.safetensors",
    ])
    _workflow_path("continuation")
    first_input = copy.deepcopy(job_input)
    first_input["duration_seconds"] = segments[0]["duration_seconds"]
    first_mode = str(job_input.get("director_initial_mode") or "text_to_video").lower()
    if first_mode in {"reference_to_video", "ingredients_to_video", "character_to_video"}:
        workflow, first_settings = build_ltx25_ingredients(first_input)
    elif first_mode in {"image_to_video", "first_frame_to_video", "ltx25_i2v"}:
        workflow, first_settings = build_ltx25_i2v(first_input)
    elif first_mode in {"first_last_frame_to_video", "first_and_last_frame_to_video"}:
        workflow, first_settings = build_ltx25_first_last(first_input)
    else:
        workflow, first_settings = build_ltx25_t2v(first_input)
    first_settings = dict(first_settings)
    first_settings.update({
        "mode": "director_video",
        "director_segments": segments,
        "director_segment_count": len(segments),
        "requested_duration_seconds": sum(item["duration_seconds"] for item in segments),
    })
    return workflow, first_settings


def _source_video_name(job_input):
    explicit = str(job_input.get("source_video_name") or "").strip()
    if explicit:
        return Path(explicit).name
    for key in ("source_video", "video"):
        item = job_input.get(key)
        if isinstance(item, dict):
            name = _asset_name(item)
            if name:
                return name
        elif isinstance(item, str) and item.strip():
            return Path(item).name
    for role in ("Source", "Tail", "Video"):
        values = _videos(job_input, role)
        if values:
            return values[0]
    values = _videos(job_input)
    return values[0] if values else None


def build_ltx25_continuation(job_input):
    """Build the official V2V IC-LoRA graph for Continue/Extend.

    The worker receives a short tail context (materialised by the handler),
    conditions the real LTX V2V graph on that video, and allocates a latent
    containing context + new frames.  ``LTXVCropGuides`` removes the guided
    context after sampling, leaving a genuinely new continuation segment.
    """
    source_name = _source_video_name(job_input)
    if not source_name:
        raise ValueError("Continue Video requires the completed source MP4")
    mode = str(job_input.get("mode") or "continue_video").lower()
    added_duration = float(
        job_input.get("added_duration_seconds", job_input.get("duration_seconds", 2))
    )
    if added_duration < 1 or added_duration > 20:
        raise ValueError("Added continuation duration must be between 1 and 20 seconds")
    context_duration = float(job_input.get("continuation_context_seconds", 2))
    if context_duration < 1 or context_duration > 8:
        raise ValueError("Continuation context must be between 1 and 8 seconds")
    fps = int(job_input.get("fps", 24))
    if fps != 24:
        raise ValueError("MiRROR Motion 1 currently supports 24 fps only")
    aspect = str(job_input.get("aspect_ratio", "16:9"))
    quality = str(job_input.get("quality", "480p"))
    width, height = _stage1_dimensions(quality, aspect)
    added_frames = _snap_frames(added_duration, fps)
    context_frames = _snap_frames(context_duration, fps)
    total_frames = context_frames + added_frames - 1
    # A sum of two valid (1 mod 8) frame counts minus one remains valid.
    if total_frames % 8 != 1:
        raise CapabilityUnavailable("Continuation context and added duration do not form a valid LTX frame grid")
    follow_on = str(job_input.get("follow_on_prompt") or "").strip()
    if not follow_on:
        follow_on = (
            "Continue this exact shot naturally from the final state. Preserve the same subjects, "
            "identity, environment, lighting, camera, motion direction and visual style. "
            "Introduce only plausible next motion that follows directly from the source video."
        )
    if len(follow_on) > 12000:
        raise ValueError("input.follow_on_prompt is too long (max 12000 characters)")
    negative = str(job_input.get("negative_prompt") or DEFAULT_NEGATIVE_PROMPT).strip()
    deblur_lora = _require_any_model(DEBLUR_LORA_FILENAMES, "V2V continuation")
    _require_models([
        "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
        "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
        "ltx-2.5-video-vae-bf16.safetensors",
        "ltx-2.5-audio-vae-bf16.safetensors",
    ])
    workflow = _prepare_official("continuation")
    _set_prompt_nodes(workflow, follow_on, negative)
    _set_common_generation(workflow, fps, total_frames, width, height, int(job_input.get("seed", 42)), "mirrorvidgen_continuation")
    _set_models(workflow, deblur_lora)
    for _, node in iter_nodes(workflow, "LoadVideo"):
        node.setdefault("inputs", {})["video"] = source_name
    # V2V dimensions/fps normally come from the guide. The handler prepares
    # the source tail at the selected native dimensions and 24 fps; these
    # explicit latent/audio lengths add the new segment beyond that context.
    for _, node in iter_nodes(workflow, "EmptyLTXVLatentVideo"):
        node.setdefault("inputs", {}).update({"width": width, "height": height, "length": total_frames})
    for _, node in iter_nodes(workflow, "LTXVEmptyLatentAudio"):
        node.setdefault("inputs", {}).update({"frames_number": total_frames, "frame_rate": fps})
    for _, node in iter_nodes(workflow, "LTXAddVideoICLoRAGuide"):
        node.setdefault("inputs", {}).update({
            "frame_idx": 0,
            "strength": 1.0,
            "crop": "disabled",
            "use_tiled_encode": False,
            "tile_size": 256,
            "tile_overlap": 64,
        })
    return workflow, {
        "mode": "extend_video" if mode == "extend_video" else "continue_video",
        "reference_conditioning": "source_video_temporal_v2v",
        "prompt_enhancer": False,
        "fps": fps,
        "requested_added_duration_seconds": added_duration,
        "actual_added_duration_seconds": (added_frames - 1) / fps,
        "context_duration_seconds": (context_frames - 1) / fps,
        "context_frames": context_frames,
        "added_frames": added_frames,
        "total_latent_frames": total_frames,
        "aspect_ratio": aspect,
        "quality": quality,
        "stage1_width": width,
        "stage1_height": height,
        "output_width": width * 2,
        "output_height": height * 2,
        "seed": int(job_input.get("seed", 42)),
        "source_video_name": source_name,
        "follow_on_prompt": follow_on,
        "deblur_lora": deblur_lora,
    }
