import copy
import json
from pathlib import Path

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
