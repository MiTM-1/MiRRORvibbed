import copy
import json
from pathlib import Path

TEMPLATE_PATH = Path("/ltx25-t2v-template.json")

DEFAULT_NEGATIVE_PROMPT = (
    "cartoon, illustration, video game, low quality, distorted, text, watermark"
)

# Stage-1 dimensions. The latent spatial upscaler produces 2x output.
# Every stage-1 dimension is divisible by 32, as required by LTX-2.5.
STAGE1_PRESETS = {
    "smoke": {
        "16:9": (320, 192),
        "9:16": (192, 320),
        "1:1": (256, 256),
    },
    "480p": {
        "16:9": (448, 256),
        "9:16": (256, 448),
        "1:1": (256, 256),
    },
    "720p": {
        "16:9": (640, 352),
        "9:16": (352, 640),
        "1:1": (352, 352),
    },
    "1080p": {
        "16:9": (960, 544),
        "9:16": (544, 960),
        "1:1": (544, 544),
    },
}


def _snap_frames(seconds, fps):
    requested = max(1, int(round(seconds * fps)))
    if requested % 8 == 1:
        return requested
    return ((requested - 1 + 7) // 8) * 8 + 1


def build_ltx25_t2v(job_input):
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
    if quality not in STAGE1_PRESETS:
        raise ValueError("input.quality must be one of: smoke, 480p, 720p, 1080p")
    if aspect_ratio not in STAGE1_PRESETS[quality]:
        raise ValueError("input.aspect_ratio must be one of: 16:9, 9:16, 1:1")

    seed = int(job_input.get("seed", 42))
    if seed < 0 or seed > 0x7FFFFFFFFFFFFFFF:
        raise ValueError("input.seed is outside the supported range")

    frames = _snap_frames(duration_seconds, fps)
    width, height = STAGE1_PRESETS[quality][aspect_ratio]

    template = json.loads(TEMPLATE_PATH.read_text())
    workflow = copy.deepcopy(template["input"]["workflow"])

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
    }
    return workflow, settings
