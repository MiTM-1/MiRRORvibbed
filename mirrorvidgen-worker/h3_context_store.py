"""Job-scoped, atomic paired-latent persistence for the candidate worker."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from h3_addons import AddonUnavailable, MOTION_REVISION, safe_job_id


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class ContextStore:
    def __init__(self, root=Path("/runpod-volume/h3-addons-context"), output=Path("/comfyui/output")):
        self.root, self.output = Path(root), Path(output)

    def read(self, job_id):
        folder = self.root / safe_job_id(job_id)
        manifest = json.loads((folder / "manifest.json").read_text())
        latent = folder / "context.safetensors"
        if manifest.get("job_id") != job_id or manifest.get("complete") is not True:
            raise AddonUnavailable("Source context is incomplete or has the wrong identity")
        if not latent.is_file() or digest(latent) != manifest.get("latent_sha256"):
            raise AddonUnavailable("Source latent is missing or failed integrity validation")
        return manifest, latent

    def stage(self, source_id, target_id):
        manifest, latent = self.read(source_id)
        target = self.output / "h3_addons" / safe_job_id(target_id) / "source.safetensors"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise AddonUnavailable("Job context staging already exists; use a new job ID")
        shutil.copyfile(latent, target)
        return manifest, str(target.relative_to(self.output))

    def commit(self, job_id, settings, actual_frames):
        identity = safe_job_id(job_id)
        latent = self.output / "h3_addons" / identity / "clip_00001.safetensors"
        if not latent.is_file() or latent.stat().st_size == 0:
            raise AddonUnavailable("Generation completed without its paired latent checkpoint")
        plan = settings["addons"]
        if actual_frames != plan["delivered_frames"]:
            raise AddonUnavailable("Decoded output does not match the planned overlap duration")
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.root / identity
        if target.exists():
            raise AddonUnavailable("Existing context checkpoint is immutable")
        temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=self.root))
        try:
            copied = temporary / "context.safetensors"
            shutil.copyfile(latent, copied)
            with copied.open("rb") as handle:
                os.fsync(handle.fileno())
            manifest = {"job_id": identity, "complete": True, "workflow": "ref2va",
                        "width": settings["width"], "height": settings["height"], "fps": 24,
                        "motion_revision": MOTION_REVISION, "latent_sha256": digest(copied),
                        "latent_bytes": copied.stat().st_size, "delivered_frames": actual_frames,
                        "source_job_id": plan["source_job_id"], "generation_speed": plan["generation_speed"]}
            with (temporary / "manifest.json").open("w") as handle:
                json.dump(manifest, handle, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.rename(temporary, target)
            return manifest
        except Exception:
            shutil.rmtree(temporary)
            raise
