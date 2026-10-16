"""CPU-only image validation. It never submits H3 or contacts production."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    assert shutil.which("ffmpeg") and shutil.which("ffprobe")
    assert os.path.isfile("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    from mirrorvision_dispatch import _healthy, _capabilities
    assert _healthy()
    probe = _capabilities({"status": "capabilities", "capabilities": {"mirrorromax_h3": {"available": True}}})
    vision = probe["capabilities"]["mirrorvision"]
    assert probe["capabilities"]["mirrorromax_h3"]["available"] is True
    assert vision["rendererVersion"] == 1 and vision["native2d"] is True
    assert vision["advanced3d"] is False and vision["previousFrameConditioning"] is False
    subprocess.run([sys.executable, "-m", "py_compile", "/handler.py", "/mirrorvision_renderer.py", "/audio_features.py"], check=True)
    import mirrorvision_renderer as renderer
    renderer.dimensions = lambda _: (320, 180)
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder); audio = root / "selected.wav"; image = root / "art.png"; output = root / "teaser.mp4"
        Image.new("RGB", (320, 180), (28, 68, 140)).save(image)
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=80:sample_rate=12000:duration=5", str(audio)], check=True)
        plan = {"mode":"Mystery Teaser","duration":5,"aspect":"16:9","words":[],"ordered":True,"audio":{},"scenes":[]}
        renderer.render(plan, str(audio), [str(image)], [], str(output))
        probe = json.loads(subprocess.check_output(["ffprobe","-v","error","-show_streams","-show_format","-of","json",str(output)]))
        assert {s["codec_type"] for s in probe["streams"]} >= {"audio", "video"}
        assert abs(float(probe["format"]["duration"])-5) < .12
    print(json.dumps({"gpu_used": False, "paid_generation": False, "rendered_fixture_mp4": True, "mirrorvision": vision}, indent=2))


if __name__ == "__main__":
    main()
