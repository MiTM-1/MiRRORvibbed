"""CPU-only image validation. It never submits H3 or contacts production."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import math
import wave
import struct
import statistics
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
    assert vision["rendererVersion"] == 1 and vision["qualityVersion"] == 2 and vision["native2d"] is True
    assert vision["advanced3d"] is False and vision["previousFrameConditioning"] is True
    assert vision["previousFrameMechanism"] == "decoded-final-frame-reference"
    subprocess.run([sys.executable, "-m", "py_compile", "/handler.py", "/mirrorvision_renderer.py", "/audio_features.py"], check=True)
    import mirrorvision_renderer as renderer
    renderer.dimensions = lambda _: (320, 180)
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder); audio = root / "selected.wav"; image = root / "art.png"; output = root / "teaser.mp4"
        Image.new("RGB", (320, 180), (28, 68, 140)).save(image)
        with wave.open(str(audio), "wb") as track:
            track.setnchannels(1); track.setsampwidth(2); track.setframerate(12000)
            track.writeframes(b"".join(struct.pack("<h", int(16000*math.sin(2*math.pi*80*i/12000)*math.exp(-((i/12000)%0.5)*18))) for i in range(5*12000)))
        plan = {"mode":"Mystery Teaser","duration":5,"aspect":"16:9","words":[],"ordered":True,"audio":{},"scenes":[]}
        renderer.render(plan, str(audio), [str(image)], [], str(output))
        probe = json.loads(subprocess.check_output(["ffprobe","-v","error","-show_streams","-show_format","-of","json",str(output)]))
        assert {s["codec_type"] for s in probe["streams"]} >= {"audio", "video"}
        assert abs(float(probe["format"]["duration"])-5) < .12
        measured = renderer.analyse(str(audio), 5)
        raw = subprocess.check_output(["ffmpeg","-v","error","-i",str(output),"-vf","scale=1:1","-f","rawvideo","-pix_fmt","rgb24","pipe:1"])
        levels = [sum(raw[i:i+3])/3 for i in range(0,len(raw),3)]
        bass = [renderer.feature_at(measured,"bassDrive",i/24) for i in range(len(levels))]
        sync = statistics.correlation(levels[:-8], bass[:-8])
        assert sync > .55, f"Bass/visual synchronization regressed: {sync}"
        assert max(levels[:12])-min(levels[:12]) > 2, "Missing first-event visual response"
    print(json.dumps({"bass_visual_correlation": sync, "gpu_used": False, "paid_generation": False, "rendered_fixture_mp4": True, "mirrorvision": vision}, indent=2))


if __name__ == "__main__":
    main()
