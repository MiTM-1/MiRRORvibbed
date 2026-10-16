"""Actual decoded PCM fixtures: silence, bands, onset clock and attack/decay."""
import importlib.util, math, struct, tempfile, wave
from pathlib import Path
spec=importlib.util.spec_from_file_location('audio','mirrorvidgen-worker/audio_features.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
with tempfile.TemporaryDirectory() as temp:
    def fixture(name, frequency):
        path=Path(temp)/(name+'.wav')
        with wave.open(str(path),'wb') as out:
            out.setnchannels(1);out.setsampwidth(2);out.setframerate(12000)
            out.writeframes(b''.join(struct.pack('<h',int(14000*math.sin(i*2*math.pi*frequency/12000))) if 12000<=i<18000 else b'\0\0' for i in range(36000)))
        return a.analyse(path,3)
    low=fixture('bass',80); high=fixture('treble',4000); silence=fixture('silence',0)
    assert not silence['onsets'] and max(silence['bands']['envelope'])==0
    assert low['bands']['bass'][120]>low['bands']['highs'][120]*3
    assert high['bands']['highs'][120]>high['bands']['bass'][120]*3
    assert any(1<=t<=1.04 for t in low['onsets']),low['onsets']
    assert a.at(low,'bass',.9)==0 and a.at(low,'bass',1.04)>.3
    assert a.at(low,'bass',2)<.001
    assert low['decodedPcmSha256']!=high['decodedPcmSha256']
    print('PASS actual PCM: silence, band separation, onset within 40ms, first-event response, decay, input hash')
