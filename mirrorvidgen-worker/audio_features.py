"""Measure the actual supplied trimmed soundtrack; no synthetic beat grid."""
import array
import hashlib
import math
import subprocess
import sys

RATE = 12000
HOP = 120  # 10ms, on the selected clip's clock

def analyse(audio, duration):
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(audio),
        '-t', str(duration), '-ac', '1', '-ar', str(RATE), '-f', 'f32le', 'pipe:1'])
    samples = array.array('f'); samples.frombytes(raw)
    if sys.byteorder != 'little': samples.byteswap()
    if len(samples) < round(duration * RATE) - RATE // 10:
        raise ValueError('Selected soundtrack is shorter than the requested video')
    bands = {key: [] for key in ('bass', 'mids', 'highs', 'envelope')}
    low = upper = 0.
    a = 1 - math.exp(-2 * math.pi * 150 / RATE)
    b = 1 - math.exp(-2 * math.pi * 2000 / RATE)
    for start in range(0, len(samples), HOP):
        sums = [0., 0., 0., 0.]
        block = samples[start:start + HOP]
        for value in block:
            low += a * (value - low); upper += b * (value - upper)
            for i, v in enumerate((low, upper-low, value-upper, value)):
                sums[i] += v*v
        for key, total in zip(bands, sums):
            bands[key].append(math.sqrt(total / max(1, len(block))))
    # A shared scale retains relative band strength. Silence stays zero.
    peak = max(bands['envelope'], default=0.)
    scale = max(.02, peak)
    for key in bands:
        previous = 0.
        for i, value in enumerate(bands[key]):
            target = min(1., value / scale) if value > .0001 else 0.
            previous += (.8 if target > previous else .16) * (target-previous)
            bands[key][i] = round(previous, 6)
    levels = bands['envelope']; onsets = []
    for i in range(1, len(levels)):
        baseline = sum(levels[max(0, i-20):i]) / min(i, 20)
        if levels[i] > max(.08, baseline*1.7) and levels[i]-levels[i-1] > .06:
            if not onsets or i*.01-onsets[-1] >= .12: onsets.append(round(i*.01, 3))
    # Tempo is measured from onset intervals. It is intentionally omitted when
    # the clip does not contain enough stable evidence; the renderer never
    # manufactures a beat grid for silence or ambiguous material.
    candidates = []
    for i, left in enumerate(onsets):
        for right in onsets[i + 1:i + 9]:
            gap = right - left
            while gap and 60 / gap < 70: gap /= 2
            while gap and 60 / gap > 180: gap *= 2
            if gap: candidates.append(60 / gap)
    detected = None
    beats = []
    if len(candidates) >= 3:
        candidates.sort(); median = candidates[len(candidates)//2]
        stable = [v for v in candidates if abs(v-median) <= max(4, median*.06)]
        if len(stable) >= 3:
            detected = round(sum(stable)/len(stable), 2)
            period = 60/detected
            anchor = onsets[0]
            beats = [round(anchor+i*period, 3) for i in range(math.ceil((duration-anchor)/period)) if anchor+i*period < duration]
    return {'version': 1, 'hopSeconds': .01, 'bands': bands, 'onsets': onsets,
        'beats': beats, 'detectedBpm': detected,
        'decodedPcmSha256': hashlib.sha256(raw).hexdigest(),
        'method': 'Measured selected-clip PCM RMS, first-order 150Hz/2kHz crossover and onset-derived tempo; no semantic or genre inference'}

def at(features, band, time):
    values = features['bands'][band]
    return values[min(len(values)-1, max(0, int(time / features['hopSeconds'])))] if values else 0.
