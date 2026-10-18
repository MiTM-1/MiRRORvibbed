"""Measure the actual supplied trimmed soundtrack; no synthetic beat grid.

Version 2 keeps each frequency band on its own robust scale.  The original
implementation divided every band by the full-range waveform peak.  That made
real cymbal/snare energy almost disappear on bass-heavy masters and was the
main reason the renderer only appeared to breathe very slightly.
"""
import array
import hashlib
import math
import subprocess
import sys

RATE = 12000
HOP = 120  # 10ms, on the selected clip's clock

def analyse(audio, duration):
    decoded = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(audio),
        '-t', str(duration), '-ac', '1', '-ar', str(RATE), '-f', 'f32le', 'pipe:1'])
    samples = array.array('f'); samples.frombytes(decoded)
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
    raw = {key: list(values) for key, values in bands.items()}

    def percentile(values, fraction):
        ordered = sorted(values)
        if not ordered: return 0.
        return ordered[min(len(ordered)-1, max(0, round((len(ordered)-1)*fraction)))]

    # Mastered music often has a narrow dynamic range.  Robust per-band drive
    # signals expose real changes.  The original physical-relative bands are
    # also retained so analysis/tests never pretend quiet highs equal loud bass.
    drive={}
    for key, original in raw.items():
        values=list(original)
        floor = percentile(values, .08)
        ceiling = max(floor + .0005, percentile(values, .97))
        previous = 0.
        for i, value in enumerate(values):
            target = min(1., max(0., (value-floor)/(ceiling-floor)))
            target = math.sqrt(target)
            previous += (.72 if target > previous else .12) * (target-previous)
            values[i] = round(previous, 6)
        drive[key+'Drive']=values
    peak=max(raw['envelope'],default=0.);scale=max(.02,peak)
    for key,values in bands.items():
        previous=0.
        for i,value in enumerate(values):
            target=min(1.,value/scale) if value>.0001 else 0.
            previous += (.8 if target>previous else .16)*(target-previous)
            values[i]=round(previous,6)
    bands.update(drive)

    def impacts(key, lookback, threshold):
        values = raw[key]; result=[]; previous=0.
        for i, value in enumerate(values):
            before=values[max(0,i-lookback):i]
            baseline=sum(before)/len(before) if before else value
            ceiling=max(.0005, percentile(values[max(0,i-120):i+1], .90))
            target=max(0., min(1., (value-baseline-threshold*ceiling)/(ceiling*.7)))
            previous += (.92 if target>previous else .22)*(target-previous)
            result.append(round(previous,6))
        return result

    # These are measured transient envelopes, not inferred instrument labels.
    # Names describe the visual mapping: low-band impacts drive bass motion;
    # mid/high impacts drive sharper accent flashes.
    bands['bassImpact'] = impacts('bass', 18, .025)
    combined=[math.sqrt(raw['mids'][i]**2+raw['highs'][i]**2) for i in range(len(raw['mids']))]
    raw['accent']=combined
    values=raw['accent']; result=[]; previous=0.
    for i,value in enumerate(values):
        before=values[max(0,i-12):i]; baseline=sum(before)/len(before) if before else value
        ceiling=max(.0005, percentile(values[max(0,i-100):i+1],.90))
        target=max(0.,min(1.,(value-baseline-.02*ceiling)/(ceiling*.62)))
        previous += (.94 if target>previous else .28)*(target-previous)
        result.append(round(previous,6))
    bands['accentImpact']=result
    levels = bands['envelopeDrive']; impact=bands['accentImpact']; bass_impact=bands['bassImpact']; onsets = []
    for i in range(2, len(levels)-3):
        baseline = sum(levels[max(0, i-20):i]) / min(i, 20)
        strength=max(impact[i],bass_impact[i])
        local_peak=strength>=max(max(impact[i-2:i+3]),max(bass_impact[i-2:i+3]))
        if (strength > .48 and local_peak) or (levels[i] > max(.16, baseline*1.38) and levels[i]-levels[i-1] > .06):
            if not onsets or i*.01-onsets[-1] >= .18: onsets.append(round(i*.01, 3))
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
    # Broad measured energy movement gives the compositor deliberate build/drop
    # points instead of random periodic transitions.
    builds=[]; drops=[]; window=80
    for i in range(window,len(levels)-window,20):
        before=sum(levels[i-window:i])/window; after=sum(levels[i:i+window])/window
        if after-before>.16: builds.append(round(i*.01,3))
        elif before-after>.18: drops.append(round(i*.01,3))
    return {'version': 2, 'hopSeconds': .01, 'bands': bands, 'onsets': onsets,
        'beats': beats, 'detectedBpm': detected,
        'builds': builds, 'drops': drops,
        'decodedPcmSha256': hashlib.sha256(decoded).hexdigest(),
        'method': 'Measured selected-clip PCM with per-band robust normalization, low/accent transient envelopes and onset-derived tempo; no semantic or genre inference'}

def at(features, band, time):
    values = features['bands'][band]
    return values[min(len(values)-1, max(0, int(time / features['hopSeconds'])))] if values else 0.
