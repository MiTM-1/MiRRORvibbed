"""MiRRORvidgen-owned CPU post-production. Never calls RunPod or H3.

FFmpeg minterpolate uses block motion estimation and motion compensation.
It creates intermediate frames; no fps-only duplication or blending filter.
Lanczos resampling increases encoded dimensions, not invented AI detail.
"""
import json,subprocess,time
from fractions import Fraction
from pathlib import Path

def probe(path):
    data=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)]))
    video=next(s for s in data['streams'] if s['codec_type']=='video')
    return {'width':video['width'],'height':video['height'],'fps':float(Fraction(video['avg_frame_rate'])),'duration':float(data['format']['duration']),'audio':any(s['codec_type']=='audio' for s in data['streams'])}

def enhance(source,output,resolution,fps,notify=lambda *args:None):
    start=time.time();original=probe(source)
    if resolution not in {'original','720p','1080p'} or fps not in {'original',60}:raise ValueError('Unsupported output')
    if resolution=='original' and fps=='original':raise ValueError('Choose an enhancement')
    # Quality targets keep the source aspect. Landscape = tier high, portrait
    # = tier wide. The UI receives actual encoded dimensions, never CSS size.
    tier=original['height'] if resolution=='original' else int(resolution[:-1])
    if resolution=='original':w,h=original['width'],original['height']
    elif original['width']>=original['height']:h=tier;w=round(h*original['width']/original['height']/2)*2
    else:w=tier;h=round(w*original['height']/original['width']/2)*2
    filters=[]
    if fps==60:filters.append('minterpolate=fps=60:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1')
    if resolution!='original':filters.append(f'scale={w}:{h}:flags=lanczos')
    filters.append('format=yuv420p')
    notify('analysing_motion' if fps==60 else 'enhancing_detail',0)
    command=['ffmpeg','-y','-hide_banner','-loglevel','error','-i',str(source),'-vf',','.join(filters),'-map','0:v:0','-map','0:a?','-c:v','libx264','-preset','medium','-crf','18','-c:a','copy','-movflags','+faststart','-progress','pipe:1','-nostats',str(output)]
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    for line in process.stdout:
        if line.startswith('out_time_us='):
            elapsed=int(line.split('=')[1])/1000000
            notify('creating_intermediate_frames' if fps==60 else 'enhancing_detail',min(97,elapsed/original['duration']*100))
    error=process.stderr.read();code=process.wait();process.stdout.close();process.stderr.close()
    if code:raise RuntimeError('Video encoding failed')
    notify('finishing',98);result=probe(output)
    if fps==60 and abs(result['fps']-60)>.05:raise RuntimeError('Frame-rate validation failed')
    if abs(result['duration']-original['duration'])>max(.15,2/original['fps']):raise RuntimeError('Duration validation failed')
    if result['width']!=w or result['height']!=h:raise RuntimeError('Dimension validation failed')
    if original['audio'] and not result['audio']:raise RuntimeError('Audio validation failed')
    return {'originalResolution':f"{original['width']}x{original['height']}",'originalFps':original['fps'],'enhancedResolution':f"{w}x{h}",'enhancedFps':result['fps'],'interpolationMethod':'FFmpeg minterpolate MCI/AOBMC' if fps==60 else None,'interpolationVersion':subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0],'upscaler':'Lanczos' if resolution!='original' else None,'upscalerVersion':'FFmpeg','enhancementProcessingTime':time.time()-start,'durationSeconds':result['duration'],'audioPreserved':result['audio']}
