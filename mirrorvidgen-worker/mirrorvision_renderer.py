"""Native CPU renderer adapter. Install alongside (not instead of) the H3 handler.
Requires FFmpeg + Pillow + a licensed DejaVu Sans font. No browser or GPU.
Run locally: python mirrorvision_renderer.py manifest.json output.mp4
The worker entry point is handle(input); it only accepts mirrorvision_render.
"""
import json, math, os, random, subprocess, tempfile, urllib.request
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps, ImageChops
import importlib.util
_spec = importlib.util.spec_from_file_location('mirrorvision_audio', Path(__file__).with_name('audio_features.py'))
_audio = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_audio)
analyse, feature_at = _audio.analyse, _audio.at

FONT = os.environ.get('MIRRORVISION_FONT', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')
FPS = 24
HTTP_HEADERS = {'User-Agent': 'MiRRORvidgen-Worker/1.0'}

def run(args):
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', *args], check=True, timeout=1800)

def extract_boundary(clip, output):
    """Exact decoded final frame; prepared for a future chained H3 reference upload.
    Not currently wired into music scene orchestration; do not claim conditioning.
    """
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames','-of','json',clip]))
    count=int(info['streams'][0]['nb_read_frames'])
    if count<1: raise ValueError('Previous segment has no frames')
    run(['-i',clip,'-vf',f'select=eq(n\\,{count-1})','-frames:v','1',output])

def validate(plan, images, clips):
    mode = plan.get('mode')
    if mode not in ['Full Music Video', 'Hybrid Music Video', 'Mystery Teaser']:
        raise ValueError('Invalid creation mode')
    durations = [5, 10, 15, 20, 30] if mode == 'Mystery Teaser' else [15, 30, 45, 60, 75, 90]
    if plan.get('duration') not in durations:
        raise ValueError('Invalid render duration')
    if plan.get('aspect') not in ['9:16', '16:9', '1:1', '3:4']:
        raise ValueError('Invalid aspect ratio')
    words = plan.get('words', [])
    if len(words) > 80 or any(not isinstance(w, str) or len(w) > 300 for w in words):
        raise ValueError('Text exceeds renderer limits')
    if mode == 'Mystery Teaser' and not images and not words:
        raise ValueError('Add an image or text')
    if mode == 'Hybrid Music Video' and not (images or clips):
        raise ValueError('Hybrid Music Video needs source artwork or generated footage')
    if mode == 'Full Music Video' and not clips:
        raise ValueError('Generated footage is missing')

def dimensions(aspect):
    return {'9:16': (480, 854), '16:9': (854, 480), '1:1': (640, 640), '3:4': (480, 640)}[aspect]

def draw_text(text, size, width):
    """Rasterize supplied Unicode strings, never generate their letters with H3."""
    font = ImageFont.truetype(FONT, size)
    missing = bytes(font.getmask(chr(0x10FFFF)))
    for char in text:
        if not char.isspace() and bytes(font.getmask(char)) == missing:
            raise ValueError('The installed font cannot render a supplied character: '+repr(char))
    # Greedy wrapping preserves every character, including punctuation/capitalization.
    lines, current = [], ''
    for char in text:
        if char == '\n':
            lines.append(current); current = ''; continue
        if font.getlength(current + char) > width and current:
            lines.append(current); current = ''
        current += char
    lines.append(current)
    h = int(len(lines) * size * 1.38 + size)
    layer = Image.new('RGBA', (width, h))
    draw = ImageDraw.Draw(layer)
    for i, line in enumerate(lines):
        draw.text((width / 2, i * size * 1.38), line, font=font, anchor='mt', fill=(249, 235, 208), stroke_width=1, stroke_fill=(27, 22, 26))
    return layer

def measured_regions(picture):
    """Connected saturated colour regions; no object/subject labels are inferred."""
    small=picture.resize((48,48)).convert('HSV'); seen=set(); found=[]
    for y in range(48):
        for x in range(48):
            if (x,y) in seen: continue
            hue,sat,value=small.getpixel((x,y))
            if sat<110 or value<65: continue
            todo=[(x,y)];seen.add((x,y));points=[]
            while todo:
                px,py=todo.pop();points.append((px,py))
                for nx,ny in ((px-1,py),(px+1,py),(px,py-1),(px,py+1)):
                    if not (0<=nx<48 and 0<=ny<48) or (nx,ny) in seen: continue
                    nh,ns,nv=small.getpixel((nx,ny));delta=abs(nh-hue)
                    if ns>=110 and nv>=65 and min(delta,256-delta)<20:
                        seen.add((nx,ny));todo.append((nx,ny))
            if 8<=len(points)<=1000:
                found.append((len(points),min(a for a,b in points),min(b for a,b in points),max(a for a,b in points)+1,max(b for a,b in points)+1))
    return [tuple(round(v/48*picture.size[i%2]) for i,v in enumerate(item[1:])) for item in sorted(found,reverse=True)[:4]]

def regional_motion(picture, regions, bass):
    # Radial source-coordinate warp, smoothly zero at each region's boundary.
    # No generated pixels, semantic masks, or invented objects.
    if not regions or bass<.001: return picture
    w,h=picture.size; mesh=[];step=24
    def source(x,y):
        dx=dy=0.
        for left,top,right,bottom in regions:
            cx=(left+right)/2;cy=(top+bottom)/2;rx=max(12,(right-left)*.65);ry=max(12,(bottom-top)*.65)
            distance=((x-cx)/rx)**2+((y-cy)/ry)**2
            strength=.055*bass*max(0,1-distance)**2
            dx+=(cx-x)*strength;dy+=(cy-y)*strength
        return (max(0,min(w,x+dx)),max(0,min(h,y+dy)))
    for y in range(0,h,step):
        for x in range(0,w,step):
            right=min(w,x+step);bottom=min(h,y+step)
            quad=(*source(x,y),*source(x,bottom),*source(right,bottom),*source(right,y))
            mesh.append(((x,y,right,bottom),quad))
    return picture.transform((w,h),Image.Transform.MESH,mesh,Image.Resampling.BICUBIC)

def render(plan, audio, images, clips, output):
    validate(plan, images, clips)
    w, h = dimensions(plan['aspect']); duration = plan['duration']
    features = analyse(audio, duration)
    # Persist measured render evidence beside the output for local QA; never log asset URLs.
    Path(str(output)+'.analysis.json').write_text(json.dumps(features))
    analysis = plan.get('audio', {}); beats = features.get('beats', []); onsets = features['onsets']
    changes = analysis.get('changes', []); energy = []
    transitions = sorted(set([0., duration] + [float(v) for v in changes if 0 < v < duration] + [float(v) for v in beats[::8] if 0 < v < duration]))
    if len(transitions) < 3: transitions = [duration * i / 4 for i in range(5)]
    pictures = [ImageOps.pad(Image.open(path).convert('RGB'), (w, h), color=(8,10,17)) for path in images]
    regions = [measured_regions(picture) for picture in pictures]
    rng = random.Random(42)
    particles = [(rng.random(), rng.random(), rng.random(), rng.uniform(.3, 1.)) for _ in range(70)]
    # This is measured palette extraction, not semantic image/object analysis.
    palette = pictures[0].resize((1, 1)).getpixel((0, 0)) if pictures else (79, 142, 155)
    words = plan.get('words', [])
    if not plan.get('ordered', False) and len(words) > 3:
        words = sorted(words[:-2], key=len) + words[-2:]
    text_layers = []
    for text in words:
        size = max(22, int(w * .105))
        layer = draw_text(text, size, int(w * .82))
        while layer.height > h * .55 and size > 12:
            size -= 2; layer = draw_text(text, size, int(w * .82))
        if layer.height > h*.55:
            raise ValueError('This phrase is too long for safe readable typography. Shorten the phrase or split it into separate lines.')
        text_layers.append(layer)
    with tempfile.TemporaryDirectory() as temp:
        footage = []
        for i, clip in enumerate(clips):
            folder = Path(temp) / str(i); folder.mkdir()
            run(['-i', clip, '-an', '-vf', f'fps=24,scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2', str(folder / '%06d.jpg')])
            footage.append(sorted(folder.glob('*.jpg')))
        encoder = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', '24', '-i', 'pipe:0', '-i', audio, '-map', '0:v', '-map', '1:a', '-c:v', 'libx264', '-preset', 'fast', '-crf', '19', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '256k', '-t', str(duration), '-movflags', '+faststart', output], stdin=subprocess.PIPE)
        try:
            for frame in range(round(duration * FPS)):
                t = frame / FPS; progress = t / duration
                section = max(i for i, v in enumerate(transitions[:-1]) if v <= t)
                scenes = plan.get('scenes') or [{'start':0,'end':duration}]
                scene_index = max(i for i, scene in enumerate(scenes) if float(scene.get('start',0)) <= t)
                pulse = max([max(0., 1 - (t-b)*9) for b in onsets + beats if 0 <= t-b < .12] or [0.])
                level = feature_at(features, 'envelope', t)
                bass = feature_at(features, 'bass', t); highs = feature_at(features, 'highs', t)
                base = Image.new('RGB', (w, h), (8, 10, 17))
                if pictures:
                    picture_index=section % len(pictures)
                    pic = regional_motion(pictures[picture_index], regions[picture_index], bass)
                    # Preserve all original subjects, lettering and composition from frame one.
                    from PIL import ImageEnhance
                    base = ImageEnhance.Brightness(pic).enhance(.94 + .12*bass)
                    # A restrained breathing camera move keeps the complete source art in view.
                    # It is composited over a blurred full-frame plate so no edge content is lost.
                    drift = .004 + .008*level
                    zoom = 1 + drift*(.5+.5*math.sin(t*.31))
                    scaled = pic.resize((math.ceil(w*zoom), math.ceil(h*zoom)), Image.Resampling.LANCZOS)
                    x = (scaled.width-w)//2 + round(math.sin(t*.23)*w*.004)
                    y = (scaled.height-h)//2 + round(math.cos(t*.19)*h*.004)
                    moving = scaled.crop((x,y,x+w,y+h))
                    plate = pic.filter(ImageFilter.GaussianBlur(max(2, int(min(w,h)*.012))))
                    base = Image.blend(plate, ImageEnhance.Brightness(moving).enhance(.94+.12*bass), .94)
                # No semantic subject verifier is installed. Quarantine generated pixels
                # whenever original artwork exists rather than claiming they were checked.
                if footage and not pictures:
                    sequence = footage[min(len(footage)-1, scene_index)]
                    if sequence:
                        shot = Image.open(sequence[min(len(sequence)-1, max(0, round((t-float(scenes[scene_index].get('start',0)))*FPS)))]).convert('RGB')
                        base = Image.blend(base, shot, .93)
                elif footage and pictures:
                    # Generated footage is permitted only as an abstract light field over supplied
                    # artwork. Heavy blur + monochrome colourisation prevents generated subjects
                    # (including unrelated people) from entering the finished composition.
                    sequence = footage[min(len(footage)-1, scene_index)]
                    if sequence:
                        local_start = float(scenes[scene_index].get('start',0))
                        local_frame = max(0, round((t-local_start)*FPS))
                        shot = Image.open(sequence[min(len(sequence)-1, local_frame)]).convert('L')
                        shot = shot.filter(ImageFilter.GaussianBlur(max(18, int(min(w,h)*.055))))
                        field = ImageOps.colorize(shot, black=(0,0,0), white=palette)
                        field = ImageEnhance.Contrast(field).enhance(.58)
                        base = Image.blend(base, ImageChops.screen(base, field), .025+.045*level)
                glow = Image.new('RGB', (w, h)); gd = ImageDraw.Draw(glow)
                # Perspective-projected light rays and particles use real beat/onset impulses.
                for j in range(9):
                    angle = t*.17 + j*.7
                    point = (w/2 + math.sin(angle)*w*.9, h*.3 + math.cos(angle*.7)*h*.8)
                    color = tuple(min(255, int(c*(level*.3+pulse*.7))) for c in palette)
                    gd.line((w/2, h*.76, *point), fill=color, width=max(1, int(1+level*2)))
                for x, y, z, speed in particles:
                    depth = .3 + ((z-t*.07*speed) % 1)
                    px = w/2 + (x-.5)*w/depth; py = h/2+(y-.5)*h/depth
                    radius = 1 + highs*2
                    gd.ellipse((px-radius, py-radius, px+radius, py+radius), fill=tuple(int(c*highs) for c in palette))
                base = Image.blend(base, ImageChops.add(base, glow.filter(ImageFilter.GaussianBlur(5))), .06*level)
                base = Image.blend(base, ImageChops.add(base, glow), .05*pulse)
                # Cinematic 2D finish: music-linked bloom, moving highlight and vignette.
                bloom = base.filter(ImageFilter.GaussianBlur(max(2, int(min(w,h)*.008))))
                base = Image.blend(base, ImageChops.screen(base, bloom), .018+.035*level)
                sheen = Image.new('RGB',(w,h)); sd=ImageDraw.Draw(sheen)
                sx = int(((t/max(.001,duration))*1.4-.2)*w)
                sd.polygon([(sx-int(w*.22),h),(sx-int(w*.07),0),(sx+int(w*.07),0),(sx+int(w*.22),h)],fill=tuple(min(255,int(c*(.18+.3*highs))) for c in palette))
                base = Image.blend(base, ImageChops.screen(base,sheen.filter(ImageFilter.GaussianBlur(max(8,int(w*.035))))), .02+.035*highs)
                vignette = Image.new('L',(w,h)); vd=ImageDraw.Draw(vignette)
                vd.ellipse((-w*.12,-h*.08,w*1.12,h*1.08),fill=255)
                vignette=vignette.filter(ImageFilter.GaussianBlur(max(12,int(min(w,h)*.11))))
                base = Image.composite(base, Image.new('RGB',(w,h),(3,4,8)), vignette)
                if text_layers:
                    at = min(len(text_layers)-1, int(progress*len(text_layers)))
                    local = progress*len(text_layers)-at
                    layer = text_layers[at].copy()
                    # Each exact glyph raster is revealed through a sharp moving light mask.
                    alpha = layer.getchannel('A'); mask = Image.new('L', layer.size)
                    ImageDraw.Draw(mask).rectangle((0, 0, int(layer.width*min(1, local*5)), layer.height), fill=255)
                    layer.putalpha(ImageChops.multiply(alpha, mask))
                    base.paste(layer, ((w-layer.width)//2, int((h-layer.height)/2 + (1-min(1, local*6))*12)), layer)
                edge = min(1., max(0., (duration-t)/.3))
                if edge < 1: base = Image.blend(Image.new('RGB', (w,h)), base, edge)
                encoder.stdin.write(base.tobytes())
        finally:
            encoder.stdin.close()
            if encoder.wait(timeout=1800): raise RuntimeError('MP4 encoding failed')
    verify(output, duration)

def verify(output, duration):
    info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', output]))
    if not {'video','audio'} <= {s['codec_type'] for s in info['streams']} or abs(float(info['format']['duration'])-duration) > .12:
        raise RuntimeError('Final MP4 failed stream/duration verification')

def handle(payload):
    """Called by the EXISTING worker dispatcher after matching action; never replace H3."""
    if payload.get('action') != 'mirrorvision_render': raise ValueError('Unsupported action')
    from urllib.parse import urlparse
    output_url = payload['outputUrl']; origin = urlparse(output_url)
    allowed = os.environ.get('MIRRORVISION_SITE_ORIGIN', 'https://mirrorvidgen.mitm1.chatgpt.site')
    if f'{origin.scheme}://{origin.netloc}' != allowed or not origin.path.endswith('/render-output'):
        raise ValueError('Untrusted output URL')
    def download(url, path):
        target = urlparse(url)
        if f'{target.scheme}://{target.netloc}' != allowed or not target.path.startswith('/api/'):
            raise ValueError('Untrusted media URL')
        # Do not follow redirects to arbitrary hosts.
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args): return None
        with urllib.request.build_opener(NoRedirect).open(urllib.request.Request(url, headers=HTTP_HEADERS), timeout=90) as response, open(path, 'wb') as dest:
            size=0
            while chunk:=response.read(1024*1024):
                size+=len(chunk)
                if size>200*1024*1024: raise ValueError('Media too large')
                dest.write(chunk)
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp); audio=str(root/'audio.wav'); download(payload['audio'],audio)
        images=[]; clips=[]
        for group,key in [(images,'images'),(clips,'clips')]:
            for i,url in enumerate(payload.get(key,[])):
                path=str(root/f'{key}-{i}');download(url,path);group.append(path)
        output=str(root/'final.mp4');render(payload['plan'],audio,images,clips,output)
        data=Path(output).read_bytes()
        request=urllib.request.Request(output_url,data=data,headers={**HTTP_HEADERS, 'Content-Type':'video/mp4'},method='POST')
        with urllib.request.urlopen(request,timeout=180) as response:
            if response.status!=200: raise RuntimeError('Final storage rejected upload')
        return {'rendererVersion':1,'uploaded':True,'analysis':{
            'method': features.get('method') if (features:=analyse(audio, payload['plan']['duration'])) else None,
            'detectedBpm': features.get('detectedBpm'),
            'onsetCount': len(features.get('onsets', [])),
            'decodedPcmSha256': features.get('decodedPcmSha256'),
        }}

if __name__=='__main__':
    import sys
    manifest=json.loads(Path(sys.argv[1]).read_text())
    render(manifest['plan'],manifest['audio'],manifest.get('images',[]),manifest.get('clips',[]),sys.argv[2])
