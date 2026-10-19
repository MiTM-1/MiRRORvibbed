"""Native CPU renderer adapter. Install alongside (not instead of) the H3 handler.
Requires FFmpeg + Pillow + a licensed DejaVu Sans font. No browser or GPU.
Run locally: python mirrorvision_renderer.py manifest.json output.mp4
The worker entry point is handle(input); it only accepts mirrorvision_render.
"""
import json, math, os, random, re, subprocess, tempfile, time, urllib.error, urllib.request
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps, ImageChops, ImageEnhance, ImageStat
import importlib.util
_spec = importlib.util.spec_from_file_location('mirrorvision_audio', Path(__file__).with_name('audio_features.py'))
_audio = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_audio)
analyse, feature_at = _audio.analyse, _audio.at

FONT = os.environ.get('MIRRORVISION_FONT', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')
FPS = 24
RENDERER_VERSION = 3
HTTP_HEADERS = {'User-Agent': 'MiRRORvidgen-Worker/1.0'}

def upload_result(url, data):
    """Retry only transient callback races; never report a render as complete
    until the durable Site store has accepted the exact MP4 bytes."""
    for attempt in range(8):
        request=urllib.request.Request(url,data=data,headers={**HTTP_HEADERS, 'Content-Type':'video/mp4'},method='POST')
        try:
            with urllib.request.urlopen(request,timeout=180) as response:
                if response.status==200: return
                raise RuntimeError(f'Final storage rejected upload (HTTP {response.status})')
        except urllib.error.HTTPError as error:
            detail=error.read(1024).decode('utf-8','replace')
            transient=error.code in (409,429,500,502,503,504) or (
                error.code==422 and 'processing an update' in detail.lower())
            if not transient or attempt==7:
                raise RuntimeError(f'Final storage rejected upload (HTTP {error.code})') from error
            time.sleep(min(8, .5 * (2 ** attempt)))

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
    # Standard HD-friendly dimensions.  All are even for H.264/yuv420p.
    return {'9:16': (720, 1280), '16:9': (1280, 720), '1:1': (900, 900), '3:4': (720, 960)}[aspect]

def draw_text(text, size, width, palette=(235, 72, 118)):
    """Rasterize supplied Unicode strings, never generate their letters with H3."""
    font = ImageFont.truetype(FONT, size)
    missing = bytes(font.getmask(chr(0x10FFFF)))
    for char in text:
        if not char.isspace() and bytes(font.getmask(char)) == missing:
            raise ValueError('The installed font cannot render a supplied character: '+repr(char))
    # Prefer word boundaries so exact phrases never split into awkward orphan
    # letters. Only a single word wider than the safe area falls back to
    # character wrapping; punctuation and capitalization are preserved.
    lines = []
    for paragraph in text.split('\n'):
        if not paragraph:
            lines.append(''); continue
        current = ''
        for token in re.findall(r'\S+\s*', paragraph):
            candidate = current + token
            if current and font.getlength(candidate.rstrip()) > width:
                lines.append(current.rstrip()); current = token.lstrip()
            else:
                current = candidate
            while current and font.getlength(current.rstrip()) > width:
                cut = 1
                while cut < len(current) and font.getlength(current[:cut + 1]) <= width:
                    cut += 1
                lines.append(current[:cut].rstrip()); current = current[cut:].lstrip()
        lines.append(current.rstrip())
    h = int(len(lines) * size * 1.34 + size*.85)
    layer = Image.new('RGBA', (width, h))
    mask=Image.new('L',(width,h)); draw=ImageDraw.Draw(mask)
    for i, line in enumerate(lines):
        draw.text((width / 2, i * size * 1.34), line, font=font, anchor='mt', fill=255, stroke_width=max(1,size//55), stroke_fill=255)
    # Deterministic dimensional typography: a deep extrusion, dark outline,
    # metallic vertical gradient and a narrow specular highlight.  The glyph
    # mask still comes only from the exact supplied string.
    shadow=Image.new('RGBA',(width,h)); sm=mask.filter(ImageFilter.GaussianBlur(max(3,size//16)))
    shadow.putalpha(sm.point(lambda v:int(v*.72)))
    shadow.paste((5,4,10,0),(0,0,width,h),shadow)
    extrusion=Image.new('RGBA',(width,h))
    for depth in range(max(3,size//18),0,-1):
        coloured=Image.new('RGBA',(width,h),(22,8,15,255)); coloured.putalpha(mask)
        extrusion.alpha_composite(coloured,(depth,depth))
    gradient=Image.new('RGBA',(width,h)); pixels=gradient.load()
    warm=tuple(max(80,min(255,int(v))) for v in palette)
    for y in range(h):
        phase=y/max(1,h-1)
        shine=1.34 if .22<phase<.36 else (.92+.20*(1-phase))
        colour=tuple(max(0,min(255,int(c*shine))) for c in warm)
        ImageDraw.Draw(gradient).line((0,y,width,y),fill=(*colour,255))
    gradient.putalpha(mask)
    outline=mask.filter(ImageFilter.MaxFilter(max(3,(size//18)*2+1)))
    rim=ImageChops.subtract(outline,mask)
    rim_layer=Image.new('RGBA',(width,h),(255,220,170,255));rim_layer.putalpha(rim)
    layer.alpha_composite(shadow)
    layer.alpha_composite(extrusion)
    layer.alpha_composite(rim_layer)
    layer.alpha_composite(gradient)
    return layer

def glyph_layers(text,size,width,palette):
    """Exact glyph sprites for staggered kinetic type; spaces retain real advances."""
    if '\n' in text:return []
    font=ImageFont.truetype(FONT,size);total=font.getlength(text)
    if total>width*.94:return []
    entries=[];cursor=(width-total)/2
    for char in text:
        advance=font.getlength(char)
        if not char.isspace():
            gw=max(round(advance+size*.62),round(size*.75))
            entries.append((draw_text(char,size,gw,palette),cursor+advance/2-gw/2))
        cursor+=advance
    return entries

def diagonal_reveal(size,ease):
    """Diagonal wipe which is completely empty at 0 and completely filled at 1."""
    width,height=size;progress=max(0.,min(1.,ease))
    if progress<=0:return Image.new('L',size)
    if progress>=1:return Image.new('L',size,255)
    edge=(-.25+1.5*progress)*width
    reveal=Image.new('L',size);draw=ImageDraw.Draw(reveal)
    draw.polygon([(0,0),(edge+width*.25,0),(edge-width*.25,height),(0,height)],fill=255)
    return reveal

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

def artwork_masks(picture):
    """Create conservative measured masks without pretending to identify objects.

    The masks isolate saturated artwork, luminous details and a broad foreground
    depth plane.  They let suitable parts move independently while the untouched
    original remains beneath them, so logos and lettering cannot be destroyed.
    """
    w,h=picture.size
    hsv=picture.convert('HSV'); saturation=hsv.getchannel('S'); value=hsv.getchannel('V')
    vivid=saturation.point(lambda v:255 if v>105 else 0).filter(ImageFilter.GaussianBlur(max(2,w//180)))
    lights=value.point(lambda v:255 if v>184 else 0).filter(ImageFilter.GaussianBlur(max(3,w//150)))
    # Local contrast is a useful content-independent proxy for a foreground plane.
    grey=picture.convert('L'); local=grey.filter(ImageFilter.GaussianBlur(max(7,w//35)))
    edges=ImageChops.difference(grey,local).point(lambda v:min(255,v*5)).filter(ImageFilter.GaussianBlur(max(3,w//120)))
    fg=Image.new('L',(w,h));d=ImageDraw.Draw(fg)
    d.ellipse((-w*.12,h*.28,w*1.12,h*1.17),fill=220)
    fg=ImageChops.multiply(fg,edges.point(lambda v:max(22,v)))
    # Split the strongest colour islands into separate measured layers.  This is
    # deliberately content-agnostic: hearts, clothing, signs, artwork accents,
    # etc. can move independently without assigning them invented labels.
    sw,sh=64,max(64,round(64*h/w));small=picture.resize((sw,sh)).convert('HSV')
    pixels=small.load();seen=set();components=[]
    for sy in range(sh):
        for sx in range(sw):
            if (sx,sy) in seen:continue
            hue,sat,val=pixels[sx,sy]
            if sat<120 or val<72:continue
            todo=[(sx,sy)];seen.add((sx,sy));points=[]
            while todo:
                px,py=todo.pop();points.append((px,py))
                for nx,ny in ((px-1,py),(px+1,py),(px,py-1),(px,py+1)):
                    if not(0<=nx<sw and 0<=ny<sh) or (nx,ny) in seen:continue
                    nh,ns,nv=pixels[nx,ny];delta=min(abs(nh-hue),256-abs(nh-hue))
                    if ns>=112 and nv>=65 and delta<24:
                        seen.add((nx,ny));todo.append((nx,ny))
            if 10<=len(points)<=sw*sh*.26:components.append(points)
    component_masks=[]
    for points in sorted(components,key=len,reverse=True)[:7]:
        cm=Image.new('L',(sw,sh));cd=ImageDraw.Draw(cm)
        for px,py in points:cd.point((px,py),fill=255)
        cm=cm.resize((w,h),Image.Resampling.NEAREST).filter(ImageFilter.GaussianBlur(max(3,w//130)))
        component_masks.append(cm)
    protect=Image.new('L',(w,h),0);pd=ImageDraw.Draw(protect)
    pd.rectangle((0,0,w,round(h*.18)),fill=232)
    pd.rectangle((0,0,round(w*.035),h),fill=190);pd.rectangle((round(w*.965),0,w,h),fill=190)
    protect=protect.filter(ImageFilter.GaussianBlur(max(7,w//45)))
    vivid_support=expanded(saturation.point(lambda v:255 if v>82 else 0),max(4,w//70)).filter(ImageFilter.GaussianBlur(max(2,w//240)))
    dark_support=expanded(value.point(lambda v:255 if v<62 else 0),max(4,w//85)).filter(ImageFilter.GaussianBlur(max(2,w//260)))
    source_detail=ImageChops.difference(grey,grey.filter(ImageFilter.GaussianBlur(max(3,w//95))))
    source_detail=source_detail.point(lambda v:min(230,v*6))
    return {'vivid':vivid,'lights':lights,'foreground':fg,'components':component_masks,
        'vivid_support':vivid_support,'dark_support':dark_support,'protect':protect,'source_detail':source_detail}

def shifted_layer(picture, mask, scale=1., dx=0., dy=0.):
    w,h=picture.size
    rgba=picture.convert('RGBA');rgba.putalpha(mask)
    sw=max(2,round(w*scale));sh=max(2,round(h*scale))
    rgba=rgba.resize((sw,sh),Image.Resampling.LANCZOS)
    out=Image.new('RGBA',(w,h))
    out.paste(rgba,(round((w-sw)/2+dx),round((h-sh)/2+dy)),rgba)
    return out

def camera_frame(picture, zoom, dx=0., dy=0., blur_plate=True):
    w,h=picture.size; sw=math.ceil(w*zoom);sh=math.ceil(h*zoom)
    scaled=picture.resize((sw,sh),Image.Resampling.LANCZOS)
    left=max(0,min(sw-w,round((sw-w)/2+dx)));top=max(0,min(sh-h,round((sh-h)/2+dy)))
    moving=scaled.crop((left,top,left+w,top+h))
    if not blur_plate:return moving
    plate=picture.filter(ImageFilter.GaussianBlur(max(2,int(min(w,h)*.009))))
    return Image.blend(plate,moving,.97)

def source_similarity(source, shot):
    """Cheap conservative visual guard; low values quarantine unrelated H3 frames."""
    a=ImageOps.fit(source.convert('RGB'),(32,32)).convert('HSV')
    b=ImageOps.fit(shot.convert('RGB'),(32,32)).convert('HSV')
    pa=list(a.getdata());pb=list(b.getdata())
    difference=sum(min(abs(x[0]-y[0]),256-abs(x[0]-y[0]))*4.0+abs(x[1]-y[1])*.55+abs(x[2]-y[2])*.55 for x,y in zip(pa,pb))/(1024*793)
    return max(0.,1.-difference)

def structural_similarity(source, shot):
    """Low-resolution edge-layout agreement, not just matching colour palette."""
    size=(64,96)
    a=source.convert('L').resize(size,Image.Resampling.BILINEAR).filter(ImageFilter.FIND_EDGES)
    b=shot.convert('L').resize(size,Image.Resampling.BILINEAR).filter(ImageFilter.FIND_EDGES)
    difference=ImageStat.Stat(ImageChops.difference(a,b)).mean[0]
    return max(0.,min(1.,1.-difference/92.))

def expanded(mask, radius):
    """Fast morphology with an odd Pillow kernel."""
    radius=max(1,int(radius));size=radius*2+1
    if size>99:
        scale=max(1,math.ceil(size/99));small=mask.resize((max(1,mask.width//scale),max(1,mask.height//scale)),Image.Resampling.BILINEAR)
        return small.filter(ImageFilter.MaxFilter(max(3,(size//scale)|1))).resize(mask.size,Image.Resampling.BILINEAR)
    return mask.filter(ImageFilter.MaxFilter(size))

def full_h3_composite(source, native_base, shot, source_masks):
    """Admit real H3 motion while suppressing newly invented subjects.

    H3 is allowed to move colour and structure already supported by the source.
    New saturated or dark silhouettes outside that support are heavily rejected.
    A feathered source-safe title band and outer frame keep branding/composition
    visible even when H3 attempts an aggressive camera move.
    """
    w,h=source.size
    shot_hsv=shot.convert('HSV')
    source_vivid=source_masks['vivid_support']
    shot_vivid=shot_hsv.getchannel('S').point(lambda v:255 if v>105 else 0)
    vivid_novelty=ImageChops.subtract(shot_vivid,source_vivid)
    source_dark=source_masks['dark_support']
    shot_dark=shot_hsv.getchannel('V').point(lambda v:255 if v<48 else 0)
    dark_novelty=ImageChops.subtract(shot_dark,source_dark)
    rgb_difference=ImageChops.difference(source,shot).convert('L').filter(ImageFilter.GaussianBlur(max(2,w//180)))
    # Keep genuine motion, but taper areas whose appearance no longer has any
    # local source support. This specifically catches duplicate figures and
    # giant silhouettes without reducing H3 to a static source overlay.
    # Near-supported H3 pixels are fully opaque, not blended with a second
    # stationary character underneath. Unsupported changes fall rapidly to
    # zero; this removes the double-exposure/duplicate-subject failure.
    fidelity=rgb_difference.point(lambda v:max(0,min(255,255-max(0,v-28)*4)))
    fidelity=ImageChops.subtract(fidelity,vivid_novelty.point(lambda v:int(v*.82)))
    fidelity=ImageChops.subtract(fidelity,dark_novelty.point(lambda v:int(v*.90)))
    protect=source_masks['protect']
    fidelity=ImageChops.subtract(fidelity,protect.point(lambda v:int(v*.78)))
    h3=ImageEnhance.Color(shot).enhance(1.07)
    composed=Image.composite(h3,native_base,fidelity)
    # Re-lay only the crisp high-frequency source detail in protected title/
    # edge areas. This retains exact logos and spelling without ghosting a
    # second fixed character over a legitimately moving H3 subject.
    source_detail=source_masks['source_detail']
    restore_mask=ImageChops.multiply(source_detail,protect).filter(ImageFilter.GaussianBlur(1))
    restore=source.convert('RGBA');restore.putalpha(restore_mask)
    composed.paste(restore,(0,0),restore)
    return composed

def h3_environment_composite(source, native_base, shot, source_masks):
    """Reuse divergent H3 as moving atmosphere, never as replacement subjects."""
    shot_hsv=shot.convert('HSV')
    source_support=ImageChops.lighter(source_masks['vivid_support'],source_masks['dark_support'])
    source_support=ImageChops.lighter(source_support,source_masks['protect'])
    environment=ImageOps.invert(source_support).filter(ImageFilter.GaussianBlur(max(5,source.width//75)))
    vivid_novelty=ImageChops.subtract(shot_hsv.getchannel('S').point(lambda v:255 if v>92 else 0),source_masks['vivid_support'])
    dark_novelty=ImageChops.subtract(shot_hsv.getchannel('V').point(lambda v:255 if v<58 else 0),source_masks['dark_support'])
    atmosphere=environment.point(lambda v:int(v*.68))
    atmosphere=ImageChops.subtract(atmosphere,vivid_novelty.point(lambda v:int(v*.88)))
    atmosphere=ImageChops.subtract(atmosphere,dark_novelty.point(lambda v:int(v*.92)))
    # Blur the admitted H3 layer enough to turn it into cinematic moving light/
    # depth rather than a second readable person or altered logo.
    moving=shot.filter(ImageFilter.GaussianBlur(max(4,source.width//90)))
    composed=Image.composite(moving,native_base,atmosphere)
    return restore_source_detail(composed,source,source_masks,.62)

def restore_source_detail(base, source, source_masks, strength=.46):
    """Keep fine logos/lettering crisp after independent layer motion.

    This restores edge detail only, never a full static subject, so the artwork
    can still move without smearing or duplicating text and facial marks.
    """
    w,h=source.size
    whole=source_masks['source_detail'].point(lambda v:int(v*strength))
    upper=Image.new('L',(w,h));ud=ImageDraw.Draw(upper)
    ud.rectangle((0,0,w,round(h*.34)),fill=215)
    upper=upper.filter(ImageFilter.GaussianBlur(max(5,w//80)))
    exact=ImageChops.lighter(whole,ImageChops.multiply(source_masks['source_detail'],upper))
    layer=source.convert('RGBA');layer.putalpha(exact)
    out=base.convert('RGBA');out.alpha_composite(layer)
    return out.convert('RGB')

def prismatic_echo(base, mask, amount, direction=1):
    """Source-derived chromatic depth echo; never invents image content."""
    if amount<=.01:return base
    w,h=base.size;shift=max(1,round(w*.012*amount))*direction
    layer=shifted_layer(base,mask,1.0+amount*.012,shift,-shift//2)
    red=Image.new('RGBA',(w,h),(244,26,86,0));red.putalpha(layer.getchannel('A').point(lambda v:int(v*.42*amount)))
    cyan=Image.new('RGBA',(w,h),(38,211,255,0));cyan.putalpha(layer.getchannel('A').point(lambda v:int(v*.30*amount)))
    out=base.convert('RGBA');out.alpha_composite(red);out.alpha_composite(cyan,(-shift*2,shift))
    return out.convert('RGB')

def impact_cut(base, source, strength, style):
    """Brief source-preserving transition used only at measured musical cuts."""
    if strength<=.01:return base
    w,h=base.size;strength=min(1.,strength)
    if style%3==0:
        # Vertical aperture snaps open/closed around the composition centre.
        mask=Image.new('L',(w,h));d=ImageDraw.Draw(mask);half=w*(.08+.52*strength)
        d.rectangle((w/2-half,0,w/2+half,h),fill=255)
        return Image.composite(base,Image.new('RGB',(w,h),(2,2,7)),mask.filter(ImageFilter.GaussianBlur(max(2,w//180))))
    if style%3==1:
        # Depth echo on alternating slices, then immediate return to clarity.
        sliced=base.copy();band=max(12,h//11)
        for i,top in enumerate(range(0,h,band)):
            strip=source.crop((0,top,w,min(h,top+band)))
            sliced.paste(strip,(round(((-1)**i)*w*.035*(1-strength)),top))
        return Image.blend(base,sliced,.38*strength)
    # Luminous radial reveal, source remains the content.
    iris=Image.new('L',(w,h));d=ImageDraw.Draw(iris);radius=math.hypot(w,h)*(.10+.90*strength)
    d.ellipse((w/2-radius,h/2-radius,w/2+radius,h/2+radius),fill=255)
    return Image.composite(base,Image.new('RGB',(w,h),(1,2,6)),iris.filter(ImageFilter.GaussianBlur(max(4,w//90))))

def screen_colour(base, overlay, amount):
    return Image.blend(base,ImageChops.screen(base,overlay),max(0.,min(1.,amount)))

def cinematic_light(base, palette, t, duration, level, bass, highs, pulse, particles):
    w,h=base.size
    glow=Image.new('RGB',(w,h));gd=ImageDraw.Draw(glow)
    # Bass halo, accent rays and high-band particles have intentionally distinct motion.
    radius=min(w,h)*(.16+.20*bass)
    cx=w*(.5+.04*math.sin(t*.43));cy=h*(.43+.025*math.cos(t*.37))
    halo=tuple(min(255,int(c*(.65+.9*bass))) for c in palette)
    gd.ellipse((cx-radius,cy-radius,cx+radius,cy+radius),outline=halo,width=max(2,int(w*.006)))
    for j in range(12):
        angle=t*(.22+.015*j)+j*math.pi/6
        length=min(w,h)*(.45+.32*pulse)
        end=(cx+math.sin(angle)*length,cy+math.cos(angle)*length)
        colour=tuple(min(255,int(c*(.22+.75*pulse))) for c in palette)
        gd.line((cx,cy,*end),fill=colour,width=max(1,int(1+4*pulse)))
    for x,y,z,speed in particles:
        depth=.24+((z-t*(.09+.08*highs)*speed)%1)
        px=w/2+(x-.5)*w/depth;py=h/2+(y-.5)*h/depth
        r=1+highs*4+pulse*2
        gd.ellipse((px-r,py-r,px+r,py+r),fill=tuple(min(255,int(c*(.25+.8*highs))) for c in palette))
    base=screen_colour(base,glow.filter(ImageFilter.GaussianBlur(max(7,int(w*.018)))),.10+.20*level)
    base=screen_colour(base,glow,.08+.18*pulse)
    # Fast diagonal sweep is driven by musical position rather than a single pass.
    sweep=Image.new('RGB',(w,h));sd=ImageDraw.Draw(sweep)
    sx=int((((t*.24)%1.35)-.18)*w)
    sd.polygon([(sx-int(w*.24),h),(sx-int(w*.055),0),(sx+int(w*.055),0),(sx+int(w*.24),h)],fill=tuple(min(255,int(c*(.35+.65*highs))) for c in palette))
    return screen_colour(base,sweep.filter(ImageFilter.GaussianBlur(max(10,int(w*.028)))),.06+.18*highs)

def render(plan, audio, images, clips, output):
    validate(plan, images, clips)
    w, h = dimensions(plan['aspect']); duration = plan['duration']
    features = analyse(audio, duration)
    # Persist measured render evidence beside the output for local QA; never log asset URLs.
    Path(str(output)+'.analysis.json').write_text(json.dumps(features))
    analysis = plan.get('audio', {}); beats = features.get('beats', []); onsets = features['onsets']
    measured_changes=features.get('builds',[])+features.get('drops',[])
    changes = measured_changes or analysis.get('changes', [])
    transitions = sorted(set([0., duration] + [float(v) for v in changes if 0 < v < duration] + [float(v) for v in beats[::8] if 0 < v < duration]))
    if len(transitions) < 3: transitions = [duration * i / 4 for i in range(5)]
    pictures = [ImageOps.pad(Image.open(path).convert('RGB'), (w, h), color=(8,10,17)) for path in images]
    regions = [measured_regions(picture) for picture in pictures]
    masks = [artwork_masks(picture) for picture in pictures]
    rng = random.Random(42)
    particles = [(rng.random(), rng.random(), rng.random(), rng.uniform(.3, 1.)) for _ in range(145)]
    # Extract a vivid measured palette colour instead of averaging it to grey/brown.
    palette=(79,142,155)
    if pictures:
        sample=pictures[0].resize((48,48)).convert('HSV')
        candidates=[]
        for hue,sat,value in sample.getdata():
            if sat>80 and value>70:candidates.append((sat*value,hue,sat,value))
        if candidates:
            _,hue,sat,value=sorted(candidates,reverse=True)[len(candidates)//8]
            palette=Image.new('HSV',(1,1),(hue,min(255,sat),min(255,max(150,value)))).convert('RGB').getpixel((0,0))
    words = plan.get('words', [])
    if not plan.get('ordered', False) and len(words) > 3:
        words = sorted(words[:-2], key=len) + words[-2:]
    text_layers = []; text_sizes=[]
    for text in words:
        size = max(30, int(w * .15))
        layer = draw_text(text, size, int(w * .86), palette)
        # A compact title reads better as one deliberately scaled line than as
        # a near-fit with one or two orphaned letters on a second line.
        if '\n' not in text and len(text) <= 36:
            while layer.height > size * 2.5 and size > 22:
                size -= 2; layer = draw_text(text, size, int(w * .86), palette)
        while layer.height > h * .55 and size > 12:
            size -= 2; layer = draw_text(text, size, int(w * .86), palette)
        if layer.height > h*.55:
            raise ValueError('This phrase is too long for safe readable typography. Shorten the phrase or split it into separate lines.')
        text_layers.append(layer);text_sizes.append(size)
    kinetic_glyphs=[glyph_layers(text,size,int(w*.86),palette) for text,size in zip(words,text_sizes)]
    text_times=[0.]
    for index in range(1,len(text_layers)):
        target=duration*index/len(text_layers)
        candidates=[v for v in onsets if abs(v-target)<=.35 and v>text_times[-1]+.4 and v<duration-.6]
        text_times.append(min(candidates,key=lambda v:abs(v-target)) if candidates else target)
    text_times.append(float(duration))
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
                level = feature_at(features, 'envelopeDrive', t)
                bass = feature_at(features, 'bassDrive', t); mids=feature_at(features,'midsDrive',t);highs = feature_at(features, 'highsDrive', t)
                bass_hit=feature_at(features,'bassImpact',t);accent=feature_at(features,'accentImpact',t)
                onset_pulse=max([max(0.,1-(t-b)*7.5) for b in onsets if 0<=t-b<.14] or [0.])
                pulse=max(bass_hit,accent,onset_pulse)
                transition_hit=max([max(0.,1-abs(t-v)/.18) for v in transitions[1:-1]] or [0.])
                base = Image.new('RGB', (w, h), (8, 10, 17))
                if pictures:
                    picture_index=section % len(pictures)
                    pic=pictures[picture_index]; mask=masks[picture_index]
                    if plan['mode']=='Hybrid Music Video':
                        # Camera drift is deliberately slower than element motion. The
                        # image no longer reads as one global breathing layer.
                        zoom=1.016+.012*(.5+.5*math.sin(t*.29))+.018*bass_hit+.014*transition_hit
                        dx=math.sin(t*.23)*w*.010 + accent*w*.006 + math.sin(section*2.1)*transition_hit*w*.014
                        dy=math.cos(t*.19)*h*.007 - bass_hit*h*.004
                    elif plan['mode']=='Full Music Video':
                        zoom=1.014+.015*(.5+.5*math.sin(t*.29))+.018*bass_hit
                        dx=math.sin(t*.27)*w*.008;dy=math.cos(t*.21)*h*.006
                    else:
                        zoom=1.008+.010*(.5+.5*math.sin(t*.21))+.012*pulse
                        dx=math.sin(t*.17)*w*.005;dy=math.cos(t*.13)*h*.004
                    base=camera_frame(pic,zoom,dx,dy)
                    # Independent measured planes: opposing parallax, bass expansion on vivid
                    # artwork and a separate high-frequency luminous layer.
                    mode_gain=1.42 if plan['mode']=='Hybrid Music Video' else (1.18 if plan['mode']=='Mystery Teaser' else 1.10)
                    foreground=shifted_layer(pic,mask['foreground'],1.003+.015*level*mode_gain+.018*bass_hit*mode_gain,
                        -math.sin(t*.31)*w*.012*mode_gain,-math.cos(t*.23)*h*.008*mode_gain-bass_hit*h*.006*mode_gain)
                    base.paste(foreground,(0,0),foreground)
                    vivid=shifted_layer(pic,mask['vivid'],1.0+.019*bass*mode_gain+.052*bass_hit*mode_gain,
                        math.sin(t*.43)*w*.011*mode_gain,-bass_hit*h*.012*mode_gain)
                    base.paste(vivid,(0,0),vivid)
                    for layer_index,component in enumerate(mask['components']):
                        # Separate colour islands receive separate frequency mappings and
                        # phase, so suitable characters, hearts, headphones, signage and
                        # environmental accents visibly move as independent artwork layers.
                        drive=(bass_hit if layer_index%3==0 else (accent if layer_index%3==1 else highs))
                        phase=t*(.58+.11*layer_index)+layer_index*1.73
                        amplitude=mode_gain*(.012+.0035*min(layer_index,5))
                        isolated=shifted_layer(pic,component,1.0+(.035+.008*layer_index)*drive*mode_gain,
                            math.sin(phase)*w*amplitude,
                            math.cos(phase*.79)*h*amplitude*.72-drive*h*.012*mode_gain)
                        base.paste(isolated,(0,0),isolated)
                    luminous=Image.new('RGBA',(w,h)); luminous.paste(pic.convert('RGBA'),(0,0),mask['lights'])
                    luminous=luminous.filter(ImageFilter.GaussianBlur(max(3,int(w*.009))))
                    light_rgb=Image.new('RGB',(w,h));light_rgb.paste(luminous.convert('RGB'))
                    base=screen_colour(base,light_rgb,.07+.31*highs*mode_gain+.20*accent*mode_gain)
                    base=ImageEnhance.Contrast(base).enhance(1.02+.08*pulse*mode_gain)
                    base=ImageEnhance.Color(base).enhance(1.02+.12*level*mode_gain)
                    if plan['mode'] in ('Hybrid Music Video','Mystery Teaser'):
                        base=prismatic_echo(base,mask['components'][0] if mask['components'] else mask['vivid'],max(accent,bass_hit)*mode_gain,1 if section%2==0 else -1)
                        base=restore_source_detail(base,pic,mask,.52 if plan['mode']=='Hybrid Music Video' else .42)
                if footage and not pictures:
                    sequence = footage[min(len(footage)-1, scene_index)]
                    if sequence:
                        shot = Image.open(sequence[min(len(sequence)-1, max(0, round((t-float(scenes[scene_index].get('start',0)))*FPS)))]).convert('RGB')
                        base = Image.blend(base, shot, .93)
                elif footage and pictures:
                    sequence = footage[min(len(footage)-1, scene_index)]
                    if sequence:
                        local_start = float(scenes[scene_index].get('start',0))
                        local_frame = max(0, round((t-local_start)*FPS))
                        shot = Image.open(sequence[min(len(sequence)-1, local_frame)]).convert('RGB')
                        similarity=source_similarity(pictures[picture_index],shot)
                        structure=structural_similarity(pictures[picture_index],shot)
                        if plan['mode']=='Full Music Video' and similarity>=.56 and structure>=.56:
                            base=full_h3_composite(pictures[picture_index],base,shot,mask)
                        elif plan['mode']=='Full Music Video':
                            base=h3_environment_composite(pictures[picture_index],base,shot,mask)
                        elif plan['mode']=='Hybrid Music Video' and similarity>=.66:
                            base=Image.blend(base,shot,.24)
                        else:
                            # Frames which no longer resemble the supplied artwork are quarantined,
                            # preventing unrelated people or replacement scenes entering the result.
                            rejected=ImageOps.colorize(shot.convert('L').filter(ImageFilter.GaussianBlur(max(22,int(w*.06)))),black=(0,0,0),white=palette)
                            base=screen_colour(base,rejected,.025+.05*level)
                strength=1.0 if plan['mode']=='Mystery Teaser' else (1.18 if plan['mode']=='Hybrid Music Video' else 1.30)
                base=cinematic_light(base,palette,t,duration,min(1,level*strength),min(1,bass*strength),min(1,highs*strength),min(1,pulse*strength),particles)
                if plan['mode']=='Hybrid Music Video' and max(accent,transition_hit)>.18:
                    # Brief beat-cut refraction: measured horizontal bands travel in opposing
                    # directions, then settle immediately. The original artwork remains below.
                    sliced=base.copy();band=max(8,h//9);amount=max(accent,transition_hit)
                    for band_index,top in enumerate(range(0,h,band)):
                        strip=base.crop((0,top,w,min(h,top+band)))
                        shift=round(((-1)**band_index)*w*.026*amount)
                        sliced.paste(strip,(shift,top))
                    base=Image.blend(base,sliced,.42*amount)
                    # The source itself drives a brief beat-cut transition; no random
                    # unrelated footage or blur replaces it.
                    base=impact_cut(base,pic if pictures else base,transition_hit,section)
                elif plan['mode']=='Mystery Teaser':
                    # Measured phrase/cut timing controls shutters and irises so even an
                    # image-only teaser has dramatic, deliberate visual progression.
                    phrase_cut=max([max(0.,1-abs(t-v)/.22) for v in text_times[1:-1]] or [0.])
                    base=impact_cut(base,pic if pictures else base,max(transition_hit,phrase_cut),section+len(text_layers))
                bloom = base.filter(ImageFilter.GaussianBlur(max(2, int(min(w,h)*.006))))
                base = screen_colour(base,bloom,.025+.10*level+.06*pulse)
                vignette = Image.new('L',(w,h)); vd=ImageDraw.Draw(vignette)
                vd.ellipse((-w*.12,-h*.08,w*1.12,h*1.08),fill=255)
                vignette=vignette.filter(ImageFilter.GaussianBlur(max(12,int(min(w,h)*.11))))
                base = Image.composite(base, Image.new('RGB',(w,h),(3,4,8)), vignette)
                if text_layers:
                    at = max(i for i,v in enumerate(text_times[:-1]) if v<=t)
                    local = (t-text_times[at])/max(.001,text_times[at+1]-text_times[at])
                    layer=text_layers[at].copy(); alpha=layer.getchannel('A')
                    enter=min(1,local*3.35);leave=min(1,max(0,(1-local)*8.5));ease=1-(1-enter)**3
                    style=at%4
                    if style==0:
                        # A luminous slit opens across the exact glyphs.
                        reveal=Image.new('L',layer.size);rd=ImageDraw.Draw(reveal)
                        centre=layer.width/2;half=layer.width*.58*ease
                        rd.rectangle((centre-half,0,centre+half,layer.height),fill=255)
                    elif style==1:
                        reveal=diagonal_reveal(layer.size,ease)
                    elif style==2:
                        reveal=Image.new('L',layer.size);ImageDraw.Draw(reveal).rectangle((0,0,layer.width*ease,layer.height),fill=255)
                    else:
                        reveal=Image.new('L',layer.size);rd=ImageDraw.Draw(reveal);rd.ellipse((layer.width/2-layer.width*.65*ease,layer.height/2-layer.height*ease,layer.width/2+layer.width*.65*ease,layer.height/2+layer.height*ease),fill=255)
                    layer.putalpha(ImageChops.multiply(alpha,reveal).point(lambda v:int(v*leave)))
                    # A deeper camera approach and measured beat impact gives the type
                    # real kinetic weight while keeping its baseline stable.
                    scale=.58+.42*ease+.050*pulse
                    tw=max(2,round(layer.width*scale));th=max(2,round(layer.height*scale));layer=layer.resize((tw,th),Image.Resampling.LANCZOS)
                    anchor=(.31,.49,.68,.43)[at%4]
                    y=int(h*anchor-th/2+(1-ease)*h*((-1 if at%2 else 1)*.08))
                    # Feathered local stage + travelling fine lines: a cinematic title
                    # treatment, not the previous full-width flat banner.
                    ph=int(th*1.78);panel=Image.new('RGBA',(w,ph));pd=ImageDraw.Draw(panel)
                    left=int(w*(.075+.055*(1-ease)));right=int(w*(.925-.055*(1-ease)))
                    pd.rounded_rectangle((left,int(ph*.10),right,int(ph*.90)),radius=max(12,int(ph*.22)),fill=(3,3,9,int(72+62*(1-ease))),outline=(*palette,int(65+80*pulse)),width=max(1,w//240))
                    line_y=int(ph*(.18+.64*ease));pd.line((left+w*.04,line_y,right-w*.04,line_y),fill=(255,228,194,int(95+130*accent)),width=max(1,w//210))
                    panel=panel.filter(ImageFilter.GaussianBlur(max(4,w//170)))
                    panel_y=max(0,int(h*anchor-panel.height/2));base.paste(panel,(0,panel_y),panel)
                    # The outgoing phrase physically expands and dissolves into the incoming
                    # one, producing a continuous word-morph transition instead of a cut.
                    if at>0 and local<.24:
                        old=text_layers[at-1].copy();fade=max(0.,1-local/.24)
                        old.putalpha(old.getchannel('A').point(lambda v:int(v*fade*.72)))
                        oscale=1+.22*(1-fade);ow=max(2,round(old.width*oscale));oh=max(2,round(old.height*oscale))
                        old=old.resize((ow,oh),Image.Resampling.BICUBIC)
                        oy=int(h*anchor-oh/2)
                        base.paste(old,((w-ow)//2,oy),old)
                    if kinetic_glyphs[at]:
                        origin=(w-int(w*.86))/2
                        count=max(1,len(kinetic_glyphs[at]))
                        for glyph_index,(glyph,glyph_x) in enumerate(kinetic_glyphs[at]):
                            gp=max(0.,min(1.,local*3.05-glyph_index*.10))
                            ge=1-(1-gp)**3
                            gscale=.48+.52*ge+.045*pulse
                            gw=max(2,round(glyph.width*gscale));gh=max(2,round(glyph.height*gscale))
                            sprite=glyph.resize((gw,gh),Image.Resampling.LANCZOS)
                            angle=((glyph_index%3)-1)*18*(1-ge)
                            if abs(angle)>.4:sprite=sprite.rotate(angle,resample=Image.Resampling.BICUBIC,expand=True)
                            sprite.putalpha(sprite.getchannel('A').point(lambda v:int(v*leave)))
                            scatter=((glyph_index%2)*2-1)*(1-ge)*w*.055
                            gx=round(origin+glyph_x+(glyph.width-sprite.width)/2+scatter)
                            arc=math.sin((glyph_index/max(1,count-1))*math.pi)*h*.018
                            gy=round(h*anchor-sprite.height/2-arc*ge+((glyph_index%2)*2-1)*(1-ge)*h*.082)
                            gl=sprite.copy();gl.putalpha(sprite.getchannel('A').filter(ImageFilter.GaussianBlur(max(5,w//110))))
                            base.paste(gl,(gx,gy),gl);base.paste(sprite,(gx,gy),sprite)
                    else:
                        glow=layer.copy();glow.putalpha(layer.getchannel('A').filter(ImageFilter.GaussianBlur(max(5,w//110))))
                        base.paste(glow,((w-tw)//2,y),glow)
                        base.paste(layer,((w-tw)//2,y),layer)
                    # Specular bar sweeps the letters on measured accents.
                    if accent>.12:
                        flare=Image.new('RGB',(w,h));fd=ImageDraw.Draw(flare);fx=int((.1+.8*ease)*w)
                        fd.line((fx-w*.08,y,fx+w*.02,y+th),fill=tuple(min(255,int(c*1.5)) for c in palette),width=max(2,w//130))
                        base=screen_colour(base,flare.filter(ImageFilter.GaussianBlur(max(7,w//90))),.18*accent)
                # Short measured-impact flashes make cuts visible; no random strobe grid.
                if pulse>.82:
                    wash=Image.new('RGB',(w,h),tuple(min(255,int(35+c*.45)) for c in palette))
                    base=Image.blend(base,wash,.055*(pulse-.82)/.18)
                edge = min(1., max(0., (duration-t)/.22))
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
        upload_result(output_url,data)
        return {'rendererVersion':RENDERER_VERSION,'uploaded':True,'analysis':{
            'method': features.get('method') if (features:=analyse(audio, payload['plan']['duration'])) else None,
            'detectedBpm': features.get('detectedBpm'),
            'onsetCount': len(features.get('onsets', [])),
            'decodedPcmSha256': features.get('decodedPcmSha256'),
        }}

if __name__=='__main__':
    import sys
    manifest=json.loads(Path(sys.argv[1]).read_text())
    render(manifest['plan'],manifest['audio'],manifest.get('images',[]),manifest.get('clips',[]),sys.argv[2])

