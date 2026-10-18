"""Real CPU media tests. No network, paid jobs, or synthetic claims of H3 output."""
import importlib.util, io, json, math, subprocess, tempfile, urllib.error, wave, struct
from pathlib import Path
from unittest.mock import patch
from PIL import Image
spec=importlib.util.spec_from_file_location('renderer','mirrorvidgen-worker/mirrorvision_renderer.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
# Low-resolution fixture tests exercise the same encoder/compositor, not a mock render.
r.dimensions=lambda _: (320,180)
def probe(path):
 return json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)]))
with tempfile.TemporaryDirectory() as temp:
 root=Path(temp);audio=root/'selected.wav';image=root/'image.png'
 with wave.open(str(audio),'wb') as out:
  out.setnchannels(1);out.setsampwidth(2);out.setframerate(12000)
  out.writeframes(b''.join(struct.pack('<h',int(14000*math.sin(i*2*math.pi*660/12000))) for i in range(30*12000)))
 Image.new('RGB',(320,180),(30,75,140)).save(image)
 assert r.source_similarity(Image.new('RGB',(64,64),'blue'),Image.new('RGB',(64,64),'blue'))>.99
 assert r.source_similarity(Image.new('RGB',(64,64),'blue'),Image.new('RGB',(64,64),'red'))<.60
 assert r.dimensions('9:16')==(320,180)  # fixture override still applies to the production render path
 clips=[]
 for color in ['red','blue']:
  path=root/(color+'.mp4');r.run(['-f','lavfi','-i',f'color={color}:s=320x180:r=24:d=15','-c:v','libx264','-pix_fmt','yuv420p',str(path)]);clips.append(str(path))
 boundary=root/'boundary.png';r.extract_boundary(clips[0],str(boundary));assert Image.open(boundary).getpixel((100,100))[0]>200
 print('PASS exact final-frame extraction helper for chained H3 reference conditioning',flush=True)
 for name,mode,duration,images,words,footage in [
  ('A-image-only','Mystery Teaser',15,[str(image)],[],[]),
  ('B-exact-words','Mystery Teaser',5,[],['MiTM','COMING SOON!'],[]),
  ('C-image-reveal','Mystery Teaser',10,[str(image)],['MiTM','The Reveal','23 October'],[]),
  ('D-hybrid','Hybrid Music Video',15,[str(image)],[],[clips[0]]),
  ('E-full-one','Full Music Video',15,[str(image)],[],[clips[0]]),
  ('F-full-two-order','Full Music Video',30,[],[],clips),
  ('G-full-two-source','Full Music Video',30,[str(image)],[],clips),
 ]:
  plan={'mode':mode,'duration':duration,'aspect':'16:9','ordered':True,'words':words,'audio':{'bpm':120,'beats':[i*.5 for i in range(duration*2)],'onsets':[i*.5 for i in range(duration*2)],'energy':[.4]*(duration*10),'changes':[duration*.5]},'scenes':[{'start':i*15,'end':min(duration,(i+1)*15)} for i in range(math.ceil(duration/15))]}
  output=root/(name+'.mp4');r.render(plan,str(audio),images,footage,str(output))
  p=probe(output);assert abs(float(p['format']['duration'])-duration)<.12
  assert {'audio','video'}<={s['codec_type'] for s in p['streams']}
  assert int(next(s for s in p['streams'] if s['codec_type']=='video')['nb_frames'])==duration*24
  pcm=subprocess.check_output(['ffmpeg','-v','error','-i',str(output),'-map','0:a:0','-ac','1','-ar','12000','-f','s16le','pipe:1'])
  values=struct.unpack('<'+'h'*(len(pcm)//2),pcm)
  crossings=sum(a<=0<b for a,b in zip(values[12000:23999],values[12001:24000]))
  assert abs(crossings-660)<5,(name,crossings)
  if name=='F-full-two-order':
   for at,channel in [(5,0),(20,2)]:
    pixel=subprocess.check_output(['ffmpeg','-v','error','-ss',str(at),'-i',str(output),'-frames:v','1','-vf','scale=1:1','-f','rawvideo','-pix_fmt','rgb24','pipe:1'])
    assert pixel[channel]>150,(at,list(pixel))
  if name in ('D-hybrid','G-full-two-source'):
   # Red unverified footage must never replace the supplied blue artwork.
   for when in [0,1,5,10]:
    pixel=subprocess.check_output(['ffmpeg','-v','error','-ss',str(when),'-i',str(output),'-frames:v','1','-vf','scale=1:1','-f','rawvideo','-pix_fmt','rgb24','pipe:1'])
    assert pixel[2]>pixel[0]*1.15 and pixel[2]>90,(when,list(pixel))
   measured=json.loads(Path(str(output)+'.analysis.json').read_text())
   assert measured['decodedPcmSha256'] and measured['hopSeconds']==.01
   # The moving compositor must produce distinct frames while source colours remain dominant.
   a=subprocess.check_output(['ffmpeg','-v','error','-ss','1','-i',str(output),'-frames:v','1','-f','md5','pipe:1'])
   b=subprocess.check_output(['ffmpeg','-v','error','-ss','5','-i',str(output),'-frames:v','1','-f','md5','pipe:1'])
   assert a!=b
  print('PASS',name,'actual MP4, frame count, duration, selected-audio frequency',flush=True)
 # The supplied string is rasterized deterministically by the installed font;
 # no generative model receives or redraws these glyphs.
 exact=r.draw_text('MiTM COMING SOON!',60,1000).getchannel('A')
 typo=r.draw_text('MiTM COMING SON!',60,1000).getchannel('A')
 assert exact.getbbox() and exact.tobytes()!=typo.tobytes()
 assert len(r.glyph_layers('MiTM COMING SOON!',60,1000,(235,72,118)))==15
 assert r.diagonal_reveal((640,180),0).getbbox() is None
 assert r.diagonal_reveal((640,180),1).getextrema()==(255,255)
 print('PASS exact deterministic raster typography and per-glyph kinetic sprites',flush=True)
 attempts=[]
 class Accepted:
  status=200
  def __enter__(self): return self
  def __exit__(self,*_): pass
 def callback(request,**_):
  attempts.append(request)
  if len(attempts)==1:
   raise urllib.error.HTTPError(request.full_url,503,'busy',{},io.BytesIO(b'{"message":"This music project is processing an update. Please retry shortly."}'))
  return Accepted()
 with patch.object(r.urllib.request,'urlopen',callback), patch.object(r.time,'sleep',lambda _:None):
  r.upload_result('https://mirrorvidgen.mitm1.chatgpt.site/api/music-react/fixture/render-output',b'video')
 assert len(attempts)==2
 print('PASS durable MP4 callback retries the transient Site update-lock race',flush=True)
