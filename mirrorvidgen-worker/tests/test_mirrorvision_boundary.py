"""No-network dispatcher tests using actual final-frame extraction and mocked storage."""
import base64, importlib.util, sys, tempfile, types
from pathlib import Path
from unittest.mock import patch
from PIL import Image
sys.path.insert(0,str(Path('mirrorvidgen-worker').resolve()))
sys.modules['runpod']=types.SimpleNamespace(serverless=types.SimpleNamespace(start=lambda **_:None))
sys.modules['h3_addon_handler_existing']=types.SimpleNamespace(handler=lambda job:job['fixture_result'])
import mirrorvision_dispatch as d
import mirrorvision_renderer as r
class Response:
 status=200
 def __enter__(self): return self
 def __exit__(self,*_): pass
with tempfile.TemporaryDirectory() as folder:
 root=Path(folder); video=root/'two.mp4'; last=root/'last.png'
 r.run(['-f','lavfi','-i','color=red:s=64x64:r=24:d=1','-f','lavfi','-i','color=blue:s=64x64:r=24:d=0.1','-filter_complex','[0:v][1:v]concat=n=2:v=1:a=0','-c:v','libx264','-pix_fmt','yuv420p',str(video)])
 result={'status':'success','outputs':[{'type':'base64','data':base64.b64encode(video.read_bytes()).decode()}]}
 seen=[]
 def upload(request,**_):
  seen.append(request);last.write_bytes(request.data);return Response()
 job={'input':{'action':'generate','workflow':'ref2va','boundary_upload':'https://mirrorvidgen.mitm1.chatgpt.site/api/music-react/test/boundary?token=fixture'},'fixture_result':result}
 with patch.object(d.urllib.request,'urlopen',upload): out=d.handler(job)
 assert out['continuity_frame_stored'] is True and len(seen)==1
 pixel=Image.open(last).getpixel((32,32)); assert pixel[2]>200 and pixel[0]<20,pixel
 with patch.object(d.urllib.request,'urlopen',side_effect=OSError('fixture failure')):
  out=d.handler(job)
 assert out['outputs'][0]['data'] and out['continuity_frame_stored'] is False
 assert d.handler({'input':{'action':'generate'},'fixture_result':{'existing':'unchanged'}})=={'existing':'unchanged'}
 print('PASS exact decoded last frame, image callback, preserved paid result on storage failure, legacy H3 delegation; callback mocked')
