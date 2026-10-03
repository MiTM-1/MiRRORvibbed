"""Independent authenticated enhancement API with durable per-job records.
Mount /data persistently. Run a single process: CPU work has one active slot.
"""
import hmac,json,os,time,uuid,threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import urlparse
import requests
from fastapi import FastAPI,Depends,Header,HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from processor import enhance

ROOT=Path(os.environ.get('ENHANCEMENT_DATA','/data'));ROOT.mkdir(parents=True,exist_ok=True)
POOL=ThreadPoolExecutor(max_workers=1)
SUBMIT_LOCK=threading.Lock()
app=FastAPI(title='MiRRORvidgen Enhancement')
def auth(authorization:str=Header(default='')):
    token=os.environ.get('ENHANCEMENT_TOKEN','')
    if not token or not hmac.compare_digest(authorization,'Bearer '+token):raise HTTPException(401,'Unauthorized')
class Request(BaseModel):
    sourceGenerationId:str
    sourceUrl:str
    resolution:str
    fps:int|str
    idempotencyKey:str

def save(folder,data):
    temp=folder/'status.tmp';temp.write_text(json.dumps(data));temp.replace(folder/'status.json')
def process(job,request):
    folder=ROOT/job;started=datetime.now(timezone.utc).isoformat();begin=time.time()
    status={'id':job,'stage':'preparing','progress':0,'sourceGenerationId':request.sourceGenerationId}
    def notify(stage,progress):status.update(stage=stage,progress=progress,elapsedSeconds=time.time()-begin);save(folder,status)
    try:
        source=folder/'source.mp4'
        response=requests.get(request.sourceUrl,timeout=90,stream=True,allow_redirects=False);response.raise_for_status()
        total=0
        with source.open('wb') as f:
            for chunk in response.iter_content(1024*1024):
                total+=len(chunk)
                if total>100*1024*1024:raise ValueError('Source exceeds limit')
                f.write(chunk)
        metadata=enhance(source,folder/'enhanced.mp4',request.resolution,request.fps,notify)
        metadata.update(sourceGenerationId=request.sourceGenerationId,enhancementStartedAt=started,enhancementFinishedAt=datetime.now(timezone.utc).isoformat())
        status.update(stage='completed',progress=100,metadata=metadata,elapsedSeconds=time.time()-begin);save(folder,status)
        source.unlink(missing_ok=True)
    except Exception:
        status.update(stage='failed',progress=None,message='Enhancement stopped. Your original video is safe.');save(folder,status)

@app.get('/health',dependencies=[Depends(auth)])
def health():return {'ready':True,'interpolation':'FFmpeg MCI/AOBMC','upscaling':'Lanczos','nativeGeneration':False}
@app.post('/jobs',dependencies=[Depends(auth)])
def submit(request:Request):
    parsed=urlparse(request.sourceUrl);allowed=os.environ.get('SOURCE_HOST','mirrorvidgen.mitm1.chatgpt.site')
    if parsed.scheme!='https' or parsed.hostname!=allowed or not parsed.path.startswith('/api/media/') or parsed.username:raise HTTPException(422,'Invalid source')
    if request.resolution not in {'original','720p','1080p'} or request.fps not in {'original',60}:raise HTTPException(422,'Invalid output')
    if request.resolution=='original' and request.fps=='original':raise HTTPException(422,'Choose enhancement')
    try:uuid.UUID(request.idempotencyKey)
    except ValueError:raise HTTPException(422,'Invalid idempotency key')
    job=str(uuid.uuid5(uuid.NAMESPACE_URL,request.sourceGenerationId+request.idempotencyKey))
    folder=ROOT/job
    with SUBMIT_LOCK:
        if (folder/'status.json').exists():return json.loads((folder/'status.json').read_text())
        folder.mkdir(exist_ok=True);status={'id':job,'stage':'queued','progress':None};save(folder,status)
        POOL.submit(process,job,request)
        return status

def folder_for(job):
    try:uuid.UUID(job)
    except ValueError:raise HTTPException(404,'Not found')
    folder=ROOT/job
    if not (folder/'status.json').exists():raise HTTPException(404,'Not found')
    return folder
@app.get('/jobs/{job}',dependencies=[Depends(auth)])
def status(job:str):return json.loads((folder_for(job)/'status.json').read_text())
@app.get('/jobs/{job}/video',dependencies=[Depends(auth)])
def video(job:str):
    path=folder_for(job)/'enhanced.mp4'
    if not path.exists():raise HTTPException(404,'Not ready')
    return FileResponse(path,media_type='video/mp4')
