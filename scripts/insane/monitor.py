"""Read-only run status/download. No create/push/retry calls."""
from pathlib import Path
import urllib.request,urllib.parse,base64,json,datetime,hashlib,re,argparse
ROOT=Path(__file__).resolve().parents[2];LAUNCH=ROOT/'outputs/insane_pilot_launch'
SLUG='drone-nav-insane-pilot'

def main(download=False):
    credentials=json.loads(Path('/Users/shapok/Downloads/kaggle.json').read_text())
    auth='Basic '+base64.b64encode((credentials['username']+':'+credentials['key']).encode()).decode()
    def api(endpoint,**params):
        params.update(userName=credentials['username'],kernelSlug=SLUG)
        req=urllib.request.Request('https://www.kaggle.com/api/v1/kernels/'+endpoint+'?'+urllib.parse.urlencode(params),
           headers={'Authorization':auth,'User-Agent':'kaggle-api/v1.7.0','Accept':'application/json'})
        with urllib.request.urlopen(req,timeout=40) as r:return json.loads(r.read(24000000))
    s=api('status');status={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':s.get('status'),'failureMessage':s.get('failureMessage')}
    (LAUNCH/'remote_status.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status),flush=True)
    if not download:return
    if str(status['status']).lower() not in ['complete','error','failed','cancelacknowledged']:raise RuntimeError('Wait until the single job is terminal')
    response=api('output',pageSize=100);files=response.get('files',[]);log=response.get('log');ids=set()
    while response.get('nextPageToken'):
        response=api('output',pageSize=100,pageToken=response['nextPageToken']);files+=response.get('files',[])
    if log:(LAUNCH/'kaggle_run_log.json').write_text(log if isinstance(log,str) else json.dumps(log,indent=2))
    got=[]
    for f in files:
        name=f.get('fileName',f.get('fileNameNullable'));url=f.get('url',f.get('urlNullable'))
        if not name or not url:continue
        # Download outputs and execution evidence, not a second copy of extracted raw inputs.
        if not name.startswith(('outputs/insane_pilot/','insane_launch/')):continue
        if Path(name).is_absolute() or '..' in Path(name).parts:raise RuntimeError('Unsafe output path')
        match=re.search(r'/([0-9]+)/',url)
        if match:ids.add(match.group(1))
        dest=ROOT/name if name.startswith('outputs/insane_pilot/') else LAUNCH/'remote_output'/name
        with urllib.request.urlopen(url,timeout=40) as r:b=r.read(16000001)
        if len(b)>16000000:raise RuntimeError('Unexpected output size')
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists() and dest.read_bytes()!=b:raise RuntimeError('Never overwrite changed remote output '+name)
        if not dest.exists():dest.write_bytes(b)
        got.append({'remote_path':name,'local_path':str(dest.relative_to(ROOT)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
    receipt={'kernel':credentials['username']+'/'+SLUG,'status':status['status'],'output_ids':sorted(ids),'files':got}
    (LAUNCH/'download_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'files_downloaded':len(got),'bytes':sum(f['bytes'] for f in got),'output_ids':sorted(ids)}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--download',action='store_true');args=p.parse_args();main(args.download)
