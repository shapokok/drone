"""Selective public INSANE acquisition via HTTP Range; no Kaggle upload or job."""
from pathlib import Path
import io,urllib.request,zipfile,hashlib,json,argparse,datetime
ROOT=Path(__file__).resolve().parents[2]
BASE='https://cns-data.aau.at/insane-dataset/'

class RemoteZip(io.RawIOBase):
    def __init__(self,url):
        self.url=url;self.pos=0;self.received=0
        with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=45) as r:
            self.size=int(r.headers['Content-Length']);self.etag=r.headers.get('ETag')
    def seekable(self):return True
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else (self.pos+offset if whence==1 else self.size+offset)
        return self.pos
    def tell(self):return self.pos
    def read(self,n=-1):
        if n<0:n=self.size-self.pos
        n=min(n,self.size-self.pos)
        if n<=0:return b''
        req=urllib.request.Request(self.url,headers={'Range':f'bytes={self.pos}-{self.pos+n-1}','If-Range':self.etag or ''})
        with urllib.request.urlopen(req,timeout=60) as r:
            if r.status!=206:raise RuntimeError('Server ignored Range; refusing whole sensor archive')
            b=r.read(n+1)
        if len(b)!=n:raise RuntimeError('Short/oversized HTTP Range')
        self.pos+=len(b);self.received+=len(b);return b


def put(path,b):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.read_bytes()!=b:raise RuntimeError('Refuse overwrite changed source '+str(path))
    if not path.exists():path.write_bytes(b)
    return {'path':str(path.relative_to(ROOT/'data/insane')),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


def acquire(sequence):
    dest=ROOT/'data/insane';record=dest/'metadata'/(sequence+'_download.json')
    if record.exists():
        old=json.loads(record.read_text())
        for f in old['files']:assert hashlib.sha256((dest/f['path']).read_bytes()).hexdigest()==f['sha256']
        print(sequence,'already verified',flush=True);return
    url=BASE+sequence+'_sensors.zip';remote=RemoteZip(url)
    with zipfile.ZipFile(remote) as z:
        names=z.namelist();wanted=[]
        for required in ['px4_imu.csv','px4_gps.csv','ground_truth_8hz.csv']:
            hits=[name for name in names if Path(name).name==required]
            if len(hits)!=1:raise RuntimeError(f'{sequence}: expected {required}, found {hits}; members={names}')
            wanted+=hits
        wanted += [name for name in names if Path(name).name.lower() in ['settings.yaml','time_info.yaml','readme.md','readme.txt','license.txt'] and name not in wanted]
        files=[]
        for name in wanted:
            relative=Path(name).name if Path(name).name in ['px4_imu.csv','px4_gps.csv','ground_truth_8hz.csv'] else Path('archive_metadata')/name
            if '..' in Path(relative).parts:raise RuntimeError('Unsafe member')
            row=put(dest/sequence/relative,z.read(name));row['archive_member']=name;files.append(row)
    meta={'sequence':sequence,'calibration_set':'KLU1' if sequence=='outdoor_1' else 'MA','url':url,
          'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'archive_bytes':remote.size,
          'archive_etag':remote.etag,'http_bytes_received':remote.received,'archive_members':names,'files':files}
    record.write_text(json.dumps(meta,indent=2)+'\n')
    print(sequence,'saved',sum(f['bytes'] for f in files),'bytes; downloaded',remote.received,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('sequences',nargs='+');args=p.parse_args()
    for sequence in args.sequences:acquire(sequence)
