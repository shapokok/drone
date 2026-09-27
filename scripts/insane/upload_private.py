"""Upload only the user-authorized private, test-free dataset once."""
from pathlib import Path
import json,datetime,hashlib,base64
ROOT=Path(__file__).resolve().parents[2]

def main():
    checks=json.loads((ROOT/'outputs/insane_preparation/package_preflight.json').read_text())
    assert checks['status']=='pass' and checks['test_physically_absent']
    folder=ROOT/'outputs/insane_data_upload';receipt=folder/'upload_receipt.json'
    if receipt.exists():raise RuntimeError('Dataset upload already recorded; do not duplicate')
    import kaggle
    api=kaggle.api
    slug='shapok/insane-navigation-pilot-data-v1'
    existing=api.dataset_list(mine=True,page=1)
    if any(getattr(x,'ref',None)==slug for x in existing):raise RuntimeError('Dataset already exists; inspect before any mutation')
    result=api.dataset_create_new(str(folder),public=False,quiet=True,convert_to_csv=False,dir_mode='skip')
    info={'dataset':slug,'public':False,'test_uploaded':False,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'bundle_sha256':hashlib.sha256(base64.b64decode((folder/'insane_payload.b64').read_bytes(),validate=True)).hexdigest(),
          'response_type':type(result).__name__}
    for key in ['url','status','error']:
        val=getattr(result,key,None)
        if val is not None:info[key]=str(val)
    receipt.write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info,indent=2))
    if info.get('error'):raise RuntimeError('Kaggle rejected dataset creation: '+info['error'])

if __name__=='__main__':main()
