"""One explicit authorized private Kaggle submission. Never auto retry."""
from pathlib import Path
import json,hashlib,datetime,shutil,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from insane.run import code_version

def main():
    launch=ROOT/'outputs/insane_pilot_launch';launch.mkdir(exist_ok=True)
    if (launch/'submission_attempt.json').exists():raise RuntimeError('Submission already attempted; inspect status, never auto-resubmit')
    check=json.loads((ROOT/'outputs/insane_preparation/preflight.json').read_text())
    version,hashes=code_version(ROOT);assert version==check['code_version'] and check['unit_tests_passed']==18
    nb=ROOT/'notebooks/kaggle_insane_pilot.ipynb';text=nb.read_text()
    assert 'RUN_TRAINING = True' in text
    source=launch/nb.name;shutil.copyfile(nb,source)
    meta={'id':'shapok/drone-nav-insane-pilot','title':'Drone Nav Insane Pilot','code_file':nb.name,'language':'python','kernel_type':'notebook',
          'is_private':True,'enable_gpu':True,'enable_tpu':False,'enable_internet':False,
          'dataset_sources':['shapok/insane-navigation-pilot-data-v1'],'competition_sources':[],'kernel_sources':[]}
    (launch/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    import kaggle
    api=kaggle.api
    matches=[d for d in api.dataset_list(mine=True,page=1) if getattr(d,'ref','')=='shapok/insane-navigation-pilot-data-v1']
    if len(matches)!=1:raise RuntimeError('Uploaded private dataset not yet visible')
    dataset=matches[0]
    private=getattr(dataset,'is_private',None)
    if private is None:private=getattr(dataset,'isPrivate',None)
    if private is not True:raise RuntimeError('Private dataset visibility not verified: '+str(private))
    attempt={'attempt_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'code_version':version,
       'notebook_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'data_private_verified':True,'single_pilot_only':True}
    (launch/'submission_attempt.json').write_text(json.dumps(attempt,indent=2)+'\n')
    result=api.kernels_push(str(launch),timeout=3600)
    record={**attempt,'kernel':meta['id'],'response_type':type(result).__name__}
    for key in ['url','ref','version_number','versionNumber','id','error','error_description','kernel_version_id']:
        value=getattr(result,key,None)
        if value is not None:record[key]=value if isinstance(value,(int,float,bool,str)) else str(value)
    (launch/'submission.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
    if record.get('error'):raise RuntimeError('Kaggle submission error: '+str(record['error']))

if __name__=='__main__':main()
