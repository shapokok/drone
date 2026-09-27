"""One explicit authorized private Kaggle submission. Never auto retry."""
from pathlib import Path
import json,hashlib,datetime,shutil,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from insane_final_test.run import source_hashes

def main():
    launch=ROOT/'outputs/insane_final_test_retry_launch';launch.mkdir(exist_ok=True)
    if (launch/'submission_attempt.json').exists():raise RuntimeError('Submission already attempted; inspect status, never auto-resubmit')
    check=json.loads((ROOT/'outputs/insane_final_test_retry_preparation/preflight.json').read_text())
    lock_path=ROOT/'configs/insane_final_test/FINAL_EVALUATION_LOCK.json'
    version=hashlib.sha256(lock_path.read_bytes()).hexdigest()
    assert version==check['final_lock_sha256'] and check['unit_tests_passed']==41
    assert source_hashes(ROOT)==json.loads(lock_path.read_text())['evaluation_source_sha256']
    nb=ROOT/'notebooks/kaggle_insane_final_test_retry.ipynb';text=nb.read_text()
    assert 'ALLOW_FINAL_TEST = True' in text
    assert hashlib.sha256(nb.read_bytes()).hexdigest()==check['notebook_sha256']
    assert check['exact_notebook_preflight_passed'] and not check['predictions_or_metrics_run_locally']
    patch_path=ROOT/'configs/insane_final_test_patch/PREFLIGHT_PATCH_MANIFEST.json'
    assert hashlib.sha256(patch_path.read_bytes()).hexdigest()==check['patch_manifest_sha256']
    for rel,h in json.loads(patch_path.read_text())['patch_source_sha256'].items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h
    source=launch/nb.name;shutil.copyfile(nb,source)
    meta={'id':'shapok/drone-nav-insane-final-test','title':'Drone Nav Insane Final Test','code_file':nb.name,'language':'python','kernel_type':'notebook',
          'is_private':True,'enable_gpu':True,'enable_tpu':False,'enable_internet':False,
          'dataset_sources':['shapok/insane-locked-final-test-inputs'],'competition_sources':[],'kernel_sources':[]}
    (launch/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    import kaggle
    api=kaggle.api
    matches=[d for d in api.dataset_list(mine=True,page=1) if getattr(d,'ref','')=='shapok/insane-locked-final-test-inputs']
    if len(matches)!=1:raise RuntimeError('Uploaded private dataset not yet visible')
    dataset=matches[0]
    private=getattr(dataset,'is_private',None)
    if private is None:private=getattr(dataset,'isPrivate',None)
    if private is not True:raise RuntimeError('Private dataset visibility not verified: '+str(private))
    attempt={'attempt_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'final_lock_sha256':version,
       'notebook_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'data_private_verified':True,'single_job_only':True,'authorized_retry_of_preflight_failure':'353128919','patch_manifest_sha256':check['patch_manifest_sha256'],'planned_trainings':0,'planned_model_evaluations':48,'planned_baseline_evaluations':24,'seeds':[0,1,2]}
    (launch/'submission_attempt.json').write_text(json.dumps(attempt,indent=2)+'\n')
    result=api.kernels_push(str(launch),timeout=3600)
    record={**attempt,'kernel':meta['id'],'response_type':type(result).__name__}
    for key in ['url','ref','version_number','versionNumber','id','error','error_description','kernel_version_id']:
        value=getattr(result,key,None)
        if value is not None:record[key]=value if isinstance(value,(int,float,bool,str)) else str(value)
    (launch/'submission.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
    if record.get('error'):raise RuntimeError('Kaggle submission error: '+str(record['error']))

if __name__=='__main__':main()
