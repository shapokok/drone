"""One private final-test input upload, explicitly authorized by the user."""
from pathlib import Path
import json,datetime,hashlib,traceback
ROOT=Path(__file__).resolve().parents[2]

def main():
    folder=ROOT/'outputs/insane_final_test_data_upload';check=json.loads((ROOT/'outputs/insane_final_test_preparation/preflight.json').read_text())
    assert check['status']=='pass' and check['unit_tests_passed']==33 and check['exact_notebook_preflight_passed']
    if (folder/'upload_attempt.json').exists():raise RuntimeError('Upload already attempted; STOP, no automatic retry')
    import kaggle
    api=kaggle.api;slug='shapok/insane-locked-final-test-inputs'
    if any(getattr(x,'ref',None)==slug for x in api.dataset_list(mine=True,page=1)):raise RuntimeError('Dataset already exists; STOP')
    attempt={'dataset':slug,'public':False,'test_flights':['mars_6','mars_7'],'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'final_lock_sha256':check['final_lock_sha256']}
    # Keep operational receipts outside the dataset folder.
    prep=ROOT/'outputs/insane_final_test_preparation'
    if (prep/'upload_attempt.json').exists():raise RuntimeError('Upload already attempted')
    (prep/'upload_attempt.json').write_text(json.dumps(attempt,indent=2)+'\n')
    try:
        result=api.dataset_create_new(str(folder),public=False,quiet=True,convert_to_csv=False,dir_mode='skip')
        record={**attempt,'response_type':type(result).__name__}
        for key in ['url','status','error']:
            value=getattr(result,key,None)
            if value is not None:record[key]=str(value)
        (prep/'upload_receipt.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
        if record.get('error'):raise RuntimeError('Dataset upload rejected')
    except Exception:
        (prep/'upload_failure.log').write_text(traceback.format_exc());raise

if __name__=='__main__':main()
