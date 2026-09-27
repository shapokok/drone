"""Freeze existing models, deterministic test support, and private input bytes.
No model forward, metric computation, training, or public upload.
"""
from pathlib import Path
import sys,json,hashlib,shutil,datetime,zipfile,base64,io
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from insane.data import sha
from insane_final_test.protocol import prepare_test
from insane_final_test.run import source_hashes

def write_new(path,obj):
    text=json.dumps(obj,indent=2,allow_nan=False)+'\n';path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.read_text()!=text:raise RuntimeError('Never overwrite frozen artifact: '+str(path))
    if not path.exists():path.write_text(text)

def main():
    final=ROOT/'configs/insane_final_test';prep=ROOT/'outputs/insane_final_test_preparation';upload=ROOT/'outputs/insane_final_test_data_upload'
    if (final/'FINAL_EVALUATION_LOCK.json').exists():raise RuntimeError('Final lock exists; do not regenerate test protocol')
    adaptive=ROOT/'outputs/insane_adaptive';raw=ROOT/'data/insane';frozen=prep/'bundle/frozen';frozen.mkdir(parents=True)
    cfg=json.loads((adaptive/'config.json').read_text());scaler=json.loads((adaptive/'scalers.json').read_text());download=json.loads((adaptive/'download_manifest.json').read_text())
    oldhash=json.loads((adaptive/'source_hashes.json').read_text())
    for rel,h in oldhash.items():
        assert sha(adaptive/'source_snapshot'/rel)==h and sha(ROOT/rel)==h,rel
        dst=frozen/'source_snapshot'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(adaptive/'source_snapshot'/rel,dst)
    mappings={'config.json':'config.json','scalers.json':'scalers.json','download_manifest.json':'download_manifest.json',
       'damping_lock.json':'damping_lock.json','status.json':'adaptive_status.json','source_hashes.json':'adaptive_source_hashes.json'}
    for source,target in mappings.items():shutil.copyfile(adaptive/source,frozen/target)
    shutil.copyfile(ROOT/'METHODS_DRAFT.md',frozen/'methods_before_test.md');(frozen/'checkpoints').mkdir()
    checkpoints={}
    for method in ['adaptive_full','adaptive_gps_only']:
        for seed in [0,1,2]:
            name=f'{method}_seed{seed}_best.pt';p=adaptive/'checkpoints'/name;checkpoints[name]=sha(p);shutil.copyfile(p,frozen/'checkpoints'/name)
    assert json.loads((frozen/'damping_lock.json').read_text())['tau_s']==5
    # This computes only support/input metadata, never predictions or errors.
    test_manifest,_=prepare_test(raw,cfg,scaler,download);write_new(final/'test_manifest.json',test_manifest)
    raw_names=set(download['support_files'])
    for seq in ['mars_6','mars_7']:
        raw_names.update(f['path'] for f in download['sequences'][seq]['files'])
    raw_hash={}
    for name in sorted(raw_names):
        if any(s in Path(name).parts for s in ['mars_1','mars_2','mars_3','mars_4','mars_5','outdoor_1']):raise ValueError('Non-test flight in bundle')
        raw_hash[name]=sha(raw/name)
    frozen_hash={str(p.relative_to(prep/'bundle')):sha(p) for p in sorted(frozen.rglob('*')) if p.is_file()}
    lock={'stage':'final_test_once','locked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'checkpoint_selection':'all six best checkpoints selected on prior validation; no seed exclusion or ensemble',
      'checkpoint_sha256':checkpoints,'frozen_files_sha256':frozen_hash,'raw_files_sha256':raw_hash,
      'evaluation_source_sha256':source_hashes(ROOT),'original_source_snapshot_sha256':oldhash,
      'test_manifest_sha256':sha(final/'test_manifest.json'),'test_flights':['mars_6','mars_7'],'seeds':[0,1,2],
      'learned_methods':['adaptive_full','adaptive_gps_only'],'baselines':['held_gnss','constant_velocity_gnss','damped_cv'],'primary_model':'adaptive_full','tau_s':5.,
      'primary_metric':'sqrt(mean(||[prediction-last legal GNSS]-[GT(t)-GT(last legal fix time)]||^2)) only on original GT timestamps inside outage',
      'additional_metrics':['absolute_rmse3d_m','absolute_rmse_h_m','absolute_rmse_v_m','final_unavailable_error_m','final_relative_motion_error_m','recovery_first_5s_rmse3d_m','first_recovered_fix_error_m'],
      'controls':'outage metrics N/A; score absolute on all original supported GT points',
      'selection':'exact chronological/support-only validation selector;20s history/10s recovery/60s onset stride;10/30/60s durations;same prior gap limits;max-duration common starts;original earliest-supported fallback;longest-segment control',
      'aggregation':'episode RMSE macro mean per duration within seed, then mean/sample SD ddof1 across3seed means; separate flight and seed tables; deterministic baselines once',
      'prohibitions':['training','refit scalers','test tuning','new seeds','test retries after results','GT initialization','trajectory alignment','validation/test pooling'],
      'eskf':'not_ready; no numeric evaluation','test_metrics_seen_at_lock':False}
    write_new(final/'FINAL_EVALUATION_LOCK.json',lock)
    upload.mkdir();buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        for name in sorted(raw_names):z.write(raw/name,'bundle/raw/'+name)
        for p in sorted(frozen.rglob('*')):
            if p.is_file():z.write(p,'bundle/'+str(p.relative_to(prep/'bundle')))
    payload=buf.getvalue();(upload/'final_test_payload.b64').write_bytes(base64.b64encode(payload))
    write_new(prep/'transport.json',{'payload_sha256':hashlib.sha256(payload).hexdigest(),'encoding':'base64 ZIP; single extraction','raw_flights':['mars_6','mars_7'],
      'encoded_bytes':(upload/'final_test_payload.b64').stat().st_size,'raw_bytes':sum((raw/n).stat().st_size for n in raw_names),'checkpoint_count':6})
    shutil.copyfile(raw/'metadata/LICENSE.txt',upload/'LICENSE.txt')
    write_new(upload/'source_manifest.json',{'source':download['source'],'private':True,'raw_flights':['mars_6','mars_7'],
       'raw_files_sha256':raw_hash,'checkpoint_sha256':checkpoints,'final_lock_sha256':sha(final/'FINAL_EVALUATION_LOCK.json')})
    (upload/'README.md').write_text('''# INSANE locked final test — private research input

Only mars_6/mars_7 sensor CSVs, existing six frozen adaptive checkpoints and fixed calibration/protocol metadata.
No images. No training or public redistribution. Preserve the full LICENSE.txt (BSD-2 plus non-commercial/citation conditions).
Source: https://www.aau.at/en/smart-systems-technologies/control-of-networked-systems/datasets/insane-dataset/
Authors tools revision 9a1c8c0fdd195f2d869fff292f2ce5b273c5a03d; dataset publication arXiv:2210.09114.
Transport is base64-encoded ZIP to preserve nested calibration archives byte-for-byte under Kaggle ingestion.
''')
    write_new(upload/'dataset-metadata.json',{'title':'INSANE Locked Final Test Inputs','id':'shapok/insane-locked-final-test-inputs','licenses':[{'name':'other'}],'description':(upload/'README.md').read_text()})
    print(json.dumps({'scenarios':test_manifest['scenarios'],'unsupported':test_manifest['unsupported'],'lock_sha256':sha(final/'FINAL_EVALUATION_LOCK.json'),
       'transport':json.loads((prep/'transport.json').read_text())},indent=2))

if __name__=='__main__':main()
