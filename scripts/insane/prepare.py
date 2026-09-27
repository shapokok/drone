"""Freeze flight splits and every augmentation/evaluation episode before training."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from insane.data import load_flight,aggregate_imu,quality_segments,fit_scaler,episode_inputs,sha

def write(path,obj):
    text=json.dumps(obj,indent=2,allow_nan=False)+'\n'
    if path.exists() and path.read_text()!=text:raise RuntimeError('Frozen artifact differs: '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)

def main():
    data=ROOT/'data/insane';folder=ROOT/'configs/insane';cfg=json.loads((folder/'pilot.json').read_text())
    split={'train':['mars_1','mars_2','mars_3'],'validation':['mars_4','mars_5'],'test_reserved':['mars_6','mars_7'],
       'loader_smoke_only':['outdoor_1'],'rule':'numerical flight order; distinct recordings; selection by availability/duration only',
       'test_policy':'hash/schema/timing inventory only; no metrics, scalers, initialization, or training',
       'same_campaign_warning':'separate flights from same campaign/vehicle, not independent sites'}
    write(folder/'splits.json',split)
    files={};manifest={'source':'https://www.aau.at/en/smart-systems-technologies/control-of-networked-systems/datasets/insane-dataset/',
      'license':'metadata/LICENSE.txt','calibration':'calibration/insane_sensor_calib_preprocessed/sensor_calibration.yaml',
      'author_tools_commit':'9a1c8c0fdd195f2d869fff292f2ce5b273c5a03d','sequences':{}}
    loaded={};aggs={};inventory={};plans=[];unsupported=[];training=[]
    for seq in sum([split[k] for k in ['train','validation','test_reserved','loader_smoke_only']],[]):
        receipt=json.loads((data/'metadata'/f'{seq}_download.json').read_text());manifest['sequences'][seq]=receipt
        sensor,ref=load_flight(data,seq,receipt)
        agg=aggregate_imu(sensor,cfg['token_period_s']);segments=quality_segments(sensor,ref,cfg)
        inventory[seq]={'split':next(k for k,v in split.items() if isinstance(v,list) and seq in v),'native_imu_n':len(sensor['imu_t']),
          'native_gps_n':len(sensor['gps_t']),'native_gt_n':len(ref['t']),'imu_duration_s':float(np.ptp(sensor['imu_t'])),
          'gt_duration_s':float(np.ptp(ref['t'])),'imu_median_dt_s':float(np.median(np.diff(sensor['imu_t']))),
          'gt_median_dt_s':float(np.median(np.diff(ref['t']))),'gt_mean_dt_s':float(np.mean(np.diff(ref['t']))),
          'projection_check_max_m':sensor['projection_error_m'],'segments_s':segments,'token_n':len(agg['t']),
          'epoch_unix_s':sensor['epoch'],'origin_enu_m':sensor['origin_enu'].tolist()}
        if seq in split['train']+split['validation']+split['loader_smoke_only']:loaded[seq]=(sensor,ref);aggs[seq]=agg
    # Pin all calibration/source metadata that will be shipped. PDFs are reading references only.
    for path in list((data/'calibration').rglob('*'))+list((data/'metadata/author_tools').rglob('*'))+[data/'metadata/LICENSE.txt']:
        if path.is_file():files[str(path.relative_to(data))]={'bytes':path.stat().st_size,'sha256':sha(path)}
    manifest['support_files']=files
    write(folder/'download_manifest.json',manifest)
    scaler=fit_scaler(aggs,split['train'])
    for seq in split['validation']:
        sensor,ref=loaded[seq];agg=aggs[seq];number=0
        for a,b in inventory[seq]['segments_s']:
            # All durations share starts if 60s fits; otherwise record duration-specific unsupported.
            onset=a+cfg['history_s']+.1
            while onset+max(cfg['duration_s'])+cfg['recovery_s']+.1 <= b:
                onset=float(agg['t'][np.searchsorted(agg['t'],onset)]);number+=1
                for duration in cfg['duration_s']:
                    plans.append({'id':f'{seq}_e{number}_d{duration}','flight':seq,'split':'validation','duration_s':duration,
                      'onset_s':onset,'start_s':onset-cfg['history_s'],'end_s':onset+duration+cfg['recovery_s']})
                onset+=cfg['validation_onset_stride_s']
        if not number:
            for duration in cfg['duration_s']:
                ok=[(a,b) for a,b in inventory[seq]['segments_s'] if b-a>=cfg['history_s']+duration+cfg['recovery_s']+.2]
                if not ok:unsupported.append({'flight':seq,'duration_s':duration,'reason':'no continuous history+outage+recovery interval'});continue
                a,b=ok[0];onset=float(agg['t'][np.searchsorted(agg['t'],a+cfg['history_s']+.1)])
                plans.append({'id':f'{seq}_e1_d{duration}','flight':seq,'split':'validation','duration_s':duration,
                    'onset_s':onset,'start_s':onset-cfg['history_s'],'end_s':onset+duration+cfg['recovery_s']})
        a,b=max(inventory[seq]['segments_s'],key=lambda x:x[1]-x[0])
        plans.append({'id':f'{seq}_control','flight':seq,'split':'validation','duration_s':0,'onset_s':None,
                      'start_s':a+.1,'end_s':b-.1})
    for epoch in range(1,cfg['max_epochs']+1):
        rng=np.random.default_rng(cfg['seed']+10000+epoch);specs=[]
        for seq in split['train']:
            for duration in cfg['duration_s']:
                spans=[(a+cfg['history_s']+.1,b-duration-cfg['recovery_s']-.1) for a,b in inventory[seq]['segments_s'] if b-a>cfg['history_s']+duration+cfg['recovery_s']+.3]
                if not spans:raise ValueError(f'Unsupported train episode {seq} {duration}')
                for k in range(cfg['train_episodes_per_flight_duration_epoch']):
                    lo,hi=spans[int(rng.integers(len(spans)))];tt=aggs[seq]['t'];onset=float(tt[np.searchsorted(tt,rng.uniform(lo,hi))])
                    specs.append({'id':f'train_ep{epoch}_{seq}_d{duration}_{k}','epoch':epoch,'flight':seq,'split':'train',
                        'duration_s':duration,'onset_s':onset,'start_s':onset-cfg['history_s'],'end_s':onset+duration+cfg['recovery_s']})
        training += [specs[int(k)] for k in rng.permutation(len(specs))]
    for spec in plans+training:
        seq=spec['flight'];x,y=episode_inputs(*loaded[seq],aggs[seq],spec,scaler,cfg)
        spec['n_native_gt_outage']=int(y['outage'].sum());spec['n_native_gt_total']=len(y['t'])
        spec['token_mask_sha256']=hashlib.sha256(np.asarray(x['availability'],np.uint8).tobytes()).hexdigest()
        spec['native_gt_timestamp_sha256']=hashlib.sha256(np.asarray(y['t'],dtype='<f8').tobytes()).hexdigest()
    write(folder/'outages.json',{'selection':'chronological 60s stride; identical starts across durations/methods; no error-based selection','scenarios':plans,'unsupported':unsupported})
    write(folder/'training_episodes.json',{'seed':cfg['seed'],'episodes':training,'same_for_both_models':True})
    write(folder/'inventory.json',inventory)
    write(folder/'scaler_preflight.json',scaler)
    locks={f.name:sha(f) for f in folder.glob('*.json') if f.name!='protocol_lock.json'}
    write(folder/'protocol_lock.json',{'pilot_id':cfg['pilot_id'],'sha256':locks,'freeze_stage':'before any training/validation metric','model_comparisons_used_for_selection':False})
    print(json.dumps({'flights':split,'validation_scenarios':len(plans),'training_episodes':len(training),'unsupported':unsupported,
       'sensor_file_bytes':sum(f['bytes'] for v in manifest['sequences'].values() for f in v['files'])},indent=2))

if __name__=='__main__':main()
