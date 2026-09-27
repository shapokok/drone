"""Time/support-only selector copied from the frozen pilot preparation algorithm."""
import hashlib
import numpy as np
from insane.data import load_flight,aggregate_imu,quality_segments,episode_inputs

def select_specs(sequence,segments,t,cfg,split='test'):
    plans=[];unsupported=[];number=0
    for a,b in segments:
        onset=a+cfg['history_s']+.1
        while onset+max(cfg['duration_s'])+cfg['recovery_s']+.1<=b:
            onset=float(t[np.searchsorted(t,onset)]);number+=1
            for duration in cfg['duration_s']:
                plans.append({'id':f'{sequence}_e{number}_d{duration}','flight':sequence,'split':split,'duration_s':duration,
                    'onset_s':onset,'start_s':onset-cfg['history_s'],'end_s':onset+duration+cfg['recovery_s']})
            onset+=cfg['validation_onset_stride_s']
    if not number:
        for duration in cfg['duration_s']:
            ok=[(a,b) for a,b in segments if b-a>=cfg['history_s']+duration+cfg['recovery_s']+.2]
            if not ok:
                unsupported.append({'flight':sequence,'duration_s':duration,'reason':'no continuous history+outage+recovery interval'});continue
            a,b=ok[0];onset=float(t[np.searchsorted(t,a+cfg['history_s']+.1)])
            plans.append({'id':f'{sequence}_e1_d{duration}','flight':sequence,'split':split,'duration_s':duration,
                'onset_s':onset,'start_s':onset-cfg['history_s'],'end_s':onset+duration+cfg['recovery_s']})
    if segments:
        a,b=max(segments,key=lambda x:x[1]-x[0])
        plans.append({'id':f'{sequence}_control','flight':sequence,'split':split,'duration_s':0,'onset_s':None,'start_s':a+.1,'end_s':b-.1})
    else:unsupported.append({'flight':sequence,'duration_s':0,'reason':'no common continuous sensor/reference support for control'})
    return plans,unsupported

def prepare_test(raw,cfg,scaler,download_manifest):
    scenarios=[];unsupported=[];inventory={};cache={}
    for seq in ['mars_6','mars_7']:
        sensor,reference=load_flight(raw,seq,download_manifest['sequences'][seq]);agg=aggregate_imu(sensor,cfg['token_period_s'])
        segments=quality_segments(sensor,reference,cfg)
        specs,missing=select_specs(seq,segments,agg['t'],cfg)
        inventory[seq]={'epoch_unix_s':sensor['epoch'],'origin_enu_m':sensor['origin_enu'].tolist(),
            'segments_s':segments,'projection_error_m':sensor['projection_error_m'],'imu_n':len(sensor['imu_t']),
            'gps_n':len(sensor['gps_t']),'gt_n':len(reference['t']),'token_n':len(agg['t'])}
        for spec in specs:
            x,y=episode_inputs(sensor,reference,agg,spec,scaler,cfg)
            spec.update(n_native_gt_outage=int(y['outage'].sum()),n_native_gt_total=len(y['t']),
                token_mask_sha256=hashlib.sha256(np.asarray(x['availability'],np.uint8).tobytes()).hexdigest(),
                native_gt_timestamp_sha256=hashlib.sha256(np.asarray(y['t'],dtype='<f8').tobytes()).hexdigest())
            cache[spec['id']]=(x,y)
        scenarios+=specs;unsupported+=missing
    return {'split':'test','selection':'unchanged chronological/support-only validation algorithm, onset stride 60s; no motion/error selection',
      'scenarios':scenarios,'unsupported':unsupported,'inventory':inventory,'predictions_or_metrics_computed':False},cache
