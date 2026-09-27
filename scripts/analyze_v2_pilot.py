"""Verify and summarize saved V2 pilot outputs only; never trains or infers."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/v2_motion_pilot'
LAUNCH=ROOT/'outputs/v2_pilot_launch'
METHODS=['fusion_v2_full','fusion_v2_gps_only','lstm_v2_full','fusion_v2_full_zero_imu','held_gnss','constant_velocity_gnss']

def read_csv(p):return list(csv.DictReader(p.open()))
def write_csv(p,rows):
    with p.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def scalar(value):return None if value in [None,''] else float(value)
def rmse(x):return float(np.sqrt(np.mean(np.sum(x*x,axis=1))))

def main():
    status=json.loads((OUT/'status.json').read_text());assert status['status']=='complete'
    assert status['evaluation_rows']==60 and not status['test_used']
    for local,original in [('config.json','v2_motion_pilot.json'),('manifest.json','v2_motion_pilot_manifest.json')]:
        assert json.loads((OUT/local).read_text())==json.loads((ROOT/'configs'/original).read_text())
    cfg=json.loads((OUT/'config.json').read_text());plan=json.loads((OUT/'manifest.json').read_text())
    for name,expected in json.loads((OUT/'source_sha256.json').read_text()).items():
        assert sha(OUT/'source_snapshot'/name)==expected==sha(ROOT/name),name
    preflight=json.loads((LAUNCH/'remote_output/v2_pilot_launch/preflight.json').read_text())
    assert preflight['status']=='passed' and preflight['production_pyproj_exercised']
    history=read_csv(OUT/'training_history.csv');assert len(history)==24
    for method in METHODS[:3]:
        epochs=[int(x['epoch']) for x in history if x['method']==method]
        assert epochs==list(range(1,9))
    assert all(np.isfinite(float(x['mean_batch_training_loss'])) for x in history)
    checkpoints=list((OUT/'checkpoints').glob('*.pt'));assert len(checkpoints)==3
    assert {p.stem for p in checkpoints}=={m+'_seed0' for m in METHODS[:3]}
    summary=read_csv(OUT/'summary.csv');assert len(summary)==60
    keyed={(x['scenario_id'],x['method']):x for x in summary};assert len(keyed)==60
    expected_keys={(s['scenario_id'],m) for s in plan['scenarios'] for m in METHODS};assert set(keyed)==expected_keys
    # Rebuild native camera GT labels from raw source timestamps, independently of saved motion_error.
    raw=ROOT/'outputs/next_stage_diagnostics/raw'
    gps=pd.read_csv(raw/'OnboardGPS.csv');gt=pd.read_csv(raw/'GroundTruthAGL.csv')
    gps.columns=gps.columns.str.strip();gt.columns=gt.columns.str.strip()
    gt['t']=gt.imgid.map(gps.set_index('imgid').Timpstemp)*1e-6
    gt=gt.dropna(subset=['t']).sort_values('t',kind='stable')
    raw_gt=gt[['x_gt','y_gt','z_gt']].values
    def interp(t):return np.column_stack([np.interp(t,gt.t.values,raw_gt[:,j]) for j in range(3)])
    differences=[];max_metric_error=0.;max_target_error=0.;extras=[]
    for s in plan['scenarios']:
        sid=s['scenario_id'];held=np.load(OUT/sid/'held_gnss.npz',allow_pickle=False)
        t=held['timestamps'];mask=held['availability'];missing=np.flatnonzero(~mask)
        assert hashlib.sha256(t.astype('<f8').tobytes()).hexdigest()==plan['common_timestamps_float64_le_sha256']
        assert hashlib.sha256(mask.astype('uint8').tobytes()).hexdigest()==s['mask_uint8_sha256']
        use=missing if len(missing) else np.arange(len(t));target=held['motion_target']
        if len(missing):
            a=missing[0];anchor_t=held['source_timestamp'][a-1]
            true_motion=interp(t[missing])-interp(np.array([anchor_t]))[0]
            err=float(np.max(np.abs(true_motion-target[missing])));max_target_error=max(max_target_error,err)
            np.testing.assert_allclose(true_motion,target[missing],atol=2e-5,rtol=0)
        preds={};motion_scores={};zero_diff=None
        for method in METHODS:
            row=keyed[sid,method];d=np.load(OUT/sid/(method+'.npz'),allow_pickle=False)
            for field in ['timestamps','availability','gt','motion_target','source_timestamp','source_index']:
                np.testing.assert_array_equal(d[field],held[field])
            p=d['pred'].astype(float);truth=d['gt'].astype(float)
            assert p.shape==(2688,3) and np.isfinite(p).all() and np.isfinite(truth).all()
            assert np.all(d['source_timestamp']<=t+1e-9)
            if len(missing):
                source=d['source_timestamp'][missing]
                assert not np.any((source>=s['onset_s'])&(source<s['requested_cutoff_s']))
            meta=json.loads(str(d['metadata_json']))
            checkpoint_name=method if method!='fusion_v2_full_zero_imu' else 'fusion_v2_full'
            checkpoint=OUT/'checkpoints'/(checkpoint_name+'_seed0.pt')
            if checkpoint.exists():assert row['checkpoint_sha256']==meta['checkpoint_sha256']==sha(checkpoint)
            else:assert row['checkpoint_sha256']==''
            error=p-truth;motion_error=(p-held['pred'])-target
            np.testing.assert_allclose(motion_error,d['motion_error'],atol=1e-12,rtol=0)
            metrics={'rmse_3d_m':rmse(error[use]),'rmse_horizontal_m':rmse(error[use,:2]),
                     'rmse_vertical_m':rmse(error[use,2:]),'motion_rmse_3d_m':rmse(motion_error[missing]) if len(missing) else None}
            if len(missing):
                last=missing[-1];metrics.update(final_unavailable_error_3d_m=float(np.linalg.norm(error[last])),
                    final_unavailable_error_horizontal_m=float(np.linalg.norm(error[last,:2])),
                    final_unavailable_abs_vertical_m=float(abs(error[last,2])))
            for name,value in metrics.items():
                saved=scalar(row[name])
                if value is None:assert saved is None
                else:
                    delta=abs(value-saved);max_metric_error=max(max_metric_error,delta);assert delta<1e-10,(sid,method,name,delta)
            assert int(row['n_evaluated'])==len(use)
            preds[method]=p;motion_scores[method]=metrics['motion_rmse_3d_m']
            extras.append({'scenario_id':sid,'method':method,'duration_s':s['requested_duration_s'],
                           'motion_rmse_horizontal_m':rmse(motion_error[missing,:2]) if len(missing) else None,
                           'motion_rmse_vertical_m':rmse(motion_error[missing,2:]) if len(missing) else None})
        if not len(missing):continue
        full=motion_scores['fusion_v2_full'];g=motion_scores['fusion_v2_gps_only'];cv=motion_scores['constant_velocity_gnss'];h=motion_scores['held_gnss'];z=motion_scores['fusion_v2_full_zero_imu']
        delta=preds['fusion_v2_full_zero_imu']-preds['fusion_v2_full']
        differences.append({'scenario_id':sid,'episode_id':s['episode_id'],'duration_s':s['requested_duration_s'],'n_outage':len(missing),'seed':0,
            'full_motion_rmse_m':full,'gps_only_motion_rmse_m':g,'full_minus_gps_only_m':full-g,
            'full_minus_gps_only_percent_of_gps_only':100*(full-g)/g,'full_better_than_gps_only':full<g,
            'held_motion_rmse_m':h,'cv_motion_rmse_m':cv,'full_minus_cv_m':full-cv,'full_minus_held_m':full-h,
            'lstm_motion_rmse_m':motion_scores['lstm_v2_full'],'zero_imu_motion_rmse_m':z,
            'zero_minus_full_motion_rmse_m':z-full,'zero_minus_full_percent':100*(z-full)/full,
            'zero_full_prediction_rms_inside_m':rmse(delta[missing]),'zero_full_prediction_max_inside_m':float(np.linalg.norm(delta[missing],axis=1).max()),
            'zero_full_prediction_rms_outside_m':rmse(delta[mask]),
            'full_raw_rmse_m':float(keyed[sid,'fusion_v2_full']['rmse_3d_m']),
            'gps_only_raw_rmse_m':float(keyed[sid,'fusion_v2_gps_only']['rmse_3d_m']),
            'cv_raw_rmse_m':float(keyed[sid,'constant_velocity_gnss']['rmse_3d_m'])})
    duration=[]
    for seconds in [10,30,60]:
        subset=[s for s in plan['scenarios'] if s['requested_duration_s']==seconds]
        for m in METHODS:
            rows=[keyed[s['scenario_id'],m] for s in subset];weights=np.array([int(r['n_evaluated']) for r in rows])
            metrics=['motion_rmse_3d_m','rmse_3d_m','rmse_horizontal_m','rmse_vertical_m','final_unavailable_error_3d_m']
            values={k:float(np.mean([float(r[k]) for r in rows])) for k in metrics}
            scores=np.array([float(r['motion_rmse_3d_m']) for r in rows])
            duration.append({'duration_s':seconds,'method':m,'seed':0,'episodes':3,'n_outage_total':int(weights.sum()),
                'aggregation':'arithmetic mean of episode metric; pooled motion separately labeled',
                **{'mean_episode_'+k:v for k,v in values.items()},
                'pooled_motion_rmse_3d_m':float(np.sqrt(np.average(scores*scores,weights=weights)))})
    write_csv(OUT/'paired_differences.csv',differences);write_csv(OUT/'duration_summary.csv',duration);write_csv(OUT/'additional_motion_metrics.csv',extras)
    result={'status':'passed','evaluation_rows':60,'duplicate_keys':0,'missing_method_scenarios':0,'neural_training_rows':24,
        'seeds':[0],'checkpoint_policy':'fixed final epoch8; exactly 3 checkpoints',
        'max_recomputed_metric_abs_difference_m':max_metric_error,'max_native_GT_motion_target_abs_difference_m':max_target_error,
        'nan_inf_predictions_metrics':False,'same_scenario_masks_gt_timestamps_sources_across_methods':True,
        'source_snapshot_config_manifest_match_prepared':True,'primary_metric':'relative-motion RMSE3D inside outage',
        'full_better_than_gps_only_episodes':sum(x['full_better_than_gps_only'] for x in differences),
        'full_better_than_cv_episodes':sum(x['full_minus_cv_m']<0 for x in differences),
        'full_better_than_held_episodes':sum(x['full_minus_held_m']<0 for x in differences),
        'postprocess_only':True,'new_training_or_inference':False}
    (OUT/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
