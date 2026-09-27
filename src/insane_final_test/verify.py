"""Independent NumPy arithmetic on saved arrays; no model import or inference."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

def rms(e):return float(np.sqrt(np.mean(np.sum(e*e,axis=1))))

def verify(out,raw=None):
    out=Path(out);d=pd.read_csv(out/'summary.csv');pairs=pd.read_csv(out/'paired_differences.csv')
    manifest=json.loads((out/'test_manifest.json').read_text());specs=manifest['scenarios'];n_out=sum(s['duration_s']>0 for s in specs)
    assert len(d)==len(specs)*9 and len(pairs)==n_out*21
    assert not d.duplicated(['scenario','method','seed']).any() and (d.split=='test').all()
    assert not pairs.duplicated(['scenario','method','seed','reference']).any() and (pairs.split=='test').all()
    lock=json.loads((out/'FINAL_EVALUATION_LOCK.json').read_text());counts=0;maxdiff=0.;maxintegration=0.;native={};gps={}
    if raw is not None:
        for seq,info in manifest['inventory'].items():
            gt=pd.read_csv(Path(raw)/seq/'ground_truth_8hz.csv');gt.columns=gt.columns.str.strip()
            native[seq]=(gt.t.to_numpy()-info['epoch_unix_s'],gt[['p_x','p_y','p_z']].to_numpy()-info['origin_enu_m'])
            gn=pd.read_csv(Path(raw)/seq/'px4_gps.csv');gn.columns=gn.columns.str.strip()
            gps[seq]=(gn.t.to_numpy()-info['epoch_unix_s'],gn[['p_x','p_y','p_z']].to_numpy()-info['origin_enu_m'])
    metric_names=['relative_motion_rmse3d_m','absolute_rmse3d_m','absolute_rmse_h_m','absolute_rmse_v_m','final_unavailable_error_m',
                  'final_relative_motion_error_m','recovery_first_5s_rmse3d_m','first_recovered_fix_error_m']
    for spec in specs:
        common=None;files=sorted((out/'predictions'/spec['id']).glob('*.npz'));assert len(files)==9
        for p in files:
            with np.load(p,allow_pickle=False) as z:a={k:z[k] for k in z.files}
            assert all(np.isfinite(v).all() for v in a.values())
            for k in ['t','gt','outage','gt_native_index','anchor_gt','source_t','source_index','availability','token_t','all_token_t','all_availability']:
                if common is not None:np.testing.assert_array_equal(a[k],common[k])
            common=a
            if raw is not None:
                nt,np_=native[spec['flight']];np.testing.assert_array_equal(a['t'],nt[a['gt_native_index']]);np.testing.assert_array_equal(a['gt'],np_[a['gt_native_index']])
                st,sp=gps[spec['flight']];np.testing.assert_array_equal(a['source_t'],st[a['source_index']]);np.testing.assert_array_equal(a['held'],sp[a['source_index']])
            name=p.stem
            if '_seed' in name:method,seed=name.rsplit('_seed',1);q=d[(d.scenario==spec['id'])&(d.method==method)&(d.seed==int(seed))]
            else:method=name;q=d[(d.scenario==spec['id'])&(d.method==method)&d.seed.isna()]
            assert len(q)==1;row=q.iloc[0];mask=a['outage'];chosen=mask if mask.any() else np.ones(len(mask),bool)
            assert row.n_native_gt==chosen.sum()
            np.testing.assert_allclose(a['pred']-a['held'],a['motion'],atol=1e-7,rtol=0)
            np.testing.assert_array_equal(a['target_motion'],a['gt']-a['anchor_gt'])
            assert np.all(a['source_t']<=a['token_t']+1e-8) and np.all(a['token_t']<=a['t']+1e-8) and np.all(a['imu_source_t']<=a['token_t']+1e-8)
            err=a['pred']-a['gt'];e=err[chosen];metrics={k:None for k in metric_names}
            metrics.update(absolute_rmse3d_m=rms(e),absolute_rmse_h_m=rms(e[:,:2]),absolute_rmse_v_m=rms(e[:,2:3]))
            if spec['duration_s']:
                start=spec['onset_s'];end=start+spec['duration_s'];np.testing.assert_array_equal(mask,(a['t']>=start)&(a['t']<end))
                assert not ((a['source_t']>=start)&(a['source_t']<end)).any()
                me=(a['pred']-a['held'])-(a['gt']-a['anchor_gt'])
                metrics.update(relative_motion_rmse3d_m=rms(me[mask]),final_unavailable_error_m=float(np.linalg.norm(err[mask][-1])),final_relative_motion_error_m=float(np.linalg.norm(me[mask][-1])))
                recovery=(a['t']>=end)&(a['t']<end+5)
                if recovery.any():metrics['recovery_first_5s_rmse3d_m']=rms(err[recovery])
                returned=(a['t']>=end)&(a['source_t']>=end)
                if returned.any():metrics['first_recovered_fix_error_m']=float(np.linalg.norm(err[np.flatnonzero(returned)[0]]))
            for key,value in metrics.items():
                if value is None:assert pd.isna(row[key]),(spec['id'],method,key)
                else:
                    delta=abs(value-float(row[key]));maxdiff=max(maxdiff,delta);assert delta<1e-6,(spec['id'],method,key,delta)
            if 'g_token' in a:
                g=a['g_token'];assert np.all((g>=0)&(g<=1))
                velocity=g[:,None]*(a['velocity_prior']+a['residual_velocity_token']);np.testing.assert_array_equal(velocity,a['velocity_token'])
                total=np.cumsum(velocity.astype(float)*a['integration_dt'][:,None],axis=0)
                index=np.maximum.accumulate(np.where(a['all_availability'],np.arange(len(total)),0));motion=np.where(a['all_availability'][:,None],0.,total-total[index])
                difference=float(np.max(np.abs(motion-a['motion_token'])));maxintegration=max(maxintegration,difference);assert difference<.002
                ids=a['token_index'];native_m=a['motion_token'][ids]+velocity[ids]*(a['t']-a['all_token_t'][ids])[:,None]*(~a['availability'])[:,None]
                np.testing.assert_allclose(native_m,a['motion'],rtol=1e-6,atol=2e-5)
            else:
                age=a['t']-a['source_t'];prior=a['velocity_prior'][a['token_index']]
                coefficient=0*age if method=='held_gnss' else age if method=='constant_velocity_gnss' else -5*np.expm1(-age/5)
                np.testing.assert_allclose(a['motion'],prior*coefficient[:,None]*(~a['availability'])[:,None],atol=1e-8,rtol=0)
            counts+=1
    paired_diff=0.
    for r in pairs.itertuples():
        a=d[(d.scenario==r.scenario)&(d.method==r.method)&(d.seed==r.seed)].iloc[0]
        q=d[(d.scenario==r.scenario)&(d.method==r.reference)]
        if r.reference.startswith('adaptive_'):q=q[q.seed==r.seed]
        assert len(q)==1;b=q.iloc[0];v=a.relative_motion_rmse3d_m;w=b.relative_motion_rmse3d_m
        delta=abs((v-w)-r.difference_m);paired_diff=max(paired_diff,delta);assert delta<1e-8
        assert abs(r.rmse_m-v)<1e-8 and abs(r.reference_rmse_m-w)<1e-8 and bool(r.better)==bool(v<w)
        if w:assert abs(r.difference_percent-100*(v-w)/w)<1e-6
    assert json.loads((out/'weights_before.json').read_text())==json.loads((out/'weights_after.json').read_text())
    for name,h in lock['checkpoint_sha256'].items():assert hashlib.sha256((out/'checkpoints'/name).read_bytes()).hexdigest()==h
    return {'status':'pass','prediction_files_checked':counts,'summary_rows':len(d),'paired_rows':len(pairs),'duplicate_rows':0,
       'max_metric_absolute_discrepancy_m':maxdiff,'max_paired_difference_discrepancy_m':paired_diff,'max_gate_integral_discrepancy_m':maxintegration,
       'all_points_and_masks_equal':True,'native_gt_and_gnss_provenance_verified_against_raw':raw is not None,
       'all_predictions_finite':True,'weights_unchanged':True,'model_inference_performed_by_verifier':False,
       'control_outage_metrics_are_na':True,'seed_count':3,'independent_test_flights':2}
