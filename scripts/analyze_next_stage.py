"""Read existing R7 arrays/predictions; bounded fixed-checkpoint probes on CPU.

No network, optimization, dataset regeneration, test evaluation or Kaggle job.
Writes a separate next_stage_diagnostics/results directory, refusing overwrite.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from diagnostics.next_stage import (rmse_xyz, fit_frame, apply_frame, utm32_wgs84,
    correction_stats, pearson_features, future_gnss_exposure, permute_imu_windows)
from diagnostics.run_clean_outages import infer_windows

BASE=ROOT/'outputs/next_stage_diagnostics'
R7=BASE/'r7_inputs'
OLD=ROOT/'outputs/diagnostic_clean_outages'
OUT=BASE/'results'


def js(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def csvout(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def describe(a):
    return {'min':float(np.min(a)), 'median':float(np.median(a)), 'p95':float(np.quantile(a,.95)),
            'max':float(np.max(a)), 'mean':float(np.mean(a))}


def raw_frame():
    a={n:np.load(R7/(n+'.npy'),allow_pickle=False) for n in ['t_imu','imu','t_gps','gps_pos','t_gt','gt_pos']}
    meta=json.loads((R7/'meta.json').read_text())
    t=a['t_imu'][:21623]
    idx=np.searchsorted(a['t_gps'],t,side='right')-1
    gps=a['gps_pos'][idx].astype(float)
    gt=np.column_stack([np.interp(t,a['t_gt'],a['gt_pos'][:,j]) for j in range(3)]).astype(np.float32).astype(float)
    tr=np.arange(18935);va=np.arange(18935,21623)
    fits=[];metrics=[]
    for fit_split,fit_idx in [('train',tr),('validation_oracle',va)]:
        for kind in ['raw','translation','se2_vertical','se3','similarity_diagnostic']:
            f=fit_frame(gps[fit_idx],gt[fit_idx],kind)
            fits.append({'fit_split':fit_split,**f})
            for name,ids in [('train',tr),('validation',va)]:
                metrics.append({'fit_split':fit_split,'evaluation_split':name,'kind':kind,
                                'n':len(ids),**rmse_xyz(apply_frame(gps[ids],f),gt[ids]),
                                'translation_x_m':f['translation_m'][0],'translation_y_m':f['translation_m'][1],
                                'translation_z_m':f['translation_m'][2],'yaw_deg':f['yaw_deg'],
                                'rotation_angle_deg':f['rotation_angle_deg'],'scale':f['scale']})
    # Physical map projection comparison: no fit, same absolute origin for both.
    gps64=a['gps_pos'].astype(np.float64)
    lat=meta['lat0']+np.rad2deg(gps64[:,1]/6378137.)
    lon=meta['lon0']+np.rad2deg(gps64[:,0]/(6378137.*np.cos(np.deg2rad(meta['lat0']))))
    physical_xy=utm32_wgs84(lat,lon)-[meta['utm_x0'],meta['utm_y0']]
    physical=np.column_stack([physical_xy,a['gps_pos'][:,2]])[idx]
    for name,ids in [('train',tr),('validation',va)]:
        metrics.append({'fit_split':'none','evaluation_split':name,'kind':'common_utm_origin_no_fit',
                        'n':len(ids),**rmse_xyz(physical[ids],gt[ids])})
    physical_constant=physical-gps
    physical_summary={'meta':meta,'initial_gnss_processed_xyz':a['gps_pos'][0].tolist(),
        'initial_gt_processed_xyz':a['gt_pos'][0].tolist(),
        'projected_gnss_origin_utm_xy':utm32_wgs84([meta['lat0']],[meta['lon0']])[0].tolist(),
        'physical_minus_R7_validation_mean_xyz_m':physical_constant[va].mean(0).tolist(),
        'physical_minus_R7_validation_std_xyz_m':physical_constant[va].std(0).tolist(),
        'approx_grid_convergence_deg':float(np.degrees(np.arctan(np.tan(np.deg2rad(meta['lon0']-9))*np.sin(np.deg2rad(meta['lat0']))))),
        'train_desired_correction_mean_xyz_m':(gt[tr]-gps[tr]).mean(0).tolist(),
        'validation_desired_correction_mean_xyz_m':(gt[va]-gps[va]).mean(0).tolist(),
        'train_position_singular_values':np.linalg.svd(gps[tr]-gps[tr].mean(0),compute_uv=False).tolist(),
        'validation_position_singular_values':np.linalg.svd(gps[va]-gps[va].mean(0),compute_uv=False).tolist()}
    chunks=[]
    for split,ids in [('train',tr),('validation',va)]:
        for start in np.arange(t[ids[0]],t[ids[-1]],60):
            ix=ids[(t[ids]>=start)&(t[ids]<start+60)]
            if len(ix):chunks.append({'split':split,'start_s':float(start),'n':len(ix),
                'mean_desired_correction_xyz_m':(gt[ix]-gps[ix]).mean(0).tolist(),
                'physical_frame_mean_error_xyz_m':(gt[ix]-physical[ix]).mean(0).tolist(),
                **rmse_xyz(gps[ix],gt[ix])})
    csvout('alignment_metrics.csv',metrics);js('alignment_transforms.json',fits)
    js('frame_summary.json',physical_summary);js('bias_60s_bins.json',chunks)
    np.savez_compressed(OUT/'train_validation_frame_series.npz',timestamps=t,gps_r7=gps,
                        gps_common_utm=physical,gt=gt,split_boundary=18935)
    return a,meta,physical_summary


def timestamps(a,meta):
    raw={}
    for n in ['RawAccel','RawGyro','OnboardGPS','GroundTruthAGL']:
        df=pd.read_csv(BASE/'raw'/(n+'.csv'));df.columns=df.columns.str.strip();raw[n]=df
    acc,gyro,gps,gtruth=[raw[n] for n in ['RawAccel','RawGyro','OnboardGPS','GroundTruthAGL']]
    ta=acc.Timpstemp.to_numpy(float)*1e-6;tg=gyro.Timpstemp.to_numpy(float)*1e-6
    j=np.searchsorted(tg,ta).clip(max=len(tg)-1)
    rebuilt=np.column_stack([acc[['x','y','z']].values,gyro[['x','y','z']].values[j]]).astype(np.float32)
    gps_t=gps.Timpstemp.to_numpy(float)*1e-6
    lat,lon,alt=[gps[k].to_numpy(float) for k in ['lat','lon','alt']]
    rebuilt_gps=np.column_stack([np.deg2rad(lon-meta['lon0'])*6378137*np.cos(np.deg2rad(meta['lat0'])),
                                np.deg2rad(lat-meta['lat0'])*6378137,alt-meta['alt0']]).astype(np.float32)
    assert np.array_equal(ta,a['t_imu']) and np.array_equal(rebuilt,a['imu'])
    assert np.array_equal(gps_t,a['t_gps']) and np.array_equal(rebuilt_gps,a['gps_pos'])
    # Recover native GT timestamps and verify position export; never fit on test.
    gtruth['t']=gtruth.imgid.map(gps.set_index('imgid').Timpstemp)*1e-6
    valid=gtruth.dropna(subset=['t']).sort_values('t')
    gp=valid[['x_gt','y_gt','z_gt']].values
    assert np.array_equal((gp-gp[0]).astype(np.float32),a['gt_pos'])
    assert np.array_equal(valid.t.values,a['t_gt'])
    mapped=gps.set_index('imgid').loc[valid.imgid]
    reference=np.column_stack([utm32_wgs84(mapped.lat.values,mapped.lon.values),mapped.alt.values])
    keep=valid.t.values<a['t_imu'][21640]
    xyref=valid[['x_gps','y_gps','z_gps']].values
    projection_check=rmse_xyz(reference[keep],xyref[keep])
    lag=tg[j]-ta
    back=np.searchsorted(tg,ta,side='right')-1
    chronology=[]
    for name,sl in [('train',slice(0,18935)),('validation',slice(18935,21640))]:
        x=lag[sl];dt=np.diff(ta[sl])
        chronology.append({'split':name,'n':len(x),'next_gyro_future_n':int(np.sum(x>0)),
            'next_gyro_future_fraction':float(np.mean(x>0)),'next_gyro_lead_s':describe(x),
            'past_gyro_age_s':describe(ta[sl]-tg[back[sl]]),'dt_s':describe(dt),
            'dt_over_0_15_n':int(np.sum(dt>.15)),'elapsed_minus_fixed_dt_s':float(np.sum(dt)-.1*len(dt))})
    reversals=np.flatnonzero(np.diff(gps_t)<0)
    xyz=gps[['lat','lon','alt']].values
    change=np.r_[True,np.any(np.diff(xyz,axis=0)!=0,axis=1)]
    fixstats=[]
    for name,start,stop in [('train',ta[0],ta[18935]),('validation',ta[18935],ta[21640])]:
        ids=np.flatnonzero((gps_t>=start)&(gps_t<stop))
        fixes=ids[change[ids]]
        fixstats.append({'split':name,'rows':len(ids),'coordinate_change_events':len(fixes),
            'row_rate_hz':float((len(ids)-1)/(gps_t[ids[-1]]-gps_t[ids[0]])),
            'coordinate_change_rate_hz':float((len(fixes)-1)/(gps_t[fixes[-1]]-gps_t[fixes[0]])),
            'coordinate_change_dt_s':describe(np.diff(gps_t[fixes])),
            'fix_type_counts':{str(k):int(v) for k,v in gps.iloc[ids].fix_type.value_counts().items()},
            'num_sat_min':float(gps.iloc[ids].num_sat.min()),'num_sat_max':float(gps.iloc[ids].num_sat.max())})
    sorted_order=np.argsort(gps_t,kind='stable');sorted_t=gps_t[sorted_order]
    compare_t=ta[:21640];old=np.searchsorted(gps_t,compare_t,side='right')-1
    new=sorted_order[np.searchsorted(sorted_t,compare_t,side='right')-1]
    sync={'raw_reproduces_all_R7_processed_arrays':True,'gyro':chronology,'gps':fixstats,
        'reversals':[{'index':int(k),'time_s':float(gps_t[k]),'next_time_s':float(gps_t[k+1]),
                     'dt_s':float(gps_t[k+1]-gps_t[k])} for k in reversals],
        'sorted_vs_R7_selected_GPS_row_difference_count_train_val':int(np.sum(old!=new)),
        'sorted_vs_R7_selected_GPS_position_difference_count_train_val':int(np.sum(np.any(rebuilt_gps[old]!=rebuilt_gps[new],axis=1))),
        'GPS_UTM_export_projection_check_train_validation':projection_check,
        'GPS_UTM_export_projection_max_abs_xyz_m':np.abs(reference[keep]-xyref[keep]).max(0).tolist(),
        'GNSS_initial_minus_GT_initial_export_m':(xyref[0]-gp[0]).tolist(),
        'physical_fix_timestamp_available':False,
        'frame_sensor_extrinsic_camera_body_norm_m':float(np.linalg.norm([.07566,.02968,-.03227]))}
    js('synchronization.json',sync)
    return sync


def correction_and_windows():
    manifest=json.loads((OLD/'protocol_manifest.json').read_text())
    stats=[];correlations=[];window_rows=[];pair_stats=[];surrogates=[]
    for s in manifest['scenarios']:
        sid=s['scenario_id'];d=np.load(OLD/sid/'inputs.npz',allow_pickle=False)
        t=d['timestamps_s'];mask=d['availability_mask'];held=d['held_gnss'].astype(float);imu=d['imu']
        age=np.zeros(len(t));exposed=np.zeros(len(t),bool);fresh=exposed.copy()
        if s['requested_duration_s']:
            age[~mask]=t[~mask]-s['onset_s']
            exposed,fresh=future_gnss_exposure(t,mask,d['held_source_timestamp_s'],s['requested_cutoff_s'])
        scopes={'all':np.ones(len(t),bool),'inside_outage':~mask,'outside_outage':mask}
        features={'time_s':t,'availability':mask.astype(float),'outage_age_s':age,'fix_age_s':d['fix_age_s']}
        features.update({f'held_{k}':held[:,i] for i,k in enumerate('xyz')})
        features.update({f'imu_{i}':imu[:,i] for i in range(6)})
        predictions={}
        for method in ['fusion_full','fusion_gps_only','fusion_full_zero_imu','held_gnss','constant_velocity_gnss','fusion_lstm']:
            p=np.load(OLD/sid/method/'prediction.npz',allow_pickle=False)['pred'].astype(float)
            predictions[method]=p
            if method.startswith('fusion_') and method!='fusion_lstm':
                corr=p-held
                for scope,chosen in scopes.items():
                    if chosen.any():
                        stats.append({'scenario_id':sid,'method':method,'scope':scope,**correction_stats(corr,t,chosen)})
                        correlations.append({'scenario_id':sid,'method':method,'scope':scope,
                                             'pearson_correction_xyz_by_feature':pearson_features(corr,features,chosen)})
                if sid=='val_control':
                    # Descriptive OLS, first 7 val windows fit, last 7 held out.
                    # These fits explain the NN output; not new navigation models.
                    position=held;clock=t[:,None]; inertial=imu.astype(float)
                    designs={'constant':np.zeros((len(t),0)),'position_affine':position,
                             'time_linear':clock,'imu_linear':inertial,
                             'position_time':np.column_stack([position,clock]),
                             'position_time_imu':np.column_stack([position,clock,inertial])}
                    pos=(position-position[:1344].mean(0))/position[:1344].std(0)
                    designs['position_quadratic']=np.column_stack([pos,pos**2,pos[:,0]*pos[:,1],pos[:,0]*pos[:,2],pos[:,1]*pos[:,2]])
                    for name,x in designs.items():
                        mean=x[:1344].mean(0);scale=x[:1344].std(0);scale[scale<1e-12]=1
                        design=np.column_stack([np.ones(len(t)),(x-mean)/scale])
                        coef=np.linalg.lstsq(design[:1344],corr[:1344],rcond=None)[0]
                        estimate=np.einsum('ni,ij->nj',design,coef)
                        for scope,ids in [('fit_val_first7windows',slice(0,1344)),('heldout_val_last7windows',slice(1344,None))]:
                            y=corr[ids];e=estimate[ids]-y;den=np.sum((y-y.mean(0))**2)
                            surrogates.append({'method':method,'surrogate':name,'scope':scope,
                                'correction_rmse_3d_m':float(np.sqrt(np.mean(np.sum(e*e,axis=1)))),
                                'centered_r2':float(1-np.sum(e*e)/den) if den>0 else None})
            if s['requested_duration_s']:
                for label,choose in [('all_outage',~mask),('no_postoutage_gnss',~mask&~exposed),('postoutage_gnss_visible',exposed)]:
                    window_rows.append({'scenario_id':sid,'method':method,'subset':label,'n':int(choose.sum()),
                        'outage_n':int((~mask).sum()),'potential_post_gnss_fraction':float(exposed.sum()/(~mask).sum()),
                        'fresh_post_gnss_fraction':float(fresh.sum()/(~mask).sum()),**rmse_xyz(p[choose],d['gt'][choose])})
        for other in ['fusion_gps_only','fusion_full_zero_imu']:
            diff=predictions['fusion_full']-predictions[other]
            for scope,choose in scopes.items():
                if choose.any():
                    norm=np.linalg.norm(diff[choose],axis=1)
                    normcorr=np.linalg.norm(predictions['fusion_full'][choose]-held[choose],axis=1)
                    pair_stats.append({'scenario_id':sid,'comparison':'full_minus_'+other,'scope':scope,'n':int(choose.sum()),
                        'rms_difference_m':float(np.sqrt(np.mean(norm*norm))),'mean_norm_m':float(norm.mean()),
                        'median_norm_m':float(np.median(norm)),'max_norm_m':float(norm.max()),
                        'ratio_to_full_correction_rms':float(np.sqrt(np.mean(norm*norm)/np.mean(normcorr*normcorr)))})
        np.savez_compressed(OUT/(sid+'_series.npz'),timestamps=t,availability=mask,outage_age_s=age,
            held=held,gt=d['gt'],imu=imu,potential_post_gnss=exposed,fresh_post_gnss=fresh,
            full_minus_gps_only=predictions['fusion_full']-predictions['fusion_gps_only'],
            full_minus_zero_imu=predictions['fusion_full']-predictions['fusion_full_zero_imu'],
            **{k+'_correction':v-held for k,v in predictions.items() if k.startswith('fusion_')})
    js('correction_statistics.json',stats);js('correlations.json',correlations)
    csvout('surrogate_fits.csv',surrogates);csvout('saved_prediction_differences.csv',pair_stats)
    csvout('window_exposure_metrics.csv',window_rows)


def sensitivity():
    import torch
    from models.fusion_transformer import FusionTransformer
    torch.set_num_threads(1);torch.manual_seed(0)
    checkpoint=R7/'fusion_transformer_full_s0_outage0.0.pt'
    model=FusionTransformer().eval()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
    rows=[]
    for scenario in json.loads((OLD/'protocol_manifest.json').read_text())['scenarios']:
        sid=scenario['scenario_id'];data=np.load(OLD/sid/'inputs.npz',allow_pickle=False)
        imu=data['imu'];held=data['held_gnss'];mask=data['availability_mask'];t=data['timestamps_s']
        p={};permutations={}
        for kind in ['full_cpu','zero','shuffle','reverse']:
            x=imu
            if kind=='zero':x=np.zeros_like(imu)
            if kind in ['shuffle','reverse']:
                x,order=permute_imu_windows(imu,192,kind,1701);permutations[kind+'_permutation']=order
            p[kind]=infer_windows(model,x,held,mask,192,1,'cpu')
        saved=np.load(OLD/sid/'fusion_full/prediction.npz',allow_pickle=False)['pred']
        p['saved_gpu_full']=saved
        for kind in ['zero','shuffle','reverse','saved_gpu_full']:
            for scope,choose in [('all',np.ones(len(t),bool)),('inside_outage',~mask),('outside_outage',mask)]:
                if choose.any():
                    diff=p[kind][choose].astype(float)-p['full_cpu'][choose].astype(float)
                    norm=np.linalg.norm(diff,axis=1)
                    corr=p['full_cpu'][choose].astype(float)-held[choose]
                    row={'scenario_id':sid,'perturbation':kind,'scope':scope,'n':int(choose.sum()),
                         'rms_prediction_change_m':float(np.sqrt(np.mean(norm**2))),
                         'max_prediction_change_m':float(norm.max()),
                         'ratio_to_full_correction_rms':float(np.sqrt(np.mean(norm**2)/np.mean(np.sum(corr*corr,axis=1)))),
                         **rmse_xyz(p[kind][choose],data['gt'][choose])}
                    row['delta_rmse_3d_vs_full_cpu_m']=row['rmse_3d_m']-rmse_xyz(p['full_cpu'][choose],data['gt'][choose])['rmse_3d_m']
                    rows.append(row)
        np.savez_compressed(OUT/(sid+'_sensitivity.npz'),timestamps=t,availability=mask,**p,**permutations)
        print('Fixed-checkpoint CPU probes complete:',sid,flush=True)
    csvout('imu_sensitivity.csv',rows)
    js('sensitivity_manifest.json',{'device':'cpu','torch':torch.__version__,'numpy':np.__version__,
        'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'shuffle_seed':1701,
        'permutation':'same six-channel row permutation within each window, identical across scenarios',
        'local_full_reference':True,'training_performed':False,'GT_modified_or_used_in_inputs':False})


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=False)
    arrays,meta,_=raw_frame();timestamps(arrays,meta);correction_and_windows();sensitivity()
    js('analysis_status.json',{'status':'complete','training_performed':False,'new_kaggle_job':False,
                              'test_used_for_fitting_or_evaluation':False})
