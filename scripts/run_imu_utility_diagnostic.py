"""Local fixed-protocol physical diagnosis. No training, tuning, Kaggle or test."""
import csv,hashlib,json,sys,datetime
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from diagnostics.imu_utility import causal_gnss_velocity,estimate_initial,integrate,metrics,initial_rotation,mv
OUT=ROOT/'outputs/imu_utility_diagnostic';V2=ROOT/'outputs/v2_motion_pilot'

def js(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def csv_write(path,rows):
 with path.open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def corr(x,y):
 if len(x)<3 or np.std(x)<1e-12 or np.std(y)<1e-12:return None
 return float(np.corrcoef(x,y)[0,1])
def stats(x):return {'mean':np.mean(x,axis=0).tolist(),'std_ddof0':np.std(x,axis=0).tolist(),'median':np.median(x,axis=0).tolist(),'min':np.min(x,axis=0).tolist(),'max':np.max(x,axis=0).tolist()}
def smooth_past(t,x,width=1.):return np.array([x[(t>=t[i]-width)&(t<=t[i])].mean(0) for i in range(len(t))])

def load_inputs(manifest):
 for path,h in manifest['source_files'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h,path
 control=np.load(V2/'val_control/held_gnss.npz',allow_pickle=False)
 # Do not expose GT or motion targets to the initialization/prediction functions.
 t=control['timestamps'];held=control['pred'];st=control['source_timestamp'];si=control['source_index']
 raw=ROOT/'outputs/next_stage_diagnostics/raw';acc=pd.read_csv(raw/'RawAccel.csv');gyro=pd.read_csv(raw/'RawGyro.csv')
 for d in [acc,gyro]:d.columns=d.columns.str.strip()
 acc=acc.iloc[:21640].copy();gyro=gyro.loc[gyro.Timpstemp<=acc.Timpstemp.iloc[-1]].sort_values('Timpstemp',kind='stable').drop_duplicates('Timpstemp',keep='last')
 ta=acc.Timpstemp.values.astype(float)*1e-6;tg=gyro.Timpstemp.values.astype(float)*1e-6
 assert np.all(np.diff(ta)>0) and np.all(np.diff(tg)>0)
 j=np.searchsorted(tg,ta,side='right')-1;assert np.all(j>=0) and np.all(tg[j]<=ta)
 full_imu=np.column_stack([acc[['x','y','z']].values,gyro[['x','y','z']].values[j]])
 np.testing.assert_array_equal(ta[18935:21623],t)
 assert hashlib.sha256(t.astype('<f8').tobytes()).hexdigest()==manifest['common_timestamps_sha256']
 return t,full_imu[18935:21623],tg[j][18935:21623],held,st,si,acc,gyro

def signals(t,imu,gs,held,st,si,velocity,cfg,manifest):
 results=[];initials={}
 for scenario in manifest['scenarios']:
  eid=scenario['episode_id']
  if eid in initials:continue
  onset=scenario['onset_s'];initial=estimate_initial(t,imu,held,st,velocity,onset,cfg);initials[eid]=initial
  choose=(t>=onset-cfg['signal_history_s'])&(t<onset);ix=np.flatnonzero(choose);ts=t[ix];x=imu[ix];vel=velocity[ix]
  # Only pre-onset GNSS is examined; no unmasked GPS during/after outage.
  start=float(ts[0]+5.);di=estimate_initial(t,imu,held,st,velocity,start,cfg);query=t[(t>=start)&(t<onset)]
  qids=np.searchsorted(t,query);checks=[];main=None
  for label,factor in [('gyro_rad_s_positive',1.),('gyro_deg_s_positive',np.pi/180),('gyro_rad_s_negative',-1.)]:
   altered=imu.copy();altered[:,3:]*=factor
   trace=integrate(t,altered,gs,query,di,'dr_zero_bias',cfg['gravity_m_s2'])
   heading=np.unwrap(np.arctan2(trace['orientation'][:,1,0],trace['orientation'][:,0,0]))
   gv=velocity[qids];course=np.unwrap(np.arctan2(gv[:,1],gv[:,0]));valid=np.linalg.norm(gv[:,:2],axis=1)>=cfg['yaw_min_horizontal_speed_m_s']
   diff=np.degrees((heading-heading[0])-(course-course[0]))
   checks.append({'candidate':label,'gyro_integrated_body_heading_change_deg':float(np.degrees(heading[-1]-heading[0])),
       'gnss_course_change_deg':float(np.degrees(course[-1]-course[0])),
       'heading_change_difference_final_deg':float(diff[-1]),'heading_change_difference_rms_deg':float(np.sqrt(np.mean(diff[valid]**2))) if valid.any() else None,
       'course_speed_valid_fraction':float(valid.mean()),'selected_for_outage':label=='gyro_rad_s_positive'})
   if factor==1.:main=trace
  # GNSS acceleration proxy: difference of trailing 5s velocity estimates over >=1s.
  lag=np.searchsorted(t,query-1.,side='right')-1;accproxy=(velocity[qids]-velocity[lag])/(query-t[lag])[:,None]
  aw=smooth_past(query,main['acceleration_world']);gnss=smooth_past(query,accproxy)
  available=(query>=query[0]+1.)
  dot=float(np.sum(aw[available,:2]*gnss[available,:2]));den=float(np.linalg.norm(aw[available,:2])*np.linalg.norm(gnss[available,:2]))
  af=np.linalg.norm(x[:,:3],axis=1);ww=np.linalg.norm(x[:,3:],axis=1)
  one={'episode_id':eid,'onset_s':onset,'scope':'strictly pre-outage last30s','n':len(ix),'accel_axes_m_s2':stats(x[:,:3]),'gyro_axes_rad_s':stats(x[:,3:]),
   'accel_magnitude_m_s2':stats(af),'gyro_magnitude_rad_s':stats(ww),'horizontal_speed_m_s':stats(np.linalg.norm(vel[:,:2],axis=1)),
   'gnss_velocity_end_minus_start_m_s':(vel[-1]-vel[0]).tolist(),'gnss_course_change_30s_deg':float(np.degrees(np.unwrap(np.arctan2(vel[:,1],vel[:,0]))[-1]-np.unwrap(np.arctan2(vel[:,1],vel[:,0]))[0])),
   'gyro_convention_checks':checks,'estimated_world_acceleration_m_s2':stats(aw[available]),'causal_gnss_acceleration_proxy_m_s2':stats(gnss[available]),
   'pearson_estimated_accel_vs_gnss_proxy_xyz':[corr(aw[available,i],gnss[available,i]) for i in range(3)],
   'horizontal_vector_cosine_similarity':dot/den if den else None,
   'pearson_accel_magnitude_vs_gnss_acceleration_magnitude':corr(np.linalg.norm(imu[qids,:3],axis=1),np.linalg.norm(gnss,axis=1)),
   'gyroscope_age_s':stats(t[ix]-gs[ix]),'max_dt_s':float(np.diff(ts).max()),'dt_above_0_15_n':int(np.sum(np.diff(ts)>.15))}
  results.append(one)
  np.savez_compressed(OUT/f'e{eid}_preoutage_signal.npz',timestamps=query,estimated_world_acceleration_smoothed=aw,causal_gnss_acceleration_proxy=gnss,
    gnss_velocity=velocity[qids],orientation=main['orientation'],gyro=imu[qids,3:],accel=imu[qids,:3])
 js(OUT/'signal_diagnostics.json',results);js(OUT/'initial_states.json',initials)
 return initials

def main():
 if (OUT/'summary.csv').exists():raise RuntimeError('Refusing rerun/overwrite of completed physical diagnostic')
 cfg=json.loads((OUT/'config.json').read_text());manifest=json.loads((OUT/'manifest.json').read_text())
 assert hashlib.sha256((OUT/'config.json').read_bytes()).hexdigest()==manifest['config_sha256']
 t,imu,gs,held,st,si,raw_acc,raw_gyro=load_inputs(manifest);velocity=causal_gnss_velocity(t,held,st,si)
 # Raw scale and axes checks use only samples strictly before earliest onset, no GT.
 units={}
 for name,df in [('accel',raw_acc),('gyro',raw_gyro)]:
  d=df.loc[(df.Timpstemp*1e-6>=t[0])&(df.Timpstemp*1e-6<manifest['scenarios'][0]['onset_s'])]
  raw=d[['x_raw','y_raw','z_raw']].values*d.scaling.values[:,None];y=d[['x','y','z']].values
  design=np.column_stack([raw,np.ones(len(raw))]);coef=np.linalg.lstsq(design,y,rcond=None)[0];fit=np.einsum('ni,ij->nj',design,coef)
  units[name]={'scope':'validation before first onset only','n':len(d),'raw_to_reported_affine_diagnostic_matrix':coef.tolist(),
      'raw_affine_fit_axis_rmse':np.sqrt(np.mean((fit-y)**2,axis=0)).tolist(),'scaling_unique':np.unique(d.scaling.values).tolist(),
      'export_range_unique':np.unique(d.range_rad_s.values).tolist(),'error_count_first_last':[float(d.Error_count.iloc[0]),float(d.Error_count.iloc[-1])],
      'raw_counts_max_abs':np.max(np.abs(d[['x_raw','y_raw','z_raw']].values),axis=0).tolist(),'affine_fit_used_for_calibration':False}
 js(OUT/'units_checks.json',units)
 initials=signals(t,imu,gs,held,st,si,velocity,cfg,manifest)
 js(OUT/'status.json',{'status':'signals_complete_before_outage_integration','training':False,'kaggle':False,'parameters_frozen':True})
 all_rows=[];mechanisms=[];cvmax=0.;orthmax=0.
 for scenario in manifest['scenarios']:
  sid=scenario['scenario_id'];initial=initials[scenario['episode_id']];folder=OUT/sid;folder.mkdir()
  # Scoring references read only AFTER pre-outage initialization/signal diagnostics.
  reference=np.load(V2/sid/'constant_velocity_gnss.npz',allow_pickle=False);mask=reference['availability'];ids=np.flatnonzero(~mask);qt=t[ids]
  assert hashlib.sha256(mask.astype('uint8').tobytes()).hexdigest()==scenario['mask_uint8_sha256']
  p0=np.array(initial['p0']);v0=np.array(initial['v0']);cv=p0+(qt-initial['anchor_timestamp_s'])[:,None]*v0
  delta=float(np.max(np.abs(cv-reference['pred'][ids])));cvmax=max(cvmax,delta);assert delta<1e-10,delta
  for method in cfg['methods']:
   if method=='constant_velocity_gnss':trace={'pred':cv}
   else:
    trace=integrate(t,imu,gs,qt,initial,method,cfg['gravity_m_s2']);orthmax=max(orthmax,trace['max_rotation_orthogonality_error'])
   pred=trace['pred'];assert np.isfinite(pred).all()
   target=reference['motion_target'][ids];gt=reference['gt'][ids]
   row={'scenario_id':sid,'episode_id':scenario['episode_id'],'duration_s':scenario['requested_duration_s'],'method':method,'primary':method==cfg['primary_method'],'n_outage':len(ids),**metrics(pred,gt,target,p0)}
   all_rows.append(row)
   np.savez_compressed(folder/(method+'.npz'),timestamps=qt,gt=gt,motion_target=target,anchor=p0,initial_velocity=v0,
       full_common_availability_mask=mask,common_outage_indices=ids,initial_state_json=np.array(json.dumps(initial)),**trace)
   if method=='dr_zero_bias':
    correction=pred-cv;np.testing.assert_allclose(correction,trace['force_residual_displacement']+trace['orientation_gravity_displacement'],atol=1e-8)
    # Decomposition is mathematical bookkeeping, not separable causal errors.
    tilt=np.degrees(np.arccos(np.clip(-trace['orientation'][:,2,2],-1,1)))
    native=imu[ids];truev=np.diff(target,axis=0)/np.diff(qt)[:,None]
    mechanisms.append({'scenario_id':sid,'duration_s':scenario['requested_duration_s'],
      'initial_radial_accel_discrepancy_m_s2':initial['radial_gravity_discrepancy_m_s2'],
      'constant_radial_discrepancy_half_b_T2_m':.5*initial['radial_gravity_discrepancy_m_s2']*(qt[-1]-initial['anchor_timestamp_s'])**2,
      'primary_orientation_tilt_deg':stats(tilt),'final_orientation_det':float(np.linalg.det(trace['orientation'][-1])),
      'final_DR_minus_CV_xyz_m':correction[-1].tolist(),'final_orientation_gravity_displacement_xyz_m':trace['orientation_gravity_displacement'][-1].tolist(),
      'final_force_residual_displacement_xyz_m':trace['force_residual_displacement'][-1].tolist(),
      'accel_axes_during_outage_m_s2':stats(native[:,:3]),'gyro_axes_during_outage_rad_s':stats(native[:,3:]),
      'accel_magnitude_during_outage_m_s2':stats(np.linalg.norm(native[:,:3],axis=1)),
      'gyro_magnitude_during_outage_rad_s':stats(np.linalg.norm(native[:,3:],axis=1)),
      'dt_gaps_above_0_15':int(np.sum(np.diff(qt)>.15)),'max_dt_s':float(np.diff(qt).max()),
      'GT_interpolated_velocity_horizontal_speed_m_s':stats(np.linalg.norm(truev[:,:2],axis=1)),
      'GT_interpolated_velocity_xyz_std_m_s':truev.std(0).tolist(),
      'GT_displacement_from_anchor_xyz_m':target[-1].tolist(),'GT_used_only_for_posthoc_scoring_and_motion_description':True})
 csv_write(OUT/'summary.csv',all_rows);js(OUT/'error_mechanisms.json',mechanisms)
 pairs=[];duration=[]
 for s in manifest['scenarios']:
  rr=[x for x in all_rows if x['scenario_id']==s['scenario_id']];cvrow=next(x for x in rr if x['method']=='constant_velocity_gnss')
  for row in rr:
   if row['method']=='constant_velocity_gnss':continue
   pairs.append({'scenario_id':row['scenario_id'],'duration_s':row['duration_s'],'method':row['method'],'primary':row['primary'],
     'imu_motion_rmse_3d_m':row['motion_rmse_3d_m'],'cv_motion_rmse_3d_m':cvrow['motion_rmse_3d_m'],
     'imu_minus_cv_motion_rmse_3d_m':row['motion_rmse_3d_m']-cvrow['motion_rmse_3d_m'],
     'imu_minus_cv_percent':100*(row['motion_rmse_3d_m']/cvrow['motion_rmse_3d_m']-1),
     'imu_minus_cv_horizontal_m':row['motion_rmse_horizontal_m']-cvrow['motion_rmse_horizontal_m'],
     'imu_minus_cv_vertical_m':row['motion_rmse_vertical_m']-cvrow['motion_rmse_vertical_m'],
     'imu_minus_cv_final_motion_3d_m':row['motion_final_3d_m']-cvrow['motion_final_3d_m']})
 for sec in [10,30,60]:
  for method in cfg['methods']:
   rr=[x for x in all_rows if x['duration_s']==sec and x['method']==method];pr=[x for x in pairs if x['duration_s']==sec and x['method']==method]
   duration.append({'duration_s':sec,'method':method,'onsets':3,'mean_episode_motion_rmse_3d_m':float(np.mean([x['motion_rmse_3d_m'] for x in rr])),
     'mean_episode_motion_horizontal_m':float(np.mean([x['motion_rmse_horizontal_m'] for x in rr])),
     'mean_episode_motion_vertical_m':float(np.mean([x['motion_rmse_vertical_m'] for x in rr])),
     'mean_episode_final_motion_3d_m':float(np.mean([x['motion_final_3d_m'] for x in rr])),
     'better_than_cv_count':sum(x['imu_minus_cv_motion_rmse_3d_m']<0 for x in pr) if pr else None})
 csv_write(OUT/'paired_differences.csv',pairs);csv_write(OUT/'duration_summary.csv',duration)
 js(OUT/'status.json',{'status':'complete','protocol_id':cfg['protocol_id'],'primary_method':cfg['primary_method'],
      'scenarios':9,'methods':6,'evaluation_rows':len(all_rows),'primary_wins_vs_cv':sum(x['imu_minus_cv_motion_rmse_3d_m']<0 for x in pairs if x['primary']),
      'cv_reconstructed_max_abs_difference_m':cvmax,'max_rotation_orthogonality_error':orthmax,
      'neural_training':False,'new_seeds':False,'test_used':False,'kaggle_job':False,'parameters_selected_by_validation':False})
 print((OUT/'status.json').read_text());print('Duration means:',[(x['duration_s'],x['method'],round(x['mean_episode_motion_rmse_3d_m'],4)) for x in duration])
if __name__=='__main__':main()
