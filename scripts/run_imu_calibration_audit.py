"""Train-only forensic sensor consistency. No navigation, training or GT inputs."""
from pathlib import Path
import hashlib, itertools, json, os
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp
from scipy.ndimage import uniform_filter1d

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/imu_calibration_audit'
RAW = ROOT / 'outputs/next_stage_diagnostics/raw'
META = OUT / 'metadata'
G = 9.80665


def dump(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def stats(x):
    x = np.asarray(x)
    return {'n': len(x), 'mean': float(x.mean()), 'std': float(x.std()),
            'min': float(x.min()), 'p05': float(np.quantile(x,.05)),
            'median': float(np.median(x)), 'p95': float(np.quantile(x,.95)), 'max': float(x.max())}


def norm(x):
    return np.linalg.norm(x, axis=-1)


def angle(a,b):
    return np.degrees(np.arccos(np.clip(np.sum(a*b,axis=-1)/norm(a)/norm(b),-1,1)))


def corr(a,b):
    return [float(np.corrcoef(a[:,j],b[:,j])[0,1]) for j in range(3)]


def rms(x):
    return float(np.sqrt(np.mean(np.sum(x*x,axis=-1))))


def interp(t, st, x):
    return np.column_stack([np.interp(t,st,x[:,j]) for j in range(x.shape[1])])


def mappings():
    result=[]
    for p in itertools.permutations(range(3)):
        for s in itertools.product((-1,1),repeat=3):
            m=np.eye(3)[list(p)]*np.array(s)[:,None]
            if np.linalg.det(m) > .5:
                name=','.join(('+' if v>0 else '-')+'xyz'[j] for j,v in zip(p,s))
                result.append((name,m))
    assert len(result)==24 and len({tuple(m.ravel()) for _,m in result})==24
    for _,m in result:
        np.testing.assert_allclose(m@m.T,np.eye(3),atol=1e-15)
        assert abs(np.linalg.det(m)-1)<1e-15
    return result


def rotate(m,x):
    return np.einsum('ij,nj->ni',m,x)


def read_train(path, cutoff):
    df=pd.read_csv(path)
    df.columns=df.columns.str.strip()
    df=df.loc[:,(~df.columns.str.startswith('Unnamed')) & (df.columns!='')]
    df=df[df.Timpstemp*1e-6 < cutoff].copy()
    t=df.Timpstemp.to_numpy()*1e-6
    info={'rows_in_file_order_train':len(df),'duplicate_timestamps':int(pd.Series(t).duplicated().sum()),
          'timestamp_reversals':int(np.sum(np.diff(t)<0)),
          'nonfinite_numeric':int((~np.isfinite(df.to_numpy(dtype=float))).sum()),
          'columns':list(df.columns)}
    df=df.sort_values('Timpstemp',kind='stable').drop_duplicates('Timpstemp',keep='last')
    t=df.Timpstemp.to_numpy()*1e-6
    info.update({'retained':len(t),'time_s':[float(t[0]),float(t[-1])],
                 'dt_s':stats(np.diff(t)), 'gaps_over_1_5_median':int(np.sum(np.diff(t)>1.5*np.median(np.diff(t))))})
    return df,t,info


def main():
    OUT.mkdir(exist_ok=True)
    # Sole use of old processed data: established V2 split boundary, not signal values.
    common_t=np.load(ROOT/'outputs/next_stage_diagnostics/r7_inputs/t_imu.npy',allow_pickle=False)
    cutoff=float(common_t[18935])
    assert abs(cutoff-1905.752757)<1e-8
    fit_end=7.090906+.7*(cutoff-7.090906)
    config={'purpose':'physical_sensor_contract_only','split':'V2 train only',
      'train_end_exclusive_s':cutoff, 'train_fit_end_s':fit_end,
      'train_check':'last 30% of train duration, NOT validation/test',
      'axis_candidates':24,'lags_raw_export_s':np.arange(-.5,.501,.1).round(4).tolist(),
      'lags_raw_onboard_s':np.arange(-.2,.201,.02).round(4).tolist(),
      'gravity_smoothing_s':1.0,'orientation_windows_s':30.0,'gravity_g_m_s2':G,
      'quasi_static_proxy':'1s smooth: |norm(f)-g|<1; norm(gyro)<0.1; RMS local accel std<0.5',
      'diagnostic_interpolation':'offline linear, quaternion Slerp, centered smoothing; never navigation inputs',
      'gyro_bias_sensitivity':'mean exported raw gyro on first 70% of train, not physical calibration',
      'outage_navigation_evaluation':False,'training':False,'kaggle_job':False,
      'continuous_mount_fit':False,'selected_calibration':None}
    config['documented_mount'] = 'Fixed Figure 2 PX4-to-Onboard body transform; not an angle search or fit'
    dump('config.json',config)
    streams={}; inventory={}; units={}; source_manifest={}
    for k,fn in [('accel',RAW/'RawAccel.csv'),('gyro',RAW/'RawGyro.csv'),('pose',META/'OnboardPose.csv')]:
        df,t,info=read_train(fn,cutoff);streams[k]=(df,t);inventory[k]=info
        source_manifest[str(fn.relative_to(ROOT))]={'sha256':hashlib.sha256(fn.read_bytes()).hexdigest(),'bytes':fn.stat().st_size}
        if k!='pose':
            x=df[['x','y','z']].to_numpy();raw=df[['x_raw','y_raw','z_raw']].to_numpy()*df.scaling.to_numpy()[:,None]
            info['consecutive_equal_export_xyz']=int(np.sum(np.all(np.diff(x,axis=0)==0,axis=1)))
            units[k]={'scaling_unique':df.scaling.unique().tolist(),'range_unique':df.range_rad_s.unique().tolist(),
                'error_count_unique':df.Error_count.unique().tolist(), 'temperature_C':stats(df.temperature),
                'temperature_driver_formula_max_abs_C':float(np.max(np.abs(df.temperature-(df.temperature_raw/361+35)))),
                'export_mean_xyz':x.mean(0).tolist(),'raw_scaled_mean_xyz':raw.mean(0).tolist(),
                'export_norm':stats(norm(x)),'raw_scaled_norm':stats(norm(raw)),
                'same_row_corr_xyz':corr(raw,x),'same_row_difference_mean_xyz':(x-raw).mean(0).tolist(),
                'same_row_difference_axis_rms':np.sqrt(np.mean((x-raw)**2,axis=0)).tolist()}
    dump('input_manifest.json',source_manifest)
    dump('units_raw_export.json',units)
    a,ta=streams['accel'];w,tw=streams['gyro'];p,tp=streams['pose']
    xyz=['x','y','z'];ax=a[xyz].to_numpy();wx=w[xyz].to_numpy()
    po=p[['Omega_x','Omega_y','Omega_z']].to_numpy()
    pf=p[['Accel_x','Accel_y','Accel_z']].to_numpy()
    pb=p[['AccBias_x','AccBias_y','AccBias_z']].to_numpy()
    q=p[['Attitude_x','Attitude_y','Attitude_z','Attitude_w']].to_numpy()
    pr=Rotation.from_quat(q)
    # Nearest report timestamps, signed gyro_time - accel_time, NOT hardware delay.
    i=np.searchsorted(tw,ta);il=np.maximum(i-1,0);ir=np.minimum(i,len(tw)-1)
    near=np.where(abs(tw[il]-ta)<abs(tw[ir]-ta),il,ir)
    inventory['accel_gyro_nearest_signed_s']=stats(tw[near]-ta)
    inventory['accel_gyro_exact_timestamp_fraction']=float(np.mean(tw[near]==ta))
    dump('timestamp_inventory.json',inventory)
    maps=mappings();dump('axis_candidates.json',{name:m.tolist() for name,m in maps})
    lagrows=[];fitrows=[];aggregation=[]
    for key in ['accel','gyro']:
        df,t=streams[key];raw=df[['x_raw','y_raw','z_raw']].to_numpy()*df.scaling.to_numpy()[:,None];x=df[xyz].to_numpy()
        for lag in config['lags_raw_export_s']:
            keep=(t>.6+t[0])&(t<t[-1]-.6)
            xx=interp(t[keep]-lag,t,raw);yy=x[keep]
            for part,mask in [('train_fit',t[keep]<fit_end),('train_check',t[keep]>=fit_end)]:
                cc=corr(xx[mask],yy[mask]);err=yy[mask]-xx[mask]
                lagrows.append({'sensor':key,'part':part,'lag_s':lag,'definition':'export(t) vs scaled_raw(t-lag)',
                      **{f'corr_{j}':cc[n] for n,j in enumerate('xyz')},
                      'demeaned_rms3':rms(err-err.mean(0))})
        fit=t<fit_end;check=~fit
        for name,m in maps:
            z=rotate(m,raw)
            # Positive diagonal gain and offset only; no free 3x3 mounting fit.
            zd=z[fit]-z[fit].mean(0);yd=x[fit]-x[fit].mean(0)
            gain=np.maximum(np.sum(zd*yd,axis=0)/np.sum(zd*zd,axis=0),0)
            offset=x[fit].mean(0)-gain*z[fit].mean(0)
            fitrows.append({'sensor':key,'mapping':name,**{f'gain_{j}':gain[n] for n,j in enumerate('xyz')},
                **{f'offset_{j}':offset[n] for n,j in enumerate('xyz')},
                'fit_rms3':rms(x[fit]-(z[fit]*gain+offset)),
                'check_rms3':rms(x[check]-(z[check]*gain+offset))})
        for sec in [0,1,5]:
            if sec:
                bins=((t-t[0])//sec).astype(int)
                dat=pd.DataFrame(np.column_stack((raw,x))).groupby(bins).mean().to_numpy()
                rr,yy=dat[:,:3],dat[:,3:]
            else:rr,yy=raw,x
            cc=corr(rr,yy)
            aggregation.append({'sensor':key,'bin_s':sec,'n':len(rr),
                **{f'corr_{j}':cc[n] for n,j in enumerate('xyz')},'difference_rms3':rms(yy-rr)})
    pd.DataFrame(lagrows).to_csv(OUT/'raw_export_lag_profiles.csv',index=False)
    pd.DataFrame(fitrows).to_csv(OUT/'raw_export_discrete_fits.csv',index=False)
    pd.DataFrame(aggregation).to_csv(OUT/'raw_export_aggregation.csv',index=False)
    # Uniform train grid, clipped so all interpolated sources are within train.
    tt=np.arange(max(ta[0],tw[0],tp[0])+1,min(ta[-1],tw[-1],tp[-1])-1,.1)
    fa=interp(tt,ta,ax);ww=interp(tt,tw,wx);oo=interp(tt,tp,po)
    fs=uniform_filter1d(fa,11,axis=0,mode='nearest');ws=uniform_filter1d(ww,11,axis=0,mode='nearest')
    u=fs/norm(fs)[:,None];du=np.gradient(u,tt,axis=0)
    localstd=np.sqrt(np.maximum(uniform_filter1d(fa*fa,11,axis=0)-fs*fs,0).sum(1))
    quasi=(abs(norm(fs)-G)<1)&(norm(ws)<.1)&(localstd<.5)
    fit=tt<fit_end;check=~fit;mean_w=ww[fit].mean(0)
    axisrows=[]
    for name,m in maps:
        mapped=rotate(m,ws)
        for scope,mask in [('all_train',np.ones(len(tt),dtype=bool)),('quasi_static_proxy',quasi),('train_check',check)]:
            residual=du+np.cross(mapped,u)
            axisrows.append({'mapping':name,'scope':scope,'n':int(mask.sum()),
                  'gravity_rate_residual_rad_s_rms':rms(residual[mask]),
                  'zero_gyro_baseline_rad_s_rms':rms(du[mask])})
    pd.DataFrame(axisrows).to_csv(OUT/'relative_gyro_accel_axis_consistency.csv',index=False)
    # Common proper rotations cannot identify absolute mounting via this criterion.
    base=du+np.cross(ws,u)
    invariance=max(float(np.max(np.abs(norm(rotate(m,du)+np.cross(rotate(m,ws),rotate(m,u)))-norm(base)))) for _,m in maps)
    # Onboard comparison only: not independent attitude or calibration supervision.
    corraw=np.corrcoef(ww.T,oo.T)[:3,3:]
    onboardrows=[]
    for name,m in maps:
        z=rotate(m,ww);offset=(oo[fit]-z[fit]).mean(0)
        onboardrows.append({'mapping':name,'fit_rms3_no_offset':rms(oo[fit]-z[fit]),
             'check_rms3_no_offset':rms(oo[check]-z[check]),
             'check_rms3_train_offset':rms(oo[check]-(z[check]+offset)),
             **{f'train_offset_{j}':offset[n] for n,j in enumerate('xyz')}})
    pd.DataFrame(onboardrows).to_csv(OUT/'raw_onboard_discrete_consistency.csv',index=False)
    # Recovered from dataset paper Figure 2, independently of signal fit.
    # Raw PX4 x = (-body_x-body_y)/sqrt(2), y=(-body_x+body_y)/sqrt(2), z=-body_z.
    s=2**-.5
    documented=np.array([[-s,-s,0],[-s,s,0],[0,0,-1.]])
    np.testing.assert_allclose(documented@documented.T,np.eye(3),atol=1e-15)
    assert abs(np.linalg.det(documented)-1)<1e-15
    mount={'source':'dataset paper page 3 Figure 2, visually inspected',
           'matrix_px4_to_onboard_body':documented.tolist(),
           'not_member_of_24_signed_permutations':True,'fitted_angle':False,
           'pose_algorithm_unknown':True,'relative_offsets_not_physical_calibration':True,'comparisons':{}}
    for key,rawsig,onboardsig in [('gyro_to_Omega',ww,oo),('accel_to_Accel',fa,interp(tt,tp,pf)),('accel_to_AccBias',fa,interp(tt,tp,pb))]:
        z=rotate(documented,rawsig);offset=(onboardsig[fit]-z[fit]).mean(0)
        mount['comparisons'][key]={'train_fit_offset_onboard_minus_transformed_raw':offset.tolist(),
            'train_fit_corr_xyz':corr(z[fit],onboardsig[fit]),'train_check_corr_xyz':corr(z[check],onboardsig[check]),
            'train_check_rms3_without_offset':rms(onboardsig[check]-z[check]),
            'train_check_rms3_with_train_offset':rms(onboardsig[check]-(z[check]+offset)),
            'train_check_axis_rms_with_train_offset':np.sqrt(np.mean((onboardsig[check]-z[check]-offset)**2,axis=0)).tolist()}
    mount['uniform_10Hz_interpolated_comparisons']=mount.pop('comparisons')
    mount['comparisons']={}
    for key,t_native,rawsig,onboardsig in [('gyro_to_Omega',tw,wx,po),('accel_to_Accel',ta,ax,pf),('accel_to_AccBias',ta,ax,pb)]:
        valid=(t_native>=tp[0])&(t_native<=tp[-1])
        native=t_native[valid];z=rotate(documented,rawsig[valid]);target=interp(native,tp,onboardsig)
        fit_native=native<fit_end;check_native=~fit_native
        offset=(target[fit_native]-z[fit_native]).mean(0)
        mount['comparisons'][key]={'timeline':'native raw timestamp; offline interpolate OnboardPose only',
            'n_fit':int(fit_native.sum()),'n_check':int(check_native.sum()),
            'train_fit_offset_onboard_minus_transformed_raw':offset.tolist(),
            'train_fit_corr_xyz':corr(z[fit_native],target[fit_native]),
            'train_check_corr_xyz':corr(z[check_native],target[check_native]),
            'train_check_rms3_without_offset':rms(target[check_native]-z[check_native]),
            'train_check_rms3_with_train_offset':rms(target[check_native]-(z[check_native]+offset)),
            'train_check_axis_rms_with_train_offset':np.sqrt(np.mean((target[check_native]-z[check_native]-offset)**2,axis=0)).tolist()}
    dump('documented_mount_consistency.json',mount)
    # Sign-robust channel pair lag profiles; no fitted yaw or selected delay.
    olags=[]
    for lag in config['lags_raw_onboard_s']:
        rr=interp(tt-lag,tw,wx)
        cc=np.corrcoef(rr.T,oo.T)[:3,3:]
        olags.append({'lag_s':lag,'definition':'onboard(t) vs raw(t-lag)',
              'corr_raw_z_vs_negative_onboard_z':float(-cc[2,2]),
              'mean_abs_xy_pair_correlation':float(np.abs(cc[:2,:2]).mean())})
    pd.DataFrame(olags).to_csv(OUT/'raw_onboard_lag_profiles.csv',index=False)
    # Quaternion convention checks; never call it world ENU or absolute heading GT.
    qpitch=pr.as_euler('xyz')[:,1];vp=p.veh_pitch.to_numpy()
    dq=pr[:-1].inv()*pr[1:];dtp=np.diff(tp)
    omega_q=dq.as_rotvec()/dtp[:,None];omega_mid=.5*(po[:-1]+po[1:])
    oerror=omega_q-omega_mid
    qdeg=np.degrees(dq.magnitude())
    jumpids=np.where(qdeg>1)[0]
    pd.DataFrame([{'t0_s':tp[i],'t1_s':tp[i+1],'dt_s':dtp[i],
        'q_rotation_deg':qdeg[i],'q_delta_x_rad':dq[i].as_rotvec()[0],
        'q_delta_y_rad':dq[i].as_rotvec()[1],'q_delta_z_rad':dq[i].as_rotvec()[2],
        'omega_norm_rad_s':norm(po[i]),'native_csv_row_t0_including_header':int(i)+2}
        for i in jumpids]).to_csv(OUT/'onboard_quaternion_large_steps.csv',index=False)
    poseinfo={'q_norm':stats(norm(q)), 'pitch_quaternion_minus_veh_pitch_rad':stats(qpitch-vp),
        'quaternion_order':'CSV wxyz -> SciPy xyzw',
        'omega_xyz_mean':po.mean(0).tolist(),'accel_xyz_mean':pf.mean(0).tolist(),
        'accbias_xyz_mean':pb.mean(0).tolist(),'accbias_norm':stats(norm(pb)),
        'velocity_all_zero':bool((p[['Vel_x','Vel_y','Vel_z']].to_numpy()==0).all()),
        'azimuth_all_zero':bool((p.Azimuth==0).all()),'GPS_on_counts':p.GPS_on.value_counts().to_dict(),
        'omega_q_minus_omega_axis_rms_rad_s':np.sqrt(np.mean(oerror**2,axis=0)).tolist(),
        'omega_q_vs_omega_corr_xyz':corr(omega_q,omega_mid),
        'omega_q_error_excluding_two_steps_over_5deg_axis_rms_rad_s':np.sqrt(np.mean(oerror[qdeg<=5]**2,axis=0)).tolist(),
        'omega_q_corr_excluding_two_steps_over_5deg':corr(omega_q[qdeg<=5],omega_mid[qdeg<=5]),
        'omega_q_error_norm_rad_s':stats(norm(oerror)),
        'q_step_deg':stats(np.degrees(dq.magnitude())),
        'q_step_over_5deg_count':int(np.sum(np.degrees(dq.magnitude())>5)),
        'q_step_over_1deg_count':int(np.sum(np.degrees(dq.magnitude())>1)),
        'raw_gyro_vs_onboard_omega_correlation_matrix':corraw.tolist(),
        'quasi_static_proxy_n':int(quasi.sum()),'train_grid_n':len(tt),
        'raw_gyro_train_fit_mean_not_identified_bias_rad_s':mean_w.tolist(),
        'common_axis_rotation_invariance_max_error':invariance}
    dump('onboard_attitude_diagnostic.json',poseinfo)
    gravityrows=[]
    rt=Slerp(tp,pr)(tt)
    for label,rr,xx in [('onboard_accel',pr,pf),('onboard_accbias',pr,pb)]:
        for inverse in [False,True]:
            world=rr.apply(xx,inverse=inverse)
            for sign in [-1,1]:
                err=world-np.array([0,0,sign*G])
                gravityrows.append({'source':label,'mapping':'native','q_inverse':inverse,'expected_z_sign':sign,
                  'n':len(xx),'residual_rms_m_s2':rms(err),'horizontal_rms_m_s2':float(np.sqrt(np.mean((world[:,:2]**2).sum(1)))),
                  **{f'world_mean_{j}':world[:,n].mean() for n,j in enumerate('xyz')},
                  'norm_mean':float(norm(xx).mean())})
    for name,m in maps:
        for inverse in [False,True]:
            world=rt.apply(rotate(m,fa),inverse=inverse)
            for sign in [-1,1]:
                err=world-np.array([0,0,sign*G])
                gravityrows.append({'source':'raw_exported_accel','mapping':name,'q_inverse':inverse,'expected_z_sign':sign,
                  'n':len(fa),'residual_rms_m_s2':rms(err),'horizontal_rms_m_s2':float(np.sqrt(np.mean((world[:,:2]**2).sum(1)))),
                  **{f'world_mean_{j}':world[:,n].mean() for n,j in enumerate('xyz')},'norm_mean':float(norm(fa).mean())})
    pd.DataFrame(gravityrows).to_csv(OUT/'offline_gravity_cancellation.csv',index=False)
    # Diagnostic reference frame q; not an asserted UTM/ENU earth frame.
    fixed_world=rt.apply(rotate(documented,fa))
    onboard_world=rt.apply(interp(tt,tp,pf))
    worldinfo={}
    for label,world in [('raw_accel_documented_mount',fixed_world),('onboard_accel',onboard_world)]:
        worldinfo[label]={}
        for scope,mask in [('all_train',np.ones(len(tt),dtype=bool)),('quasi_static_proxy',quasi),('train_check',check)]:
            err=world[mask]-[0,0,G]
            worldinfo[label][scope]={'n':int(mask.sum()),'reference_frame_mean_xyz':world[mask].mean(0).tolist(),
                'gravity_residual_rms_m_s2':rms(err),'horizontal_rms_m_s2':float(np.sqrt(np.mean((err[:,:2]**2).sum(1)))),
                'vertical_rms_m_s2':float(np.sqrt(np.mean(err[:,2]**2)))}
    dump('documented_mount_gravity_cancellation.json',worldinfo)
    euler=rt.as_euler('xyz');tilt=np.degrees(np.arccos(np.clip(rt.as_matrix()[:,2,2],-1,1)))
    deps=[]
    for lo,hi in [(0,5),(5,10),(10,20),(20,45),(45,180.01)]:
        mask=(tilt>=lo)&(tilt<hi)
        if mask.any():deps.append({'onboard_tilt_deg_low':lo,'onboard_tilt_deg_high':hi,'n':int(mask.sum()),
             'accel_norm_mean':float(norm(fa[mask]).mean()),'accel_norm_std':float(norm(fa[mask]).std())})
    pd.DataFrame(deps).to_csv(OUT/'accel_norm_vs_onboard_tilt.csv',index=False)
    dump('accel_norm_diagnostics.json',{'all_train_export_norm':stats(norm(fa)),
         'quasi_static_smooth_norm':stats(norm(fs[quasi])),
         'norm_correlation_with_onboard_roll_pitch':corr(np.column_stack([norm(fa)]*3),np.column_stack([euler[:,:2],tilt])),
         'norm_correlation_temperature':float(np.corrcoef(norm(ax),a.temperature)[0,1])})
    # Fixed disjoint 30s windows on TRAIN; integrate orientation only, no p/v/GT/GNSS.
    traces=[];windows=[];raw_q_windows=[];raw_q_traces=[]
    mount_rotation=Rotation.from_matrix(documented)
    start=tt[0]
    while start+30<tt[-1]:
        take=np.where((tt>=start)&(tt<start+30))[0]
        if len(take)<290:break
        ts=tt[take];fwin=u[take];rates=ww[take]
        for method,bias in [('raw_zero',np.zeros(3)),('raw_train_mean_sensitivity',mean_w)]:
            relative=Rotation.identity();errs=[];relative_quaternions=[]
            for k in range(len(take)):
                if k:
                    relative=relative*Rotation.from_rotvec((rates[k-1]-bias)*(ts[k]-ts[k-1]))
                expected=relative.inv().apply(fwin[0])
                errs.append(float(angle(expected,fwin[k])))
                relative_quaternions.append(relative.as_quat())
            relative_body=mount_rotation*Rotation.from_quat(relative_quaternions)*mount_rotation.inv()
            estimated_q=rt[take[0]]*relative_body
            qerrors=np.degrees((rt[take].inv()*estimated_q).magnitude())
            raw_q_windows.append({'method':method,'start_s':start,
                'part':'train_fit' if start+30<fit_end else ('train_check' if start>=fit_end else 'cross_fit_check'),
                'reference':'onboard q with documented mounting; offline dependent diagnostic',
                'final_error_deg':qerrors[-1],'mean_error_deg':float(qerrors.mean())})
            for k in range(0,len(take),10):
                raw_q_traces.append({'method':method,'start_s':start,'elapsed_s':ts[k]-ts[0],'error_deg':qerrors[k]})
            for k in range(0,len(take),10):
                traces.append({'method':method,'start_s':start,'elapsed_s':ts[k]-ts[0],'error_deg':errs[k]})
            windows.append({'method':method,'start_s':start,'part':'train_fit' if start+30<fit_end else ('train_check' if start>=fit_end else 'cross_fit_check'),
                 'reference':'1s centered accel direction, not true attitude','final_error_deg':errs[-1],
                 'mean_error_deg':float(np.mean(errs))})
        # Onboard Omega integrated at its own 50Hz, compare shared estimator quaternion.
        ids=np.where((tp>=start)&(tp<start+30))[0]
        estimate=pr[ids[0]];errs=[]
        for k,idx in enumerate(ids):
            if k:estimate=estimate*Rotation.from_rotvec(po[ids[k-1]]*(tp[idx]-tp[ids[k-1]]))
            errs.append(float(np.degrees((pr[idx].inv()*estimate).magnitude())))
        for k in range(0,len(ids),50):
            traces.append({'method':'onboard_omega_to_onboard_q','start_s':start,'elapsed_s':tp[ids[k]]-tp[ids[0]],'error_deg':errs[k]})
        windows.append({'method':'onboard_omega_to_onboard_q','start_s':start,
            'part':'train_fit' if start+30<fit_end else ('train_check' if start>=fit_end else 'cross_fit_check'),
            'reference':'onboard q, not independent GT','final_error_deg':errs[-1],'mean_error_deg':float(np.mean(errs))})
        start+=30
    pd.DataFrame(windows).to_csv(OUT/'orientation_window_summary.csv',index=False)
    pd.DataFrame(traces).to_csv(OUT/'orientation_disagreement_over_time.csv',index=False)
    pd.DataFrame(raw_q_windows).to_csv(OUT/'raw_gyro_vs_onboard_orientation.csv',index=False)
    pd.DataFrame(raw_q_traces).to_csv(OUT/'raw_gyro_vs_onboard_over_time.csv',index=False)
    # Math checks: proper rotations, sign convention and gyro propagation.
    test_angle=.2
    rr=Rotation.from_rotvec([test_angle,0,0]);u0=np.array([0.,0.,-1.])
    u1=rr.inv().apply(u0)
    assert angle(u1,Rotation.from_rotvec([-test_angle,0,0]).apply(u0))<1e-6
    assert angle(u0,rr.apply(u1))<1e-6
    assert invariance<1e-12
    assert max(ta.max(),tw.max(),tp.max(),tt.max())<cutoff
    for key,hashv in source_manifest.items():assert hashlib.sha256((ROOT/key).read_bytes()).hexdigest()==hashv['sha256']
    protected=json.loads((OUT/'protected_files_before.json').read_text())
    changes=[name for name,v in protected.items() if not (ROOT/name).is_file() or hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=v['sha256']]
    assert not changes,changes
    dump('verification.json',{'proper_axis_mappings':24,'axis_orthogonality_pass':True,
         'common_rotation_invariance_pass':True,'synthetic_gyro_sign_propagation_pass':True,
         'all_signal_analysis_strictly_train':True,'protected_files_unchanged':len(protected),
         'protected_changes':changes,'validation_or_test_signal_used':False,'navigation_metrics_computed':False})
    dump('status.json',{'status':'complete','calibration_status':'B_PARTIALLY_RECOVERED',
         'training':False,'kaggle_job':False,'outage_evaluation':False,'selected_transform':None,
         'selected_bias':None,'selected_delay':None,'orientation_windows_per_method':len(windows)//3})
    # Export human-readable summary without selecting a calibration.
    print(json.dumps({'train_cutoff':cutoff,'raw_export':units,'onboard':poseinfo,
         'orientation_final_by_method':pd.DataFrame(windows).groupby('method').final_error_deg.agg(['mean','median','min','max']).to_dict()},indent=2))


if __name__=='__main__':
    main()
