"""V2 data contract. No fitting GNSS->GT, pose inputs or GT-derived features."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from diagnostics.clean_outages import clean_gnss, false_runs


def backward_indices(source_t, query_t):
    if np.any(np.diff(source_t)<0):
        raise ValueError('sort source timestamps first')
    ids=np.searchsorted(source_t,query_t,side='right')-1
    if np.any(ids<0):
        raise ValueError('no past measurement; do not clip to a future sample')
    return ids


def prepare(raw_dir,config):
    """Read four small logs only, no images; project both streams into UTM32N.

    pyproj is required, never installed automatically. Same GNSS-derived origin
    subtracted from GPS and GT. Vertical values retained in documented metres;
    no geoid conversion invented. Motion labels cancel constant height offsets.
    """
    raw_dir=Path(raw_dir)
    for name,expected in config['raw_sha256'].items():
        if hashlib.sha256((raw_dir/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('unexpected raw data: '+name)
    from pyproj import Transformer
    projection=Transformer.from_crs('EPSG:4326','EPSG:32632',always_xy=True)
    frames={name:pd.read_csv(raw_dir/(name+'.csv')) for name in ['RawAccel','RawGyro','OnboardGPS','GroundTruthAGL']}
    for frame in frames.values():frame.columns=frame.columns.str.strip()
    acc,gyro,gps,gt=[frames[name] for name in frames]
    t=acc.Timpstemp.values.astype(float)*1e-6
    if np.any(np.diff(t)<=0):raise ValueError('unexpected accel timeline')
    # Preserve original accel split indices. GPS joins GT by imgid BEFORE sorting.
    gt['t']=gt.imgid.map(gps.set_index('imgid').Timpstemp)*1e-6
    gt=gt.dropna(subset=['t']).sort_values('t',kind='stable')
    gyro=gyro.sort_values('Timpstemp',kind='stable').drop_duplicates('Timpstemp',keep='last')
    tg=gyro.Timpstemp.values.astype(float)*1e-6
    j=backward_indices(tg,t)
    imu=np.column_stack([acc[['x','y','z']].values,gyro[['x','y','z']].values[j]]).astype(float)
    gyro_age=t-tg[j]
    gps=gps.sort_values('Timpstemp',kind='stable').drop_duplicates('Timpstemp',keep='last')
    # Coordinate-change events are a documented proxy, not verified receiver epochs.
    xyz=gps[['lat','lon','alt']].values
    changed=np.r_[True,np.any(np.diff(xyz,axis=0)!=0,axis=1)]
    valid=(gps.fix_type.values>=3)&np.isfinite(xyz).all(axis=1)
    event=gps.loc[changed&valid].copy()
    event_t=event.Timpstemp.values.astype(float)*1e-6
    east,north=projection.transform(event.lon.values,event.lat.values)
    absolute=np.column_stack([east,north,event.alt.values])
    origin=absolute[0].copy()
    position=absolute-origin
    gt_t=gt.t.values.astype(float)
    gt_position=gt[['x_gt','y_gt','z_gt']].values.astype(float)-origin
    # Only train/validation region is returned. No test labels or inputs evaluated.
    stop=config['validation_original_indices'][1]
    data={'t':t[:stop], 'imu':imu[:stop], 'gyro_age':gyro_age[:stop],
          'event_t':event_t[event_t<=t[stop-1]],'event_position':position[event_t<=t[stop-1]],
          'gt_t':gt_t[gt_t<=t[stop-1]],'gt_position':gt_position[gt_t<=t[stop-1]],
          'origin_utm_e_n_msl_m':origin.tolist(),
          'frame_note':'same UTM grid and common origin; camera/antenna lever arm and exact GT height datum unresolved'}
    return data


def interpolate_gt(data,t):
    if np.any(t<data['gt_t'][0]) or np.any(t>data['gt_t'][-1]):
        raise ValueError('no GT extrapolation')
    return np.column_stack([np.interp(t,data['gt_t'],data['gt_position'][:,j]) for j in range(3)])


def fit_imu_scaler(imu,train_stop):
    x=np.asarray(imu[:train_stop],float)
    return {'mean':x.mean(0).tolist(),'std':np.maximum(x.std(0),1e-6).tolist()}


def build_inputs(data,indices,available,scaler,blocked_intervals):
    """No GT read here. Actual dt; backwards gyro; no absolute position feature."""
    t=data['t'][indices]
    event=backward_indices(data['event_t'],t)
    clean=clean_gnss(t,data['event_position'][event],event,data['event_t'][event],
                     available,blocked_intervals)
    held=clean['held_gnss'];update=clean['gnss_update_mask'];st=clean['held_source_timestamp_s']
    dt=np.r_[0.,np.diff(t)]
    velocity=np.zeros((len(t),3));delta=np.zeros_like(velocity)
    history=[];v=np.zeros(3)
    for i in range(len(t)):
        if update[i]:
            if history:delta[i]=held[i]-held[history[-1]]
            history.append(i)
            history=[k for k in history if st[k]>=t[i]-5.]
            if len(history)>=2 and st[history[-1]]-st[history[0]]>=1.:
                x=st[history]-st[history].mean();y=held[history]
                v=(x[:,None]*(y-y.mean(0))).sum(0)/(x@x)
            else:v=np.zeros(3)
        velocity[i]=v
    # Freeze CV exactly at pre-onset value; history does not update in outage.
    effective_dt=dt.copy()
    for a,b in false_runs(available):
        effective_dt[a]=t[a]-st[a-1]
    imu=(data['imu'][indices]-scaler['mean'])/scaler['std']
    imu_features=np.column_stack([imu,data['gyro_age'][indices],(dt>.15).astype(float)])
    gnss_features=np.column_stack([delta,velocity,clean['fix_age_s']/60.,
                                   available.astype(float),update.astype(float),dt])
    assert imu_features.shape[1]==8 and gnss_features.shape[1]==10
    return {'imu_features':imu_features.astype(np.float32),'gnss_features':gnss_features.astype(np.float32),
            'velocity_prior':velocity.astype(np.float32),'integration_dt':effective_dt.astype(np.float32),
            't':t,'held':held,'availability':np.asarray(available,bool),'clean':clean}


def motion_labels(data,inputs,label_period_s=1.):
    """GT used ONLY as target/reference, never in build_inputs or rollout.

    Native GT >=~1 Hz; sample relative-displacement loss about once per second,
    not as independent 10 Hz ground-truth velocity observations.
    """
    t=inputs['t'];target=np.zeros((len(t),3));chosen=np.zeros(len(t),bool)
    gt=interpolate_gt(data,t)
    for a,b in false_runs(inputs['availability']):
        anchor_t=inputs['clean']['held_source_timestamp_s'][a-1]
        anchor_gt=interpolate_gt(data,np.array([anchor_t]))[0]
        target[a:b]=gt[a:b]-anchor_gt
        times=np.arange(t[a],t[b-1]+1e-9,label_period_s)
        ids=np.unique(np.r_[np.searchsorted(t,times),b-1])
        chosen[ids[(ids>=a)&(ids<b)]]=True
    return target.astype(np.float32),chosen,gt


def baseline_predictions(inputs):
    held=inputs['held'].copy();cv=held.copy()
    for a,b in false_runs(inputs['availability']):
        anchor=held[a-1]
        st=inputs['clean']['held_source_timestamp_s'][a-1]
        cv[a:b]=anchor+(inputs['t'][a:b]-st)[:,None]*inputs['velocity_prior'][a-1]
    return held,cv
