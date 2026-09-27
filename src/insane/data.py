"""INSANE adapter: causal weighted IMU integration and mask-first GNSS features.

GT is returned separately. Its quaternion columns are deliberately never read.
Published timestamps/ENU positions are already processed; no second offset/calibration.
"""
from pathlib import Path
import hashlib,json,re
import numpy as np
import pandas as pd

IMU_COLUMNS=['t','a_x','a_y','a_z','w_x','w_y','w_z']
GPS_COLUMNS=['t','lat','long','alt','p_x','p_y','p_z','cov_p_x','cov_p_y','cov_p_z']
GT_COLUMNS=['t','p_x','p_y','p_z']

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_columns(path,columns):
    d=pd.read_csv(path);d.columns=d.columns.str.strip()
    if not set(columns)<=set(d.columns):raise ValueError('Unexpected schema '+str(path))
    x=d[columns].to_numpy(dtype=np.float64)
    if not np.isfinite(x).all():raise ValueError('Nonfinite input '+str(path))
    if np.any(np.diff(x[:,0])<=0):raise ValueError('Nonmonotonic/duplicate timestamps '+str(path))
    return x

def ecef(lla):
    lla=np.asarray(lla);lat,lon=np.deg2rad(lla[...,0]),np.deg2rad(lla[...,1]);h=lla[...,2]
    a=6378137.;e2=6.69437999014e-3;n=a/np.sqrt(1-e2*np.sin(lat)**2)
    return np.stack([(n+h)*np.cos(lat)*np.cos(lon),(n+h)*np.cos(lat)*np.sin(lon),(n*(1-e2)+h)*np.sin(lat)],axis=-1)

def enu(lla,reference):
    lat,lon=np.deg2rad(reference[:2]);s,c=np.sin,np.cos
    r=np.array([[-s(lon),c(lon),0],[-s(lat)*c(lon),-s(lat)*s(lon),c(lat)],[c(lat)*c(lon),c(lat)*s(lon),s(lat)]])
    return np.einsum('ij,nj->ni',r,ecef(lla)-ecef(np.array(reference)))

def load_flight(root,sequence,expected=None):
    root=Path(root);folder=root/sequence
    if expected:
        for item in expected['files']:
            if sha(root/item['path'])!=item['sha256']:raise ValueError('Source hash mismatch '+item['path'])
    im=read_columns(folder/'px4_imu.csv',IMU_COLUMNS)
    gps=read_columns(folder/'px4_gps.csv',GPS_COLUMNS)
    gt=read_columns(folder/'ground_truth_8hz.csv',GT_COLUMNS)
    readmes=list((folder/'archive_metadata').glob('*/README.txt'))
    if len(readmes)!=1:raise ValueError('Missing unique stream README '+sequence)
    m=re.search(r'\[Latitude,Longitude,Altitude\]\s*=\s*([\d.\-]+),\s*([\d.\-]+),\s*([\d.\-]+)',readmes[0].read_text())
    if not m:raise ValueError('Missing published ENU origin')
    rounded_ref=np.array([float(v) for v in m.groups()])
    # README rounds to six decimals; use authors' pinned extract_data.m:89-104.
    ref=np.array([30.5999294,34.867308,526.594] if sequence.startswith('mars_')
                 else [46.606867399,14.279121199,484.017])
    if np.max(np.abs(ref-rounded_ref))>1e-6:raise ValueError('Wrong location/calibration reference')
    projection_error=float(np.max(np.abs(enu(gps[:,1:4],ref)-gps[:,4:7])))
    if projection_error>.002:raise ValueError(f'ENU contract mismatch {sequence}: {projection_error}')
    if np.any(gps[:,7:10]<0):raise ValueError('Negative GNSS covariance')
    # One common rebase, using ordinary GNSS only. No GT alignment.
    epoch=float(im[0,0]);origin=gps[0,4:7].copy()
    sensor={'sequence':sequence,'epoch':epoch,'imu_t':im[:,0]-epoch,'imu':im[:,1:],
       'gps_t':gps[:,0]-epoch,'gps_p':gps[:,4:7]-origin,'gps_cov':gps[:,7:10],
       'origin_enu':origin,'published_enu_reference':ref,'projection_error_m':projection_error}
    reference={'t':gt[:,0]-epoch,'p':gt[:,1:]-origin}
    return sensor,reference

def integral_at(t,x,q):
    """Integral of left-held measurements, never a sample after q."""
    t=np.asarray(t);x=np.asarray(x);q=np.asarray(q)
    if np.any(q<t[0]) or np.any(q>t[-1]):raise ValueError('No sensor extrapolation')
    prefix=np.vstack([np.zeros(x.shape[1]),np.cumsum(np.diff(t)[:,None]*x[:-1],axis=0)])
    j=np.searchsorted(t,q,side='right')-1
    return prefix[j]+(q-t[j])[:,None]*x[j]

def aggregate_imu(sensor,period):
    t=sensor['imu_t'];start=max(t[0],sensor['gps_t'][0])
    edges=start+np.arange(int(np.floor((t[-1]-start)/period))+1)*period
    integrated=integral_at(t,sensor['imu'],edges)
    dt=np.diff(edges);query=edges[1:];values=np.diff(integrated,axis=0)/dt[:,None]
    indices=np.searchsorted(t,query,side='right')-1
    counts=np.searchsorted(t,query,side='right')-np.searchsorted(t,edges[:-1],side='right')
    return {'t':query,'imu':values,'dt':dt,'imu_source_t':t[indices],
        'imu_age':query-t[indices],'imu_count':counts,'edges':edges}

def fit_scaler(aggregated,train_names):
    total=sum(float(aggregated[s]['dt'].sum()) for s in train_names)
    mean=sum((aggregated[s]['imu']*aggregated[s]['dt'][:,None]).sum(0) for s in train_names)/total
    var=sum(((aggregated[s]['imu']-mean)**2*aggregated[s]['dt'][:,None]).sum(0) for s in train_names)/total
    return {'mean':mean.tolist(),'std':np.maximum(np.sqrt(var),1e-5).tolist(),'fit_flights':list(train_names),'weighted_seconds':total}

def blocked(t,intervals):
    out=np.zeros(len(t),bool)
    for a,b in intervals:out|=(t>=a)&(t<b)
    return out

def build_inputs(sensor,agg,indices,intervals,scaler,cfg):
    """Function deliberately accepts no reference / GT object."""
    ids=np.asarray(indices,dtype=int);t=agg['t'][ids]
    # FINAL native-source mask first. Everything derived from GNSS follows it.
    legal=~blocked(sensor['gps_t'],intervals)
    st=sensor['gps_t'][legal];sp=sensor['gps_p'][legal];sc=sensor['gps_cov'][legal]
    original_ids=np.flatnonzero(legal)
    j=np.searchsorted(st,t,side='right')-1
    if np.any(j<0):raise ValueError('No prior legal GNSS fix')
    age=t-st[j]
    availability=(~blocked(t,intervals))&(age<=cfg['gps_gap_limit_s'])
    source_index=original_ids[j]
    update=np.r_[True,np.diff(source_index)!=0]&availability
    vel=np.zeros_like(sp)
    for k in range(len(st)):
        first=np.searchsorted(st,st[k]-cfg['cv_history_s'],side='left')
        if k>first and st[k]-st[first]>=1:
            x=st[first:k+1];x=x-x.mean();y=sp[first:k+1];vel[k]=(x[:,None]*(y-y.mean(0))).sum(0)/np.sum(x*x)
    velocity=vel[j]
    delta=np.zeros((len(t),3));previous=j[0]
    for k in range(1,len(t)):
        if update[k]:delta[k]=sp[j[k]]-sp[previous];previous=j[k]
    dt=np.r_[agg['dt'][ids[0]],np.diff(t)]
    integration_dt=dt.copy()
    first_out=np.flatnonzero((~availability)&np.r_[True,availability[:-1]])
    for k in first_out:integration_dt[k]=t[k]-st[j[k]]
    imu=(agg['imu'][ids]-scaler['mean'])/scaler['std']
    common=np.column_stack([availability,update,age/60.,dt,agg['imu_age'][ids],
       agg['imu_age'][ids]>.02,agg['imu_count'][ids]/10.,np.sqrt(sc[j])/10.])
    gnss=np.column_stack([delta/5.,velocity/5.])
    assert common.shape[1]==10 and imu.shape[1]==6 and gnss.shape[1]==6
    assert np.all(st[j]<=t+1e-9) and not blocked(st[j],intervals).any()
    return {'t':t,'imu':imu.astype('float32'),'gnss':gnss.astype('float32'),'common':common.astype('float32'),
      'velocity_prior':velocity.astype('float32'),'integration_dt':integration_dt.astype('float32'),
      'availability':availability,'held':sp[j],'source_t':st[j],'source_index':source_index,
      'imu_source_t':agg['imu_source_t'][ids],'raw_imu':agg['imu'][ids], 'intervals':intervals}

def make_labels(reference,x,spec,cfg):
    """Only native GT rows are scored. One interpolated anchor is target-only."""
    select=(reference['t']>=x['t'][0])&(reference['t']<=x['t'][-1])
    gt_ids=np.flatnonzero(select);t=reference['t'][select];gt=reference['p'][select]
    index=np.searchsorted(x['t'],t,side='right')-1
    outage=blocked(t,x['intervals'])
    anchor=np.zeros_like(gt)
    for a,b in x['intervals']:
        take=(t>=a)&(t<b)
        first=np.searchsorted(x['t'],a)
        at=float(x['source_t'][first])
        j=np.searchsorted(reference['t'],at)
        if j==0 or j==len(reference['t']) or reference['t'][j]-reference['t'][j-1]>cfg['gt_gap_limit_s']:
            raise ValueError('Missing GT anchor support')
        anchor[take]=[np.interp(at,reference['t'],reference['p'][:,d]) for d in range(3)]
    target=gt-anchor
    return {'t':t,'gt':gt,'gt_native_index':gt_ids,'token_index':index,'partial_dt':t-x['t'][index],
            'outage':outage,'target_motion':target,'anchor_gt':anchor}

def specs_indices(agg,spec):
    ids=np.flatnonzero((agg['t']>=spec['start_s']-1e-9)&(agg['t']<=spec['end_s']+1e-9))
    if len(ids)<2:raise ValueError('Empty scenario')
    return ids

def episode_inputs(sensor,reference,agg,spec,scaler,cfg):
    blocks=[] if spec['duration_s']==0 else [[spec['onset_s'],spec['onset_s']+spec['duration_s']]]
    x=build_inputs(sensor,agg,specs_indices(agg,spec),blocks,scaler,cfg)
    labels=make_labels(reference,x,spec,cfg)
    if spec['duration_s'] and not labels['outage'].any():raise ValueError('No native reference observations')
    return x,labels

def quality_segments(sensor,reference,cfg):
    start=max(sensor['imu_t'][0]+cfg['token_period_s'],sensor['gps_t'][0],reference['t'][0])
    end=min(sensor['imu_t'][-1],sensor['gps_t'][-1],reference['t'][-1])
    invalid=[]
    for times,limit in [(sensor['imu_t'],cfg['imu_gap_limit_s']),(sensor['gps_t'],cfg['gps_gap_limit_s']),(reference['t'],cfg['gt_gap_limit_s'])]:
        for k in np.flatnonzero(np.diff(times)>limit):invalid.append((float(times[k]),float(times[k+1])))
    segments=[(float(start),float(end))]
    for a,b in sorted(invalid):
        split=[]
        for l,r in segments:
            if b<=l or a>=r:split.append((l,r))
            else:
                if a>l:split.append((l,a))
                if b<r:split.append((b,r))
        segments=split
    return [(a,b) for a,b in segments if b>a]
