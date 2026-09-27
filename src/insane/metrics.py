"""Native-reference timestamp metrics; no fitted trajectory alignment."""
import numpy as np
import pandas as pd

def rmse(error):
    return float(np.sqrt(np.mean(np.sum(np.asarray(error)**2,axis=-1))))

def score(spec,x,y,pred,motion,method,inference_ms=None):
    inside=y['outage'];select=inside if inside.any() else np.ones(len(inside),bool)
    err=pred-y['gt'];e=err[select]
    cut=None if spec['duration_s']==0 else spec['onset_s']+spec['duration_s']
    recovery=np.zeros(len(inside),bool) if cut is None else ((y['t']>=cut)&(y['t']<cut+5))
    row={'scenario':spec['id'],'flight':spec['flight'],'split':'validation','method':method,'seed':0 if method in ['full','gps_only'] else None,
       'duration_s':spec['duration_s'],'status':'complete','n_native_gt':int(select.sum()),
       'relative_motion_rmse3d_m':rmse((motion-y['target_motion'])[inside]) if inside.any() else None,
       'absolute_rmse3d_m':rmse(e),'absolute_rmse_h_m':rmse(e[:,:2]),'absolute_rmse_v_m':rmse(e[:,2:3]),
       'final_unavailable_error_m':float(np.linalg.norm(e[-1])) if inside.any() else None,
       'final_relative_motion_error_m':float(np.linalg.norm((motion-y['target_motion'])[inside][-1])) if inside.any() else None,
       'recovery_first_5s_rmse3d_m':rmse(err[recovery]) if recovery.any() else None,
       'first_recovered_fix_error_m':None,'inference_ms':inference_ms,'reason':None}
    if cut is not None:
        idx=y['token_index'];back=(y['t']>=cut)&(x['source_t'][idx]>=cut)
        if back.any():row['first_recovered_fix_error_m']=float(np.linalg.norm(err[np.flatnonzero(back)[0]]))
    return row

def baselines(x,y):
    ids=y['token_index'];held=x['held'][ids];available=x['availability'][ids]
    motion=x['velocity_prior'][ids]*(y['t']-x['source_t'][ids])[:,None]*(~available)[:,None]
    return {'held_gnss':(held.copy(),np.zeros_like(held)),'constant_velocity_gnss':(held+motion,motion)}

def paired(rows):
    d=pd.DataFrame(rows);out=[]
    for scenario,g in d[(d.status=='complete')&(d.duration_s>0)].groupby('scenario',sort=False):
        a=g.set_index('method');full=float(a.loc['full','relative_motion_rmse3d_m'])
        for other in ['gps_only','constant_velocity_gnss','held_gnss']:
            b=float(a.loc[other,'relative_motion_rmse3d_m'])
            out.append({'scenario':scenario,'flight':a.iloc[0].flight,'duration_s':int(a.iloc[0].duration_s),'comparison':'full_minus_'+other,
                'full_rmse_m':full,'other_rmse_m':b,'difference_m':full-b,'difference_percent':100*(full-b)/b if b>0 else None,
                'full_better':full<b,'n_native_gt':int(a.loc['full','n_native_gt'])})
    return out
