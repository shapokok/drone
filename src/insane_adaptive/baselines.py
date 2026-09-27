"""Analytic causal exponential damping; tune tau exclusively on train."""
import numpy as np
from insane.metrics import rmse

def damped_cv(x,y,tau):
    if not np.isfinite(tau) or tau<=0:raise ValueError('tau must be positive and finite')
    idx=y['token_index'];available=x['availability'][idx]
    age=np.maximum(0,y['t']-x['source_t'][idx])
    # Integral of v0 exp(-age/tau) from last legal fix to native timestamp.
    motion=x['velocity_prior'][idx]*(tau*(-np.expm1(-age/tau)))[:,None]*(~available)[:,None]
    return x['held'][idx]+motion,motion

def choose_tau(train_specs,cache,candidates,allowed_flights):
    if not train_specs or any(s['flight'] not in allowed_flights for s in train_specs):
        raise ValueError('Damping selection accepts train episodes only')
    rows=[]
    for tau in candidates:
        scores=[]
        for spec in train_specs:
            x,y=cache[spec['id']];_,m=damped_cv(x,y,tau)
            scores.append(rmse((m-y['target_motion'])[y['outage']]))
        rows.append({'tau_s':float(tau),'train_macro_relative_rmse3d_m':float(np.mean(scores)),'episodes':len(scores)})
    # Fixed ascending candidate list: lower tau wins exact ties.
    best=min(rows,key=lambda r:(r['train_macro_relative_rmse3d_m'],r['tau_s']))
    return best['tau_s'],rows
