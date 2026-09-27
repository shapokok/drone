"""Exactly six trainings in one job; no retries, test access, or auto next run."""
from pathlib import Path
import json,hashlib,shutil,platform,time,datetime,traceback
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from insane.run import preflight as original_preflight,ten,write
from insane.data import sha
from insane.model import native_motion
from insane.metrics import score,baselines
from .model import AdaptiveFusionNav,integrate_gated
from .baselines import damped_cv,choose_tau

BASES=['held_gnss','constant_velocity_gnss','damped_cv']

def code_version(root):
    root=Path(root);hashes={}
    for base in ['src/insane','src/insane_adaptive','tests/insane','tests/insane_adaptive','configs/insane','configs/insane_adaptive']:
        for p in sorted((root/base).glob('*')):
            if p.is_file() and p.suffix in ['.py','.json','.csv']:hashes[str(p.relative_to(root))]=sha(p)
    return hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),hashes

def preflight(raw,root):
    root=Path(root);cfg=json.loads((root/'configs/insane_adaptive/adaptive.json').read_text())
    lock=json.loads((root/'configs/insane_adaptive/protocol_lock.json').read_text())
    for name,h in lock['sha256'].items():
        if sha(root/name)!=h:raise ValueError('Original protocol modified: '+name)
    if sha(root/'configs/insane_adaptive/adaptive.json')!=lock['adaptive_config_sha256']:raise ValueError('Adaptive configuration changed')
    assert cfg['seeds']==[0,1,2] and cfg['methods']==['adaptive_full','adaptive_gps_only']
    original,split,val,train,scaler,cache,evidence=original_preflight(raw,root/'configs/insane')
    for key in ['learning_rate','weight_decay','batch_size','max_epochs','patience','min_delta_m','huber_delta_m','grad_clip','hidden','validation_criterion']:
        assert cfg[key]==original[key],key
    assert all(not (Path(raw)/seq).exists() for seq in split['test_reserved']) or str(raw).startswith('/Users/')
    # Local preparation can point to the original folder; no test file is read.
    evidence.update(adaptive_protocol_lock_pass=True,planned_trainings=6,gate_initial_value=.5)
    return cfg,split,val,train,scaler,cache,evidence

def state_hash(model):
    h=hashlib.sha256()
    for key,value in model.state_dict().items():h.update(key.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def evaluate(model,x,y,device):
    model.eval()
    if device=='cuda':torch.cuda.synchronize()
    start=time.perf_counter()
    with torch.inference_mode():
        dv,g,_=model(*(ten(x[k],device)[None] for k in ['imu','gnss','common']))
        m,v=integrate_gated(dv,g,ten(x['velocity_prior'],device)[None],ten(x['integration_dt'],device)[None],torch.as_tensor(x['availability'],device=device)[None])
        native=native_motion(m[0],v[0],x,y).cpu().numpy().astype(float)
        detail={'g_token':g[0,:,0].cpu().numpy(),'residual_velocity_token':dv[0].cpu().numpy(),
                'motion_token':m[0].cpu().numpy(),'velocity_token':v[0].cpu().numpy()}
    if device=='cuda':torch.cuda.synchronize()
    return x['held'][y['token_index']]+native,native,(time.perf_counter()-start)*1000,detail

def batch_loss(model,selected,device,cfg):
    def stack(key):return pad_sequence([ten(x[key],device) for x,y in selected],batch_first=True)
    av=pad_sequence([torch.as_tensor(x['availability'],device=device) for x,y in selected],batch_first=True,padding_value=True)
    dv,g,_=model(stack('imu'),stack('gnss'),stack('common'))
    m,v=integrate_gated(dv,g,stack('velocity_prior'),stack('integration_dt'),av)
    losses=[]
    for k,(x,y) in enumerate(selected):
        chosen=torch.as_tensor(y['outage'],device=device);p=native_motion(m[k],v[k],x,y)
        losses.append(torch.nn.functional.huber_loss(p[chosen],ten(y['target_motion'],device)[chosen],delta=cfg['huber_delta_m']))
    return torch.stack(losses).mean()

def paired_rows(rows):
    df=pd.DataFrame(rows);result=[]
    for scenario,group in df[df.duration_s>0].groupby('scenario',sort=False):
        base=group[group.method.isin(BASES)].set_index('method')
        for seed in [0,1,2]:
            pair=group[group.seed==seed].set_index('method')
            for method in ['adaptive_full','adaptive_gps_only']:
                own=pair.loc[method]
                others=BASES+(['adaptive_gps_only'] if method=='adaptive_full' else [])
                for other in others:
                    ref=pair.loc[other] if other.startswith('adaptive_') else base.loc[other]
                    a=float(own.relative_motion_rmse3d_m);b=float(ref.relative_motion_rmse3d_m)
                    result.append({'scenario':scenario,'flight':own.flight,'duration_s':int(own.duration_s),'seed':seed,'method':method,'reference':other,
                       'rmse_m':a,'reference_rmse_m':b,'difference_m':a-b,'difference_percent':100*(a-b)/b if b else None,'better':a<b})
    return result

def save_prediction(folder,method,seed,x,y,p,m,detail):
    idx=y['token_index'];name=method if seed is None else f'{method}_seed{seed}'
    np.savez_compressed(folder/(name+'.npz'),t=y['t'],gt=y['gt'],pred=p,motion=m,target_motion=y['target_motion'],anchor_gt=y['anchor_gt'],
      gt_native_index=y['gt_native_index'],outage=y['outage'],availability=x['availability'][idx],source_t=x['source_t'][idx],source_index=x['source_index'][idx],
      held=x['held'][idx],token_t=x['t'][idx],imu_source_t=x['imu_source_t'][idx],
      all_token_t=x['t'],all_availability=x['availability'],integration_dt=x['integration_dt'],velocity_prior=x['velocity_prior'],token_index=idx,**detail)

def execute(raw,root,out,run_training=False):
    if not run_training:raise RuntimeError('Training gate false')
    if not torch.cuda.is_available():raise RuntimeError('All training/evaluation is Kaggle GPU only')
    root=Path(root);out=Path(out)
    if out.exists():raise FileExistsError('One job only: outputs already exist')
    cfg,split,val,train,scaler,cache,checks=preflight(raw,root)
    version,hashes=code_version(root);out.mkdir(parents=True);(out/'checkpoints').mkdir();(out/'predictions').mkdir()
    status={'status':'running','test_used':False,'seeds':[0,1,2],'planned_trainings':6,'models':{},'eskf_status':'not_ready',
      'code_version':version,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'run_identity':{'kernel':'shapok/drone-nav-insane-adaptive','version':'see submission receipt'}}
    write(out/'status.json',status)
    try:
        for rel in hashes:
            dst=out/'source_snapshot'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/rel,dst)
        write(out/'source_hashes.json',hashes);write(out/'config.json',cfg);write(out/'scalers.json',scaler);write(out/'preflight.json',checks)
        for name in ['splits','outages','training_episodes','download_manifest','inventory','protocol_lock']:
            shutil.copyfile(root/'configs/insane'/f'{name}.json',out/f'{name}.json')
        shutil.copyfile(root/'configs/insane_adaptive/protocol_lock.json',out/'adaptive_protocol_lock.json')
        shutil.copyfile(root/'configs/insane_adaptive/previous_pilot_summary.csv',out/'previous_pilot_seed0.csv')
        write(out/'eskf_status.json',{'status':'not_ready','blockers':cfg['eskf']['blockers'],'evaluated':False})
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        write(out/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,
              'device':'cuda','cuda':torch.version.cuda,'hardware':torch.cuda.get_device_name(0)})
        # Only train targets are read here; absolutely no validation metric precedes this lock.
        train_cache={s['id']:cache[s['id']] for s in train}
        tau,search=choose_tau(train,train_cache,cfg['damping']['tau_candidates_s'],split['train'])
        pd.DataFrame(search).to_csv(out/'damping_train_selection.csv',index=False)
        write(out/'damping_lock.json',{'tau_s':tau,'selected_on':'train only','flights':split['train'],'episodes':len(train),
           'validation_metrics_computed_before_lock':False,'selected_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'rule':cfg['damping'],'episode_ids_sha256':hashlib.sha256(json.dumps([s['id'] for s in train]).encode()).hexdigest()})
        print('DAMPING LOCKED ON TRAIN:',tau,flush=True)
        models={};history=[];counts={};initial={}
        for seed in cfg['seeds']:
            for method in cfg['methods']:
                key=f'{method}_seed{seed}';torch.manual_seed(seed);np.random.seed(seed);torch.cuda.manual_seed_all(seed)
                model=AdaptiveFusionNav(method,cfg['hidden']).to('cuda');counts[key]=sum(p.numel() for p in model.parameters());initial[key]=state_hash(model)
                if method=='adaptive_gps_only':assert initial[key]==initial[f'adaptive_full_seed{seed}']
                write(out/'initial_weight_hashes.json',initial)
                optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
                best=float('inf');best_epoch=0;bad=0
                for epoch in range(1,cfg['max_epochs']+1):
                    start=time.perf_counter();model.train();losses=[];episodes=[s for s in train if s['epoch']==epoch]
                    for k in range(0,len(episodes),cfg['batch_size']):
                        loss=batch_loss(model,[cache[s['id']] for s in episodes[k:k+cfg['batch_size']]],'cuda',cfg)
                        if not torch.isfinite(loss):raise ValueError('Nonfinite train loss')
                        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip']);optimizer.step();losses.append(float(loss.detach().cpu()))
                    values=[]
                    for spec in val:
                        if not spec['duration_s']:continue
                        x,y=cache[spec['id']];p,m,_,_=evaluate(model,x,y,'cuda');values.append(score(spec,x,y,p,m,method)['relative_motion_rmse3d_m'])
                    criterion=float(np.mean(values))
                    if not np.isfinite(criterion):raise ValueError('Nonfinite validation criterion')
                    improved=criterion<best-cfg['min_delta_m']
                    if improved:
                        best=criterion;best_epoch=epoch;bad=0;torch.save(model.state_dict(),out/'checkpoints'/f'{key}_best.pt')
                    else:bad+=1
                    history.append({'method':method,'seed':seed,'epoch':epoch,'mean_batch_huber_train_loss':float(np.mean(losses)),
                        'validation_macro_relative_rmse3d_m':criterion,'selected_best_so_far':improved,'epochs_without_improvement':bad,
                        'epoch_wall_seconds':time.perf_counter()-start,'episode_order_sha256':hashlib.sha256(json.dumps([s['id'] for s in episodes]).encode()).hexdigest()})
                    pd.DataFrame(history).to_csv(out/'training_history.csv',index=False)
                    status['models'][key]={'last_epoch':epoch,'best_epoch':best_epoch,'best_validation_criterion_m':best};write(out/'status.json',status)
                    print(json.dumps(history[-1]),flush=True)
                    if bad>=cfg['patience']:break
                torch.save(model.state_dict(),out/'checkpoints'/f'{key}_last.pt')
                model.load_state_dict(torch.load(out/'checkpoints'/f'{key}_best.pt',map_location='cuda',weights_only=True));models[(method,seed)]=model.eval()
        assert set(counts.values())=={7548};write(out/'parameter_counts.json',counts)
        rows=[];gates=[]
        for spec in val:
            x,y=cache[spec['id']];folder=out/'predictions'/spec['id'];folder.mkdir()
            start=time.perf_counter();base=baselines(x,y);base_ms=(time.perf_counter()-start)*1000
            start=time.perf_counter();dp,dm=damped_cv(x,y,tau);dms=(time.perf_counter()-start)*1000
            predictions=[(method,None,p,m,base_ms,{}) for method,(p,m) in base.items()]+[('damped_cv',None,dp,dm,dms,{})]
            for (method,seed),model in models.items():
                p,m,ms,detail=evaluate(model,x,y,'cuda');predictions.append((method,seed,p,m,ms,detail))
            for method,seed,p,m,ms,detail in predictions:
                if not np.isfinite(p).all():raise ValueError('Nonfinite predictions')
                row=score(spec,x,y,p,m,method,ms);row['seed']=seed;rows.append(row)
                save_prediction(folder,method,seed,x,y,p,m,detail)
                if detail and spec['duration_s']:
                    mask=(x['t']>=spec['onset_s'])&(x['t']<spec['onset_s']+spec['duration_s']);g=detail['g_token'][mask]
                    gates.append({'scenario':spec['id'],'flight':spec['flight'],'duration_s':spec['duration_s'],'method':method,'seed':seed,
                      'g_mean':float(g.mean()),'g_min':float(g.min()),'g_max':float(g.max()),'g_first':float(g[0]),'g_last':float(g[-1])})
        pd.DataFrame(rows).to_csv(out/'summary.csv',index=False);pd.DataFrame(paired_rows(rows)).to_csv(out/'paired_differences.csv',index=False)
        pd.DataFrame(gates).to_csv(out/'gate_summary.csv',index=False)
        status.update(status='complete',completed_trainings=len(models),evaluation_rows=len(rows),finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),additional_jobs_started=False)
        write(out/'status.json',status)
        from .report import generate
        generate(out)
        print('ADAPTIVE COMPLETE. STOP: no test, no additional seed/job.',flush=True)
    except Exception as e:
        status.update(status='failed',error=str(e));write(out/'status.json',status);(out/'failure_traceback.txt').write_text(traceback.format_exc());raise
