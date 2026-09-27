"""One explicitly authorized test-only execution; no optimizer or training path."""
from pathlib import Path
import json,hashlib,shutil,time,platform,datetime,traceback
import numpy as np
import pandas as pd
import torch
from insane.data import sha
from insane.model import native_motion
from insane.metrics import score,baselines
from insane_adaptive.model import AdaptiveFusionNav,integrate_gated
from insane_adaptive.baselines import damped_cv
from .protocol import prepare_test

BASES=['held_gnss','constant_velocity_gnss','damped_cv']

def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def tensor(v,device):return torch.as_tensor(v,device=device,dtype=torch.float32)
def state_hash(model):
    h=hashlib.sha256()
    for key,value in model.state_dict().items():h.update(key.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def source_hashes(root):
    root=Path(root);files={}
    for base in ['src/insane','src/insane_adaptive','src/insane_final_test','tests/insane','tests/insane_adaptive','tests/insane_final_test']:
        for p in sorted((root/base).glob('*.py')):files[str(p.relative_to(root))]=sha(p)
    files['scripts/insane/prepare.py']=sha(root/'scripts/insane/prepare.py')
    return files

def preflight(bundle,root,load_models=True):
    bundle=Path(bundle);root=Path(root);lock=json.loads((root/'configs/insane_final_test/FINAL_EVALUATION_LOCK.json').read_text())
    for rel,h in lock['evaluation_source_sha256'].items():
        if sha(root/rel)!=h:raise ValueError('Evaluation source changed: '+rel)
    for rel,h in lock['frozen_files_sha256'].items():
        if sha(bundle/rel)!=h:raise ValueError('Frozen model/config/input changed: '+rel)
    for rel,h in lock['raw_files_sha256'].items():
        if sha(bundle/'raw'/rel)!=h:raise ValueError('Raw/calibration hash mismatch: '+rel)
    cfg=json.loads((bundle/'frozen/config.json').read_text());scaler=json.loads((bundle/'frozen/scalers.json').read_text())
    manifest=json.loads((bundle/'frozen/download_manifest.json').read_text())
    assert scaler['fit_flights']==['mars_1','mars_2','mars_3'] and lock['tau_s']==5
    assert lock['seeds']==[0,1,2] and lock['test_flights']==['mars_6','mars_7']
    assert json.loads((bundle/'frozen/damping_lock.json').read_text())['tau_s']==5
    for name in ['mars_1','mars_2','mars_3','mars_4','mars_5','outdoor_1']:
        if (bundle/'raw'/name).exists():raise ValueError('Non-test raw flight in final evaluation input')
    expected=json.loads((root/'configs/insane_final_test/test_manifest.json').read_text())
    if sha(root/'configs/insane_final_test/test_manifest.json')!=lock['test_manifest_sha256']:raise ValueError('Test manifest changed')
    actual,cache=prepare_test(bundle/'raw',cfg,scaler,manifest)
    assert json.dumps(actual,sort_keys=True)==json.dumps(expected,sort_keys=True),'Test scenario/support/hash mismatch'
    models={};states={}
    if load_models:
        for method in lock['learned_methods']:
            for seed in lock['seeds']:
                name=f'{method}_seed{seed}_best.pt';p=bundle/'frozen/checkpoints'/name
                assert sha(p)==lock['checkpoint_sha256'][name]
                m=AdaptiveFusionNav(method,cfg['hidden']);state=torch.load(p,map_location='cpu',weights_only=True)
                m.load_state_dict(state,strict=True);m.eval();assert sum(v.numel() for v in m.parameters())==7548
                models[(method,seed)]=m;states[name]=state_hash(m)
    checks={'status':'pass','checkpoint_compatibility_checked':load_models,'checkpoints':len(models),'all_hashes_pass':True,
      'test_manifest_exact_match':True,'train_scaler_reused_without_refit':True,'tau_s':5,'test_flights':['mars_6','mars_7'],
      'scenario_count':len(actual['scenarios']),'unsupported':actual['unsupported'],'model_predictions_computed':False,'baseline_metrics_computed':False,
      'projection_max_error_m':max(i['projection_error_m'] for i in actual['inventory'].values())}
    return cfg,scaler,actual,cache,models,states,checks,lock

def evaluate(model,x,y,device):
    model.eval()
    if device=='cuda':torch.cuda.synchronize()
    start=time.perf_counter()
    with torch.inference_mode():
        dv,g,_=model(*(tensor(x[k],device)[None] for k in ['imu','gnss','common']))
        motion,velocity=integrate_gated(dv,g,tensor(x['velocity_prior'],device)[None],tensor(x['integration_dt'],device)[None],torch.as_tensor(x['availability'],device=device)[None])
        native=native_motion(motion[0],velocity[0],x,y).cpu().numpy().astype(float)
        detail={'g_token':g[0,:,0].cpu().numpy(),'residual_velocity_token':dv[0].cpu().numpy(),
             'motion_token':motion[0].cpu().numpy(),'velocity_token':velocity[0].cpu().numpy()}
    if device=='cuda':torch.cuda.synchronize()
    return x['held'][y['token_index']]+native,native,(time.perf_counter()-start)*1000,detail

def paired_rows(rows):
    d=pd.DataFrame(rows);out=[]
    for scenario,g in d[d.duration_s>0].groupby('scenario',sort=False):
        base=g[g.method.isin(BASES)].set_index('method')
        for seed in [0,1,2]:
            pair=g[g.seed==seed].set_index('method')
            for method in ['adaptive_full','adaptive_gps_only']:
                a=pair.loc[method]
                for other in BASES+(['adaptive_gps_only'] if method=='adaptive_full' else []):
                    b=pair.loc[other] if other.startswith('adaptive_') else base.loc[other]
                    v=float(a.relative_motion_rmse3d_m);w=float(b.relative_motion_rmse3d_m)
                    out.append({'split':'test','scenario':scenario,'flight':a.flight,'duration_s':int(a.duration_s),'method':method,'seed':seed,
                      'reference':other,'rmse_m':v,'reference_rmse_m':w,'difference_m':v-w,'difference_percent':100*(v-w)/w if w else None,'better':v<w})
    return out

def save_prediction(folder,method,seed,x,y,p,m,detail):
    idx=y['token_index'];name=method if seed is None else f'{method}_seed{seed}'
    np.savez_compressed(folder/(name+'.npz'),t=y['t'],gt=y['gt'],pred=p,motion=m,target_motion=y['target_motion'],anchor_gt=y['anchor_gt'],
      gt_native_index=y['gt_native_index'],outage=y['outage'],availability=x['availability'][idx],source_t=x['source_t'][idx],source_index=x['source_index'][idx],
      held=x['held'][idx],token_t=x['t'][idx],imu_source_t=x['imu_source_t'][idx],
      all_token_t=x['t'],all_availability=x['availability'],integration_dt=x['integration_dt'],velocity_prior=x['velocity_prior'],token_index=idx,**detail)

def execute(bundle,root,out,allow_test=False):
    if not allow_test:raise RuntimeError('Explicit final test authorization is required')
    if not torch.cuda.is_available():raise RuntimeError('Full test evaluation requires Kaggle GPU')
    root=Path(root);out=Path(out);bundle=Path(bundle)
    if out.exists():raise FileExistsError('Final test already attempted; no overwrite/retry')
    out.mkdir(parents=True)
    status={'status':'preflight','split':'test','training_performed':False,'optimizer_steps':0,'test_flights':['mars_6','mars_7'],
       'seeds':[0,1,2],'model_evaluations':0,'baseline_evaluations':0,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'run_identity':{'kernel':'shapok/drone-nav-insane-final-test','version':'see submission receipt'},'retries':0,'eskf_status':'not_ready'}
    write(out/'status.json',status)
    try:
        cfg,scaler,manifest,cache,models,before,checks,lock=preflight(bundle,root)
        for name in ['FINAL_EVALUATION_LOCK.json','test_manifest.json']:
            shutil.copyfile(root/'configs/insane_final_test'/name,out/name)
        for name in ['config.json','scalers.json','download_manifest.json','damping_lock.json','adaptive_status.json','adaptive_source_hashes.json']:
            shutil.copyfile(bundle/'frozen'/name,out/name)
        shutil.copyfile(bundle/'frozen/methods_before_test.md',out/'METHODS_BEFORE_TEST.md')
        for rel in lock['evaluation_source_sha256']:
            dst=out/'source_snapshot'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/rel,dst)
        write(out/'source_hashes.json',lock['evaluation_source_sha256']);write(out/'preflight.json',checks)
        write(out/'eskf_status.json',{'status':'not_ready','evaluated':False,'blockers':cfg['eskf']['blockers']})
        write(out/'weights_before.json',before)
        # Manifest/config/scalers/weights are pinned and saved before the first prediction or baseline score.
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        for model in models.values():model.to('cuda');model.requires_grad_(False)
        env={'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,
             'device':'cuda','cuda':torch.version.cuda,'hardware':torch.cuda.get_device_name(0)}
        write(out/'environment.json',env);status['status']='evaluating';write(out/'status.json',status)
        rows=[];gates=[];(out/'predictions').mkdir()
        for spec in manifest['scenarios']:
            x,y=cache[spec['id']];folder=out/'predictions'/spec['id'];folder.mkdir()
            start=time.perf_counter();base=baselines(x,y);base_ms=(time.perf_counter()-start)*1000
            start=time.perf_counter();dp,dm=damped_cv(x,y,5.);dms=(time.perf_counter()-start)*1000
            predictions=[(method,None,p,m,base_ms,{}) for method,(p,m) in base.items()]+[('damped_cv',None,dp,dm,dms,{})]
            for (method,seed),model in models.items():
                p,m,ms,detail=evaluate(model,x,y,'cuda');predictions.append((method,seed,p,m,ms,detail));status['model_evaluations']+=1
            for method,seed,p,m,ms,detail in predictions:
                if not np.isfinite(p).all():raise ValueError('Nonfinite prediction')
                row=score(spec,x,y,p,m,method,ms);row.update(split='test',seed=seed);rows.append(row)
                if seed is None:status['baseline_evaluations']+=1
                save_prediction(folder,method,seed,x,y,p,m,detail)
                if detail and spec['duration_s']:
                    mask=(x['t']>=spec['onset_s'])&(x['t']<spec['onset_s']+spec['duration_s']);g=detail['g_token'][mask]
                    gates.append({'scenario':spec['id'],'flight':spec['flight'],'duration_s':spec['duration_s'],'method':method,'seed':seed,
                        'g_mean':float(g.mean()),'g_min':float(g.min()),'g_max':float(g.max()),'g_first':float(g[0]),'g_last':float(g[-1])})
            pd.DataFrame(rows).to_csv(out/'summary.csv',index=False);write(out/'status.json',status)
            print('Finished fixed test scenario:',spec['id'],flush=True)
        after={f'{method}_seed{seed}_best.pt':state_hash(model) for (method,seed),model in models.items()}
        assert before==after,'Loaded tensors changed during inference'
        file_after={n:sha(bundle/'frozen/checkpoints'/n) for n in lock['checkpoint_sha256']}
        assert file_after==lock['checkpoint_sha256'],'Checkpoint files changed'
        # Copy exact original checkpoint bytes for a self-contained final handoff.
        (out/'checkpoints').mkdir()
        for name in file_after:shutil.copyfile(bundle/'frozen/checkpoints'/name,out/'checkpoints'/name)
        write(out/'weights_after.json',after);write(out/'checkpoint_hashes_after.json',file_after)
        pd.DataFrame(paired_rows(rows)).to_csv(out/'paired_differences.csv',index=False);pd.DataFrame(gates).to_csv(out/'gate_summary.csv',index=False)
        status.update(status='verifying',summary_rows=len(rows),weights_unchanged=True);write(out/'status.json',status)
        from .verify import verify
        evidence=verify(out,raw=bundle/'raw');write(out/'independent_verification.json',evidence)
        status.update(status='complete',verification='pass',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());write(out/'status.json',status)
        from .report import generate
        generate(out)
        print('FINAL TEST COMPLETE. STOP. No training/retry/next experiment.',flush=True)
    except Exception as e:
        status.update(status='failed',error=str(e));write(out/'status.json',status);(out/'failure_traceback.txt').write_text(traceback.format_exc());raise
