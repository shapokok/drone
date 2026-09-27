"""One frozen two-model pilot. No execution at import, no resume or extra seeds."""
from pathlib import Path
import hashlib,json,platform,time,traceback,shutil,datetime
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from .data import load_flight,aggregate_imu,fit_scaler,episode_inputs,sha
from .model import FusionNavINSANE,integrate_motion,native_motion
from .metrics import score,baselines,paired
from .eskf import readiness
from .report import generate

def write(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def code_version(code_root):
    hashes={str(p.relative_to(code_root)):sha(p) for base in ['src/insane','tests/insane','configs/insane']
            for p in sorted((Path(code_root)/base).glob('*')) if p.is_file() and p.suffix in ['.py','.json']}
    return hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),hashes

def preflight(data_root,config_dir):
    config_dir=Path(config_dir);cfg=json.loads((config_dir/'pilot.json').read_text())
    lock=json.loads((config_dir/'protocol_lock.json').read_text())
    for name,h in lock['sha256'].items():
        if sha(config_dir/name)!=h:raise ValueError('Frozen protocol changed: '+name)
    split=json.loads((config_dir/'splits.json').read_text());manifest=json.loads((config_dir/'download_manifest.json').read_text())
    val=json.loads((config_dir/'outages.json').read_text())['scenarios']
    train=json.loads((config_dir/'training_episodes.json').read_text())['episodes']
    sets=[set(split[k]) for k in ['train','validation','test_reserved']]
    if any(sets[i]&sets[j] for i in range(3) for j in range(i)):raise ValueError('Flight split overlap')
    if cfg['seed']!=0 or cfg['max_epochs']>30 or cfg['patience']!=5:raise ValueError('Unexpected pilot plan')
    for rel,item in manifest['support_files'].items():
        if sha(Path(data_root)/rel)!=item['sha256']:raise ValueError('Calibration/reference hash mismatch '+rel)
    loaded={};aggs={}
    # Explicit allowlist: NEVER load test or loader-only flight into the run.
    for seq in split['train']+split['validation']:
        loaded[seq]=load_flight(data_root,seq,manifest['sequences'][seq]);aggs[seq]=aggregate_imu(loaded[seq][0],cfg['token_period_s'])
    scaler=fit_scaler(aggs,split['train'])
    expected=json.loads((config_dir/'scaler_preflight.json').read_text())
    np.testing.assert_allclose(scaler['mean'],expected['mean'],rtol=1e-8,atol=1e-8)
    np.testing.assert_allclose(scaler['std'],expected['std'],rtol=1e-8,atol=1e-8)
    cache={}
    for spec in val+train:
        expected_split='validation' if spec in val else 'train'
        if spec['flight'] not in split[expected_split]:raise ValueError('Wrong episode split')
        x,y=episode_inputs(*loaded[spec['flight']],aggs[spec['flight']],spec,scaler,cfg)
        if hashlib.sha256(np.asarray(x['availability'],np.uint8).tobytes()).hexdigest()!=spec['token_mask_sha256']:raise ValueError('Mask changed '+spec['id'])
        if hashlib.sha256(np.asarray(y['t'],dtype='<f8').tobytes()).hexdigest()!=spec['native_gt_timestamp_sha256']:raise ValueError('Native GT support changed')
        if int(y['outage'].sum())!=spec['n_native_gt_outage']:raise ValueError('Point count changed')
        cache[spec['id']]=(x,y)
    evidence={'status':'pass','train_flights':split['train'],'validation_flights':split['validation'],
        'test_loaded':False,'raw_calibration_hashes_pass':True,'projection_max_m':max(s[0]['projection_error_m'] for s in loaded.values()),
        'validated_training_episodes':len(train),'validated_eval_scenarios':len(val),'train_only_scaler_pass':True,
        'all_masks_and_native_gt_timestamp_hashes_pass':True}
    return cfg,split,val,train,scaler,cache,evidence

def ten(v,device):return torch.as_tensor(v,device=device,dtype=torch.float32)

def evaluate_model(model,x,y,device):
    model.eval()
    if device=='cuda':torch.cuda.synchronize()
    start=time.perf_counter()
    with torch.inference_mode():
        dv,_=model(ten(x['imu'],device)[None],ten(x['gnss'],device)[None],ten(x['common'],device)[None])
        motion,velocity=integrate_motion(dv,ten(x['velocity_prior'],device)[None],ten(x['integration_dt'],device)[None],torch.as_tensor(x['availability'],device=device)[None])
        native=native_motion(motion[0],velocity[0],x,y).cpu().numpy().astype(float)
    if device=='cuda':torch.cuda.synchronize()
    ms=(time.perf_counter()-start)*1000
    pred=x['held'][y['token_index']]+native
    return pred,native,ms

def batch_loss(model,selected,device,cfg):
    # Padding exists only inside computational batches; loss uses real native GT rows.
    def stack(key):return pad_sequence([ten(x[key],device) for x,y in selected],batch_first=True)
    imu,gnss,common=stack('imu'),stack('gnss'),stack('common')
    available=pad_sequence([torch.as_tensor(x['availability'],device=device) for x,y in selected],batch_first=True,padding_value=True)
    dv,_=model(imu,gnss,common)
    motion,velocity=integrate_motion(dv,stack('velocity_prior'),stack('integration_dt'),available)
    losses=[]
    for k,(x,y) in enumerate(selected):
        chosen=torch.as_tensor(y['outage'],device=device)
        native=native_motion(motion[k],velocity[k],x,y)
        losses.append(torch.nn.functional.huber_loss(native[chosen],ten(y['target_motion'],device)[chosen],delta=cfg['huber_delta_m']))
    return torch.stack(losses).mean()

def execute(data_root,config_dir,out,code_root,run_training=False,require_cuda=True):
    if not run_training:raise RuntimeError('Training gate is false')
    if require_cuda and not torch.cuda.is_available():raise RuntimeError('Pilot must run on Kaggle GPU, not the Mac')
    out=Path(out);code_root=Path(code_root)
    if out.exists():raise FileExistsError('No overwrite/restart/resume of one pilot: '+str(out))
    cfg,split,val,train,scaler,cache,checks=preflight(data_root,config_dir)
    version,hashes=code_version(code_root)
    out.mkdir(parents=True);(out/'checkpoints').mkdir();(out/'predictions').mkdir()
    for name in ['splits','outages','training_episodes','protocol_lock','download_manifest','inventory']:
        shutil.copyfile(Path(config_dir)/(name+'.json'),out/(name+'.json'))
    write(out/'config.json',cfg);write(out/'scalers.json',scaler);write(out/'preflight.json',checks)
    for rel in hashes:
        dst=out/'source_snapshot'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(code_root/rel,dst)
    write(out/'source_hashes.json',hashes);write(out/'eskf_status.json',readiness(cfg))
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    env={'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,
         'device':device,'cuda':torch.version.cuda,'hardware':torch.cuda.get_device_name(0) if device=='cuda' else platform.processor()}
    write(out/'environment.json',env)
    status={'status':'running','seed':0,'test_used':False,'code_version':version,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'run_identity':{'kernel':'shapok/drone-nav-insane-pilot','version':'see submission receipt'},'models':{},'eskf_status':'not_ready'}
    write(out/'status.json',status);history=[];models={};counts={}
    try:
        for variant in cfg['methods']:
            torch.manual_seed(0);np.random.seed(0);torch.cuda.manual_seed_all(0)
            model=FusionNavINSANE(variant,cfg['hidden']).to(device);counts[variant]=sum(p.numel() for p in model.parameters())
            optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
            best=float('inf');best_epoch=0;bad=0
            for epoch in range(1,cfg['max_epochs']+1):
                start=time.perf_counter();model.train();losses=[];episodes=[s for s in train if s['epoch']==epoch]
                for k in range(0,len(episodes),cfg['batch_size']):
                    loss=batch_loss(model,[cache[s['id']] for s in episodes[k:k+cfg['batch_size']]],device,cfg)
                    if not torch.isfinite(loss):raise ValueError('Nonfinite train loss')
                    optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip']);optimizer.step()
                    losses.append(float(loss.detach().cpu()))
                val_scores=[]
                for spec in val:
                    if not spec['duration_s']:continue
                    x,y=cache[spec['id']];pred,motion,ms=evaluate_model(model,x,y,device)
                    val_scores.append(score(spec,x,y,pred,motion,variant)['relative_motion_rmse3d_m'])
                criterion=float(np.mean(val_scores));improved=criterion<best-cfg['min_delta_m']
                if improved:
                    best=criterion;best_epoch=epoch;bad=0
                    torch.save(model.state_dict(),out/'checkpoints'/f'{variant}_best_seed0.pt')
                else:bad+=1
                history.append({'method':variant,'epoch':epoch,'mean_batch_huber_train_loss':float(np.mean(losses)),
                    'validation_macro_relative_rmse3d_m':criterion,'selected_best_so_far':improved,'epochs_without_improvement':bad,
                    'epoch_wall_seconds':time.perf_counter()-start})
                pd.DataFrame(history).to_csv(out/'training_history.csv',index=False)
                status['models'][variant]={'last_epoch':epoch,'best_epoch':best_epoch,'best_validation_criterion_m':best};write(out/'status.json',status)
                print(json.dumps(history[-1]),flush=True)
                if bad>=cfg['patience']:break
            torch.save(model.state_dict(),out/'checkpoints'/f'{variant}_last_seed0.pt')
            model.load_state_dict(torch.load(out/'checkpoints'/f'{variant}_best_seed0.pt',map_location=device,weights_only=True));models[variant]=model.eval()
        if counts['full']!=counts['gps_only']:raise ValueError('Parameter count mismatch')
        write(out/'parameter_counts.json',counts)
        rows=[]
        for spec in val:
            x,y=cache[spec['id']]
            start=time.perf_counter();base=baselines(x,y);base_ms=(time.perf_counter()-start)*1000
            predictions={key:(p,m,base_ms) for key,(p,m) in base.items()}
            for name,model in models.items():predictions[name]=evaluate_model(model,x,y,device)
            folder=out/'predictions'/spec['id'];folder.mkdir()
            for method,(prediction,motion,ms) in predictions.items():
                if not np.isfinite(prediction).all():raise ValueError('Nonfinite predictions')
                rows.append(score(spec,x,y,prediction,motion,method,ms));idx=y['token_index']
                np.savez_compressed(folder/(method+'.npz'),t=y['t'],gt=y['gt'],pred=prediction,motion=motion,
                  target_motion=y['target_motion'],anchor_gt=y['anchor_gt'],gt_native_index=y['gt_native_index'],outage=y['outage'],
                  availability=x['availability'][idx],source_t=x['source_t'][idx],source_index=x['source_index'][idx],
                  held=x['held'][idx],token_t=x['t'][idx],imu_source_t=x['imu_source_t'][idx])
            empty={k:None for k in rows[-1]};empty.update({'scenario':spec['id'],'flight':spec['flight'],'split':'validation',
              'method':'eskf','duration_s':spec['duration_s'],'status':'not_ready','reason':'; '.join(cfg['eskf']['blockers'])});rows.append(empty)
        pd.DataFrame(rows).to_csv(out/'summary.csv',index=False);pd.DataFrame(paired(rows)).to_csv(out/'paired_differences.csv',index=False)
        status.update(status='complete',evaluation_rows=len(rows),finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),extra_seeds_started=False)
        write(out/'status.json',status);generate(out)
        print('PILOT COMPLETE. STOP. No test / additional seeds.',flush=True)
    except Exception as e:
        status.update(status='failed',error=str(e));write(out/'status.json',status)
        (out/'failure_traceback.txt').write_text(traceback.format_exc());raise
