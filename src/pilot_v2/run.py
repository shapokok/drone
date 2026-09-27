"""Prepared pilot. Training requires explicit --run-training authorization.

No execution on import. Final-epoch checkpoints, one seed, no test split.
"""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch

from diagnostics.clean_outages import availability_from_blocks, false_runs, raw_metrics
from pilot_v2.data import prepare, build_inputs, motion_labels, fit_imu_scaler, baseline_predictions
from pilot_v2.model import CausalMotion, rollout_motion


def json_write(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def csv_write(path,rows):
    with Path(path).open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def tensor(array,device,dtype=torch.float32):
    return torch.as_tensor(array,dtype=dtype,device=device)


def training_specs(data,cfg,epoch):
    w=cfg['train_window_samples'];end=cfg['train_stop_index']
    rng=np.random.default_rng(cfg['seed']+10000+epoch)
    specs=[]
    for start in range(0,end-w+1,cfg['train_stride_samples']):
        ids=np.arange(start,start+w);t=data['t'][ids]
        duration=float(rng.choice(cfg['outage_durations_s']))
        # Reserve two maximal sample gaps for snapping onset and recovery.
        low=t[0]+cfg['train_warmup_s'];high=t[-1]-duration-cfg['train_recovery_s']-2*np.max(np.diff(t))
        if high<=low:raise ValueError('training window too short for fixed outage curriculum')
        a=int(np.searchsorted(t,rng.uniform(low,high)));onset=float(t[a]);cutoff=onset+duration
        b=int(np.searchsorted(t,cutoff))
        if b>=len(t) or t[-1]-t[b]<cfg['train_recovery_s']:
            raise ValueError('outage has insufficient recovery')
        specs.append({'indices':ids,'available':availability_from_blocks(w,[(a,b)]),
                      'blocked':[[onset,cutoff]],'start_index':start,'onset_s':onset,
                      'cutoff_s':cutoff,'duration_s':duration})
    order=rng.permutation(len(specs));return [specs[i] for i in order]


def eval_prediction(model,inputs,device,zero_imu=False):
    imu=inputs['imu_features']
    if zero_imu:imu=np.zeros_like(imu)
    model.eval()
    with torch.inference_mode():
        dv,_=model(tensor(imu,device)[None],tensor(inputs['gnss_features'],device)[None])
        motion=rollout_motion(dv,tensor(inputs['velocity_prior'],device)[None],
            tensor(inputs['integration_dt'],device)[None],tensor(inputs['availability'],device,torch.bool)[None])
    return inputs['held']+motion[0].cpu().numpy(),motion[0].cpu().numpy()


def execute(config_path,raw_dir,out_dir,run_training=False):
    if not run_training:
        raise RuntimeError('Pilot is prepared only. Separate user command is required; pass --run-training afterward.')
    config_path=Path(config_path);cfg=json.loads(config_path.read_text())
    plan=json.loads(config_path.with_name('v2_motion_pilot_manifest.json').read_text())
    if hashlib.sha256(config_path.read_bytes()).hexdigest()!=plan['config_sha256']:
        raise ValueError('pilot config differs from prepared manifest')
    out=Path(out_dir)
    if out.exists():raise FileExistsError('no overwrite/resume: '+str(out))
    data=prepare(raw_dir,cfg)
    scaler=fit_imu_scaler(data['imu'],cfg['train_stop_index'])
    ids=np.arange(*cfg['validation_common_indices'])
    if len(ids)!=2688:raise ValueError('unexpected common validation support')
    # Validate every planned scenario and label support BEFORE optimizing.
    prepared=[]
    for s in plan['scenarios']:
        mask=availability_from_blocks(len(ids),s['unavailable_index_runs'])
        if s['requested_duration_s']:
            actual=(data['t'][ids]>=s['onset_s'])&(data['t'][ids]<s['requested_cutoff_s'])
            if not np.array_equal(actual,~mask):raise ValueError('fixed validation timestamps changed')
        x=build_inputs(data,ids,mask,scaler,s['source_blocked_intervals_s'])
        labels,label_mask,gt=motion_labels(data,x)
        prepared.append((s,x,labels,gt))
    for e in range(cfg['epochs']):
        for spec in training_specs(data,cfg,e):
            x=build_inputs(data,spec['indices'],spec['available'],scaler,spec['blocked'])
            motion_labels(data,x)
    out.mkdir(parents=True)
    (out/'checkpoints').mkdir()
    # Preserve the exact implementation actually used by the future pilot.
    code_root=Path(__file__).resolve().parents[2]
    source_hashes={}
    for relative in ['src/pilot_v2/data.py','src/pilot_v2/model.py','src/pilot_v2/run.py',
                     'src/diagnostics/clean_outages.py']:
        payload=(code_root/relative).read_bytes()
        destination=out/'source_snapshot'/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(payload)
        source_hashes[relative]=hashlib.sha256(payload).hexdigest()
    json_write(out/'source_sha256.json',source_hashes)
    json_write(out/'status.json',{'status':'running','seed':cfg['seed'],'test_used':False})
    json_write(out/'config.json',cfg);json_write(out/'manifest.json',plan);json_write(out/'imu_scaler_train_only.json',scaler)
    json_write(out/'coordinate_contract.json',{'origin':data['origin_utm_e_n_msl_m'],'note':data['frame_note'],
                                             'target':'camera relative motion; no fit/alignment of GT to inputs'})
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    import pandas,pyproj
    json_write(out/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,
        'numpy':np.__version__,'pandas':pandas.__version__,'pyproj':pyproj.__version__,
        'device':device,'cuda':torch.version.cuda,
        'device_name':torch.cuda.get_device_name(0) if device=='cuda' else platform.processor()})
    models={};history=[];augmentation=[];count={}
    try:
        for variant in cfg['methods']:
            torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
            model=CausalMotion(cfg['hidden'],variant['ablation'],variant['backbone']).to(device)
            count[variant['id']]=sum(p.numel() for p in model.parameters())
            optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
            for epoch in range(cfg['epochs']):
                model.train();specs=training_specs(data,cfg,epoch);losses=[]
                for offset in range(0,len(specs),cfg['batch_size']):
                    selected=specs[offset:offset+cfg['batch_size']]
                    xx=[];yy=[];mm=[]
                    for spec in selected:
                        x=build_inputs(data,spec['indices'],spec['available'],scaler,spec['blocked'])
                        y,m,_=motion_labels(data,x)
                        xx.append(x);yy.append(y);mm.append(m)
                        if variant==cfg['methods'][0]:
                            augmentation.append({'epoch':epoch,**{k:v for k,v in spec.items() if k not in ('indices','available','blocked')}})
                    stack=lambda key:tensor(np.stack([x[key] for x in xx]),device)
                    availability=stack('availability').bool()
                    dv,_=model(stack('imu_features'),stack('gnss_features'))
                    motion=rollout_motion(dv,stack('velocity_prior'),stack('integration_dt'),availability)
                    chosen=tensor(np.stack(mm),device,torch.bool)
                    target=tensor(np.stack(yy),device)
                    loss=torch.nn.functional.huber_loss(motion[chosen],target[chosen],delta=cfg['huber_delta_m'])
                    if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                    optimizer.zero_grad();loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip']);optimizer.step()
                    losses.append(float(loss.detach().cpu()))
                history.append({'method':variant['id'],'epoch':epoch+1,'mean_batch_training_loss':float(np.mean(losses))})
                print(history[-1],flush=True)
            checkpoint=out/'checkpoints'/(variant['id']+'_seed0.pt')
            torch.save(model.state_dict(),checkpoint)
            models[variant['id']]=model.eval()
        json_write(out/'parameter_counts.json',count)
        json_write(out/'training_augmentation_manifest.json',augmentation)
        csv_write(out/'training_history.csv',history)
        rows=[]
        for scenario,x,target,gt in prepared:
            sid=scenario['scenario_id'];folder=out/sid;folder.mkdir()
            held,cv=baseline_predictions(x)
            predictions={'held_gnss':held,'constant_velocity_gnss':cv}
            for mid,model in models.items():predictions[mid]=eval_prediction(model,x,device)[0]
            predictions['fusion_v2_full_zero_imu']=eval_prediction(models['fusion_v2_full'],x,device,True)[0]
            # Primary motion metric removes initial fix error ONLY in labels, not model state.
            mask=x['availability'];anchor_base=x['held'].copy()
            for a,b in false_runs(mask):anchor_base[a:b]=held[a-1]
            for mid,pred in predictions.items():
                metric=raw_metrics(pred,gt,mask)
                motion_error=(pred-anchor_base)-target
                selected=~mask
                motion_rmse=float(np.sqrt(np.mean(np.sum(motion_error[selected]**2,axis=1)))) if selected.any() else None
                checkpoint=(out/'checkpoints'/((mid if mid!='fusion_v2_full_zero_imu' else 'fusion_v2_full')+'_seed0.pt'))
                checkpoint_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint.exists() else None
                meta={'scenario':scenario,'method':mid,'seed':cfg['seed'] if checkpoint_hash else None,
                      'checkpoint_sha256':checkpoint_hash,'frame':data['frame_note'],
                      'target':'relative camera motion from last legal fix timestamp; no GT input/state',
                      'non_independent_validation':True,'zero_imu_is_sensitivity':mid.endswith('zero_imu')}
                np.savez_compressed(folder/(mid+'.npz'),pred=pred,gt=gt,timestamps=x['t'],
                    availability=mask,motion_target=target,motion_error=motion_error,
                    source_timestamp=x['clean']['held_source_timestamp_s'],
                    source_index=x['clean']['held_source_index'],metadata_json=np.array(json.dumps(meta)))
                rows.append({'scenario_id':sid,'method':mid,'seed':meta['seed'],
                             'checkpoint_sha256':checkpoint_hash,'motion_rmse_3d_m':motion_rmse,**metric})
        csv_write(out/'summary.csv',rows)
        json_write(out/'status.json',{'status':'complete','seed':cfg['seed'],'evaluation_rows':len(rows),
                                      'checkpoint_selection':'fixed final epoch, no validation selection','test_used':False})
    except Exception as error:
        json_write(out/'status.json',{'status':'failed','error':str(error)});raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--raw-dir',required=True)
    parser.add_argument('--out-dir',default='outputs/v2_motion_pilot')
    parser.add_argument('--run-training',action='store_true')
    a=parser.parse_args();execute(a.config,a.raw_dir,a.out_dir,a.run_training)
