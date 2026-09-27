"""Automatic complete pilot report: all episodes retained, no winner selection."""
from pathlib import Path
import json
import pandas as pd

def table(frame):
    headers=list(frame.columns);lines=['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']
    for row in frame.itertuples(index=False,name=None):
        vals=[]
        for v in row:
            if pd.isna(v):vals.append('—')
            elif isinstance(v,float):vals.append(f'{v:.6f}')
            else:vals.append(str(v))
        lines.append('| '+' | '.join(vals)+' |')
    return '\n'.join(lines)

def generate(out,run_info=None):
    out=Path(out);d=pd.read_csv(out/'summary.csv');pair=pd.read_csv(out/'paired_differences.csv');hist=pd.read_csv(out/'training_history.csv')
    cfg=json.loads((out/'config.json').read_text());split=json.loads((out/'splits.json').read_text())
    env=json.loads((out/'environment.json').read_text());status=json.loads((out/'status.json').read_text())
    readiness=json.loads((out/'eskf_status.json').read_text())
    clean=d[(d.status=='complete')&(d.duration_s>0)]
    means=clean.groupby(['duration_s','method'],sort=False).relative_motion_rmse3d_m.mean().unstack('method').reset_index()
    flights=clean.groupby(['flight','duration_s','method'],sort=False).relative_motion_rmse3d_m.mean().unstack('method').reset_index()
    main=pair[pair.comparison=='full_minus_gps_only'];wins=int(main.full_better.sum())
    sections=[
      '# INSANE_PILOT_REPORT',
      '**Status: '+status['status']+'. STOP after one seed=0 pilot. No test metrics or additional seeds.**',
      '## Run and frozen protocol',
      'Run identity: `'+json.dumps(run_info or status.get('run_identity',{}),ensure_ascii=False)+'`.',
      'Code version: `'+status['code_version']+'`. Config/manifests/source snapshot are stored alongside this report. No hyperparameter or dataset change after training.',
      'Train: '+', '.join(split['train'])+'. Validation: '+', '.join(split['validation'])+'. Reserved test: '+', '.join(split['test_reserved'])+' (not loaded by the run). Outdoor_1 was loader smoke only.',
      '## Data / reference contract',
      'Inputs: published PX4 IMU specific force (m/s²) and angular rate (rad/s), ordinary PX4 GNSS ENU position/covariance only. IMU is aggregated with causal time-weighted integration at 20 Hz; dt/sample age/count and all GNSS availability/quality features are shared with gps_only. No GT attitude, RTK, onboard pose, magnetometer, or absolute position is a learned input. Published timestamp corrections are already applied; they are not applied twice.',
      'Scoring uses original rows/timestamps of ground_truth_8hz.csv; the nominal filename does not imply 8 independent observations each second. Reference positions are at the PX4 IMU, produced by the authors from dual RTK and magnetometer geometry. Their timing alignment is postprocessed and GT is not error-free. Ordinary GNSS antenna lever arm is absent from the supplied calibration; its offset/dynamic lever effect remains a limitation. No GT-derived rotation or fitted trajectory alignment is applied.',
      'Source: [INSANE data](https://www.aau.at/en/smart-systems-technologies/control-of-networked-systems/datasets/insane-dataset/), [authors tools](https://github.com/aau-cns/insane_dataset_tools/tree/9a1c8c0fdd195f2d869fff292f2ce5b273c5a03d), [Brommer et al.](https://arxiv.org/abs/2210.09114). License and attribution are included in the private data package.',
      '## Architecture and training',
      'Pilot architecture, not established novelty: IMU 6→16, GNSS displacement/velocity 6→16, shared quality/time 10→8; concat40→GRU32→velocity residual3. Matching full/gps_only parameter count: '+str(json.loads((out/'parameter_counts.json').read_text()))+'. gps_only zeros only the six normalized sensor values; shared features remain available.',
      f"AdamW lr={cfg['learning_rate']}, weight_decay={cfg['weight_decay']}, batch={cfg['batch_size']}, max_epochs={cfg['max_epochs']}, patience={cfg['patience']}, Huber delta={cfg['huber_delta_m']}m. Both methods start from scratch with seed0 and identical frozen augmentation/order. Selection: {cfg['validation_criterion']}; strict decrease, min_delta=0. No test selection.",
      table(hist),
      '## Metric definitions',
      'Primary = sqrt(mean(||(prediction−last_legal_GNSS_position)−(GT(t)−GT(last_legal_fix_time))||²)) inside each artificial outage. The anchor GT is interpolated only for the target/reference; scoring rows are original GT timestamps. The state/position is never initialized from GT. Predictions at native GT times use the most recent causal token plus its velocity over the remaining fraction of dt. Absolute3D/H/V and final unavailable error use raw positions without alignment. CV uses least-squares velocity from the last5s of permitted fixes. No new hidden GNSS values or covariance enter outage features.',
      'Summary values are macro arithmetic means of per-episode RMSE, not pooled-point RMSE. Nested durations/overlapping episodes are dependent. One seed, two validation flights, same vehicle/campaign: no confidence interval, significance or generalization claim.',
      '## All fixed validation scenarios',
      table(d[['scenario','method','status','n_native_gt','relative_motion_rmse3d_m','absolute_rmse3d_m','absolute_rmse_h_m','absolute_rmse_v_m','final_unavailable_error_m','recovery_first_5s_rmse3d_m']]),
      '## By flight and duration — primary RMSE in metres',table(flights),
      '## By duration — primary RMSE in metres',table(means),
      '## Paired full comparisons',table(pair),
      f'Full is lower than matched gps_only in **{wins}/{len(main)}** outage scenarios. Negative full-minus-other means lower full error. Percent =100×(full−other)/other. No cherry-picking of epochs beyond the predefined early-stopping criterion or of favorable scenarios.',
      '## ESKF readiness',
      'A separate 15-state ESKF (p,v,right attitude error,ba,bg) has synthetic stationary/constant-velocity/turn/irregular-dt/Joseph-update tests. Real-data ESKF is **not ready**: '+ '; '.join(readiness['blockers'])+'. Its rows are explicitly not_ready with no numeric metrics. Unknown heading is not replaced by GNSS course, GT or an arbitrary confident yaw. No network superiority over a calibrated ESKF is claimed. Noise values in config are synthetic engineering defaults, not fitted flight calibration.',
      '## Runtime / recovery',
      'Environment: `'+json.dumps(env,ensure_ascii=False)+'`. Model inference_ms is one measured synchronized forward/readout on that hardware, excludes preprocessing/training and is not a repeated latency benchmark. Held/CV rows share the measured time of their joint calculation; those numbers are not separately timed per-baseline latencies. Recovery is first5s after nominal outage end; first_recovered_fix_error uses the first scored point after an actual returned fix. Decoder returns to legal GNSS on availability; no post-outage trajectory alignment.',
      table(d[d.status=='complete'][['scenario','method','inference_ms','first_recovered_fix_error_m']]),
      '## Conclusion and next action',
      'This pilot estimates whether IMU helps under this fixed small-data protocol. A small difference is not proof of IMU utility. Inspect the complete paired table, duration means and replication across flights; a single-seed validation-selected result cannot establish generalization. **STOP. Do not run more seeds, change metrics, or swap flights automatically.**',
      'Artifacts: config/splits/outage/training manifests, train-only scaler, source hashes, environment, best/last checkpoints, training history, each native-timestamp prediction/reference/mask, summary.csv, paired_differences.csv, preflight.json, eskf_status.json, status.json. All test flights remain reserved.'
    ]
    (out/'INSANE_PILOT_REPORT.md').write_text('\n\n'.join(sections)+'\n')
