"""Descriptive report from every saved episode/seed; no scientific claim selection."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

PRIMARY='relative_motion_rmse3d_m'

def table(df):
    def fmt(x):
        if pd.isna(x):return '—'
        if isinstance(x,(float,np.floating)):return f'{x:.6f}'
        return str(x).replace('|','/')
    return '| '+' | '.join(df.columns)+' |\n|'+ '|'.join(['---']*len(df.columns))+'|\n'+''.join('| '+' | '.join(fmt(v) for v in row)+' |\n' for row in df.itertuples(index=False,name=None))

def aggregate(df,keys):
    valid=df[df.duration_s>0];model=valid[valid.method.str.startswith('adaptive_')]
    # First macro-average episodes within seed, then summarize variation of seed means.
    seed_means=model.groupby(keys+['method','seed'])[PRIMARY].mean().reset_index()
    neural=seed_means.groupby(keys+['method'])[PRIMARY].agg(mean_m='mean',seed_sd_m='std',n_seeds='count').reset_index()
    baseline=valid[~valid.method.str.startswith('adaptive_')].groupby(keys+['method'])[PRIMARY].mean().reset_index(name='mean_m')
    baseline['seed_sd_m']=np.nan;baseline['n_seeds']=0
    return pd.concat([neural,baseline],ignore_index=True).sort_values(keys+['method']),seed_means

def generate(out):
    out=Path(out);d=pd.read_csv(out/'summary.csv');pairs=pd.read_csv(out/'paired_differences.csv');hist=pd.read_csv(out/'training_history.csv')
    status=json.loads((out/'status.json').read_text());cfg=json.loads((out/'config.json').read_text());damp=json.loads((out/'damping_lock.json').read_text())
    env=json.loads((out/'environment.json').read_text());gates=pd.read_csv(out/'gate_summary.csv')
    by_duration,seeds=aggregate(d,['duration_s']);by_flight,flight_seeds=aggregate(d,['flight','duration_s'])
    by_duration.to_csv(out/'duration_summary.csv',index=False);by_flight.to_csv(out/'flight_duration_summary.csv',index=False)
    seeds.to_csv(out/'seed_duration_summary.csv',index=False);flight_seeds.to_csv(out/'seed_flight_duration_summary.csv',index=False)
    imu=pairs[(pairs.method=='adaptive_full')&(pairs.reference=='adaptive_gps_only')]
    imu_ep=imu.groupby(['flight','scenario','duration_s']).agg(mean_delta_m=('difference_m','mean'),seed_sd_delta_m=('difference_m','std'),wins_of_3=('better','sum')).reset_index()
    # Percentage aggregates are ratio of macro means; do not average episode percentages.
    comparison=[]
    for (flight,duration,method,ref),q in pairs.groupby(['flight','duration_s','method','reference'],sort=False):
        a=float(q.rmse_m.mean());b=float(q.reference_rmse_m.mean())
        comparison.append({'flight':flight,'duration_s':duration,'method':method,'reference':ref,'mean_rmse_m':a,'reference_mean_rmse_m':b,
          'delta_m':a-b,'delta_percent':100*(a-b)/b if b else None,'wins_episode_seed':int(q.better.sum()),'episode_seed_pairs':len(q)})
    comparisons=pd.DataFrame(comparison);comparisons.to_csv(out/'flight_comparisons.csv',index=False)
    imu_ep.to_csv(out/'imu_episode_seed_summary.csv',index=False)
    title='# INSANE adaptive experiment\n\n'
    text=title+f"Status **{status['status']}**. Six scheduled trainings, seeds 0/1/2; no test use. Kernel `shapok/drone-nav-insane-adaptive`; exact version/output ID in launch receipt. Code SHA256 `{status['code_version']}`.\n\n"
    text+='## Two separate questions\n\nA. Does adaptive prediction improve on Held, CV and train-selected Damped-CV? All flight/duration comparisons below use every fixed episode and all three seeds. Lower is better; improvements must be judged separately on each flight.\n\n'
    text+=table(comparisons[comparisons.reference!='adaptive_gps_only'])
    text+='\nB. Does IMU improve matched gps_only?\n\n'+table(comparisons[comparisons.reference=='adaptive_gps_only'])
    counts=imu.groupby('seed').better.agg(['sum','count']).reset_index();counts.columns=['seed','full_wins','outage_episodes']
    text+='\n'+table(counts)+'\n'+table(imu_ep)
    text+=f"\nAdaptive full has lower error in {int(imu.better.sum())}/{len(imu)} episode–seed pairs. All-three-seed wins occur in {int((imu_ep.wins_of_3==3).sum())}/9 fixed episodes. Three seeds are repeat trainings on the same two validation flights, not independent flights or evidence of generalization. No practical-effect threshold or significance claim is invented after results.\n"
    text+='\n## Every episode and seed, before aggregate tables\n\nDeterministic baselines appear once per episode; their seed is blank. Controls are retained, primary outage metric is undefined for them.\n\n'
    text+=table(d[['flight','scenario','method','seed','duration_s','n_native_gt',PRIMARY,'absolute_rmse3d_m','absolute_rmse_h_m','absolute_rmse_v_m','final_unavailable_error_m','recovery_first_5s_rmse3d_m']])
    text+='\n## Aggregates\n\nFirst average episode RMSE within a seed, then report mean and sample SD (ddof=1) of the three seed means. SD is not a confidence interval. Episode durations are nested, episodes can overlap. Overall duration means weight three episodes equally (two mars_4, one mars_5); flight tables make that imbalance explicit. Baselines have no seed SD.\n\n'+table(by_duration)+'\n'+table(by_flight)
    text+='\n## Gate behavior\n\n![Every gate trace](gate_traces.png)\n\nEach panel shows all six models on one fixed outage; traces include permitted history and recovery, shaded region is the outage. Gate observations describe prediction control, not a stop detector or causal explanation. Complete token timestamps, gate, residual velocity, integrated velocity/motion and native predictions are in NPZ.\n\n'+table(gates)
    text+='\n## Damped-CV selection\n\n'+f"Tau = **{damp['tau_s']:g} s**, selected exclusively on 540 frozen train episodes before any adaptive validation metric. Fixed candidates: {cfg['damping']['tau_candidates_s']}. Criterion: train macro relative-motion RMSE3D, smaller tau on ties. v(age)=v_CV exp(-age/tau), displacement=v_CV tau(1−exp(-age/tau)); age starts at last legal fix. This is a bounded baseline calibration, not a neural hyperparameter search.\n\n"+table(pd.read_csv(out/'damping_train_selection.csv'))
    text+='\n## Training and checks\n\nThe encoder and all original preprocessing/scenario/scaler hashes are preserved. Both variants have 7548 parameters. For each seed, initial tensor hashes and frozen training episode order match. AdamW lr0.001, weight_decay0.0001, batch4, clip1, Huber delta1m, max30 epochs, patience5, strict decrease in fixed validation macro RMSE, selected best checkpoint. Histories include all attempted epochs; same episode prefix is consumed until each model independently early-stops.\n\n'
    text+=table(pd.DataFrame([{'model_seed':k,**v} for k,v in status['models'].items()]))
    text+='\nPreflight validates source/calibration hashes, ENU projection, train-only scaler, all 540 train and 11 validation scenarios, exact native timestamps and masks. Mandatory original and adaptive tests cover gate0/gate1, irregular dt, state carry, future input perturbation, hidden GNSS mutation, shared inputs and backward gradients. Logs are in launch artifacts.\n'
    text+='\n## Previous pilot — separate seed0 only\n\nThese saved results are descriptive context; they are never pooled with new seeds. No ungated seeds1/2 were trained, so a controlled three-seed estimate of the isolated gate effect is unavailable.\n\n'
    previous=pd.read_csv(out/'previous_pilot_seed0.csv');previous=previous[(previous.duration_s>0)&previous.method.isin(['full','gps_only'])]
    text+=table(previous[['flight','scenario','method','seed',PRIMARY]])
    text+='\n## Metric / data / reference limitations\n\nNative GT timestamps are scored without trajectory fitting. Primary is sqrt(mean(norm((pred−last legal GNSS)−(GT(t)−GT(last legal fix time)))²)) inside outage. Only the anchor is interpolated as a target/reference; no interpolated GT samples are counted as observations. Raw absolute3D/H/V, final unavailable error and recovery-first5s are saved. No GT attitude/position or RTK enters the encoder or initialization. Original timestamp corrections are already applied. GT is authors’ dual-RTK/magnetometer-derived PX4 IMU reference; ordinary GNSS antenna lever arm is missing. Filename 8hz does not guarantee 8 independent points/s. Test mars_6/7 remains absent from the Kaggle data.\n\n'
    text+='ESKF remains **not_ready**, with no invented scores: missing ordinary-GNSS lever arm and validated absolute heading initialization. No superiority over INS/EKF is claimed. Validation selects checkpoints and is reused for this follow-up hypothesis after viewing the prior pilot; this is exploratory validation, not untouched confirmatory testing.\n\n'
    text+='Environment: `'+json.dumps(env)+'`. Saved inference_ms is synchronized single forward/readout, not repeated benchmark; Held/CV share their joint timing. No training or full evaluation on Mac.\n\n**STOP. No test, additional seeds/jobs, parameter changes or automatic next version.**\n'
    (out/'INSANE_ADAPTIVE_REPORT.md').write_text(text)
    methods=f'''# Methods draft — completed INSANE adaptive experiment

This draft describes the executed exploratory protocol; it does not establish architectural novelty or guarantee improved navigation.

## Data and split

The existing INSANE Sensor Data exports and author-provided calibrations were reused. Inputs are `px4_imu.csv` accelerometer and gyroscope (specific force m/s², angular rate rad/s, published PX4 MAVROS body axes) and ordinary `px4_gps.csv` positions/covariance. GT is `ground_truth_8hz.csv` position at the PX4 IMU reference, derived by the dataset authors from dual RTK and magnetometer geometry. No RTK, magnetometer, onboard pose, GT orientation or GT position enters learned inputs or initial state. Train flights are mars_1–3; validation mars_4–5. Test mars_6–7 was neither uploaded nor evaluated. Outdoor_1 was used only in the preceding loader check.

Public sources: [dataset](https://www.aau.at/en/smart-systems-technologies/control-of-networked-systems/datasets/insane-dataset/), [authors' tools, pinned revision](https://github.com/aau-cns/insane_dataset_tools/tree/9a1c8c0fdd195f2d869fff292f2ce5b273c5a03d), [dataset publication](https://arxiv.org/abs/2210.09114). Download manifests, calibration hashes and license were retained. No new dataset was selected for this experiment.

## Causal preparation

The unchanged pilot adapter aggregates native approximately 200Hz IMU to 20Hz by time-weighted integration of past-held measurements, not row dropping. Published timestamp corrections are not applied twice. WGS84 GPS positions are checked against documented ENU coordinates, with one common first-ordinary-GNSS origin per flight, no GT-fitted alignment. The unknown ordinary-GNSS-to-IMU lever arm is not guessed. No GT rotation or additional intrinsic correction is applied to IMU.

The final outage mask is applied to native GNSS before position deltas, CV velocity, covariance, availability and fix age are computed. CV uses the most recent permitted 5s GNSS history. No hidden GNSS fix enters derived inputs. IMU scalers are fitted on train flights only. The exact previous episode manifests, native GT row support and mask hashes are preserved. All 540 training episodes (18/epoch for 30 epochs) are shared in the same order across seeds/variants; only initial weights vary across seeds. Validation has nine fixed outages (three each of 10/30/60s), plus two GNSS controls, with 20s history and 10s recovery; two onset positions are on mars_4 and one on mars_5. Flights are never concatenated.

## Pilot architecture and integration

IMU6→Linear16+tanh; GNSS displacement/velocity6→Linear16+tanh; shared quality/time10→Linear8+tanh; concatenated40→causal GRU32. The retained residual head maps hidden32→velocity3. The sole architectural addition maps hidden32→scalar sigmoid gate g∈[0,1]. Residual head starts at zero and gate head at zero weight/bias (g=0.5). Each variant has 7548 parameters. gps_only zeros only the six normalized IMU sensor values; availability, fix age, actual dt, IMU timing/count and GNSS quality remain shared. Paired variants begin with identical tensor hashes for each seed.

Velocity is g(t)[v_CV+residual(t)]. Position integrates this velocity by the actual per-step time interval; the first unavailable token includes lag since the last legal fix. Zero gate freezes accumulated displacement rather than resetting it; gate1 reduces to the original decoder. The state is carried through outages. On legal GNSS return the position decoder resumes that permitted fix while GRU state persists. Independent episodes start from zero hidden state with permitted history; no true-state resets occur. At a native GT timestamp, prediction uses the latest past token and its velocity over the remaining partial interval. The gate uses only the causal hidden state, never GT at inference. It is prediction attenuation, not an established drone stop detector.

## Training and checkpoint choice

Six from-scratch trainings were scheduled in one Kaggle job: adaptive_full and adaptive_gps_only for seeds0,1,2. AdamW lr=0.001, weight_decay=0.0001, batch4, gradient clip1, Huber delta1m on native outage relative-motion targets; max30 epochs, patience5. Batching padding contributes no reference measurements or loss terms. The target is GT(t)−GT(last legal fix time); the single anchor may be interpolated for the reference only. No checkpoint from Zurich or the previous INSANE pilot initializes these models.

The predefined selection criterion is the macro mean of the nine validation episode relative-motion RMSE3D values, strict decrease (min_delta=0). Each model retains its own best checkpoint; paired models see the same training episode prefix until independent early stopping. Actual selected/last epochs are:

'''+table(pd.DataFrame([{'model_seed':k,**v} for k,v in status['models'].items()]))+f'''
## Baselines and evaluation

Held-GNSS retains the last legal fix. CV extrapolates its permitted-history velocity. Damped-CV uses v(age)=v_CV exp(−age/tau), integrated analytically; tau={damp['tau_s']:g}s was selected only on all 540 frozen train episodes from {cfg['damping']['tau_candidates_s']}, minimizing train macro relative-motion RMSE3D before any validation scoring. No adaptive model hyperparameter sweep was performed. ESKF remains not_ready and was not numerically evaluated.

Primary metric: sqrt(mean(||(prediction−last legal GNSS position)−(GT(t)−GT(last legal fix time))||²)) on original GT rows strictly within each outage. Additional metrics are unaligned absolute3D, horizontal and vertical RMSE, final unavailable error, first5s recovery RMSE and error after actual fix return. All methods use identical native timestamps and masks. Gate traces and inference timing are saved; gates are descriptive, not causal explanations.

Reports expose each episode, flight and seed before aggregation. Episode RMSEs are averaged within seed; then mean and sample SD across three seed means are reported. SD is not a confidence interval; seeds are not independent flights. Overall means weight the three episodes equally; per-flight tables are also reported. Baselines are deterministic and not duplicated as independent seed observations. The prior pilot is reported separately as seed0 only; no estimate of an isolated gate effect across three ungated seeds is available.

## Reproducibility and limits

Run status: {status['status']}; code SHA256 `{status['code_version']}`. Environment: `{json.dumps(env)}`. Configuration, source snapshot/hashes, original manifests, scalers, train damping-selection log, initial weight hashes, histories, best/last checkpoints, prediction/reference/mask/gate arrays and all comparisons are saved. Mandatory tests precede training. Actual run identifiers are recorded in the submission/download receipts.

The two validation flights are a small correlated sample, outage durations overlap, and checkpoints are selected on these same episodes. The adaptive hypothesis follows inspection of the previous validation pilot. Thus results are exploratory and do not establish generalization, significance or physical interpretation of g. Reference postprocessing and absent ordinary-GNSS lever arm limit interpretation. No test results, INS/EKF superiority or proven novelty are claimed. After the six trainings, execution stops.
'''
    (out/'METHODS_DRAFT.md').write_text(methods)
    plot_gates(out)

def plot_gates(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    specs=[s for s in json.loads((out/'outages.json').read_text())['scenarios'] if s['duration_s']]
    fig,axes=plt.subplots(3,3,figsize=(15,10),sharey=True)
    for ax,s in zip(axes.flat,specs):
        for method,color in [('adaptive_full','tab:blue'),('adaptive_gps_only','tab:orange')]:
            for seed,style in zip([0,1,2],['-','--',':']):
                with np.load(out/'predictions'/s['id']/f'{method}_seed{seed}.npz',allow_pickle=False) as z:
                    ax.plot(z['all_token_t']-s['onset_s'],z['g_token'],color=color,ls=style,lw=1,label=f'{method}, seed{seed}')
        ax.axvspan(0,s['duration_s'],color='grey',alpha=.12);ax.set_title(s['id']);ax.set_ylim(0,1);ax.set_xlabel('Time from outage onset (s)');ax.set_ylabel('g(t)')
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=9)
    fig.suptitle('Learned motion attenuation: descriptive gate traces, not stop detection')
    fig.tight_layout(rect=[0,.06,1,.96]);fig.savefig(out/'gate_traces.png',dpi=160);plt.close(fig)
