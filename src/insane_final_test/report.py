"""All test cases first, then descriptive aggregation; no validation pooling."""
from pathlib import Path
import json,math
import numpy as np
import pandas as pd
from insane_adaptive.report import table,aggregate

PRIMARY='relative_motion_rmse3d_m'

def generate(out):
    out=Path(out);d=pd.read_csv(out/'summary.csv');pairs=pd.read_csv(out/'paired_differences.csv')
    manifest=json.loads((out/'test_manifest.json').read_text());status=json.loads((out/'status.json').read_text());lock=json.loads((out/'FINAL_EVALUATION_LOCK.json').read_text())
    env=json.loads((out/'environment.json').read_text());ver=json.loads((out/'independent_verification.json').read_text())
    duration,seed_duration=aggregate(d,['duration_s']);flight,seed_flight=aggregate(d,['flight','duration_s'])
    for name,frame in [('duration_summary',duration),('per_flight_summary',flight),('per_seed_summary',seed_duration),('per_flight_per_seed_summary',seed_flight)]:frame.to_csv(out/(name+'.csv'),index=False)
    imu=pairs[(pairs.method=='adaptive_full')&(pairs.reference=='adaptive_gps_only')]
    wins=imu.groupby('seed').better.agg(['sum','count']).reset_index();wins.columns=['seed','full_wins','outage_episodes']
    ep=imu.groupby(['flight','scenario','duration_s']).agg(mean_difference_m=('difference_m','mean'),seed_sd_difference_m=('difference_m','std'),full_wins_of_3=('better','sum')).reset_index()
    ep.to_csv(out/'paired_episode_summary.csv',index=False)
    contrasts=[]
    for (seq,seconds,method,ref),q in pairs.groupby(['flight','duration_s','method','reference'],sort=False):
        a=float(q.rmse_m.mean());b=float(q.reference_rmse_m.mean())
        contrasts.append({'flight':seq,'duration_s':seconds,'method':method,'reference':ref,'mean_m':a,'reference_mean_m':b,
            'difference_m':a-b,'difference_percent':100*(a-b)/b if b else None,'wins_episode_seed':int(q.better.sum()),'pairs':len(q)})
    contrast=pd.DataFrame(contrasts);contrast.to_csv(out/'per_flight_comparisons.csv',index=False)
    text='# INSANE FINAL TEST REPORT\n\n'
    text+=f"Status: **{status['status']}**. Один финальный test-only job: {status['model_evaluations']} model×scenario evaluations + {status['baseline_evaluations']} deterministic baseline×scenario evaluations, всего {len(d)}. Обучение/optimizer steps: 0; repeats: 0. Run ID и Kaggle version — в download/submission receipt. Test: mars_6/mars_7; validation в эти метрики не включена.\n\n"
    text+='## 1. Все test-полёты, эпизоды и seeds\n\nPrimary — relative-motion RMSE3D внутри outage, метры. Baselines представлены один раз; seed пуст. Controls без искусственного outage сохраняются с N/A в outage-метриках. Ничего не исключено по ошибкам или движению.\n\n'
    text+=table(d[['flight','scenario','method','seed','duration_s','n_native_gt',PRIMARY,'absolute_rmse3d_m','absolute_rmse_h_m','absolute_rmse_v_m','final_unavailable_error_m','final_relative_motion_error_m','recovery_first_5s_rmse3d_m','first_recovered_fix_error_m']])
    text+='\n## 2. Агрегация\n\nСначала среднее episode RMSE каждой длительности внутри seed; затем mean и sample SD (ddof=1) трёх seed-средних. SD не является confidence interval. Baselines детерминированы: n_seeds=0, SD=N/A. Наложенные длительности и seeds не являются независимыми полётами.\n\n'+table(duration)+'\n'+table(flight)+'\n'+table(seed_duration)+'\n'+table(seed_flight)
    text+='\n## 3. A — full против matched gps_only\n\n'+table(wins)+'\n'+table(ep)+'\n'+table(contrast[contrast.reference=='adaptive_gps_only'])
    n=int(imu.better.sum());total=len(imu);all3=int((ep.full_wins_of_3==3).sum())
    text+=f'\nFull имеет меньшую ошибку в **{n}/{total}** episode×seed сравнениях; в **{all3}/{len(ep)}** эпизодах выигрывает при всех трёх seeds. Знак full−gps_only показан для каждого test-полёта и длительности: отрицательный означает меньшую ошибку full. Эти counts описывают две отложенные записи, а не статистическое доказательство generalization.\n'
    text+='\n## 4. B — сравнение с Held/CV/Damped-CV по полётам\n\n'+table(contrast[(contrast.method=='adaptive_full')&(contrast.reference!='adaptive_gps_only')])
    text+='\nПреимущество full над gps_only само по себе не означает превосходство над простыми baselines. Полётные таблицы имеют приоритет перед общим средним для этой интерпретации. Damped-CV τ=5 с зафиксировано ранее по train; здесь не подбиралось.\n'
    text+='\n## 5. C — научные выводы и ограничения\n\nПоложительный matched contrast может поддержать пользу IMU в этих конкретных полётах; ухудшения, смена знака между seeds/полётами и выигрыши baselines ограничивают этот вывод. Результаты test не использованы для изменения модели, выбора seed, ensemble или новых параметров. Все шесть checkpoints выбраны ранее на validation. Два полёта одной кампании/платформы — ограниченный перенос, не доказательство общей применимости. Validation и test не объединяются.\n\nИзолированный эффект gate не установлен: matched ungated control по трём seeds не проводился. g(t) — коэффициент управления прогнозом движения, не детектор остановки и не причинное объяснение результата. ESKF остаётся not_ready: отсутствуют документированное плечо ordinary GNSS и проверенная heading initialization; чисел ESKF нет. Нельзя заявлять превосходство над INS/EKF, доказанную новизну или готовность к Q2 на основании этих метрик.\n'
    text+='\n## 6. Фиксация и проверки\n\n`FINAL_EVALUATION_LOCK.json` содержит SHA256 шести best checkpoints, scalers/config, исходников, raw/calibration файлов и test manifest; он сохранён до inference. Scalers повторно не обучались. Mask-first GNSS features, causal IMU aggregation, actual dt, native GT support и разрешённая история остались прежними. В новом CSV split=test; старые CSV не изменялись.\n\n'
    text+='Test manifest: '+str(len(manifest['scenarios']))+' сценариев; unsupported: `'+json.dumps(manifest['unsupported'],ensure_ascii=False)+'`. Выбор: chronological onset stride60s, history20s/recovery10s, общие начала10/30/60s, прежние gap thresholds; fallback на поддерживаемые длительности только при отсутствии общего60s интервала, как в исходном алгоритме. Контроль — самый длинный непрерывный интервал каждого полёта. Нет padding наблюдений/склейки полётов.\n\n'
    text+='Общие точки/маски, все восемь метрик из всех NPZ, paired deltas/percent, GT/GNSS native row provenance и g-интеграция независимо проверены. Веса в памяти и SHA256 файлов после inference совпали с зафиксированными. Verification: `'+json.dumps(ver)+'`.\n\n'
    text+='Reference — опубликованная авторами dual-RTK/magnetometer позиция PX4 IMU. GT attitude не поступает в модель. Смещение ordinary GNSS antenna→IMU не документировано, поэтому остаётся ограничение геометрии/абсолютных ошибок. Существующие time corrections не применяются повторно. Primary использует один interpolated target anchor в момент последнего legal fix, но все оцениваемые GT timestamps исходные. Ни fitted alignment, ни GT reset нет.\n'
    text+='\n## 7. Поведение gate и ресурсы\n\n![All test gate traces](gate_traces.png)\n\n'+table(pd.read_csv(out/'gate_summary.csv'))
    text+='\nОкружение: `'+json.dumps(env)+'`. Inference_ms — один synchronized forward/readout, не повторный latency benchmark. Held/CV разделяют время совместного расчёта. При возврате legal GNSS позиционный decoder использует legal fix, скрытое GRU-состояние сохраняется.\n\n**STOP: финальный test завершён; повторного test или следующей версии автоматически не запускать.**\n'
    (out/'INSANE_FINAL_TEST_REPORT.md').write_text(text)
    # A distinct methods version; original METHODS_DRAFT.md is retained untouched.
    methods='''# METHODS_DRAFT — final held-out INSANE evaluation

This version extends the completed adaptive experiment with one locked test-only evaluation. It supersedes the previous statement that test was not yet evaluated; the previous draft is preserved as METHODS_BEFORE_TEST.md. No new training, model revision or calibration search was performed at the final stage.

## Data and split

INSANE exports px4_imu.csv, px4_gps.csv and ground_truth_8hz.csv were used with the previously downloaded calibration and provenance manifests. Train was mars_1–3, validation mars_4–5. Previously reserved mars_6 and mars_7 form the final test. The files were originally downloaded and inventoried for schema/timing only; no prior model scores were calculated on them. Only these two raw flights were included in the new private Kaggle input, together with frozen model artifacts and calibration/license metadata. No images were required.

Sources remain the [INSANE dataset](https://www.aau.at/en/smart-systems-technologies/control-of-networked-systems/datasets/insane-dataset/), [authors' pinned tools](https://github.com/aau-cns/insane_dataset_tools/tree/9a1c8c0fdd195f2d869fff292f2ce5b273c5a03d), and [dataset publication](https://arxiv.org/abs/2210.09114). No new bibliographic claims are added.

## Inputs, synchronization and reference

The unchanged adapter uses PX4 accelerometer specific force in m/s² and gyroscope angular rate in rad/s in published MAVROS body axes. Native IMU is causally time-weighted aggregated to20Hz using past-held samples, not row decimation. Ordinary GNSS supplies permitted position history/covariance in the documented ENU frame. One common first-ordinary-GNSS origin is subtracted; no GT alignment is fitted. Exported timing corrections are already applied and are not applied twice. IMU normalization reuses the exact saved train-only scaler fitted on mars_1–3.

Final GNSS masks precede all GNSS-derived displacement/velocity/quality features. CV velocity uses the preceding permitted5s GNSS history and is frozen during outage. GT position is an authors' postprocessed dual-RTK/magnetometer reference at the PX4 IMU origin. RTK, magnetometer, GT position/attitude and onboard pose are never model inputs or initial states. The absent ordinary-GNSS antenna lever arm remains a reference-point limitation.

## Model and previously completed training

The encoder is unchanged: IMU6→Linear16+tanh, GNSS6→Linear16+tanh, common timing/quality10→Linear8+tanh, concat40→causal GRU32. Heads output residual velocity3 and scalar sigmoid g. Each variant has7548 parameters. adaptive_gps_only zeros only six normalized IMU sensor values; common availability, age, dt and quality features remain shared. Predicted velocity is g(t)[v_CV+residual(t)]. Its actual-dt prefix integral carries displacement through outages; g=0 freezes it and g=1 equals the ungated decoder. The first absent token integrates lag since the last legal fix. At fix return the position decoder resumes the legal GNSS position while the recurrent state persists. Separate episodes have zero hidden state plus20s permitted history, never GT initialization/reset.

Previously, six models were trained from scratch: paired full/gps_only for seeds0/1/2 with matched initial weights and episode order. AdamW lr0.001, weight_decay0.0001, batch4, clip1, native-relative-motion Huber delta1m, max30 epochs, patience5. The best checkpoint used strict improvement of the nine-episode validation macro RMSE. Final test reuses all six selected checkpoints; no seed selection or ensemble is introduced.

'''
    old=json.loads((out/'adaptive_status.json').read_text());methods+=table(pd.DataFrame([{'model_seed':k,**v} for k,v in old['models'].items()]))
    methods+='''
## Locked final evaluation

Before any prediction or baseline metric, FINAL_EVALUATION_LOCK.json fixed checkpoint/config/scaler/source hashes, the τ=5s Damped-CV parameter, methods, metric definitions, aggregation and the deterministic test manifest. The selector matches the original validation algorithm: valid continuous spans from unchanged IMU/GNSS/GT timestamp gap limits,20s history,10s recovery, onset stride60s and durations10/30/60s. Onsets are snapped to existing causal token timestamps. All durations share onsets where60s fits; otherwise the original earliest-supported-interval fallback applies and unsupported durations are recorded. Controls use the longest valid segment. No movement/error selection, flight stitching, synthetic GT points or padding-as-data is allowed.

Each selected scenario is evaluated once by each of the six frozen models and once by Held-GNSS, CV and Damped-CV. Damped-CV uses v(age)=v_CV exp(−age/5), with analytic displacement v_CV×5×[1−exp(−age/5)] from the last legal fix. The time constant was chosen only in the previous train-stage search; no test fitting is performed. ESKF is not_ready and has no test scores.

## Metrics and aggregation

Prediction at each original GT timestamp uses only the latest past model token and its velocity over the remaining partial interval. Primary relative-motion RMSE3D is sqrt(mean(||[pred−last legal GNSS]−[GT(t)−GT(last legal fix time)]||²)) strictly within outage. Only the target anchor is interpolated; scoring observations are original rows. Additional metrics are raw absolute3D/H/V RMSE, final unavailable error, final relative-motion error, first5s recovery RMSE and first scored error after actual fix return. Controls have N/A outage metrics. Every method uses identical native GT points/masks; no fitted trajectory alignment is applied.

All episode/flight/seed rows are presented before aggregation. The main summary averages episode RMSE within each seed and duration, then gives mean and sample SD of the three seed means. Flight-specific and seed-specific summaries are retained. Deterministic baselines are not duplicated across seeds. SD is not a confidence interval; repeated seeds and nested outages are not independent flights. Validation results remain separate from test.

## Execution, verification and limitations

'''
    methods+=f"The final job completed {status['model_evaluations']} model evaluations and {status['baseline_evaluations']} baseline evaluations over {len(manifest['scenarios'])} scenarios, with zero training/optimizer steps. Unsupported entries: {json.dumps(manifest['unsupported'])}. Runtime environment: `{json.dumps(env)}`. Every NPZ metric and paired comparison was independently checked; raw reference/source provenance and before/after weight hashes matched. Exact Kaggle identity is attached in the final receipts.\n\n"
    methods+='''Two test flights from the same vehicle/campaign do not establish generalization beyond these recordings. The adaptive hypothesis was developed on earlier validation results, but no test-dependent revision was made. No matched ungated three-seed control exists; therefore an isolated learned-gate effect cannot be claimed. Gate traces describe attenuation, not stop detection or causal explanation. Missing ordinary-GNSS extrinsics and postprocessed GT limit absolute interpretation. No superiority over INS/EKF, proven novelty or publication-readiness claim follows automatically. Execution stops after this single test job.
'''
    (out/'METHODS_DRAFT_FINAL_TEST.md').write_text(methods)
    results='# RESULTS_DRAFT — held-out INSANE test\n\nThis draft reports the locked test run only; validation values are not pooled. The primary metric is native-timestamp relative-motion RMSE3D inside outage, metres.\n\n'
    results+='## All observations\n\n'+table(d[['flight','scenario','method','seed','duration_s',PRIMARY,'absolute_rmse3d_m','absolute_rmse_h_m','absolute_rmse_v_m','final_unavailable_error_m']])
    results+='\n## Duration and flight summaries\n\nMean and sample SD across three seed-level episode means; SD is not a confidence interval.\n\n'+table(duration)+'\n'+table(flight)
    results+='\n## Matched sensor ablation\n\n'+table(wins)+'\n'+table(ep)
    results+=f'\nFull is lower than gps_only in {n}/{total} episode–seed pairs, with all-three-seed wins in {all3}/{len(ep)} episodes. These are repeated trainings evaluated on two flights, not independent replicated test flights.\n'
    results+='\n## Comparison with deterministic baselines by flight\n\n'+table(contrast[(contrast.method=='adaptive_full')&(contrast.reference!='adaptive_gps_only')])
    results+='\n## Interpretation limits\n\nAll fixed episodes and seeds are retained, including unfavorable differences. A full/gps_only difference is distinct from an advantage over Held/CV/Damped-CV. Results support or limit claims only within these two held-out flights. No test-driven checkpoint selection, tuning or retraining occurred. No isolated gate effect, stop-detector behavior, INS/EKF superiority, statistical significance or generalization beyond these recordings is established. Exact scientific interpretation must respect the signs and magnitudes in the per-flight tables.\n'
    (out/'RESULTS_DRAFT.md').write_text(results)
    plot_gates(out,manifest)

def plot_gates(out,manifest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    specs=[s for s in manifest['scenarios'] if s['duration_s']]
    if not specs:return
    fig,axes=plt.subplots(math.ceil(len(specs)/3),3,figsize=(15,3.3*math.ceil(len(specs)/3)),squeeze=False,sharey=True)
    for ax,s in zip(axes.flat,specs):
        for method,color in [('adaptive_full','tab:blue'),('adaptive_gps_only','tab:orange')]:
            for seed,style in zip([0,1,2],['-','--',':']):
                with np.load(out/'predictions'/s['id']/f'{method}_seed{seed}.npz',allow_pickle=False) as z:
                    ax.plot(z['all_token_t']-s['onset_s'],z['g_token'],color=color,ls=style,lw=1,label=f'{method}, seed{seed}')
        ax.axvspan(0,s['duration_s'],color='grey',alpha=.12);ax.set_title(s['id']);ax.set_ylim(0,1);ax.set_xlabel('Time from outage onset (s)');ax.set_ylabel('g(t)')
    for ax in list(axes.flat)[len(specs):]:ax.set_visible(False)
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=8)
    fig.suptitle('Held-out test gate traces: descriptive attenuation, not stop detection');fig.tight_layout(rect=[0,.09,1,.95]);fig.savefig(out/'gate_traces.png',dpi=160);plt.close(fig)
