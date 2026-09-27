# METHODS_DRAFT — final held-out INSANE evaluation

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

| model_seed | last_epoch | best_epoch | best_validation_criterion_m |
|---|---|---|---|
| adaptive_full_seed0 | 30 | 26 | 11.575646 |
| adaptive_gps_only_seed0 | 23 | 18 | 14.576295 |
| adaptive_full_seed1 | 23 | 18 | 12.542395 |
| adaptive_gps_only_seed1 | 23 | 18 | 14.576983 |
| adaptive_full_seed2 | 30 | 30 | 13.074474 |
| adaptive_gps_only_seed2 | 23 | 18 | 14.616175 |

## Locked final evaluation

Before any prediction or baseline metric, FINAL_EVALUATION_LOCK.json fixed checkpoint/config/scaler/source hashes, the τ=5s Damped-CV parameter, methods, metric definitions, aggregation and the deterministic test manifest. The selector matches the original validation algorithm: valid continuous spans from unchanged IMU/GNSS/GT timestamp gap limits,20s history,10s recovery, onset stride60s and durations10/30/60s. Onsets are snapped to existing causal token timestamps. All durations share onsets where60s fits; otherwise the original earliest-supported-interval fallback applies and unsupported durations are recorded. Controls use the longest valid segment. No movement/error selection, flight stitching, synthetic GT points or padding-as-data is allowed.

Each selected scenario is evaluated once by each of the six frozen models and once by Held-GNSS, CV and Damped-CV. Damped-CV uses v(age)=v_CV exp(−age/5), with analytic displacement v_CV×5×[1−exp(−age/5)] from the last legal fix. The time constant was chosen only in the previous train-stage search; no test fitting is performed. ESKF is not_ready and has no test scores.

## Metrics and aggregation

Prediction at each original GT timestamp uses only the latest past model token and its velocity over the remaining partial interval. Primary relative-motion RMSE3D is sqrt(mean(||[pred−last legal GNSS]−[GT(t)−GT(last legal fix time)]||²)) strictly within outage. Only the target anchor is interpolated; scoring observations are original rows. Additional metrics are raw absolute3D/H/V RMSE, final unavailable error, final relative-motion error, first5s recovery RMSE and first scored error after actual fix return. Controls have N/A outage metrics. Every method uses identical native GT points/masks; no fitted trajectory alignment is applied.

All episode/flight/seed rows are presented before aggregation. The main summary averages episode RMSE within each seed and duration, then gives mean and sample SD of the three seed means. Flight-specific and seed-specific summaries are retained. Deterministic baselines are not duplicated across seeds. SD is not a confidence interval; repeated seeds and nested outages are not independent flights. Validation results remain separate from test.

## Execution, verification and limitations

The final job completed 48 model evaluations and 24 baseline evaluations over 8 scenarios, with zero training/optimizer steps. Unsupported entries: []. Runtime environment: `{"python": "3.12.13", "torch": "2.10.0+cu128", "numpy": "2.0.2", "pandas": "2.3.3", "device": "cuda", "cuda": "12.8", "hardware": "Tesla T4"}`. Every NPZ metric and paired comparison was independently checked; raw reference/source provenance and before/after weight hashes matched. Exact Kaggle identity is attached in the final receipts.

Two test flights from the same vehicle/campaign do not establish generalization beyond these recordings. The adaptive hypothesis was developed on earlier validation results, but no test-dependent revision was made. No matched ungated three-seed control exists; therefore an isolated learned-gate effect cannot be claimed. Gate traces describe attenuation, not stop detection or causal explanation. Missing ordinary-GNSS extrinsics and postprocessed GT limit absolute interpretation. No superiority over INS/EKF, proven novelty or publication-readiness claim follows automatically. Execution stops after this single test job.

## Technical preflight repair and execution identity

The first job stopped before inference on brittle whole-JSON equality. One explicitly authorized retry changed only diagnostic manifest comparison: projection_error_m uses absolute tolerance1e-8m and rtol0; all scientific fields/source hashes stay exact. The original evaluation lock and scientific source files remain byte-identical; separate PREFLIGHT_PATCH_MANIFEST pins the preflight wrapper. Both regenerated manifests/diffs are saved. The first job produced zero evaluations; the retry produced the first held-out scores.

`{"kernel": "shapok/drone-nav-insane-final-test", "version": 2, "output_run_ids": ["353132214"], "status": "complete", "model_evaluations": 48, "baseline_evaluations": 24, "training_performed": false, "authorized_retry_after_failed_run": "353128919", "evaluation_passes": 1, "downloaded_artifacts": 148, "downloaded_bytes": 5583197}`.
