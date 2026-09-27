# Methods draft — completed INSANE adaptive experiment

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

| model_seed | last_epoch | best_epoch | best_validation_criterion_m |
|---|---|---|---|
| adaptive_full_seed0 | 30 | 26 | 11.575646 |
| adaptive_gps_only_seed0 | 23 | 18 | 14.576295 |
| adaptive_full_seed1 | 23 | 18 | 12.542395 |
| adaptive_gps_only_seed1 | 23 | 18 | 14.576983 |
| adaptive_full_seed2 | 30 | 30 | 13.074474 |
| adaptive_gps_only_seed2 | 23 | 18 | 14.616175 |

## Baselines and evaluation

Held-GNSS retains the last legal fix. CV extrapolates its permitted-history velocity. Damped-CV uses v(age)=v_CV exp(−age/tau), integrated analytically; tau=5s was selected only on all 540 frozen train episodes from [1, 2, 5, 10, 20, 30, 60, 120], minimizing train macro relative-motion RMSE3D before any validation scoring. No adaptive model hyperparameter sweep was performed. ESKF remains not_ready and was not numerically evaluated.

Primary metric: sqrt(mean(||(prediction−last legal GNSS position)−(GT(t)−GT(last legal fix time))||²)) on original GT rows strictly within each outage. Additional metrics are unaligned absolute3D, horizontal and vertical RMSE, final unavailable error, first5s recovery RMSE and error after actual fix return. All methods use identical native timestamps and masks. Gate traces and inference timing are saved; gates are descriptive, not causal explanations.

Reports expose each episode, flight and seed before aggregation. Episode RMSEs are averaged within seed; then mean and sample SD across three seed means are reported. SD is not a confidence interval; seeds are not independent flights. Overall means weight the three episodes equally; per-flight tables are also reported. Baselines are deterministic and not duplicated as independent seed observations. The prior pilot is reported separately as seed0 only; no estimate of an isolated gate effect across three ungated seeds is available.

## Reproducibility and limits

Run status: complete; code SHA256 `ea0480ffdb853952ef1184283b480646927be865c15e83eae71778abc644560c`. Environment: `{"python": "3.12.13", "torch": "2.10.0+cu128", "numpy": "2.0.2", "pandas": "2.3.3", "device": "cuda", "cuda": "12.8", "hardware": "Tesla T4"}`. Configuration, source snapshot/hashes, original manifests, scalers, train damping-selection log, initial weight hashes, histories, best/last checkpoints, prediction/reference/mask/gate arrays and all comparisons are saved. Mandatory tests precede training. Actual run identifiers are recorded in the submission/download receipts.

The two validation flights are a small correlated sample, outage durations overlap, and checkpoints are selected on these same episodes. The adaptive hypothesis follows inspection of the previous validation pilot. Thus results are exploratory and do not establish generalization, significance or physical interpretation of g. Reference postprocessing and absent ordinary-GNSS lever arm limit interpretation. No test results, INS/EKF superiority or proven novelty are claimed. After the six trainings, execution stops.

## Execution identity

Kaggle version 1; output run ID 353123708; six completed trainings. Local integrity verification: 99 prediction files, maximum saved-metric difference 7.11e-15 m. No model forward, training or test access during verification.
