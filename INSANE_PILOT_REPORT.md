# INSANE_PILOT_REPORT

## Итог завершённого pilot

Kaggle version 3, output run ID `353114895`, GPU Tesla T4. Один завершённый training pilot: full и gps_only, seed=0, по 7515 параметров. До него версии 1/2 остановились **до обучения** на mount path и изменении вложенного ZIP при распаковке Kaggle; обе причины и исправления сохранены в `outputs/insane_pilot_launch/infrastructure_repair.json`. Проверки не обходились, научный протокол не менялся. Локально и на Kaggle прошли 18 tests.

Full: 11 epochs, выбран epoch 6; gps_only: 9 epochs, выбран epoch 4. Early stopping и выбор checkpoint выполнены по заранее заданному validation-критерию. Test не использован и не загружен в Kaggle.

Основная метрика ниже — среднее трёх episode relative-motion RMSE3D внутри outage, метры. Процент = 100 × (среднее full − среднее gps_only) / среднее gps_only; это отношение средних, не среднее процентов эпизодов.

| Outage, с | Full, м | gps_only, м | Full−gps_only, м | Разница, % | CV, м | Held, м |
|---|---|---|---|---|---|---|
| 10 | 5.709303 | 5.718232 | -0.008929 | -0.156 | 5.736834 | 6.550193 |
| 30 | 15.314450 | 15.300446 | +0.014004 | +0.092 | 15.573896 | 13.067700 |
| 60 | 36.798934 | 37.291693 | -0.492759 | -1.321 | 38.334729 | 27.846606 |

**Устойчивое практическое преимущество IMU не подтверждено.** Full лучше gps_only в 4 из 9 эпизодов; все выигрыши на mars_4. На mars_5 full хуже во всех трёх длительностях. Среднее улучшение на 60 с около 0.493 м (1.32%) не доказывает generalization. Held-GNSS в среднем лучше обеих сетей на 30/60 с. Это validation, используемая также для checkpoint selection; длительности вложены, эпизоды зависимы, seed один.

**STOP: дополнительные seeds и test evaluation не запускать автоматически.** ESKF остаётся not_ready: отсутствуют документированное плечо ordinary GNSS к PX4 IMU и проверенная инициализация абсолютного yaw из разрешённых входов. Синтетические проверки механизации прошли, но численного сравнения real-data ESKF нет. Это не свидетельство преимущества сети.

**Status: complete. STOP after one seed=0 pilot. No test metrics or additional seeds.**

## Run and frozen protocol

Run identity: `{"kernel": "shapok/drone-nav-insane-pilot", "version": "see submission receipt"}`.

Code version: `cb79659e02d8032173e7c2866f610337fdc1bb2bdcc4bdbfcf3627c9a0ef4106`. Config/manifests/source snapshot are stored alongside this report. No hyperparameter or dataset change after training.

Train: mars_1, mars_2, mars_3. Validation: mars_4, mars_5. Reserved test: mars_6, mars_7 (not loaded by the run). Outdoor_1 was loader smoke only.

## Data / reference contract

Inputs: published PX4 IMU specific force (m/s²) and angular rate (rad/s), ordinary PX4 GNSS ENU position/covariance only. IMU is aggregated with causal time-weighted integration at 20 Hz; dt/sample age/count and all GNSS availability/quality features are shared with gps_only. No GT attitude, RTK, onboard pose, magnetometer, or absolute position is a learned input. Published timestamp corrections are already applied; they are not applied twice.

Scoring uses original rows/timestamps of ground_truth_8hz.csv; the nominal filename does not imply 8 independent observations each second. Reference positions are at the PX4 IMU, produced by the authors from dual RTK and magnetometer geometry. Their timing alignment is postprocessed and GT is not error-free. Ordinary GNSS antenna lever arm is absent from the supplied calibration; its offset/dynamic lever effect remains a limitation. No GT-derived rotation or fitted trajectory alignment is applied.

Source: [INSANE data](https://www.aau.at/en/smart-systems-technologies/control-of-networked-systems/datasets/insane-dataset/), [authors tools](https://github.com/aau-cns/insane_dataset_tools/tree/9a1c8c0fdd195f2d869fff292f2ce5b273c5a03d), [Brommer et al.](https://arxiv.org/abs/2210.09114). License and attribution are included in the private data package.

## Architecture and training

Pilot architecture, not established novelty: IMU 6→16, GNSS displacement/velocity 6→16, shared quality/time 10→8; concat40→GRU32→velocity residual3. Matching full/gps_only parameter count: {'full': 7515, 'gps_only': 7515}. gps_only zeros only the six normalized sensor values; shared features remain available.

AdamW lr=0.001, weight_decay=0.0001, batch=4, max_epochs=30, patience=5, Huber delta=1.0m. Both methods start from scratch with seed0 and identical frozen augmentation/order. Selection: macro mean episode relative-motion RMSE3D on fixed validation outages; strict decrease, min_delta=0. No test selection.

| method | epoch | mean_batch_huber_train_loss | validation_macro_relative_rmse3d_m | selected_best_so_far | epochs_without_improvement | epoch_wall_seconds |
|---|---|---|---|---|---|---|
| full | 1 | 11.920627 | 19.513384 | True | 0 | 1.333703 |
| full | 2 | 12.418767 | 19.535557 | False | 1 | 0.095276 |
| full | 3 | 8.716373 | 19.521928 | False | 2 | 0.091697 |
| full | 4 | 11.828317 | 19.435665 | True | 0 | 0.092656 |
| full | 5 | 9.815551 | 19.365134 | True | 0 | 0.082412 |
| full | 6 | 11.447382 | 19.274229 | True | 0 | 0.080588 |
| full | 7 | 9.214000 | 19.417377 | False | 1 | 0.079780 |
| full | 8 | 11.182856 | 19.649149 | False | 2 | 0.077756 |
| full | 9 | 11.045358 | 20.317605 | False | 3 | 0.076017 |
| full | 10 | 8.605634 | 20.814853 | False | 4 | 0.077602 |
| full | 11 | 8.181526 | 20.650350 | False | 5 | 0.078457 |
| gps_only | 1 | 11.918765 | 19.488459 | True | 0 | 0.075469 |
| gps_only | 2 | 12.418712 | 19.511089 | False | 1 | 0.092601 |
| gps_only | 3 | 8.715076 | 19.502016 | False | 2 | 0.078423 |
| gps_only | 4 | 11.825362 | 19.436790 | True | 0 | 0.080532 |
| gps_only | 5 | 9.806073 | 19.479296 | False | 1 | 0.079420 |
| gps_only | 6 | 11.468374 | 19.562717 | False | 2 | 0.078426 |
| gps_only | 7 | 9.257993 | 19.874212 | False | 3 | 0.079131 |
| gps_only | 8 | 11.236299 | 20.218943 | False | 4 | 0.078829 |
| gps_only | 9 | 11.121937 | 20.924896 | False | 5 | 0.078158 |

## Metric definitions

Primary = sqrt(mean(||(prediction−last_legal_GNSS_position)−(GT(t)−GT(last_legal_fix_time))||²)) inside each artificial outage. The anchor GT is interpolated only for the target/reference; scoring rows are original GT timestamps. The state/position is never initialized from GT. Predictions at native GT times use the most recent causal token plus its velocity over the remaining fraction of dt. Absolute3D/H/V and final unavailable error use raw positions without alignment. CV uses least-squares velocity from the last5s of permitted fixes. No new hidden GNSS values or covariance enter outage features.

Summary values are macro arithmetic means of per-episode RMSE, not pooled-point RMSE. Nested durations/overlapping episodes are dependent. One seed, two validation flights, same vehicle/campaign: no confidence interval, significance or generalization claim.

## All fixed validation scenarios

| scenario | method | status | n_native_gt | relative_motion_rmse3d_m | absolute_rmse3d_m | absolute_rmse_h_m | absolute_rmse_v_m | final_unavailable_error_m | recovery_first_5s_rmse3d_m |
|---|---|---|---|---|---|---|---|---|---|
| mars_4_e1_d10 | held_gnss | complete | 29.000000 | 0.505639 | 2.693868 | 1.993519 | 1.811851 | 3.000196 | 2.942734 |
| mars_4_e1_d10 | constant_velocity_gnss | complete | 29.000000 | 1.488202 | 2.454870 | 2.374678 | 0.622328 | 2.803890 | 2.932554 |
| mars_4_e1_d10 | full | complete | 29.000000 | 2.023852 | 1.476657 | 1.225572 | 0.823704 | 1.601282 | 2.882028 |
| mars_4_e1_d10 | gps_only | complete | 29.000000 | 1.659391 | 1.804973 | 1.660594 | 0.707358 | 1.887097 | 2.891390 |
| mars_4_e1_d10 | eskf | not_ready | — | — | — | — | — | — | — |
| mars_4_e1_d30 | held_gnss | complete | 115.000000 | 5.346875 | 6.005079 | 5.283607 | 2.853853 | 17.189280 | 2.598311 |
| mars_4_e1_d30 | constant_velocity_gnss | complete | 115.000000 | 9.552351 | 9.454672 | 6.364851 | 6.991387 | 23.816958 | 2.598311 |
| mars_4_e1_d30 | full | complete | 115.000000 | 10.038588 | 9.048420 | 3.673650 | 8.269112 | 22.427525 | 2.598311 |
| mars_4_e1_d30 | gps_only | complete | 115.000000 | 9.466744 | 8.833305 | 4.412050 | 7.652522 | 22.362216 | 2.598311 |
| mars_4_e1_d30 | eskf | not_ready | — | — | — | — | — | — | — |
| mars_4_e1_d60 | held_gnss | complete | 231.000000 | 24.967087 | 25.865798 | 24.548413 | 8.149533 | 49.253469 | 1.681252 |
| mars_4_e1_d60 | constant_velocity_gnss | complete | 231.000000 | 31.544700 | 31.992685 | 26.578197 | 17.808182 | 57.935246 | 1.681252 |
| mars_4_e1_d60 | full | complete | 231.000000 | 28.533322 | 28.448345 | 19.579295 | 20.638788 | 50.888499 | 1.681252 |
| mars_4_e1_d60 | gps_only | complete | 231.000000 | 29.467426 | 29.643437 | 22.654632 | 19.118079 | 54.030752 | 1.681252 |
| mars_4_e1_d60 | eskf | not_ready | — | — | — | — | — | — | — |
| mars_4_e2_d10 | held_gnss | complete | 48.000000 | 18.557776 | 18.686558 | 18.527404 | 2.433668 | 25.511162 | 3.062152 |
| mars_4_e2_d10 | constant_velocity_gnss | complete | 48.000000 | 14.789161 | 15.603278 | 15.007244 | 4.271407 | 20.162089 | 3.062152 |
| mars_4_e2_d10 | full | complete | 48.000000 | 13.399618 | 14.196455 | 13.352374 | 4.822183 | 18.058970 | 3.062152 |
| mars_4_e2_d10 | gps_only | complete | 48.000000 | 14.145136 | 14.942112 | 14.235087 | 4.541916 | 19.157615 | 3.062152 |
| mars_4_e2_d10 | eskf | not_ready | — | — | — | — | — | — | — |
| mars_4_e2_d30 | held_gnss | complete | 120.000000 | 33.256678 | 33.488176 | 33.362416 | 2.899504 | 58.065121 | 3.250258 |
| mars_4_e2_d30 | constant_velocity_gnss | complete | 120.000000 | 33.683586 | 34.617940 | 33.790171 | 7.525032 | 68.974943 | 3.250258 |
| mars_4_e2_d30 | full | complete | 120.000000 | 30.238074 | 31.170561 | 29.884677 | 8.860585 | 62.936701 | 3.250258 |
| mars_4_e2_d30 | gps_only | complete | 120.000000 | 31.805966 | 32.735760 | 31.687473 | 8.217911 | 65.542590 | 3.250258 |
| mars_4_e2_d30 | eskf | not_ready | — | — | — | — | — | — | — |
| mars_4_e2_d60 | held_gnss | complete | 244.000000 | 52.242040 | 52.696970 | 52.429473 | 5.302927 | 30.689360 | 2.984149 |
| mars_4_e2_d60 | constant_velocity_gnss | complete | 244.000000 | 71.280993 | 72.290811 | 71.256854 | 12.182865 | 106.563506 | 2.984149 |
| mars_4_e2_d60 | full | complete | 244.000000 | 65.644896 | 66.642670 | 65.034822 | 14.550515 | 100.932683 | 2.984149 |
| mars_4_e2_d60 | gps_only | complete | 244.000000 | 67.847799 | 68.852112 | 67.522254 | 13.466941 | 102.084608 | 2.984149 |
| mars_4_e2_d60 | eskf | not_ready | — | — | — | — | — | — | — |
| mars_4_control | held_gnss | complete | 711.000000 | — | 2.944914 | 1.530340 | 2.516064 | — | — |
| mars_4_control | constant_velocity_gnss | complete | 711.000000 | — | 2.944914 | 1.530340 | 2.516064 | — | — |
| mars_4_control | full | complete | 711.000000 | — | 2.944914 | 1.530340 | 2.516064 | — | — |
| mars_4_control | gps_only | complete | 711.000000 | — | 2.944914 | 1.530340 | 2.516064 | — | — |
| mars_4_control | eskf | not_ready | — | — | — | — | — | — | — |
| mars_5_e1_d10 | held_gnss | complete | 40.000000 | 0.587165 | 6.203632 | 3.596204 | 5.054935 | 6.220601 | 5.608907 |
| mars_5_e1_d10 | constant_velocity_gnss | complete | 40.000000 | 0.933140 | 5.185223 | 3.452228 | 3.868936 | 4.382373 | 5.515861 |
| mars_5_e1_d10 | full | complete | 40.000000 | 1.704438 | 5.167228 | 3.754277 | 3.550444 | 4.561807 | 5.523827 |
| mars_5_e1_d10 | gps_only | complete | 40.000000 | 1.350168 | 5.116049 | 3.555291 | 3.678840 | 4.342682 | 5.514259 |
| mars_5_e1_d10 | eskf | not_ready | — | — | — | — | — | — | — |
| mars_5_e1_d30 | held_gnss | complete | 122.000000 | 0.599547 | 6.216036 | 3.594904 | 5.071072 | 6.223684 | 6.377110 |
| mars_5_e1_d30 | constant_velocity_gnss | complete | 122.000000 | 3.485750 | 4.052330 | 3.200578 | 2.485492 | 3.243321 | 6.217585 |
| mars_5_e1_d30 | full | complete | 122.000000 | 5.666687 | 5.479108 | 4.838139 | 2.571583 | 7.528581 | 6.479783 |
| mars_5_e1_d30 | gps_only | complete | 122.000000 | 4.628628 | 4.587171 | 3.880121 | 2.446793 | 5.360386 | 6.322453 |
| mars_5_e1_d30 | eskf | not_ready | — | — | — | — | — | — | — |
| mars_5_e1_d60 | held_gnss | complete | 226.000000 | 6.330691 | 6.426742 | 4.980892 | 4.061248 | 2.973026 | 7.593655 |
| mars_5_e1_d60 | constant_velocity_gnss | complete | 226.000000 | 12.178496 | 9.214092 | 4.953948 | 7.769034 | 13.750179 | 8.302607 |
| mars_5_e1_d60 | full | complete | 226.000000 | 16.218585 | 13.716282 | 9.552164 | 9.843401 | 19.354606 | 8.979110 |
| mars_5_e1_d60 | gps_only | complete | 226.000000 | 14.559854 | 11.823319 | 7.740489 | 8.937321 | 16.575941 | 8.621471 |
| mars_5_e1_d60 | eskf | not_ready | — | — | — | — | — | — | — |
| mars_5_control | held_gnss | complete | 348.000000 | — | 6.648175 | 3.486030 | 5.660904 | — | — |
| mars_5_control | constant_velocity_gnss | complete | 348.000000 | — | 6.648175 | 3.486030 | 5.660904 | — | — |
| mars_5_control | full | complete | 348.000000 | — | 6.648175 | 3.486030 | 5.660904 | — | — |
| mars_5_control | gps_only | complete | 348.000000 | — | 6.648175 | 3.486030 | 5.660904 | — | — |
| mars_5_control | eskf | not_ready | — | — | — | — | — | — | — |

## By flight and duration — primary RMSE in metres

| flight | duration_s | held_gnss | constant_velocity_gnss | full | gps_only |
|---|---|---|---|---|---|
| mars_4 | 10 | 9.531708 | 8.138682 | 7.711735 | 7.902263 |
| mars_4 | 30 | 19.301777 | 21.617969 | 20.138331 | 20.636355 |
| mars_4 | 60 | 38.604564 | 51.412846 | 47.089109 | 48.657613 |
| mars_5 | 10 | 0.587165 | 0.933140 | 1.704438 | 1.350168 |
| mars_5 | 30 | 0.599547 | 3.485750 | 5.666687 | 4.628628 |
| mars_5 | 60 | 6.330691 | 12.178496 | 16.218585 | 14.559854 |

## By duration — primary RMSE in metres

| duration_s | held_gnss | constant_velocity_gnss | full | gps_only |
|---|---|---|---|---|
| 10 | 6.550193 | 5.736834 | 5.709303 | 5.718232 |
| 30 | 13.067700 | 15.573896 | 15.314450 | 15.300446 |
| 60 | 27.846606 | 38.334729 | 36.798934 | 37.291693 |

## Paired full comparisons

| scenario | flight | duration_s | comparison | full_rmse_m | other_rmse_m | difference_m | difference_percent | full_better | n_native_gt |
|---|---|---|---|---|---|---|---|---|---|
| mars_4_e1_d10 | mars_4 | 10 | full_minus_gps_only | 2.023852 | 1.659391 | 0.364462 | 21.963582 | False | 29 |
| mars_4_e1_d10 | mars_4 | 10 | full_minus_constant_velocity_gnss | 2.023852 | 1.488202 | 0.535650 | 35.993119 | False | 29 |
| mars_4_e1_d10 | mars_4 | 10 | full_minus_held_gnss | 2.023852 | 0.505639 | 1.518214 | 300.256554 | False | 29 |
| mars_4_e1_d30 | mars_4 | 30 | full_minus_gps_only | 10.038588 | 9.466744 | 0.571845 | 6.040566 | False | 115 |
| mars_4_e1_d30 | mars_4 | 30 | full_minus_constant_velocity_gnss | 10.038588 | 9.552351 | 0.486237 | 5.090235 | False | 115 |
| mars_4_e1_d30 | mars_4 | 30 | full_minus_held_gnss | 10.038588 | 5.346875 | 4.691713 | 87.746824 | False | 115 |
| mars_4_e1_d60 | mars_4 | 60 | full_minus_gps_only | 28.533322 | 29.467426 | -0.934104 | -3.169955 | True | 231 |
| mars_4_e1_d60 | mars_4 | 60 | full_minus_constant_velocity_gnss | 28.533322 | 31.544700 | -3.011378 | -9.546384 | True | 231 |
| mars_4_e1_d60 | mars_4 | 60 | full_minus_held_gnss | 28.533322 | 24.967087 | 3.566235 | 14.283744 | False | 231 |
| mars_4_e2_d10 | mars_4 | 10 | full_minus_gps_only | 13.399618 | 14.145136 | -0.745518 | -5.270490 | True | 48 |
| mars_4_e2_d10 | mars_4 | 10 | full_minus_constant_velocity_gnss | 13.399618 | 14.789161 | -1.389543 | -9.395688 | True | 48 |
| mars_4_e2_d10 | mars_4 | 10 | full_minus_held_gnss | 13.399618 | 18.557776 | -5.158158 | -27.795131 | True | 48 |
| mars_4_e2_d30 | mars_4 | 30 | full_minus_gps_only | 30.238074 | 31.805966 | -1.567891 | -4.929552 | True | 120 |
| mars_4_e2_d30 | mars_4 | 30 | full_minus_constant_velocity_gnss | 30.238074 | 33.683586 | -3.445511 | -10.229052 | True | 120 |
| mars_4_e2_d30 | mars_4 | 30 | full_minus_held_gnss | 30.238074 | 33.256678 | -3.018604 | -9.076684 | True | 120 |
| mars_4_e2_d60 | mars_4 | 60 | full_minus_gps_only | 65.644896 | 67.847799 | -2.202903 | -3.246831 | True | 244 |
| mars_4_e2_d60 | mars_4 | 60 | full_minus_constant_velocity_gnss | 65.644896 | 71.280993 | -5.636097 | -7.906872 | True | 244 |
| mars_4_e2_d60 | mars_4 | 60 | full_minus_held_gnss | 65.644896 | 52.242040 | 13.402856 | 25.655307 | False | 244 |
| mars_5_e1_d10 | mars_5 | 10 | full_minus_gps_only | 1.704438 | 1.350168 | 0.354270 | 26.238926 | False | 40 |
| mars_5_e1_d10 | mars_5 | 10 | full_minus_constant_velocity_gnss | 1.704438 | 0.933140 | 0.771298 | 82.656238 | False | 40 |
| mars_5_e1_d10 | mars_5 | 10 | full_minus_held_gnss | 1.704438 | 0.587165 | 1.117273 | 190.282643 | False | 40 |
| mars_5_e1_d30 | mars_5 | 30 | full_minus_gps_only | 5.666687 | 4.628628 | 1.038059 | 22.426919 | False | 122 |
| mars_5_e1_d30 | mars_5 | 30 | full_minus_constant_velocity_gnss | 5.666687 | 3.485750 | 2.180937 | 62.567221 | False | 122 |
| mars_5_e1_d30 | mars_5 | 30 | full_minus_held_gnss | 5.666687 | 0.599547 | 5.067141 | 845.162114 | False | 122 |
| mars_5_e1_d60 | mars_5 | 60 | full_minus_gps_only | 16.218585 | 14.559854 | 1.658731 | 11.392499 | False | 226 |
| mars_5_e1_d60 | mars_5 | 60 | full_minus_constant_velocity_gnss | 16.218585 | 12.178496 | 4.040090 | 33.173964 | False | 226 |
| mars_5_e1_d60 | mars_5 | 60 | full_minus_held_gnss | 16.218585 | 6.330691 | 9.887895 | 156.189823 | False | 226 |

Full is lower than matched gps_only in **4/9** outage scenarios. Negative full-minus-other means lower full error. Percent =100×(full−other)/other. No cherry-picking of epochs beyond the predefined early-stopping criterion or of favorable scenarios.

## ESKF readiness

A separate 15-state ESKF (p,v,right attitude error,ba,bg) has synthetic stationary/constant-velocity/turn/irregular-dt/Joseph-update tests. Real-data ESKF is **not ready**: ordinary GNSS antenna to PX4 IMU lever arm not supplied; absolute initial yaw unavailable from permitted sensors without validated dynamic alignment. Its rows are explicitly not_ready with no numeric metrics. Unknown heading is not replaced by GNSS course, GT or an arbitrary confident yaw. No network superiority over a calibrated ESKF is claimed. Noise values in config are synthetic engineering defaults, not fitted flight calibration.

## Runtime / recovery

Environment: `{"python": "3.12.13", "torch": "2.10.0+cu128", "numpy": "2.0.2", "pandas": "2.3.3", "device": "cuda", "cuda": "12.8", "hardware": "Tesla T4"}`. Model inference_ms is one measured synchronized forward/readout on that hardware, excludes preprocessing/training and is not a repeated latency benchmark. Held/CV rows share the measured time of their joint calculation; those numbers are not separately timed per-baseline latencies. Recovery is first5s after nominal outage end; first_recovered_fix_error uses the first scored point after an actual returned fix. Decoder returns to legal GNSS on availability; no post-outage trajectory alignment.

| scenario | method | inference_ms | first_recovered_fix_error_m |
|---|---|---|---|
| mars_4_e1_d10 | held_gnss | 0.102689 | 2.953028 |
| mars_4_e1_d10 | constant_velocity_gnss | 0.102689 | 2.953028 |
| mars_4_e1_d10 | full | 1.675726 | 2.953028 |
| mars_4_e1_d10 | gps_only | 1.662534 | 2.953028 |
| mars_4_e1_d30 | held_gnss | 0.080913 | 2.546093 |
| mars_4_e1_d30 | constant_velocity_gnss | 0.080913 | 2.546093 |
| mars_4_e1_d30 | full | 1.799397 | 2.546093 |
| mars_4_e1_d30 | gps_only | 1.737212 | 2.546093 |
| mars_4_e1_d60 | held_gnss | 0.084070 | 1.354477 |
| mars_4_e1_d60 | constant_velocity_gnss | 0.084070 | 1.354477 |
| mars_4_e1_d60 | full | 2.179666 | 1.354477 |
| mars_4_e1_d60 | gps_only | 2.175280 | 1.354477 |
| mars_4_e2_d10 | held_gnss | 0.071789 | 2.432964 |
| mars_4_e2_d10 | constant_velocity_gnss | 0.071789 | 2.432964 |
| mars_4_e2_d10 | full | 1.711680 | 2.432964 |
| mars_4_e2_d10 | gps_only | 1.612550 | 2.432964 |
| mars_4_e2_d30 | held_gnss | 0.084489 | 2.664977 |
| mars_4_e2_d30 | constant_velocity_gnss | 0.084489 | 2.664977 |
| mars_4_e2_d30 | full | 1.874635 | 2.664977 |
| mars_4_e2_d30 | gps_only | 1.848224 | 2.664977 |
| mars_4_e2_d60 | held_gnss | 0.094327 | 2.701501 |
| mars_4_e2_d60 | constant_velocity_gnss | 0.094327 | 2.701501 |
| mars_4_e2_d60 | full | 2.225986 | 2.701501 |
| mars_4_e2_d60 | gps_only | 2.186916 | 2.701501 |
| mars_4_control | held_gnss | 0.131988 | — |
| mars_4_control | constant_velocity_gnss | 0.131988 | — |
| mars_4_control | full | 3.840506 | — |
| mars_4_control | gps_only | 3.342976 | — |
| mars_5_e1_d10 | held_gnss | 0.077630 | 5.611469 |
| mars_5_e1_d10 | constant_velocity_gnss | 0.077630 | 5.611469 |
| mars_5_e1_d10 | full | 1.950372 | 5.611469 |
| mars_5_e1_d10 | gps_only | 1.608263 | 5.611469 |
| mars_5_e1_d30 | held_gnss | 0.081084 | 6.217556 |
| mars_5_e1_d30 | constant_velocity_gnss | 0.081084 | 6.217556 |
| mars_5_e1_d30 | full | 1.865569 | 6.217556 |
| mars_5_e1_d30 | gps_only | 1.841216 | 6.217556 |
| mars_5_e1_d60 | held_gnss | 0.090929 | 7.475862 |
| mars_5_e1_d60 | constant_velocity_gnss | 0.090929 | 7.475862 |
| mars_5_e1_d60 | full | 2.194583 | 7.475862 |
| mars_5_e1_d60 | gps_only | 2.147171 | 7.475862 |
| mars_5_control | held_gnss | 0.091028 | — |
| mars_5_control | constant_velocity_gnss | 0.091028 | — |
| mars_5_control | full | 2.314011 | — |
| mars_5_control | gps_only | 2.136982 | — |

## Conclusion and next action

This pilot estimates whether IMU helps under this fixed small-data protocol. A small difference is not proof of IMU utility. Inspect the complete paired table, duration means and replication across flights; a single-seed validation-selected result cannot establish generalization. **STOP. Do not run more seeds, change metrics, or swap flights automatically.**

Artifacts: config/splits/outage/training manifests, train-only scaler, source hashes, environment, best/last checkpoints, training history, each native-timestamp prediction/reference/mask, summary.csv, paired_differences.csv, preflight.json, eskf_status.json, status.json. All test flights remain reserved.


## Verified local handoff

Kaggle kernel: shapok/drone-nav-insane-pilot; version 3; output IDs: 353114895. Downloaded 84 artifacts. Independent recomputation: 44 prediction files, maximum metric discrepancy 1.42e-14; native reference timestamps and identical masks verified. No test reads, model forwards, or additional training.

Local verification: `outputs/insane_pilot/independent_verification.json`; source/run receipts in `outputs/insane_pilot_launch/`.
