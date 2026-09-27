# V2_PILOT_REPORT

Дата: 2026-09-25 UTC. **Pilot завершён. Практическое преимущество IMU не подтверждено. Рекомендация: STOP; пять seeds сейчас не запускать.**

full лучше matched gps_only по основной метрике в **3 из 9** outage scenarios; все три выигрыша относятся к одному onset e2 с вложенными 10/30/60 s интервалами. Для каждого duration средняя ошибка full выше. Ни архитектура, ни preprocessing, target, fixed scenarios, seeds, epochs или hyperparameters после получения результатов не менялись. Выполнен только один Kaggle job, три предусмотренных обучения, seed 0, final epoch 8. Test не использован.

## 1. Run status и Kaggle run ID

| Поле | Фактически выполнено |
| --- | --- |
| Kaggle notebook | [shapok/drone-nav-v2-motion-pilot](https://www.kaggle.com/code/shapok/drone-nav-v2-motion-pilot) |
| Notebook version / output-session ID | **version 1 / 352778350** |
| Submission UTC | 2026-09-25T18:33:06.554083+00:00 |
| Kaggle final status | `complete`, failureMessage пуст |
| Pilot status | `complete`, evaluation_rows=60, seed=0, test_used=false |
| Три обучения | `fusion_v2_full`, `fusion_v2_gps_only`, `lstm_v2_full`; по 8 epochs |
| Checkpoint selection | Fixed final epoch8, без validation selection/early stopping |
| Evaluation | 10 fixed validation scenarios × 6 методов = 60 rows; test отсутствует |
| Device / environment | Tesla T4, CUDA12.8, PyTorch2.10.0+cu128, Python3.12.13, NumPy2.0.2, pandas2.3.3, pyproj3.7.2 |
| Ограничение запуска | Private notebook, internet off, timeout1800s; одна submission, без перезапуска |
| Скачано | 97 файлов, 7,395,478 bytes: outputs, source snapshot и preflight/code files |

Источник: [submission.json](outputs/v2_pilot_launch/submission.json), [remote_status.json](outputs/v2_pilot_launch/remote_status.json), [download_receipt.json](outputs/v2_pilot_launch/download_receipt.json), [status.json](outputs/v2_motion_pilot/status.json), [environment.json](outputs/v2_motion_pilot/environment.json), [Kaggle log](outputs/v2_pilot_launch/kaggle_run_log.json). Лог содержит все 24 epoch records; последняя epoch LSTM около90.9s от начала log clock, финальный export около102.8s. Это наблюдаемая длительность лога, не отдельный benchmark обучения/inference.

Отправлен подготовленный notebook `notebooks/kaggle_v2_motion_pilot.ipynb` в копии [launch/kaggle_v2_motion_pilot.ipynb](outputs/v2_pilot_launch/kaggle_v2_motion_pilot.ipynb). Изменения копии: включён разрешённый `RUN_TRAINING=True`, сохранение stdout/stderr тестов и дополнительные pre-training assertions. **Embedded model/data/training code и config/manifest совпадают с подготовленными побайтно.** Исходный подготовленный notebook с False не переписан. Список изменений и хеши: [pre_submission.json](outputs/v2_pilot_launch/pre_submission.json).

## 2. Проверки preprocessing / hashes / tests

**Все обязательные проверки прошли до optimizer steps; обходов и повторного запуска не было.**

- Локально прошли 17/17 tests; на Kaggle — те же 17/17, `Ran 17 tests in 0.576s`, `OK`. Лог: [unit_tests.log](outputs/v2_pilot_launch/remote_output/v2_pilot_launch/unit_tests.log).
- Config SHA256: `f413f38984fd816e6ade634dcf329488c3225117085f7116f041c1622e43c6b0`.
- Manifest SHA256: `37b1837afb89f371d291125d4f35b26782a8353375e1dc213211e3f670e70900`.
- Common validation timestamp SHA256: `65f65ce416cd8b5a2b4d4e89fefb8b81cea2da9ecf5149d38e4c637eb5d1b110`. Все 10 mask hashes и fixed onset/cutoff intervals совпали с manifest.
- Четыре фактических raw CSV совпали с pinned version2 hashes; никакого другого dataset/подмены CSV не было.

| Raw file | SHA256 |
| --- | --- |
| RawAccel.csv | `2609d0d936c75c7ed2566b258f87c7508faa28848e5cd366f55fbf069e557c11` |
| RawGyro.csv | `3d39a569265dcc9682fe254c5246a7b7b84c14f338c29a99f25d435d683d7bec` |
| OnboardGPS.csv | `8e58193ea9ff4e085a61241a66179d46f66110074d59cc1b97a7167fca1181a5` |
| GroundTruthAGL.csv | `50e95abc70461567a915a76ba17231ae3088b834dbcb433737135eacb9c9c552` |

Реальная production проекция **pyproj** проверена на Kaggle: WGS84→UTM32N, `always_xy=True`. Первый GNSS UTM=(465670.706847399,5247978.033436624) м; максимальное отличие от опубликованной reference point **0.000001376 м**. Максимальное отличие pyproj от независимой аналитической UTM по train+validation GPS **0.000322549 м**, при заранее установленном допуске **0.001 м**. Проверка центрального меридиана также прошла. Общий XYZ origin=(465670.706847399,5247978.033436624,464.91) м. Никакой fitted alignment не применялся.

До обучения проверены все **280 training window plans** (35 windows×8epochs) и 10 validation scenarios с настоящим pyproj. Gyro только из прошлого; actual dt, GNSS availability/new-fix/age и источники fixes сохранены. Norm scalers fit по train `[0,18935)`; validation common `[18935,21623)`, 2688 точек. Loss labels используют GT только как target; GT не инициализирует состояние модели. Полный протокол не менялся относительно подготовленного V2.

Preflight evidence: [preflight.json](outputs/v2_pilot_launch/remote_output/v2_pilot_launch/preflight.json), [coordinate_contract.json](outputs/v2_motion_pilot/coordinate_contract.json), [imu_scaler_train_only.json](outputs/v2_motion_pilot/imu_scaler_train_only.json). Тесты проверяют причинность prefix/state carry с ненулевым head, train-only scaler, отсутствие GT/absolute-position input, legal source restrictions, actual anchor dt и zero-head=CV.

**Проверки после скачивания:** все 60 predictions заново сопоставлены с GT, availability и timestamp masks. Дубликатов и потерянных сочетаний scenario/method нет; NaN/Inf нет. Все сохранённые RMSE/final-point metrics независимо пересчитаны с **максимальным расхождением 0.0**. Motion target дополнительно восстановлен из native raw GT и legal source timestamp: max разность **0.000003659 м**, соответствует float32 label rounding. Across methods полностью совпадают timestamps, GT, motion target, availability, source indices/timestamps; source fixes внутри запрещённого интервала не обнаружены.

Код проверки: [analyze_v2_pilot.py](scripts/analyze_v2_pilot.py); итог: [independent_verification.json](outputs/v2_motion_pilot/independent_verification.json). Проверка читает сохранённые данные, не делает новых forwards, не обучает модели и не использует test. Source snapshot/config/manifest фактического запуска совпадают с подготовленными.

## 3. Training history трёх моделей

Общая настройка: AdamW lr0.001, weight_decay0.0001, batch4, clip1, 8epochs, seed0; hidden32, Huber delta1м. Train windows1024/stride512, warmup30s/recovery≥5s, один outage10/30/60s на окно; одинаковый RNG/masks/order между full и gps_only. Parameter counts: GRU full/gps_only **6755/6755**, LSTM **8867**. Полные параметры: [config.json](outputs/v2_motion_pilot/config.json); фактические train masks: [training_augmentation_manifest.json](outputs/v2_motion_pilot/training_augmentation_manifest.json).

Ниже `mean_batch_training_loss`: среднее batch Huber loss на относительном displacement, **не RMSE в метрах и не validation metric**. Augmentation меняется по epoch, поэтому рост loss в epoch7 не означает ухудшение на одном фиксированном наборе. Между epochs validation checkpoint selection не выполнялась.

| Epoch | GRU full loss | GRU gps_only loss | LSTM full loss |
| --- | --- | --- | --- |
| 1 | 4.946623629993862 | 4.936162286334568 | 4.953267521328396 |
| 2 | 5.486692428588867 | 5.485921170976427 | 5.460528426700169 |
| 3 | 4.841339495446947 | 4.833662006590101 | 4.834373116493225 |
| 4 | 5.101632846726312 | 5.090096010102166 | 5.147574583689372 |
| 5 | 4.2099208566877575 | 4.1704980333646136 | 4.2346088488896685 |
| 6 | 4.2778592175907555 | 4.213802509837681 | 4.312578075461918 |
| 7 | 6.533825000127156 | 6.260347949133979 | 6.437880410088433 |
| 8 | 3.8892925447887845 | 3.6392545898755393 | 3.9097521702448526 |

Источник: [training_history.csv](outputs/v2_motion_pilot/training_history.csv). Сохранены ровно три final checkpoints, перечисленные ниже; ни best-run, ни усреднение checkpoints не использовались.

| Final epoch8 checkpoint | SHA256 |
| --- | --- |
| fusion_v2_full_seed0.pt | `98f12886505b4314b55b2abeb9aed8f20fd5b770ae9628df8ca45a7e183d11a6` |
| fusion_v2_gps_only_seed0.pt | `5db75defa05c47eff7297d117c12e9ef209211466ce451c7b97fbb25676f10dd` |
| lstm_v2_full_seed0.pt | `2248050cd0f60b4961901c422f5c5203f1bc45d35d551fd1390cd8adbeb30ab9` |

## 4. Итоговые validation metrics

**Primary:** relative-motion RMSE3D **только внутри outage**:

`motion_target(t) = GT(t) − GT(t_last_legal_source)`;
`predicted_motion(t) = prediction(t) − held_GNSS_anchor`;
`RMSE = sqrt(mean(||predicted_motion − motion_target||²))`.

Anchor GT нужен только для target/metric; модель его не получает. Нет rigid/scale alignment. Constant initial GNSS error не входит в primary metric, но остаётся в raw absolute error. GT~1Hz интерполирован на IMU timeline; 100/300/600 временных точек не являются независимыми GT observations.

**Secondary:** raw absolute RMSE3D/H/V, без alignment, по той же outage mask. H=`sqrt(mean(dx²+dy²))`, V=`sqrt(mean(dz²))`. Final unavailable error — norm ошибки **в последней недоступной точке**, не после восстановления и не RMSE последнего окна. Ниже «final3D/H/|V|» — эти три скалярные ошибки. Все единицы — метры.

Для control primary и final unavailable errors не определены (пустые поля). Raw control на всех2688 точках у **всех шести методов одинаков:** 3D **7.758991374**, H **6.059848044**, V **4.845636059** м. Это ожидаемо по decoder: при доступном GNSS prediction сбрасывается к legal held fix. Равенство control не свидетельствует об обученном fusion улучшении.

Таблицы ниже округлены до6знаков; [summary.csv](outputs/v2_motion_pilot/summary.csv) содержит полную сохранённую precision для всех60строк. Никакие ±, доверительные интервалы или статистическая значимость не вычислялись. Все нейросетевые числа относятся к одному seed0; baselines детерминированы и не обучались.

## 5. Таблица всех 9 outage scenarios

По каждой строке сравниваются одни и те же точки и legal GNSS sources. e1/e2/e3 — заранее выбранные onsets, а три duration на каждом onset вложены.

| Scenario | Onset, с | Cutoff, с | N outage | full motion | gps_only motion | LSTM motion | zero_IMU motion | held motion | CV motion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val_e1_d10s | 1975.869753 | 1985.869753 | 100 | 2.614048 | 1.808004 | 2.750512 | 2.614379 | 6.755859 | 3.343932 |
| val_e1_d30s | 1975.869753 | 2005.869753 | 300 | 8.383131 | 5.935091 | 8.718845 | 8.306534 | 20.061148 | 10.351077 |
| val_e1_d60s | 1975.869753 | 2035.869753 | 600 | 16.688310 | 11.556814 | 17.258029 | 16.267230 | 40.249201 | 20.446518 |
| val_e2_d10s | 2015.981753 | 2025.981753 | 100 | 2.441036 | 2.474611 | 2.890241 | 2.507252 | 6.041596 | 3.104924 |
| val_e2_d30s | 2015.981753 | 2045.981753 | 300 | 7.481590 | 7.818578 | 8.886792 | 7.724283 | 16.625546 | 9.455026 |
| val_e2_d60s | 2015.981753 | 2075.981753 | 600 | 14.521529 | 15.744118 | 16.755514 | 15.218191 | 35.330335 | 17.001577 |
| val_e3_d10s | 2055.997753 | 2065.997753 | 100 | 3.079475 | 2.435430 | 2.863194 | 3.019796 | 7.272853 | 3.124455 |
| val_e3_d30s | 2055.997753 | 2085.997753 | 300 | 8.961734 | 7.259304 | 8.354507 | 8.875569 | 21.670429 | 9.059666 |
| val_e3_d60s | 2055.997753 | 2115.997753 | 600 | 16.636747 | 12.790114 | 16.148710 | 16.514841 | 40.537868 | 18.302894 |

Полные raw absolute метрики всех методов, включая control (N=2688) и outages (N=100/300/600):

| Scenario | Метод | N | raw3D | rawH | rawV | final3D | finalH | final \|V\| |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val_control | full | 2688 | 7.758991 | 6.059848 | 4.845636 | — | — | — |
| val_control | gps_only | 2688 | 7.758991 | 6.059848 | 4.845636 | — | — | — |
| val_control | LSTM | 2688 | 7.758991 | 6.059848 | 4.845636 | — | — | — |
| val_control | full_zero_IMU | 2688 | 7.758991 | 6.059848 | 4.845636 | — | — | — |
| val_control | held | 2688 | 7.758991 | 6.059848 | 4.845636 | — | — | — |
| val_control | CV | 2688 | 7.758991 | 6.059848 | 4.845636 | — | — | — |
| val_e1_d10s | full | 100 | 6.827874 | 2.543859 | 6.336295 | 4.891590 | 2.910616 | 3.931407 |
| val_e1_d10s | gps_only | 100 | 7.415021 | 2.508724 | 6.977739 | 6.055397 | 2.828752 | 5.354063 |
| val_e1_d10s | LSTM | 100 | 6.763541 | 2.567065 | 6.257449 | 4.827325 | 3.073769 | 3.722232 |
| val_e1_d10s | full_zero_IMU | 100 | 6.839634 | 2.560964 | 6.342085 | 4.912509 | 2.938218 | 3.936956 |
| val_e1_d10s | held | 100 | 10.059837 | 6.002641 | 8.072709 | 12.711990 | 10.196078 | 7.591751 |
| val_e1_d10s | CV | 100 | 6.395484 | 2.490809 | 5.890508 | 4.222216 | 3.127571 | 2.836442 |
| val_e1_d30s | full | 300 | 6.028547 | 4.128188 | 4.393341 | 9.148963 | 6.888644 | 6.020807 |
| val_e1_d30s | gps_only | 300 | 6.207201 | 4.104971 | 4.656024 | 7.087247 | 6.968133 | 1.293904 |
| val_e1_d30s | LSTM | 300 | 6.051041 | 4.069127 | 4.478538 | 9.180659 | 6.354533 | 6.626040 |
| val_e1_d30s | full_zero_IMU | 300 | 6.063924 | 4.239759 | 4.335391 | 9.105714 | 7.161001 | 5.624419 |
| val_e1_d30s | held | 300 | 20.360340 | 19.097463 | 7.059062 | 34.241989 | 33.875135 | 4.998901 |
| val_e1_d30s | CV | 300 | 6.503140 | 4.041056 | 5.095165 | 11.033600 | 5.995986 | 9.262208 |
| val_e1_d60s | full | 600 | 11.742227 | 6.909738 | 9.493968 | 20.539045 | 10.376247 | 17.725289 |
| val_e1_d60s | gps_only | 600 | 8.550133 | 6.973269 | 4.947554 | 12.881938 | 10.296213 | 7.741596 |
| val_e1_d60s | LSTM | 600 | 12.073806 | 6.412791 | 10.230000 | 22.079618 | 10.740319 | 19.291321 |
| val_e1_d60s | full_zero_IMU | 600 | 11.495438 | 7.213408 | 8.950521 | 19.866153 | 10.912758 | 16.600474 |
| val_e1_d60s | held | 600 | 39.628034 | 39.203562 | 5.784622 | 66.084880 | 65.979145 | 3.736809 |
| val_e1_d60s | CV | 600 | 14.657261 | 6.482213 | 13.145958 | 27.613792 | 12.179324 | 24.782767 |
| val_e2_d10s | full | 100 | 6.980814 | 5.929427 | 3.684245 | 6.594654 | 6.387089 | 1.641510 |
| val_e2_d10s | gps_only | 100 | 7.033310 | 5.983538 | 3.696582 | 6.660179 | 6.474660 | 1.561014 |
| val_e2_d10s | LSTM | 100 | 7.086881 | 6.186517 | 3.457006 | 7.036501 | 6.956254 | 1.059665 |
| val_e2_d10s | full_zero_IMU | 100 | 6.949941 | 5.945631 | 3.598770 | 6.551292 | 6.393774 | 1.427963 |
| val_e2_d10s | held | 100 | 6.609073 | 3.593402 | 5.546828 | 7.218527 | 4.528404 | 5.621448 |
| val_e2_d10s | CV | 100 | 7.286108 | 6.414922 | 3.454873 | 7.558133 | 7.475828 | 1.112372 |
| val_e2_d30s | full | 300 | 8.022566 | 7.356414 | 3.200740 | 9.811326 | 7.818118 | 5.927828 |
| val_e2_d30s | gps_only | 300 | 8.246662 | 7.476858 | 3.479084 | 10.468391 | 8.006388 | 6.744254 |
| val_e2_d30s | LSTM | 300 | 9.240053 | 8.413146 | 3.820675 | 12.435707 | 9.859475 | 7.578757 |
| val_e2_d30s | full_zero_IMU | 300 | 8.106142 | 7.342738 | 3.434201 | 10.231078 | 7.798312 | 6.622784 |
| val_e2_d30s | held | 300 | 13.508558 | 12.157602 | 5.888451 | 24.794361 | 23.974659 | 6.322662 |
| val_e2_d30s | CV | 300 | 10.096314 | 9.419482 | 3.634406 | 13.751564 | 11.776436 | 7.100779 |
| val_e2_d60s | full | 600 | 12.322714 | 7.167679 | 10.023655 | 21.780811 | 5.025993 | 21.192997 |
| val_e2_d60s | gps_only | 600 | 13.443924 | 7.332108 | 11.268509 | 24.301577 | 5.178017 | 23.743521 |
| val_e2_d60s | LSTM | 600 | 14.845454 | 9.173351 | 11.672066 | 25.259678 | 8.121218 | 23.918553 |
| val_e2_d60s | full_zero_IMU | 600 | 12.953813 | 7.201491 | 10.767535 | 23.096853 | 5.455261 | 22.443367 |
| val_e2_d60s | held | 600 | 31.483642 | 30.932839 | 5.863374 | 59.404651 | 59.222233 | 4.651847 |
| val_e2_d60s | CV | 600 | 15.614519 | 11.203122 | 10.876730 | 25.033714 | 11.676804 | 22.143602 |
| val_e3_d10s | full | 100 | 8.598740 | 6.050038 | 6.110268 | 9.162044 | 4.566917 | 7.942689 |
| val_e3_d10s | gps_only | 100 | 7.927205 | 5.987916 | 5.194751 | 7.712829 | 4.429385 | 6.314133 |
| val_e3_d10s | LSTM | 100 | 8.951577 | 6.508042 | 6.146228 | 9.763425 | 5.536890 | 8.041599 |
| val_e3_d10s | full_zero_IMU | 100 | 8.503998 | 6.027526 | 5.998910 | 8.991770 | 4.519000 | 7.773710 |
| val_e3_d10s | held | 100 | 6.047818 | 4.900331 | 3.544412 | 7.466938 | 6.732837 | 3.228632 |
| val_e3_d10s | CV | 100 | 9.661648 | 7.018075 | 6.640336 | 11.089647 | 6.560981 | 8.940570 |
| val_e3_d30s | full | 300 | 11.142843 | 4.300629 | 10.279472 | 15.818672 | 3.629716 | 15.396609 |
| val_e3_d30s | gps_only | 300 | 8.638408 | 4.187070 | 7.555828 | 11.140656 | 3.434122 | 10.598161 |
| val_e3_d30s | LSTM | 300 | 11.706907 | 5.067284 | 10.553403 | 16.588556 | 4.341806 | 16.010275 |
| val_e3_d30s | full_zero_IMU | 300 | 10.916689 | 4.314321 | 10.027998 | 15.490038 | 3.799200 | 15.016902 |
| val_e3_d30s | held | 300 | 16.694979 | 16.440939 | 2.901354 | 29.692678 | 29.635774 | 1.837395 |
| val_e3_d30s | CV | 300 | 13.567153 | 6.093952 | 12.121526 | 19.719582 | 5.859879 | 18.828801 |
| val_e3_d60s | full | 600 | 17.811586 | 5.751216 | 16.857524 | 28.358134 | 9.672272 | 26.657661 |
| val_e3_d60s | gps_only | 600 | 12.690498 | 5.599622 | 11.388282 | 19.565785 | 9.517272 | 17.095071 |
| val_e3_d60s | LSTM | 600 | 18.753560 | 5.952758 | 17.783719 | 29.892753 | 8.474201 | 28.666437 |
| val_e3_d60s | full_zero_IMU | 600 | 17.519706 | 5.925798 | 16.487116 | 27.977926 | 10.065593 | 26.104562 |
| val_e3_d60s | held | 600 | 34.963949 | 34.887562 | 2.309923 | 61.958630 | 61.954524 | 0.713213 |
| val_e3_d60s | CV | 600 | 22.282328 | 7.115348 | 21.115727 | 35.868996 | 9.363738 | 34.625212 |

## 6. Сводка отдельно 10/30/60 s

Это **арифметическое среднее трёх episode RMSE** для каждой длительности. Оно отличается от pooled RMSE всех точек; pooled значения сохранены отдельно в [duration_summary.csv](outputs/v2_motion_pilot/duration_summary.csv). Три onsets не считаются независимыми полётами. Процент ниже =100×(mean full−mean gps_only)/mean gps_only, а не среднее трёх процентных изменений.

| Outage, с | full | gps_only | LSTM | zero_IMU | held | CV | full−gps_only, м | full−gps_only, % | full выигрывает |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 2.711519 | 2.239348 | 2.834649 | 2.713809 | 6.690103 | 3.191104 | 0.472171 | 21.085 | 1 / 3 |
| 30 | 8.275485 | 7.004324 | 8.653381 | 8.302129 | 19.452374 | 9.621923 | 1.271161 | 18.148 | 1 / 3 |
| 60 | 15.948862 | 13.363682 | 16.720751 | 16.000087 | 38.705801 | 18.583663 | 2.585180 | 19.345 | 1 / 3 |

По raw absolute RMSE3D средние full/gps_only: 10s **7.469143/7.458512** м; 30s **8.397985/7.697424** м; 60s **13.958842/11.561519** м. Mean final unavailable 3D error full/gps_only: 10s **6.882762/6.809468**, 30s **11.592987/9.565431**, 60s **23.559330/18.916433** м. Таким образом, отрицательный средний вывод не ограничивается одной primary метрикой.

## 7. Full vs gps_only

Знак Δ=`full−gps_only`: **отрицательное значение означает преимущество full**. Процент знаменателем использует gps_only RMSE конкретного episode.

| Scenario | full, м | gps_only, м | Δ, м | Δ, % | Full лучше? |
| --- | --- | --- | --- | --- | --- |
| val_e1_d10s | 2.614048 | 1.808004 | 0.806043 | 44.582 | нет |
| val_e1_d30s | 8.383131 | 5.935091 | 2.448040 | 41.247 | нет |
| val_e1_d60s | 16.688310 | 11.556814 | 5.131496 | 44.402 | нет |
| val_e2_d10s | 2.441036 | 2.474611 | -0.033575 | -1.357 | да |
| val_e2_d30s | 7.481590 | 7.818578 | -0.336988 | -4.310 | да |
| val_e2_d60s | 14.521529 | 15.744118 | -1.222589 | -7.765 | да |
| val_e3_d10s | 3.079475 | 2.435430 | 0.644045 | 26.445 | нет |
| val_e3_d30s | 8.961734 | 7.259304 | 1.702430 | 23.452 | нет |
| val_e3_d60s | 16.636747 | 12.790114 | 3.846633 | 30.075 | нет |

**A. Full выигрывает 3/9**, e2/10, e2/30, e2/60. Это один и тот же onset, а не три независимых подтверждения.

**B. Разница неоднородна:** выигрыш full от **0.033575 м (1.357%)** до **1.222589 м (7.765%)**; проигрыш — от **0.644045 м (26.445%)** до **5.131496 м (44.402%)**. Малый выигрыш 3.4см в e2/10 не назван доказанной практической пользой.

**C. На 30/60s преимущество только в e2:** −0.336988/−1.222589 м. В e1 и e3 full хуже на обеих длительностях. По среднему всех трёх эпизодов full хуже на **1.271161м /18.148%** (30s) и **2.585180м /19.345%** (60s).

По secondary raw absolute RMSE3D full лучше gps_only в5/9, включая e1/10 и e1/30, где primary хуже. Это показывает влияние начального GNSS anchor error и его компенсации ошибкой движения. Primary заранее выбран; заменять его более привлекательной raw metric после просмотра результата нельзя. Пример e1/30: primary full/gps=8.383131/5.935091м, raw full/gps=6.028547/6.207201м.

Источник: [paired_differences.csv](outputs/v2_motion_pilot/paired_differences.csv). Этот файл создан **локальным анализом скачанных predictions/summary**, не является дополнительным Kaggle экспериментом; метрики сверены с исходным summary. Analysis script не меняет исходные результаты.

## 8. Full vs CV / held-GNSS

Таблица показывает заранее установленную primary motion metric; Δ отрицательное — full лучше baseline.

| Scenario | full | CV | full−CV | held | full−held |
| --- | --- | --- | --- | --- | --- |
| val_e1_d10s | 2.614048 | 3.343932 | -0.729885 | 6.755859 | -4.141811 |
| val_e1_d30s | 8.383131 | 10.351077 | -1.967946 | 20.061148 | -11.678016 |
| val_e1_d60s | 16.688310 | 20.446518 | -3.758209 | 40.249201 | -23.560891 |
| val_e2_d10s | 2.441036 | 3.104924 | -0.663888 | 6.041596 | -3.600560 |
| val_e2_d30s | 7.481590 | 9.455026 | -1.973437 | 16.625546 | -9.143957 |
| val_e2_d60s | 14.521529 | 17.001577 | -2.480049 | 35.330335 | -20.808806 |
| val_e3_d10s | 3.079475 | 3.124455 | -0.044980 | 7.272853 | -4.193378 |
| val_e3_d30s | 8.961734 | 9.059666 | -0.097932 | 21.670429 | -12.708695 |
| val_e3_d60s | 16.636747 | 18.302894 | -1.666147 | 40.537868 | -23.901121 |

**E. CV не побеждает full на длинных outages в V2:** full лучше CV в9/9 по primary, в том числе во всех30/60s сценариях. На60s full=16.688310/14.521529/16.636747м, CV=20.446518/17.001577/18.302894м. По raw absolute ошибке CV также хуже full во всех30/60s сценариях; единственное raw преимущество CV среди9 — e1/10 (6.395484 против6.827874м).

Однако **gps_only тоже лучше CV в9/9** по primary и в среднем лучше full на каждой длительности. Следовательно, neural improvement над CV в этом pilot не требует доказанного вклада IMU. Full лучше held в9/9 по relative motion; простое удержание позиции не восстанавливает движение.

LSTM full по среднему primary проигрывает matched GRU full на всех длительностях (2.834649/8.653381/16.720751м против2.711519/8.275485/15.948862м), хотя в отдельных e3 episodes LSTM лучше. У LSTM нет matched gps_only counterpart в этом pilot и другое число параметров, поэтому его нельзя использовать как отдельное доказательство IMU benefit.

Сравнение с R7 не является architecture-only ablation: изменились координаты, синхронизация, target, причинность и обучение outages. Отрицательный R7 результат CV/full нельзя механически объединять с V2 таблицей.

## 9. Zero-IMU sensitivity

Это тот же final full checkpoint с обнулёнными **8 нормализованными IMU-branch features** (6 accel/gyro каналов + gyro age + gap flag). Нули в стандартизованных sensor каналах соответствуют train mean, не физическому нулю ускорения. Общие GNSS/time features неизменны. Это предусмотренный inference probe, не переобученная модель, не causal importance и не изолированная проверка только шести raw sensor каналов.

Δ=`zero_IMU−full`; положительное означает, что обнуление ухудшило primary RMSE. Prediction RMS — `sqrt(mean(||p_zero−p_full||²))` внутри outage; он не равен изменению RMSE относительно GT.

| Scenario | full motion | zero motion | Δ motion, м | Δ motion, % | RMS change predictions, м | Max change predictions, м |
| --- | --- | --- | --- | --- | --- | --- |
| val_e1_d10s | 2.614048 | 2.614379 | 0.000332 | 0.013 | 0.057950 | 0.094166 |
| val_e1_d30s | 8.383131 | 8.306534 | -0.076598 | -0.914 | 0.214988 | 0.489973 |
| val_e1_d60s | 16.688310 | 16.267230 | -0.421079 | -2.523 | 0.657321 | 1.255031 |
| val_e2_d10s | 2.441036 | 2.507252 | 0.066216 | 2.713 | 0.134510 | 0.235092 |
| val_e2_d30s | 7.481590 | 7.724283 | 0.242694 | 3.244 | 0.433428 | 0.778916 |
| val_e2_d60s | 14.521529 | 15.218191 | 0.696662 | 4.797 | 0.883488 | 1.419005 |
| val_e3_d10s | 3.079475 | 3.019796 | -0.059678 | -1.938 | 0.130021 | 0.191510 |
| val_e3_d30s | 8.961734 | 8.875569 | -0.086165 | -0.961 | 0.297259 | 0.449899 |
| val_e3_d60s | 16.636747 | 16.514841 | -0.121906 | -0.733 | 0.473009 | 0.749383 |

**D. Выход full меняется заметнее, чем у R7:** inside-outage RMS разности **0.057950–0.883488м**, максимальная pointwise разность до **1.419005м**. Однако влияние на точность непоследовательно: zero хуже full в4/9 (e1/10 и триe2), лучше в5/9. Самое большое ухудшение от zero — e2/60 **+0.696662м (+4.797%)**; самое большое улучшение — e1/60 **−0.421079м (−2.523%)**.

Средний primary zero/full: 10s2.713809/2.711519м; 30s8.302129/8.275485м; 60s16.000087/15.948862м. Средние изменения **+0.002290/+0.026644/+0.051226м** малы и не являются устойчивым практическим доказательством пользы. Вне outage prediction difference равен0 по конструкции decoder (позиция=held fix), что ничего не говорит о внутренних recurrent activations.

Чувствительность к IMU-ветви действительно появилась, но «выход зависит от входа» и «этот вход улучшает прогноз» — разные выводы. Matched обученная gps_only всё равно лучше full в среднем и в6/9 scenarios.

## 10. Вывод: есть ли практический IMU benefit

**Не подтверждён.** V2 full проявляет измеримую зависимость от IMU-branch input и выигрывает в одном onset, но не даёт повторяемого преимущества над matched gps_only на нескольких разных заранее заданных onset. В среднем full хуже на всех трёх длительностях, а проигрыши e1/e3 превосходят выигрыши e2.

**F. Это не простое повторение R7 static GNSS→GT correction task:** V2 не получает absolute XYZ/time/GT, обучается относительному движению и выдаёт интегрируемый residual velocity; constant GT translation не меняет target. На доступном GNSS residual position принудительно0. Поэтому выигрыш над CV относится к прогнозу движения в данной постановке, а не к fitted absolute frame offset. Но это **не доказывает использование физической инерциальной динамики**: сеть может улучшать GNSS velocity prior по GNSS history, а IMU может влиять на выход без практической пользы. gps_only демонстрирует именно такую возможность; zero probe не отделяет sensor values от двух дополнительных IMU-branch timing features.

Корректная формулировка результата: «В одно-seed validation pilot на одном полёте matched GRU full не превзошла gps_only устойчиво; средняя relative-motion RMSE была выше на21.09%,18.15%,19.34% для10/30/60s. Чувствительность full к обнулению IMU выявлена, но не дала устойчивого выигрыша в точности. Обе модели превзошли causal GNSS CV на выбранных эпизодах по primary metric».

Нельзя формулировать: «IMU бесполезна вообще», «V2 доказала generalization», «GRU научилась физическому INS», «5 seeds подтвердят преимущество» или «отрицательный результат вызван именно архитектурой». Причины отрицательного результата этим одним pilot не изолированы.

## 11. Ограничения

- Один seed и один flight; temporal validation уже использовалась в диагностике. Статистического доказательства generalization нет. Три duration на одном onset вложены;9 scenarios не9 независимых episodes/полётов.
- Target — camera relative motion; GT около1Hz и interpolated, photogrammetric reference связан с GNSS. Точные GT height datum, GNSS antenna extrinsics и physical receiver fix epoch по-прежнему не установлены.
- Coordinate-change events — proxy GNSS fixes. Common frame исправляет горизонтальные оси/origin, но не завершает вертикальную/extrinsic калибровку. Relative labels устраняют constant offsets в target, не выдают их за физически исправленные inputs.
- Train/eval dynamic target, raw anchor-biased error и R7 absolute correction — разные протоколы. Их значения не объединяются в общий leaderboard. Raw error иногда маскирует плохой motion из-за взаимной компенсации ошибок.
- 8epochs и маленькая модель — зафиксированный дешёвый pilot, не доказательство оптимальности или окончательный предел возможностей IMU. Loss curve со сменой augmentation не является learning curve на фиксированной выборке.
- Zero-IMU probe меняет все8 features IMU ветви и может вводить непривычные сочетания inputs. Это sensitivity, не causal attribution. LSTM не имеет matched ablation здесь.
- В выполненном config/manifest сохранены подготовительные поля `prepared_only: true` и `training_executed: false` как часть неизменяемого **плана**; они не обновлялись, чтобы не нарушить frozen hashes. Факт исполнения подтверждают отдельный `status.json`, 24history rows, checkpoints, Kaggle logs и60predictions. Плановые флаги не следует читать как actual run status.

Сохранённый комплект:

- [status.json](outputs/v2_motion_pilot/status.json), [summary.csv](outputs/v2_motion_pilot/summary.csv), [paired_differences.csv](outputs/v2_motion_pilot/paired_differences.csv), [training_history.csv](outputs/v2_motion_pilot/training_history.csv).
- [config.json](outputs/v2_motion_pilot/config.json), [manifest.json](outputs/v2_motion_pilot/manifest.json), [duration_summary.csv](outputs/v2_motion_pilot/duration_summary.csv), [independent_verification.json](outputs/v2_motion_pilot/independent_verification.json).
- `outputs/v2_motion_pilot/checkpoints/`: три final epoch8 checkpoints. `outputs/v2_motion_pilot/val_*/`:60prediction NPZ с GT, timestamps, masks, motion targets/errors и source metadata. `source_snapshot/`, `source_sha256.json`, environment/scaler/mask manifest также скачаны.
- Для передачи ChatGPT: [V2_PILOT_CHATGPT_HANDOFF.zip](outputs/v2_motion_pilot/V2_PILOT_CHATGPT_HANDOFF.zip). Внутри этот отчёт, основные CSV/JSON, preflight/tests и run provenance; большие predictions/checkpoints остаются отдельно и доступны для проверки. Секретов/credentials в архиве нет.

## 12. Рекомендация: STOP или переход к 5 seeds

**STOP. Переход к 5 seeds сейчас не рекомендован:** заранее установленный критерий повторяемого практического преимущества full над gps_only на нескольких разных fixed episodes не выполнен. Не выбирали другой epoch/checkpoint, не меняли параметры после результатов, не запускали дополнительные модели, seeds или probes сверх подготовленного pilot.

Следующий шаг — передать сохранённый отчёт/комплект ChatGPT для отдельного обсуждения отрицательного результата. Любое последующее обучение требует нового явного решения пользователя; этот запуск завершён и остановлен.
