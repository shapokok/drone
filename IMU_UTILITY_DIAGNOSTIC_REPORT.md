# IMU_UTILITY_DIAGNOSTIC_REPORT

Дата: 2026-09-26 (Asia/Almaty). Локальная физическая диагностика, протокол `imu_utility_fixed_v1`. **Нового обучения, seeds, изменения V2 checkpoints, Kaggle jobs и использования test не было.**

## 1. Executive summary

**Практическая польза IMU сверх causal GNSS constant velocity при текущей доступной калибровке не подтверждена.** Заранее выбранный primary strapdown с zero bias проиграл CV в **9/9** fixed outage scenarios. Заранее заданные mean-bias, median-bias, CV+inertial-acceleration-residual и horizontal-only варианты также не выиграли ни одного 3D сравнения; ни один не улучшил horizontal RMSE относительно CV.

Средний relative-motion RMSE3D primary DR/CV: **24.658498/3.191104м (10s)**, **550.016576/9.621923м (30s)**, **3806.981579/18.583663м (60s)**. Большие значения — сохранённый отрицательный результат, не исправлялись подбором параметров. Математика SO(3) и метрики проверены отдельно.

Диагноз относится к **пригодности текущих IMU сигналов и доступной инициализации для strapdown**, а не к фундаментальному отсутствию информации в датчике. До outage уже наблюдается несовместимость propagated gyro attitude с направлением low-pass accelerometer gravity: финальная угловая разность примерно39–44° за25s. Есть radial force discrepancy 0.42–0.56м/с² и ненулевая средняя gyro. Истинный body yaw из GNSS course мультикоптера не наблюдается. Все эти факторы препятствуют интерпретации «датчик плохой».

Итоговые ответы: **A — NO under current usable calibration; B — MAYBE, после проверки физической калибровки; C — исправить physical calibration/preprocessing.** Новое neural training сейчас не нужно. Это diagnostic utility check, не новый основной baseline статьи и не настроенный EKF.

## 2. Input data / causal protocol

Контекст: [PROJECT_HANDOFF.md](PROJECT_HANDOFF.md), [PROTOCOL_FIX_REPORT.md](PROTOCOL_FIX_REPORT.md), [NEXT_STAGE_DIAGNOSTIC_REPORT.md](NEXT_STAGE_DIAGNOSTIC_REPORT.md), [V2_PILOT_REPORT.md](V2_PILOT_REPORT.md). Числа текущего отчёта получены из новых локальных артефактов, не из заявлений старых заметок.

Использован **V2**, не R7 preprocessing:

- GNSS — **точные сохранённые UTM32N координаты V2** из `outputs/v2_motion_pilot/val_control/held_gnss.npz`, поля `pred`, `source_timestamp`, `source_index`. Это baseline held-GNSS, не нейросетевой прогноз. Проекция была выполнена и проверена реальным pyproj3.7.2 в V2 Kaggle output **352778350**; здесь она не аппроксимировалась и не вычислялась заново.
- Общий origin V2: (465670.706847399,5247978.033436624,464.91)м, world X/East, Y/North, Z/up. Нет fitted alignment, смены geoid или дополнительного zeroing координат.
- IMU заново прочитана из pinned `RawAccel.csv` и `RawGyro.csv`. Stable sort/dedup gyro, backward join `source_t<=accel_t`, SI exported x/y/z без нормализации и без повторного scaling. Это та же causal синхронизация V2; R7 future-gyro lookup не используется.
- Common timestamps точно совпали с V2 `[18935,21623)` (2688 samples). Фиксированные маски/onsets/cutoffs взяты из выполненного V2 manifest. Никакой случайной генерации и новых seeds.
- Во время outage интегратор принимает **только IMU, timestamps и заранее оценённое initial state**; в его аргументах нет GT, GNSS, onboard pose или post-outage measurements. Последующие GNSS значения не корректируют состояние.
- Initial states используют только историю до onset; gravity/bias samples дополнительно ограничены временем последнего legal GNSS source. Для каждого отдельного сценария pre-history считается доступной; сценарии независимы как отдельные маски, но интервалы на одном flight перекрываются и не являются независимыми полётами.
- GT и `motion_target` из V2 загружаются после pre-outage initialization/signal stage только для scoring и описания движения. GT никогда не сбрасывает INS position, velocity или attitude.

Manifest содержит hashes input files, config и всех9fixed scenarios: [manifest.json](outputs/imu_utility_diagnostic/manifest.json). Параметры и primary variant сохранены **до расчёта outage errors** в [config.json](outputs/imu_utility_diagnostic/config.json), копия [configs/imu_utility_diagnostic.json](configs/imu_utility_diagnostic.json). Фактический статус: [status.json](outputs/imu_utility_diagnostic/status.json).

Траектории сохранены только на outage timestamps, но каждая NPZ также содержит исходную common availability mask и outage indices. Методы оцениваются на одинаковых 100/300/600 points; источники и время начала проверены хешами.

## 3. Initial state estimation

**Position:** последний разрешённый GNSS fix. **Velocity:** точно повторена V2 causal regression последних5секунд по уникальным разрешённым fixes, включая float32 velocity rounding V2. Все9CV trajectories воспроизведены с **max difference0.0м** относительно сохранённого V2 CV. Это исключает смену initial velocity как объяснение различий.

**Roll/pitch:** среднее specific force за5секунд до anchor; его направление отображается в world up. R построена как proper orthonormal body→world rotation. **Yaw:** body-forward считается направленным по GNSS course. Заранее фиксированные критерии: horizontal speed≥0.5м/с и course std за5s≤30°. Если не выполнены — fallback east yaw=0°, с записью причины. Здесь все три course оценки прошли численные пороги; fallback не потребовался.

**Это не доказательство истинного body yaw.** Мультикоптер может двигаться боком или вращаться независимо от course. Поле `course_reliable_for_course_only` намеренно относится только к course; sensor/body mounting и yaw остаются допущением. Разрешённых данных для независимой проверки истинного heading нет.

| Onset | Last legal source, с | v0 XYZ, м/с | H speed | Course yaw east-CCW, ° | Course std, ° | Gravity samples | Initial down-axis tilt, ° |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1975.763195 | (0.61141, -1.14674, 0.47512) | 1.299553 | -61.934774 | 4.597473 | 38 | 6.925992 |
| 2 | 2015.765278 | (0.26605, -1.14794, -0.44554) | 1.178373 | -76.951216 | 11.446249 | 50 | 7.233267 |
| 3 | 2055.772174 | (0.32515, -1.13150, 0.56378) | 1.177292 | -73.967184 | 6.889954 | 50 | 7.305539 |

**Bias варианты заранее фиксированы**, без выбора лучшего по validation:

| Вариант | Определение | Статус |
| --- | --- | --- |
| zero | ba=0, bg=0 | **Primary**: не предполагает, что движение/вращение до outage отсутствует |
| mean | ba=mean(f_pre)−R0ᵀ(0,0,g), bg=mean(ω_pre) | Sensitivity; условная quasi-static pseudo-bias оценка |
| median | ba=componentwise median(f_pre)−R0ᵀ(0,0,g), bg=median(ω_pre) | Sensitivity; устойчивее к выбросам, но та же неопределённость истинного движения |

R0 одинаковая между bias вариантами, из mean gravity. Среднее IMU на движущемся дроне нельзя безусловно называть физическим bias: оно смешивает bias с реальной angular rate/acceleration и ошибкой tilt. Эти варианты проверяют зависимость от доступной causal correction, не идентифицируют истинную калибровку.

| Onset | Mean accel XYZ, м/с² | Norm mean accel | Norm−g, м/с² | ba mean XYZ, м/с² | bg mean XYZ, rad/s |
| --- | --- | --- | --- | --- | --- |
| 1 | (0.87882, -0.71427, -9.32282) | 9.391356 | -0.415294 | (-0.03886, 0.03159, 0.41226) | (-0.01615, -0.02476, 0.02452) |
| 2 | (0.44418, -1.07634, -9.17428) | 9.247872 | -0.558778 | (-0.02684, 0.06504, 0.55433) | (-0.01314, -0.02048, 0.03631) |
| 3 | (0.60781, -1.01812, -9.24917) | 9.324866 | -0.481784 | (-0.03140, 0.05260, 0.47787) | (-0.02239, -0.02691, 0.03424) |

Все R0 matrices, p0/v0, median/mean/std и точные границы initial histories — [initial_states.json](outputs/imu_utility_diagnostic/initial_states.json).

## 4. IMU units, frames, gravity and bias checks

Первичные источники: [официальное описание Zurich MAV](https://rpg.ifi.uzh.ch/zurichmavdataset.html) указывает RawAccel/RawGyro и timestamp clock PX4; [PX4 SensorAccel](https://docs.px4.io/main/en/msg_docs/SensorAccel) описывает exported acceleration в м/с² и FRD board frame; [PX4 SensorGyro](https://docs.px4.io/main/en/msg_docs/SensorGyro) — rad/s и FRD board frame. Современная PX4 документация поддерживает рабочую SI/FRD интерпретацию, но не доказывает конкретную mounting/calibration конфигурацию старого полёта. В локальном `raw/readme.txt` перечислены columns/scaling/raw counts, но не полный IMU calibration contract.

Проверки единиц до интеграции:

- Accel norm около9.3м/с², dominant z отрицательная; это согласуется с **specific force в down-positive body frame**, а не с уже очищенным world acceleration. g=9.80665м/с² фиксировано, world gravity=(0,0,−g). Используется `a_world=R(f−ba)+gravity`.
- Exported range акселерометра156.9064≈16g; gyro range34.906586rad/s≈2000deg/s. Raw scaling поля0.0047884034 и0.0010642195 не умножаются на exported x/y/z повторно. Ошибочный heading fit не является основанием объявить gyro в degrees/s.
- Static convention test: при FRD `[0,0,−g]` и R=diag(1,−1,−1) world acceleration точно0; body-forward при yaw90° указывает north; положительный body-z gyro при FRD даёт отрицательный world east-CCW yaw. RᵀR=I, detR=+1.
- Простая affine зависимость scaled raw counts→exported XYZ на pre-first-onset данных не восстановилась с малым residual. Axis RMSE fit accel=(0.229733,0.216897,0.545573)м/с²; gyro=(0.070964,0.041159,0.018747)rad/s. Эти поля могут различаться внутренней обработкой/временем, но причина не установлена. **Raw counts не подменяли exported signals**, fit не использовали для калибровки. Источник: [units_checks.json](outputs/imu_utility_diagnostic/units_checks.json).

Из-за возможной неоднозначности отдельно сделан **pre-outage convention consistency test**, без GT и без outage RMSE: положительный rad/s (основная трактовка по metadata), положительный degrees/s и отрицательный rad/s. Последние два — только проверки того, насколько информативен GNSS course; они не запускались как alternative outage DR и не выбирались по ошибке. На последних≈25s pre-outage (после5s initialization):

| Onset | Конвенция | Integrated body heading Δ, ° | GNSS course Δ, ° | Конечное расхождение Δ, ° | RMS расхождения, ° | Применена в DR? |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | gyro_rad_s_positive | -49.843590 | -1.317778 | -48.525812 | 31.142708 | да |
| 1 | gyro_deg_s_positive | -0.661719 | -1.317778 | 0.656059 | 15.893687 | нет |
| 1 | gyro_rad_s_negative | 41.640064 | -1.317778 | 42.957841 | 27.404234 | нет |
| 2 | gyro_rad_s_positive | -80.771387 | -12.136116 | -68.635271 | 55.271984 | да |
| 2 | gyro_deg_s_positive | -1.269868 | -12.136116 | 10.866248 | 10.591993 | нет |
| 2 | gyro_rad_s_negative | 84.597277 | -12.136116 | 96.733393 | 51.420506 | нет |
| 3 | gyro_rad_s_positive | -67.102625 | 55.251472 | -122.354097 | 126.115846 | да |
| 3 | gyro_deg_s_positive | -0.983107 | 55.251472 | -56.234580 | 89.210582 | нет |
| 3 | gyro_rad_s_negative | 60.419786 | 55.251472 | 5.168314 | 58.102184 | нет |

**Данные не валидируют body-heading convention однозначно.** Degrees/s случайно ближе к course вe1, смена знака ближе по final change вe3; ни одна интерпретация не даёт достаточного основания менять units/sign. GNSS course не равен body yaw, и sensor biases тоже влияют. Вывод — явная неопределённость, а не молчаливое угадывание или подбор «лучших единиц».

Независимая от GNSS course проверка tilt: propagated orientation сравнивается с текущим trailing1s mean accelerometer direction. Это quasi-static gravity consistency, не GT attitude; translational acceleration может нарушать предположение. Тем не менее, наклон из accel остаётся≈5.5–10°, тогда как zero-bias gyro propagation за≈25s даёт32–38° down-axis tilt:

| Onset | Gyro propagated tilt в конце, ° | Accel-implied tilt, ° | Gravity-direction disagreement median, ° | Disagreement final, ° | GNSS acceleration H RMS, м/с² |
| --- | --- | --- | --- | --- | --- |
| 1 | 36.857351 | 7.223550 | 22.218458 | 44.067106 | 0.162288 |
| 2 | 32.193820 | 7.113873 | 19.644148 | 38.984558 | 0.250907 |
| 3 | 38.071860 | 5.653632 | 21.951470 | 43.541497 | 0.560307 |

Источник: [gravity_consistency.json](outputs/imu_utility_diagnostic/gravity_consistency.json), сохранённые `e*_preoutage_signal.npz`. Проверка указывает на проблему согласованности calibration/export/attitude propagation; не доказывает, что именно gyro bias является единственной причиной.

## 5. Signal utility diagnostics

Signal stage выполнен **до scoring основной outage интеграции**, только на30секундах разрешённой pre-outage history. На нём не меняли параметры. Ниже mean/std по axes; std=ddof0, не seed uncertainty. Все подробности/magnitudes/min/max — [signal_diagnostics.json](outputs/imu_utility_diagnostic/signal_diagnostics.json).

| Onset | Accel mean XYZ, м/с² | Accel std XYZ | Gyro mean XYZ, rad/s | Gyro std XYZ |
| --- | --- | --- | --- | --- |
| 1 | (0.88658, -0.83605, -9.27656) | (0.23015, 0.21428, 0.49213) | (-0.01627, -0.02740, 0.03076) | (0.07171, 0.04003, 0.02071) |
| 2 | (0.71319, -0.99468, -9.19856) | (0.36864, 0.37657, 0.57954) | (-0.01605, -0.02614, 0.04786) | (0.09476, 0.05502, 0.07206) |
| 3 | (0.62509, -1.09736, -9.24218) | (0.27397, 0.26425, 0.45243) | (-0.01719, -0.02548, 0.04158) | (0.08464, 0.05945, 0.05482) |

GNSS velocity proxy — причинная5s regression. Acceleration proxy — изменение этой velocity за предыдущую≈1s, делённое на actual elapsed time. Estimated world IMU acceleration и GNSS proxy сглажены только trailing1s mean. Не применяются centered filters, будущие fixes или GT. Correlations ниже — diagnostic consistency, **не causal proof** и не supervision для настройки:

| Onset | GNSS velocity Δ за30s XYZ, м/с | GNSS course Δ30s, ° | Pearson estimated accel ↔ GPS proxy XYZ | Pearson magnitudes | Accel magnitude mean/std, м/с² |
| --- | --- | --- | --- | --- | --- |
| 1 | (0.12827, -0.25913, 0.74826) | -0.495091 | (-0.05713, 0.22360, -0.24688) | 0.029242 | 9.362576 / 0.472006 |
| 2 | (-0.26150, -0.20259, -0.64511) | -16.114698 | (0.23767, -0.16975, 0.25215) | 0.008546 | 9.296299 / 0.551329 |
| 3 | (1.59714, -0.41509, 2.97831) | 76.643729 | (0.39623, 0.61311, 0.15184) | 0.047011 | 9.336681 / 0.434595 |

Есть изменчивость сигнала, но надёжного consistent matching нет: axis correlations наe1/e2 слабы и меняют знак; e3 показывает положительную horizontal association (0.396/0.613), при этом orientation consistency именно там тоже нарушена. Correlation magnitudes около0.009–0.047 почти не показывает связи между величиной specific force и этим шумным GNSS acceleration proxy. Метровый GNSS noise и дифференцирование ограничивают интерпретацию.

**Манёвры и насколько близок CV.** Доe1 course за30s почти не меняется (−0.50°), доe2 меняется−16.11°, доe3 +76.64°. Это изменения GNSS course, не доказанные body turns: multipath/velocity noise могут усиливать видимый манёвр. GT используется только после интеграции для независимого описания outage motion. На60s mean/std horizontal speed из interpolated GT: e1 **1.127/0.198**, e2 **1.092/0.211**, e3 **1.154/0.225**м/с. Компоненты velocity также меняются; участки не абсолютно constant-velocity, но horizontal CV уже сравнительно хорош.

Raw accel magnitude в60s outages mean/std: e1 **9.303/0.536**, e2 **9.332/0.443**, e3 **9.328/0.523**м/с²; gyro magnitude mean/std≈0.122/0.064,0.106/0.051,0.104/0.047rad/s. Это подтверждает ненулевой изменяющийся signal, но не отделяет полезную motion информацию от biases/noise/aliasing. Exact per-duration statistics и GT velocity description: [error_mechanisms.json](outputs/imu_utility_diagnostic/error_mechanisms.json).

У CV средний60s horizontal RMSE≈5.995м, vertical≈17.472м; его3D ошибка сильно связана с вертикальной скоростью initial GNSS regression. Поэтому «CV тоже ошибается» не означает, что корректно калиброванный IMU обязательно исправит этот конкретный reference/anchor error. В то же время наличие изменений движения не позволяет списать провал DR только на отсутствие манёвров.

## 6. Physical dead-reckoning implementation

Код: [src/diagnostics/imu_utility.py](src/diagnostics/imu_utility.py), запуск [run_imu_utility_diagnostic.py](scripts/run_imu_utility_diagnostic.py). Исходный EKF и V2 не редактировались и не использовались как настроенная INS реализация.

Состояние `(p,v,R)`, R body→world. Для каждого фактического интервала h:

```
R_mid  = R · Exp((gyro − bg) · h/2)
a_world = R_mid · (specific_force − ba) + (0,0,−g)
p_next = p + v·h + 0.5·a_world·h²
v_next = v + a_world·h
R_next = R · Exp((gyro − bg) · h)
```

Rodrigues exponential сохраняет SO(3). IMU выбирается строго с **левого конца** интервала; интеграция делится по accel timestamps, gyro уже matched backward. Midpoint orientation использует тот же прошлый gyro, не будущую выборку. dt не фиксируется0.1 и gaps не обрезаются. Первый шаг начинается с actual last legal GNSS source timestamp; timestamps вывода точно те же, что у V2.

| Метод | Отличие |
| --- | --- |
| CV | p=lastGNSS+v0×actual elapsed; IMU не используется |
| DR zero PRIMARY | Полный strapdown, ba=bg=0 |
| DR mean / median | Тот же R0/p0/v0, pre-history pseudo-bias из раздела3 |
| CV+acc residual | Тот же zero-bias gyro propagation; из a_world вычитается замороженное pre-outage `a_ref=R0 mean(f_pre)+g_world` |
| Horizontal DR | Тот же zero-bias strapdown, world acceleration Z=0, vertical velocity остаётся v0_Z |

Почему C не дублирует B: просто назвать `v0 + integral(acceleration)` CV+IMU означало бы тот же INS. Здесь residual означает **изменение acceleration относительно pre-outage уровня**, явно определённое до результатов. Оно снимает постоянный начальный world-force offset, но не исправляет дальнейший gyro-induced gravity leakage. Horizontal вариант отделяет влияние вертикальной интеграции, не исправляя горизонтальный tilt/yaw.

Проверены9 synthetic tests: gravity/FRD sign, yaw/right multiplication, irregular dt, постоянное ускорение, causal prefix при изменении будущей IMU, decomposition identity, horizontal-only, known bias removal, causal GNSS velocity. [tests/test_imu_utility.py](tests/test_imu_utility.py). Max rotation orthogonality error во всех реальных traces **6.8834e−15**, detR≈1. Это исключает численное разрушение rotation matrix, но не доказывает её физическую правильность.

Полная независимая проверка54rows: [verification.json](outputs/imu_utility_diagnostic/verification.json). Saved predictions finite, дубликатов нет, метрики пересчитаны с расхождением0.0; все89зафиксированных V2 artifacts сохранили hashes. Повторных neural forwards/обучений не выполнялось.

## 7. Results for all 9 outages

**Primary metric:** relative-motion RMSE3D внутри outage, как V2: `(p(t)−last_GNSS_anchor)−(GT(t)−GT(last_source_time))`. Horizontal=`sqrt(mean(ex²+ey²))`, vertical=`sqrt(mean(ez²))`. Final motion error —3D norm в последней недоступной точке; raw final error —norm `(p−GT)` в той же точке. Это разные ошибки из-за GNSS anchor error. Все числа в метрах; никакого alignment.

Ниже все54сочетания. CSV содержит полную precision, дополнительно raw RMSE3D/H/V и horizontal/vertical final errors: [summary.csv](outputs/imu_utility_diagnostic/summary.csv). Округление таблицы до6знаков. N=100/300/600 для10/30/60s, один flight; нейросетевых seeds в этом расчёте нет.

| Scenario | Метод | Motion RMSE3D | Motion H | Motion V | Final motion3D | Final raw3D |
| --- | --- | --- | --- | --- | --- | --- |
| val_e1_d10s | CV | 3.343932 | 0.944873 | 3.207662 | 5.924440 | 4.222216 |
| val_e1_d10s | DR zero PRIMARY | 20.660293 | 18.134893 | 9.898149 | 52.575152 | 58.109776 |
| val_e1_d10s | DR mean | 4.194369 | 4.108489 | 0.844423 | 9.682003 | 11.794225 |
| val_e1_d10s | DR median | 7.411410 | 7.371210 | 0.770885 | 18.164767 | 19.238187 |
| val_e1_d10s | CV+acc residual | 18.161736 | 18.134893 | 0.987074 | 46.959901 | 49.665900 |
| val_e1_d10s | Horizontal DR | 18.416391 | 18.134893 | 3.207662 | 47.195614 | 48.405139 |
| val_e1_d30s | CV | 10.351077 | 2.671338 | 10.000438 | 18.201894 | 11.033600 |
| val_e1_d30s | DR zero PRIMARY | 516.838928 | 478.005278 | 196.553890 | 1346.576710 | 1351.089483 |
| val_e1_d30s | DR mean | 55.649313 | 53.736012 | 14.466757 | 135.247500 | 137.481069 |
| val_e1_d30s | DR median | 141.441531 | 139.900268 | 20.823580 | 359.618662 | 360.474247 |
| val_e1_d30s | CV+acc residual | 491.481336 | 478.005278 | 114.301611 | 1286.449380 | 1289.948807 |
| val_e1_d30s | Horizontal DR | 478.109877 | 478.005278 | 10.000438 | 1243.636421 | 1244.903492 |
| val_e1_d60s | CV | 20.446518 | 5.153008 | 19.786526 | 34.975857 | 27.613792 |
| val_e1_d60s | DR zero PRIMARY | 3605.661297 | 3213.400203 | 1635.497637 | 9032.162564 | 9036.874551 |
| val_e1_d60s | DR mean | 325.487941 | 316.753135 | 74.898939 | 809.716245 | 811.145627 |
| val_e1_d60s | DR median | 983.408345 | 970.926158 | 156.186963 | 2532.946795 | 2533.546473 |
| val_e1_d60s | CV+acc residual | 3468.946968 | 3213.400203 | 1306.771671 | 8696.636167 | 8700.804094 |
| val_e1_d60s | Horizontal DR | 3213.461121 | 3213.400203 | 19.786526 | 7916.015088 | 7916.719767 |
| val_e2_d10s | CV | 3.104924 | 2.123466 | 2.265269 | 5.620092 | 7.558133 |
| val_e2_d10s | DR zero PRIMARY | 28.438564 | 23.723748 | 15.682337 | 68.948241 | 71.829214 |
| val_e2_d10s | DR mean | 7.629507 | 7.536883 | 1.185226 | 17.621206 | 23.793975 |
| val_e2_d10s | DR median | 8.793846 | 8.366953 | 2.706628 | 20.690832 | 25.872387 |
| val_e2_d10s | CV+acc residual | 23.873295 | 23.723748 | 2.667956 | 59.696018 | 65.305825 |
| val_e2_d10s | Horizontal DR | 23.831653 | 23.723748 | 2.265269 | 59.480614 | 65.301903 |
| val_e2_d30s | CV | 9.455026 | 6.352726 | 7.002884 | 15.051979 | 13.751564 |
| val_e2_d30s | DR zero PRIMARY | 565.012052 | 520.378949 | 220.100814 | 1449.999438 | 1453.098940 |
| val_e2_d30s | DR mean | 123.146375 | 123.139154 | 1.333532 | 313.184432 | 319.281628 |
| val_e2_d30s | DR median | 157.303672 | 156.385580 | 16.970430 | 404.915176 | 409.921517 |
| val_e2_d30s | CV+acc residual | 531.457766 | 520.378949 | 107.949553 | 1370.615971 | 1374.847833 |
| val_e2_d30s | Horizontal DR | 520.426067 | 520.378949 | 7.002884 | 1334.068035 | 1339.582059 |
| val_e2_d60s | CV | 17.001577 | 8.096187 | 14.950097 | 28.466371 | 25.033714 |
| val_e2_d60s | DR zero PRIMARY | 3901.133823 | 3422.261409 | 1872.691099 | 9875.486091 | 9876.609486 |
| val_e2_d60s | DR mean | 915.214345 | 914.085139 | 45.449504 | 2415.150698 | 2420.850262 |
| val_e2_d60s | DR median | 1200.949688 | 1191.045747 | 153.916157 | 3172.443640 | 3177.158099 |
| val_e2_d60s | CV+acc residual | 3709.165743 | 3422.261409 | 1430.397620 | 9394.871872 | 9396.608316 |
| val_e2_d60s | Horizontal DR | 3422.294064 | 3422.261409 | 14.950097 | 8464.098702 | 8468.496924 |
| val_e3_d10s | CV | 3.124455 | 0.847221 | 3.007396 | 5.232819 | 11.089647 |
| val_e3_d10s | DR zero PRIMARY | 24.876637 | 22.577029 | 10.446283 | 62.893136 | 65.609963 |
| val_e3_d10s | DR mean | 3.212745 | 1.858936 | 2.620322 | 7.361783 | 15.284818 |
| val_e3_d10s | DR median | 8.879820 | 8.520593 | 2.500139 | 20.660117 | 20.439911 |
| val_e3_d10s | CV+acc residual | 22.614899 | 22.577029 | 1.308227 | 57.608186 | 62.123840 |
| val_e3_d10s | Horizontal DR | 22.776449 | 22.577029 | 3.007396 | 57.828344 | 62.673910 |
| val_e3_d30s | CV | 9.059666 | 2.868498 | 8.593560 | 15.631042 | 19.719582 |
| val_e3_d30s | DR zero PRIMARY | 568.198747 | 530.801488 | 202.730354 | 1457.840042 | 1461.110526 |
| val_e3_d30s | DR mean | 83.898514 | 83.548565 | 7.654937 | 230.355153 | 234.979172 |
| val_e3_d30s | DR median | 165.320970 | 165.267604 | 4.200259 | 417.860572 | 414.340161 |
| val_e3_d30s | CV+acc residual | 541.382732 | 530.801488 | 106.513108 | 1393.046430 | 1397.075422 |
| val_e3_d30s | Horizontal DR | 530.871048 | 530.801488 | 8.593560 | 1357.580525 | 1362.646089 |
| val_e3_d60s | CV | 18.302894 | 4.736114 | 17.679512 | 31.640647 | 35.868996 |
| val_e3_d60s | DR zero PRIMARY | 3914.149617 | 3488.234185 | 1775.609612 | 9906.367175 | 9909.868735 |
| val_e3_d60s | DR mean | 816.938382 | 814.502598 | 63.038373 | 2228.410622 | 2231.148090 |
| val_e3_d60s | DR median | 1140.125297 | 1135.618756 | 101.270597 | 2932.139873 | 2927.336524 |
| val_e3_d60s | CV+acc residual | 3756.480053 | 3488.234185 | 1394.046147 | 9512.646072 | 9516.647245 |
| val_e3_d60s | Horizontal DR | 3488.278988 | 3488.234185 | 17.679512 | 8653.740391 | 8659.919193 |

Scenario NPZ находятся в `outputs/imu_utility_diagnostic/val_*/`: predictions, timestamps, GT/targets только для scoring, p0/v0, полная исходная mask/outage indices, orientation/velocity/acceleration, bias и источник IMU interval. Ни один исходный V2 файл не перезаписан.

## 8. IMU DR vs constant-velocity

Δ=`DR−CV`; отрицательное означало бы выигрыш DR. Главный вывод основан на **заранее назначенном primary zero-bias**, не на выборе лучшего sensitivity метода.

| Scenario | Primary DR3D, м | CV3D, м | Δ3D, м | Δ, % | Δ horizontal, м | Δ vertical, м |
| --- | --- | --- | --- | --- | --- | --- |
| val_e1_d10s | 20.660293 | 3.343932 | 17.316361 | 517.844 | 17.190020 | 6.690487 |
| val_e1_d30s | 516.838928 | 10.351077 | 506.487851 | 4893.093 | 475.333940 | 186.553452 |
| val_e1_d60s | 3605.661297 | 20.446518 | 3585.214779 | 17534.598 | 3208.247196 | 1615.711111 |
| val_e2_d10s | 28.438564 | 3.104924 | 25.333640 | 815.918 | 21.600282 | 13.417068 |
| val_e2_d30s | 565.012052 | 9.455026 | 555.557026 | 5875.785 | 514.026223 | 213.097930 |
| val_e2_d60s | 3901.133823 | 17.001577 | 3884.132246 | 22845.717 | 3414.165223 | 1857.741003 |
| val_e3_d10s | 24.876637 | 3.124455 | 21.752182 | 696.191 | 21.729808 | 7.438887 |
| val_e3_d30s | 568.198747 | 9.059666 | 559.139081 | 6171.741 | 527.932990 | 194.136794 |
| val_e3_d60s | 3914.149617 | 18.302894 | 3895.846723 | 21285.414 | 3483.498071 | 1757.930099 |

Среднее **трёх episode RMSE**, не pooled RMSE и не среднее независимых полётов:

| Outage, с | Метод | Mean motion3D | Mean H | Mean V | Mean final motion3D | Лучше CV /3 |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | CV | 3.191104 | 1.305187 | 2.826776 | 5.592450 | — |
| 10 | DR zero PRIMARY | 24.658498 | 21.478557 | 12.008923 | 61.472176 | 0 |
| 10 | DR mean | 5.012207 | 4.501436 | 1.549990 | 11.554997 | 0 |
| 10 | DR median | 8.361692 | 8.086252 | 1.992551 | 19.838572 | 0 |
| 10 | CV+acc residual | 21.549977 | 21.478557 | 1.654419 | 54.754701 | 0 |
| 10 | Horizontal DR | 21.674831 | 21.478557 | 2.826776 | 54.834857 | 0 |
| 30 | CV | 9.621923 | 3.964188 | 8.532294 | 16.294972 | — |
| 30 | DR zero PRIMARY | 550.016576 | 509.728572 | 206.461686 | 1418.138730 | 0 |
| 30 | DR mean | 87.564734 | 86.807910 | 7.818409 | 226.262361 | 0 |
| 30 | DR median | 154.688724 | 153.851151 | 13.998090 | 394.131470 | 0 |
| 30 | CV+acc residual | 521.440611 | 509.728572 | 109.588091 | 1350.037260 | 0 |
| 30 | Horizontal DR | 509.802331 | 509.728572 | 8.532294 | 1311.761660 | 0 |
| 60 | CV | 18.583663 | 5.995103 | 17.472045 | 31.694292 | — |
| 60 | DR zero PRIMARY | 3806.981579 | 3374.631933 | 1761.266116 | 9604.671943 | 0 |
| 60 | DR mean | 685.880223 | 681.780291 | 61.128939 | 1817.759188 | 0 |
| 60 | DR median | 1108.161110 | 1099.196887 | 137.124572 | 2879.176769 | 0 |
| 60 | CV+acc residual | 3644.864255 | 3374.631933 | 1377.071813 | 9201.384704 | 0 |
| 60 | Horizontal DR | 3374.678057 | 3374.631933 | 17.472045 | 8344.618060 | 0 |

Источники: [paired_differences.csv](outputs/imu_utility_diagnostic/paired_differences.csv) —45парных сравнения, [duration_summary.csv](outputs/imu_utility_diagnostic/duration_summary.csv). У всех5IMUвариантов 0/3 на каждой длительности, то есть0/9 по3D; horizontal улучшений тоже0/9. Mean-bias сильно уменьшает ошибки относительно primary, но остаётся хуже CV: **5.012/87.565/685.880м** для10/30/60s. Он не назначается новым primary постфактум.

Есть отдельные улучшения **vertical компоненты**: например, mean-bias e1/10 V=0.844м противCV3.208м, e2/30 V=1.334м против7.003м. Они не дают полезного3D DR из-за горизонтального дрейфа и не являются устойчивым доказательством IMU utility. Horizontal-only оставляет те же плохие horizontal predictions и потому не спасает результат.

![Drift vs duration](outputs/imu_utility_diagnostic/drift_vs_duration.png)

Логарифмическая ось ошибок; каждый маркер —среднее3onset RMSE. Это saved-result plot, не новый experiment.

![Error over 60 second outages](outputs/imu_utility_diagnostic/error_over_outage.png)

На каждой панели pointwise3D relative-motion error на фиксированном60s episode; логарифмическая ось. Взрыв ошибки не является следствием большого dt gap именно внутри outage.

## 9. Error mechanism analysis

**Accel bias / scale / dynamic mean.** Средняя норма initial force меньше g на0.415294/0.558778/0.481784м/с². Если условно считать это постоянной vertical acceleration error, оценка0.5·b·T² на60s даёт около **−748/−1011/−872м**. Это scale-of-error diagnostic, не измеренный истинный bias. Часть расхождения может быть реальной динамикой или ошибкой signal export; движущаяся pre-history не даёт полной observability.

**Gyro bias / attitude consistency.** Средняя gyro за5s порядка(−0.013…−0.022,−0.020…−0.027,+0.025…+0.036)rad/s. При интеграции60s малые систематические угловые ошибки превращаются в большой tilt. На primary60s maximum body-down tilt достигает **67.72°,84.62°,81.52°**; до outage gyro/accel gravity disagreement уже39–44°. Mean-bias subtraction существенно уменьшает drift, что согласуется с вкладом gyro/force offsets, но не изолирует их причинно: оно одновременно удаляет возможное истинное вращение и acceleration.

**Gravity leakage.** В primary каждый acceleration contribution дополнительно разложен математически:

`a = [R(t)·R0ᵀ·up·g − up·g] + R(t)·[f(t) − R0ᵀ·up·g]`.

Первое слагаемое отражает rotation of initial reference gravity, второе residual force. Их двойные интегралы **точно** складываются в DR−CV displacement; это bookkeeping, не разделённые истинные sensor errors. На60s:

| Scenario | Final DR−CV XYZ, м | Reference-gravity displacement XYZ, м | Residual-force displacement XYZ, м |
| --- | --- | --- | --- |
| val_e1_d60s | (2364.22962, -7543.34352, -4382.67950) | (2187.74080, -7625.96574, -3375.80201) | (176.48883, 82.62222, -1006.87750) |
| val_e2_d60s | (817.47359, -8417.21310, -5060.42121) | (1020.25339, -8725.98950, -4317.04141) | (-202.77980, 308.77640, -743.37980) |
| val_e3_d60s | (2719.88862, -8212.11601, -4852.56012) | (3019.39972, -8800.05871, -4486.15636) | (-299.51110, 587.94269, -366.40377) |

В этой декомпозиции горизонтальный дрейф в основном сопровождает изменение reference-gravity направления. Это объясняет, почему удаление только world vertical acceleration не решает проблему. Такая декомпозиция сама по себе не доказывает, какая часть physical rotation ошибочна; independent attitude отсутствует.

**Wrong yaw / frame convention.** Course-aligned yaw не observable body yaw у multicopter. Начальная yaw ошибка меняет направление horizontal acceleration; она не объясняет одна весь рост tilt и vertical drift. Body/sensor mount и связь exported raw/scaled channels полностью не установлены; математический FRD unit test подтверждает кодовую конвенцию, а не аппаратную калибровку. До следующего обучения нужен разбор именно этой неопределённости.

**dt gaps.** Actual dt сохранены; внутри всех9outage intervals нет dt>0.15s. Известные секундные gaps в других частях flight не объясняют текущий резкий рост ошибки. Causal gyro age и pre-history dt statistics сохранены отдельно. Интерполяция/усечение gaps не использовались.

**Sensor noise и sampling.** Pre-outage accel std до≈0.58м/с² на оси и gyro std до≈0.095rad/s; эти std включают реальное движение, не являются noise density. Экспорт около10Hz не предоставляет высокочастотных delta-angle/delta-velocity и полного filter/calibration описания. Нельзя разделить noise, aliasing, offset, температурную коррекцию и true motion по этому одному trace. Ошибки счетчиков в рассмотренном pre-first-onset участке0, saturation по показанным range/counts не выявлена; это не гарантирует качественный INS signal.

**Lever arm.** В описании датасета приведён body→camera offset с нормой≈0.08745м, но это не полная GNSS antenna↔IMU↔camera калибровка. Такой масштаб не объясняет километровый drift сам по себе; rotation-dependent camera motion всё же ограничивает точное сравнение. Источник: [статья Zurich MAV](https://rpg.ifi.uzh.ch/docs/IJRR17_Majdik.pdf), camera/body geometry; GNSS antenna extrinsic здесь неизвестен.

Наиболее поддержанный вывод — **непригодность текущей неопределённой калибровки/attitude initialization для свободной интеграции**. Утверждать «IMU плохая» или «gyro точно в неверных единицах» оснований недостаточно.

## 10. Limitations

1. Один flight, три onset с вложенными и перекрывающимися durations; это не9независимых trials. Confidence intervals/значимость по этим трём onset не вычислялись.
2. Нет независимого body attitude/heading. Course-aligned yaw —явное допущение; GNSS heading comparison нельзя использовать как абсолютный тест gyro sign/units на мультикоптере.
3. Gravity alignment и mean/median bias требуют quasi-static допущений. Их правдоподобие ограничено движением, GNSS noise и sensor export mismatch; physical ba/bg здесь не идентифицированы.
4. GT —camera photogrammetry примерно1Hz, interpolated; точные height datum и extrinsics неизвестны. GT acceleration не использовалось как sensor input или calibration target.
5. Использованы ровно exported V2 SI channels и10Hz timeline. Более сложный calibrated INS с attitude aiding, высокочастотными интегралами или дополнительными датчиками не проверялся. Отрицательный простой DR не устанавливает верхнюю границу возможностей всех fusion методов.
6. Данные GNSS внутри outage не размаскировались для prediction или pre-outage signal test. Posthoc GT movement description и scoring не являются доступным onboard сигналом.
7. Параметры не подбирались ни по validation outage error, ни по test. Варианты заданы до scoring; даже существенно более удачный mean-bias остался sensitivity. Ни rerun с другими units/sign, ни search yaw/bias/noise не проводились.

Артефакты: config/manifest/status, [summary.csv](outputs/imu_utility_diagnostic/summary.csv), [paired_differences.csv](outputs/imu_utility_diagnostic/paired_differences.csv), duration summary,54scenario/method NPZ, initial states, signal/unit/gravity/error-mechanism JSON,9tests и два рисунка. Каталог: [outputs/imu_utility_diagnostic](outputs/imu_utility_diagnostic). Полный hash index и компактный комплект передачи: [artifact_manifest.json](outputs/imu_utility_diagnostic/artifact_manifest.json), [IMU_UTILITY_CHATGPT_HANDOFF.zip](outputs/imu_utility_diagnostic/IMU_UTILITY_CHATGPT_HANDOFF.zip).

## 11. Decision for the paper

**A. Есть ли evidence полезной IMU информации сверх CV? — NO under current usable calibration.** Ни primary, ни заранее заданные sensitivity варианты не улучшили3D или horizontal RMSE на этих9scenarios. Есть изменчивость IMU, отдельные correlations и vertical component improvements; это не устойчивый navigation benefit. Вопрос о внутреннем информационном содержании правильно откалиброванных сенсоров остаётся не решённым этим экспериментом.

**B. Продолжать neural IMU+GNSS fusion на этом dataset? — MAYBE, после физической проверки.** Сейчас переход к новому neural training не обоснован: V2 уже не показала matched IMU benefit, а физическая диагностика выявила сильную несогласованность gyro/accel без external attitude. Новая сеть могла бы лишь компенсировать неизвестные offsets/frames. Это основание приостановить IMU-fusion claim, но ещё не доказательство необходимости навсегда отказаться от датасета.

Для статьи допустим проверенный отрицательный вывод: на фиксированных validation outages доступный causal strapdown diagnostic с declared course-aligned initialization не превзошёл GNSS CV; mean/median pseudo-bias помогают относительно zero-bias, но не обеспечивают useful DR. Нельзя объявлять это calibrated state-of-the-art INS benchmark, доказательством отсутствия motion signal или результатом настройки EKF.

## 12. Recommended next single step

**Выбран один шаг: исправить physical calibration/preprocessing.** Сначала восстановить документированный контракт raw/exported IMU: units, axis/mounting transform, timestamps/filtering и уже применённые calibration offsets. На train/pre-flight участках проверить согласованность gyro-propagated gravity с accelerometer, используя физические ограничения и metadata, без выбора по этим9outage errors. Если metadata недоступны, явно зафиксировать неидентифицируемые yaw/bias и не выдавать их за найденную калибровку.

Этот будущий этап здесь **не выполнялся как tuning**. Новый V3, новые neural seeds и Kaggle запуск не нужны сейчас и не подготовлены как автоматическое продолжение. R7/V2 checkpoints и результаты сохранены без изменений.
