# IMU_CALIBRATION_AUDIT

Дата: 2026-09-26. Репозиторий: `main`, commit `766575a07886e335c1d6cac664c5e140550942aa`. **Статус: B — PARTIALLY RECOVERED.** Это forensic-аудит сигналов и документации, не новый navigation benchmark.

## 1. Executive summary

**Главная находка: mounting частично перестал быть неизвестным.** В Figure 2 исходной статьи о Zurich Urban MAV явно различаются PX4 frame для RawAccel/RawGyro и body frame для OnboardPose: между горизонтальными осями есть 45°, Z противоположны. Это нельзя исправить одной перестановкой XYZ. Документированный переход подтверждается корреляцией gyro с onboard Omega на отложенной части **train**: **0.925512 / 0.963339 / 0.997562** по XYZ. Сам угол по данным не подбирался.

**Полная физическая калибровка не восстановлена.** Неизвестны фактические offsets/scales/filter settings, исходная частота до экспорта, точная firmware и алгоритм OnboardPose. Согласованность после постоянной коррекции улучшается, но это не независимая оценка bias. На 18 внутренних train-check окнах по 30s среднее финальное расхождение gyro с направлением акселерометра составляет **50.556829°** без коррекции и **3.011342°** после вычитания среднего gyro раннего train. Направление акселерометра также содержит реальные ускорения; это не attitude ground truth.

Сохранены отрицательные выводы предыдущих этапов: V2 full выиграла matched gps_only только в 3/9 эпизодов одного onset; physical DR с прежней доступной калибровкой проиграл CV в 9/9. Здесь эти navigation metrics **не пересчитывались**, и новые данные их не отменяют. Источники: [V2_PILOT_REPORT.md](V2_PILOT_REPORT.md), [IMU_UTILITY_DIAGNOSTIC_REPORT.md](IMU_UTILITY_DIAGNOSTIC_REPORT.md). Контекст R7 и исправлений: [PROJECT_HANDOFF.md](PROJECT_HANDOFF.md), [PROTOCOL_FIX_REPORT.md](PROTOCOL_FIX_REPORT.md), [NEXT_STAGE_DIAGNOSTIC_REPORT.md](NEXT_STAGE_DIAGNOSTIC_REPORT.md).

Обучение, V3, seeds, Kaggle jobs, test/validation calibration и девять outage evaluations не выполнялись. Проверена неизменность **334** ранее существовавших файлов R7 inputs, clean-outage diagnostic, V2, physical diagnostic и пяти отчётов: [verification.json](outputs/imu_calibration_audit/verification.json). Новые материалы находятся отдельно в [outputs/imu_calibration_audit](outputs/imu_calibration_audit).

## 2. Dataset IMU metadata

Обозначения: **[Документировано]** — первичный источник; **[Данные]** — воспроизводимая локальная проверка; **[Гипотеза]** — согласуется с наблюдениями, но не установлено для полёта; **[Неизвестно]** — необходимое подтверждение отсутствует.

| Материал | Что установлено | Ограничение / доказательство |
|---|---|---|
| RawAccel/RawGyro CSV | Timestamp, error count, exported XYZ, temperature, range, scaling, integer raw XYZ/temperature | [README](outputs/next_stage_diagnostics/raw/readme.txt), строки31–35; schema/data в [timestamp_inventory.json](outputs/imu_calibration_audit/timestamp_inventory.json) |
| OnboardPose.csv | Raw/estimated поля Pixhawk: Omega, Accel, Vel, AccBias, quaternion wxyz, Height/Altitude, veh_pitch, tether fields, GPS_on | README27–29; алгоритм фильтра и смысл каждого производного поля не описаны |
| calibration_data.npz | Только `intrinsic_matrix` (3×3) и `distCoeff` (1×5) | README51–52; [файл](outputs/imu_calibration_audit/metadata/calibration_data.npz) прочитан с `allow_pickle=False`; это camera intrinsics |
| MATLAB scripts | Читают camera GT и рисуют GPS/GT path; IMU calibration не задают | [loadGroundTruthAGL.m](outputs/next_stage_diagnostics/raw/loadGroundTruthAGL.m), строки53–55; [plotPath.m](outputs/next_stage_diagnostics/raw/plotPath.m) |
| write_ros_bag.py | Читает GPS, image matching, camera GT, изображения и camera calibration | [код](outputs/imu_calibration_audit/metadata/write_ros_bag.py), строки21–35,100–124; RawAccel/RawGyro/OnboardPose вообще не экспортирует |
| Исторический PX4 v1.0.0 | Структуры sensor reports совпадают с CSV; показаны SI, rotation, calibration и filtering | Это **reference implementation, не найденная flight firmware**; [receipt](outputs/imu_calibration_audit/metadata/px4_reference_receipt.json) |
| Sensor IDs / parameters | В доступных IMU CSV нет device ID, serial, parameter dump, coefficients фильтра | `Error_count=0` на всём train — не доказательство правильной калибровки |

Новые OnboardPose, camera NPZ и ROS writer получены из **того же Kaggle dataset version2** отдельными файлами, без job и без изображений/полного датасета. URL, размеры и SHA256: [download_receipt.json](outputs/imu_calibration_audit/metadata/download_receipt.json). OnboardPose: 52,802,195 bytes, SHA256 `0e0e74976ddda2753f3cb5dcee3c384b2ff6b858737690fd0b2a5d2f28e8064a`. Hashes исходных IMU: [input_manifest.json](outputs/imu_calibration_audit/input_manifest.json).

**[Документировано]** Платформа — tethered Fotokite; yaw дрона и pitch камеры управляются отдельно от перемещения. Figure 2 показывает PX4 axes и body axes; дана camera/body translation (75.66,29.68,−32.27)mm в описанном авторами базисе. Камерная геометрия не заменяет IMU calibration и не задаёт постоянный camera pitch. Сигналы имеют общий PX4 board clock; timestamps выражены в микросекундах. Источник: [статья о датасете, pp.2–3](https://rpg.ifi.uzh.ch/docs/IJRR17_Majdik.pdf), локальные [PDF](outputs/imu_calibration_audit/metadata/dataset_paper.pdf) и [просмотренная страница Figure 2](outputs/imu_calibration_audit/metadata/dataset_paper_page3.png). Поэтому GNSS course не идентифицирует истинный body heading.

ROS writer **не запускался**: у него запись bag при импорте. Его строка108 передаёт GT angles непосредственно в `quaternion_from_euler`, тогда как README описывает их в градусах и другом порядке. Этот экспорт нельзя принимать за проверенное преобразование camera/body orientation.

## 3. Raw vs exported sensor channels

Все численные проверки ниже ограничены `t < 1905.752757s` — train V2. Отрезок до **1336.1542017s** используется для диагностического fit, остаток — `train_check`, **не validation и не test**. Старый `t_imu.npy` прочитан только для границы split. GNSS/GT positions, checkpoints и outage predictions не являются входами аудита. CSV целиком читаются для извлечения train, но измерения остальных split отбрасываются до анализа.

Код: [run_imu_calibration_audit.py](scripts/run_imu_calibration_audit.py), функции `read_train`, `mappings`, `main`; настройки: [config.json](outputs/imu_calibration_audit/config.json). Критерии — sensor consistency, без подбора по navigation errors. Сопоставление потоков офлайн может использовать интерполяцию и двустороннее сглаживание **внутри train**; это не causal preprocessing для будущей модели.

| Сигнал, native train | N | Mean exported XYZ | Mean scaled raw XYZ | RMS3(export−scaled raw) | Same-row Pearson XYZ |
|---|---:|---|---|---:|---|
| Accel, м/с² | 18935 | (0.953228,−0.652202,−9.159334) | (0.948913,−0.658439,−9.163919) | 5.299591 | (0.142175,0.085364,−0.516953) |
| Gyro, rad/s | 18936 | (−0.015300,−0.026782,0.028786) | (−0.015007,−0.023296,0.029235) | 0.287550 | (0.148515,0.283649,0.810941) |

Источники: [units_raw_export.json](outputs/imu_calibration_audit/units_raw_export.json), [raw_export_aggregation.csv](outputs/imu_calibration_audit/raw_export_aggregation.csv). RMS3 здесь `sqrt(mean(sum((export−scaled_raw)²)))`, без навигационного смысла; это не среднее трёх axis RMSE.

Средние близки, отдельные samples существенно различаются. При усреднении по 5s accel XY корреляции растут до **0.913346/0.877131**, но Z остаётся **−0.230036**; gyro Z растёт до **0.981665**, XY остаются **−0.329243/−0.063771**. Следовательно, `export = raw × scaling` не воспроизводит экспорт, а простой постоянный offset не устраняет различие.

На train-fit перебраны ровно **24 proper signed permutations**, с отдельным диагностическим положительным diagonal gain и offset; все результаты сохранены в [raw_export_discrete_fits.csv](outputs/imu_calibration_audit/raw_export_discrete_fits.csv). Например, минимальный fit gyro даёт identity с gains (0.0826,0.1416,0.6557), а accel — `+y,+x,−z` с gains около0.05–0.066 и offset Z≈−9.62. Это подавление динамики регрессией, **не физически восстановленные scales/оси**. Ни такой transform, ни gain/offset не выбран для навигации.

**[Гипотеза с сильной поддержкой]** exported XYZ — SI sensor reports после обработки драйвером; это не naked ADC counts. В [историческом MPU6000 reference](https://raw.githubusercontent.com/PX4/PX4-Autopilot/v1.0.0/src/drivers/mpu6000/mpu6000.cpp) raw counts сохраняются до пользовательского rotation, затем применяются SI scale, offset, axis scale и low-pass (локальный [код](outputs/imu_calibration_audit/metadata/px4_v1_mpu6000.cpp), строки1732–1784). Отсутствующие внутренние samples и параметры не позволяют обратить фильтрацию по 10Hz CSV. Конкретную цепочку данного полёта этот reference не доказывает.

## 4. Units and scaling

| Поле | Accel | Gyro | Вывод |
|---|---|---|---|
| `scaling`, постоянно на train | 0.004788403399288654 | 0.0010642195120453835 | Примерно g/2048 м/с²/count и (π/180)/16.4 rad/s/count |
| `range_rad_s` из CSV | 156.90640258789062 | 34.906585693359375 | **Accel header ошибочен:** около16g м/с²; gyro около2000deg/s в rad/s |
| Температура | 5.362883…9.867037°C | 5.362883…9.872578°C | Диапазон этого train, не отдельная калибровка температуры |
| `raw_temp/361 + 35` | Max error0.000001765°C | Max error0.000001765°C | Совпадение формулы старого MPU6000 reference до float rounding |
| `Error_count` | Только0 | Только0 | Ошибок указанного счётчика нет; остальные ошибки не исключаются |

Числа полей можно проверить в строке2 обоих raw CSV; all-train unique values и ошибки формулы — [units_raw_export.json](outputs/imu_calibration_audit/units_raw_export.json). Структуры [accel_report](https://raw.githubusercontent.com/PX4/PX4-Autopilot/v1.0.0/src/drivers/drv_accel.h) и [gyro_report](https://raw.githubusercontent.com/PX4/PX4-Autopilot/v1.0.0/src/drivers/drv_gyro.h), локальные строки58–71, описывают XYZ как SI в board axes. Совпадение структуры, range/scaling и необычной температурной формулы — сильный fingerprint семейства драйвера, **не sensor serial/device ID и не точная версия firmware**.

Нет оснований повторно умножать exported XYZ на scaling либо объявлять gyro degrees/s. Старое выражение «NED board axis» в заголовке драйвера относится к базису платы, а не к уже повернутому world NED acceleration. Чувствительность к bias не является тестом другой системы единиц.

## 5. Axis/frame/mounting analysis

### Документированный переход, без fit угла

Figure 2 первоисточника явно подписывает body frame для OnboardPose и PX4 frame для RawAccel/RawGyro. Body — forward/left/up; PX4 X/Y идут по диагональным направлениям относительно body, Z вниз. Для column-vector `v_PX4` переход к body из схемы:

```text
v_body = C × v_PX4
C = [ −1/√2  −1/√2   0 ]
    [ −1/√2  +1/√2   0 ]
    [    0      0   −1 ]
CᵀC = I, det(C) = +1.
```

Это **документированный фиксированный transform**, отдельно от перебора24 mappings; непрерывный поиск yaw/mounting не выполнялся. Одной смены Z без других осей недостаточно: она меняет handedness. Конкретное применение этого C для ещё неизвестных стадий внутренней firmware не предполагается автоматически.

Сопоставление выполнено на native raw timestamps, OnboardPose интерполирован офлайн. Offset `δ = mean(onboard − C×raw)` оценён только на ранних70% train; нижние ошибки — на последних30% train:

| Каналы | N fit/check | Pearson XYZ на check | δ XYZ на fit | Check RMS3 без δ / с δ |
|---|---|---|---|---|
| RawGyro→Omega, rad/s | 13252/5683 | 0.925512 /0.963339 /0.997562 | (−0.028076,0.007177,0.030393) | 0.054665 /0.035371 |
| RawAccel→Accel, м/с² | 13251/5684 | 0.911015 /0.902904 /0.487808 | (0.415923,0.240555,0.184137) | 0.790966 /0.598050 |
| RawAccel→AccBias, м/с² | 13251/5684 | 0.897757 /0.888313 /0.436870 | (0.418430,0.242489,0.186293) | 0.841969 /0.663995 |

Источник: [documented_mount_consistency.json](outputs/imu_calibration_audit/documented_mount_consistency.json), ключ `comparisons`; дополнительный ключ `uniform_10Hz_interpolated_comparisons` имеет **другую timeline** и намеренно не смешан с таблицей. Сильная связь подтверждает смысл documented frame и общий sensor source; δ не объявляется истинным ba/bg, поскольку onboard тоже обработан неизвестным фильтром.

### Что дали только24 discrete candidates

[axis_candidates.json](outputs/imu_calibration_audit/axis_candidates.json) содержит все24 unique right-handed матрицы; отражений нет. На 10Hz diagnostic grid определено `u = smooth(f)/||smooth(f)||`, проверяется `u_dot + (Cω)×u`. Quasi-static proxy: 1s smoothing, `|norm(f)−g|<1м/с²`, norm gyro<0.1rad/s, local accel std norm<0.5м/с²; это **кандидат на малую динамику, не доказанный покой**.

На13828 таких точках два наименьших residual — `+y,−x,+z`: **0.046292rad/s** и `−x,+y,−z`: **0.046547rad/s**. Identity: **0.049424rad/s**; baseline `ω=0`: **0.033902rad/s**. Малые различия и выигрыш нулевого gyro не позволяют выбрать физический winner: [relative_gyro_accel_axis_consistency.csv](outputs/imu_calibration_audit/relative_gyro_accel_axis_consistency.csv).

При одинаковом proper rotation accel и gyro норма этого residual вообще инвариантна (численно max отличие **2.78×10⁻¹⁷**). Gravity-only consistency **принципиально не восстанавливает общий absolute mounting/yaw**. Даже документированный C сам по себе не устранит raw gyro/gravity drift при согласованном вращении обоих сигналов. Он исправляет связь базисов и ошибочное понятие board-forward, а не заменяет bias calibration.

## 6. Timestamp/filter-delay analysis

| Train stream | N | Median dt / nominal CSV rate | Mean dt | Max gap | Gaps>1.5×median |
|---|---:|---|---:|---:|---:|
| RawAccel | 18935 | 0.100000s /10Hz | 0.100273s | 1.099012s | 12 |
| RawGyro | 18936 | 0.100000s /10Hz | 0.100272s | 1.101011s | 10 |
| OnboardPose | 94575 | 0.020000s /50Hz | 0.020077s | 0.979995s | 22 |

На train timestamp duplicates, reversals и numeric NaN/Inf отсутствуют. У raw XYZ нет подряд одинаковых полных векторов. Это не доказывает аппаратную частоту или отсутствие пропущенных внутренних samples. Источник: [timestamp_inventory.json](outputs/imu_calibration_audit/timestamp_inventory.json).

Nearest gyro timestamp − accel timestamp: median **−1.012ms**, P05 **−8.003ms**, P95 **0ms**, диапазон **−12.997…+10.971ms**; совпадают16.4774% timestamps. Это разность времён зарегистрированных reports, **не измеренная задержка физического сенсора**. У raw/exported XYZ внутри одного CSV одна временная метка: по ней нельзя узнать фазовую задержку фильтра.

Raw→exported lag profile проверен на заранее ограниченном диапазоне−0.5…+0.5s, шаг0.1s: [raw_export_lag_profiles.csv](outputs/imu_calibration_audit/raw_export_lag_profiles.csv). Accel минимум demeaned RMS меняется с +0.1s на train-fit до−0.2s на train-check; residual остаётся4.36/6.23м/с². Gyro минимум в0s, но сохраняется существенное XY несогласование. **Единой достоверной raw/export задержки не найдено.**

Raw gyro Z и отрицательный onboard Omega Z: на offline10Hz grid корреляция максимальна в0s (**0.991010**), в−20/+20ms —0.988408/0.988882. Profile: [raw_onboard_lag_profiles.csv](outputs/imu_calibration_audit/raw_onboard_lag_profiles.csv). Это поддерживает близкую report-time alignment, но не означает нулевую физическую задержку или точность20ms: raw частота10Hz, использована интерполяция, лаг смешан с фильтрацией.

В reference MPU6000 заданы defaults1000Hz, driver LPF30Hz, on-chip filter42Hz (строки167–174), но параметры изменяемы. **Эти defaults не приписываются данному полёту.** По редкому экспорту нельзя отделить aliasing/decimation, hardware LPF, software LPF, report phase и реальные колебания. Изменения timing для навигации не внесены.

## 7. Offline attitude diagnostic

**Происхождение установлено только до уровня Pixhawk/PX4 onboard estimate.** Доступные README/публикация не называют точный estimator, firmware commit, sensor selection, aiding, reset semantics или frame reference для quaternion. `GPS_on=1` во всех94575 train rows, но это не доказывает, что quaternion фильтр использует GNSS. Магнитометр, tether aiding, GNSS/course aiding и estimator bias corrections нельзя ни подтвердить, ни исключить по этим файлам. Поэтому onboard quaternion/Omega применены только как **зависимая диагностическая оценка**, не как calibration truth или навигационный вход.

| Проверка | Результат |
|---|---|
| CSV quaternion order | Документирован `w,x,y,z`; для SciPy переставлен в `x,y,z,w` |
| Norm quaternion | 0.999999889…1.000000168 |
| `Vel_x/y/z`, `Azimuth` | Все0 на train; не показание истинного покоя или истинного heading |
| `AccBias` mean XYZ | (0.203617,−0.891496,9.344619); mean norm9.421835 |
| `Accel` mean XYZ | (0.203090,−0.892757,9.342981) |
| Euler pitch(q)−veh_pitch | Mean−0.110900rad, std0.092263rad; простой equivalence не подтверждён |

Источник: [onboard_attitude_diagnostic.json](outputs/imu_calibration_audit/onboard_attitude_diagnostic.json). В частности, **AccBias содержит gravity-sized vector**, а не очевидный малый constant sensor bias. Его точный смысл неизвестен; использовать его как `ba` нельзя.

При active rotation `R(q)` onboard Accel ближе к +g по reference Z: residual RMS3 **0.829570м/с²**, против1.613651 при `R(q)ᵀ`. Это согласуется с body→reference interpretation, но не устанавливает Earth/UTM heading и не доказывает качество attitude. Оба направления и оба знака сохранены: [offline_gravity_cancellation.csv](outputs/imu_calibration_audit/offline_gravity_cancellation.csv).

## 8. Gyro consistency

Quaternion-derived rate вычислена как `Log(q[i]⁻¹ q[i+1])/actual_dt`, сравнивается со средней соседних onboard Omega. На всём train axis RMS residual **0.029264/0.026128/0.388081rad/s**. Большая Z RMS обусловлена прежде всего двумя крупными discontinuities:

| Native CSV row начала, включая header | Интервал PX4 seconds | Δt | Quaternion rotation | Norm onboard Omega в начале |
|---:|---|---:|---:|---:|
| 38189 | 774.479985→774.494986 | 0.015001s | 87.346578° | 0.064566rad/s |
| 87906 | 1772.128870→1772.157839 | 0.028969s | 105.165423° | 0.072382rad/s |

Это не sign flip представления q: вычислена относительная SO(3) rotation. Gyro не поддерживает такие мгновенные вращения. Причина может быть reset/reference change/logging; **конкретная причина неизвестна**. Все шаги>1°: [onboard_quaternion_large_steps.csv](outputs/imu_calibration_audit/onboard_quaternion_large_steps.csv).

При отдельном описательном исключении только этих двух шагов>5° axis RMS становится **0.013553/0.012030/0.003400rad/s**, корреляции **0.981679/0.985754/0.998520**. Полные метрики выше сохраняются; из orientation propagation окна с jumps не исключались. Это поддерживает SI/sign conventions и shared estimator dynamics, а не независимость quaternion.

### Disagreement во времени — только train

Заданы63 неперекрывающихся окна30s от начала доступной train history. Raw gyro интегрируется на общей offline10Hz сетке; dt — разность её timestamp, accel gravity direction — centered11-sample mean. Onboard Omega отдельно интегрируется на native примерно50Hz timestamp с actual dt, стартуя из onboard q начала окна. Позиции/скорости не интегрируются. Это sensor diagnostic с ресемплингом, **не реализация causal strapdown для deployment**; gaps/interpolation также могут влиять на ошибки.

Среднее raw gyro первого70% train: **(−0.015344,−0.027020,0.028712)rad/s**. Оно фиксировано один раз и в sensitivity вычитается на всех окнах; reset bias на каждом окне отсутствует. Из63 окон44 целиком в train-fit,18 в train-check,1 пересекает внутреннюю границу.

| Метод / reference | Check windows | Mean final angular error | Median | Min…max |
|---|---:|---:|---:|---|
| Raw zero offset / accel direction | 18 | 50.556829° | 50.522883° | 38.154874…59.443235° |
| Raw train-mean sensitivity / accel direction | 18 | 3.011342° | 2.919169° | 0.071428…6.892012° |
| Onboard Omega / onboard q | 18 | 10.193986° | 4.547239° | 1.628043…105.123357° |

На44 fit windows средние raw zero/mean соответственно52.312030°/3.418290°. Сильное улучшение на позднем train поддерживает устойчивый constant component, но средняя rate смешивает true motion и bias, а gravity не наблюдает yaw. Это **не новая навигационная польза IMU** и не recovered calibration.

Дополнительно тот же integrated **raw gyro** переведён через документированный C и сопоставлен с onboard quaternion, начиная каждый30s интервал из его начального q. На18 train-check окнах mean/median final полной SO(3) разности: zero offset **73.384691°/73.805247°**, train-mean sensitivity **9.740260°/4.293309°**. Максимум sensitivity **101.448604°** связан с окном, содержащим discontinuity onboard q; никакие окна не удалялись. Это сравнение полной orientation, поэтому его нельзя объединять с gravity-direction метрикой таблицы. Источники: [raw_gyro_vs_onboard_orientation.csv](outputs/imu_calibration_audit/raw_gyro_vs_onboard_orientation.csv), [полные временные кривые](outputs/imu_calibration_audit/raw_gyro_vs_onboard_over_time.csv). Onboard initialization здесь разрешена только как offline diagnostic, не как доступное состояние будущей навигации.

Полные63×3 rows: [orientation_window_summary.csv](outputs/imu_calibration_audit/orientation_window_summary.csv); time curves с шагом около1s: [orientation_disagreement_over_time.csv](outputs/imu_calibration_audit/orientation_disagreement_over_time.csv). [Рисунок диагностики](outputs/imu_calibration_audit/sensor_consistency_diagnostics.png), [SVG](outputs/imu_calibration_audit/sensor_consistency_diagnostics.svg), [скрипт](scripts/plot_imu_calibration_audit.py). На рисунке полосаP05–P95 распределения окон, **не confidence interval**.

Разделение причин: единицы и documented relative frame сильно поддержаны; constant gyro component существенно влияет; yaw/mounting нельзя заменить GNSS course; software filtering и report interpolation влияют на high-frequency соответствие; q discontinuities принадлежат onboard output; true translational acceleration и true rotation не отделены независимой attitude truth. Нельзя объяснять весь старый drift одной ошибкой знака или одним фиксированным lag.

## 9. Accelerometer/gravity consistency

**[Данные + физическая интерпретация]** Raw exported Z отрицательна, body/Onboard Z положительна после C. Это согласуется со specific force: при условном покое в down-positive PX4 Z наблюдается−g. Рабочая формула при известном body→world `R`: `a_world = R(f−ba) + gravity_world`. Ни raw XYZ, ни onboard Accel нельзя принимать за уже очищенное world acceleration.

Native train norm exported accel: mean **9.265153м/с²**, std0.480738, P05–P95 **8.529493…10.036742**. Mean norm ниже9.80665 примерно на**0.541497м/с² (5.52%)**. Scaled raw counts имеют mean norm9.795393, но std3.832356 и P05–P95 3.875019…15.488862. Это не доказательство «raw counts исправляют scale»: норма нелинейна, фильтрация/колебания меняют её среднее; mean vectors raw/exported почти одинаковы.

На13828 low-dynamics proxy точках norm сглаженного accel: mean **9.264157**, std0.134351м/с²; реальный покой независимо не установлен. Rotation сохраняет norm, поэтому сменой осей этот deficit не устранить. Разделить accel scale, offset, vibration/filtering и реальное вертикальное движение по этому flight однозначно нельзя.

Offline gravity cancellation после **documented C**, затем `R(q)`, относительно (0,0,+9.80665) в reference frame q:

| Сигнал / scope на общей10Hz сетке | N | Mean rotated XYZ, м/с² | RMS3 residual | H RMS | V RMS |
|---|---:|---|---:|---:|---:|
| RawAccel, весь train | 18966 | (−0.096187,−0.579578,9.201374) | 1.285833 | 1.058554 | 0.729953 |
| RawAccel, low-dynamics proxy | 13828 | (−0.061865,−0.632473,9.207861) | 1.245086 | 1.043113 | 0.679819 |
| Onboard Accel, весь train | 18966 | (0.005561,−0.343900,9.392333) | 0.828466 | 0.661445 | 0.498844 |
| Onboard Accel, low-dynamics proxy | 13828 | (0.029805,−0.366643,9.402132) | 0.800888 | 0.648063 | 0.470569 |

Источник: [documented_mount_gravity_cancellation.json](outputs/imu_calibration_audit/documented_mount_gravity_cancellation.json). Эти residuals **не acceleration estimation errors относительно независимой truth**: реальные ускорения и estimator attitude error входят в них. Уменьшение residual у onboard может отражать общую фильтрацию/коррекцию; оно не разрешает использовать onboard как будущий input.

Mean norm по onboard tilt bins0–5°,5–10°,10–20°: **9.265871/9.258523/9.251775м/с²**, N=10294/8565/107. Корреляции norm с onboard roll/pitch/tilt примерно0.02768/−0.00199/−0.02335, с temperature0.03487. Сильной простой зависимости не обнаружено, но диапазон tilt узок и pose зависим: это не полноценная six-position calibration. Источники: [accel_norm_vs_onboard_tilt.csv](outputs/imu_calibration_audit/accel_norm_vs_onboard_tilt.csv), [accel_norm_diagnostics.json](outputs/imu_calibration_audit/accel_norm_diagnostics.json).

## 10. What is known vs unknown

| Элемент | Статус | Что можно использовать / чего нельзя заключать |
|---|---|---|
| Timestamp units и board clock | Документировано, data intervals согласованы | µs→s; учитывать gaps; hardware sample instant неизвестен |
| SI exported units и ADC scaling | Сильное согласование metadata, старого driver layout и данных | м/с²,rad/s; не применять scaling повторно |
| Raw PX4→Onboard body axes | Восстановлено по Figure 2, подтверждено сигналами | Фиксированный C с45°/Z reversal; не приравнивать board X к body forward |
| Body→Earth heading / true attitude | Не восстановлено | Course и quaternion с неизвестным reference не независимая truth |
| Реальные ba/bg, axis gains, температурная calibration | Не восстановлены | Train means/relative onboard offsets — sensitivity, не calibration constants |
| Конкретный IMU/device ID и flight firmware | Не установлены | MPU6000 lineage — fingerprint; actual device не идентифицирован |
| Filter chain/sample rate до CSV/anti-alias | Частично понятен возможный механизм | Reference driver не подменяет flight parameters |
| Физический lag и timestamp age | Не идентифицированы однозначно | Report times близки; универсальная delay correction не выбрана |
| Onboard estimator/sensor aiding/resets | Не установлены | Общий IMU source сильно поддержан; GNSS/mag/tether use неизвестен |
| AccBias semantics | Не установлены | Gravity-sized quantity, нельзя трактовать как малый sensor bias |
| Camera calibration | Документирована как intrinsics и часть геометрии | Не IMU six-axis calibration; динамический camera pitch отдельно |
| Сохранённые R7/V2 navigation results | Не изменены | Старые protocol limitations остаются; новой победы IMU нет |

## 11. Calibration status: A/B/C

**B — PARTIALLY RECOVERED.** Это не C для всех свойств: units, relative axes, exported/report distinction и конкретные onboard discontinuities установлены существенно лучше. Но до A недостаточно доказательств bias/scale/timing/filter rules и независимой attitude initialization.

Статус вычислительной процедуры — `complete`, статус калибровки — `B_PARTIALLY_RECOVERED`: [status.json](outputs/imu_calibration_audit/status.json). Подготовка «исправленного физического diagnostic» как подтверждённо calibrated INS **не выполняется**, поскольку условие A не выполнено. Новая матрица не внедрена в production preprocessing. Выбор bias, delay или discrete mapping по минимуму остатка остановлен; дальнейшего tuning и9outage rerun нет.

Проверки:24 unique proper mappings, orthogonality/determinants, SO(3) sign на синтетическом вращении, invariance common rotation, строго train-only ranges и неизменность protected inputs прошли. Это проверки математики/целостности, **не сертификат физической калибровки**. Код и все новые результаты индексируются в [artifact_manifest.json](outputs/imu_calibration_audit/artifact_manifest.json).

## 12. Consequences for the paper

Продолжать анализ этого dataset научно допустимо с ограниченным claim: воспроизводимая causal validation, отрицательный matched full/gps_only результат и forensic diagnosis неполного sensor contract. Нельзя сейчас заявлять эффективное calibrated IMU+GNSS восстановление 3D при длительных GNSS outages, отсутствие полезной информации в IMU, superiority над корректно настроенным INS/EKF или generalization по одному flight/seed.

Старое допущение «raw board forward можно совместить с GNSS course как heading дрона» не обосновано и теперь имеет конкретное документированное возражение. Но исправление только этого frame не доказывает улучшение DR: bias и временно-фильтровая цепочка остаются неопределёнными, а общий rotation не меняет gyro/gravity disagreement. Поэтому нельзя переписать старые отрицательные результаты как «исправленные» без отдельного разрешённого и зафиксированного evaluation.

**Другой dataset не нужен для завершения этого аудита.** Для основной статьи с положительным claim о физически корректном IMU fusion нужен dataset с проверяемой IMU calibration, timing и независимым reference, **если оригинальный sensor contract Zurich не удастся получить**. Тогда Zurich можно оставить дополнительным ограниченным case study. Конкретный второй dataset здесь не выбран и не скачан. **Новое neural training сейчас не обосновано:** оно не заменит недостающие физические доказательства. Величина метрик сама по себе не устанавливает готовность к Q2-публикации.

## 13. Recommended single next step

**Получить архив фактической flight firmware / parameter dump / sensor calibration от источника датасета.** Нужны device selection/IDs, calibration offsets/scales, board rotation, sensor sample/LPF/logging settings, timestamp semantics и реализация OnboardPose с aiding/reset rules. Это один шаг восстановления provenance, а не новый sweep. Обращение к авторам автоматически не отправлялось.

Если такого архива нет, зафиксировать невозможность дальнейшего физически достоверного восстановления по доступному экспорту и менять основной dataset/claim; не компенсировать пробелы подбором на9validation outages. Сейчас **STOP: ни новых seeds, ни V3, ни Kaggle jobs, ни нового neural training**.

Для передачи следующему ассистенту достаточно этого отчёта и компактного [IMU_CALIBRATION_AUDIT_HANDOFF.zip](outputs/imu_calibration_audit/IMU_CALIBRATION_AUDIT_HANDOFF.zip): скрипты, config/status, таблицы/JSON, рисунок, источник Figure 2 и hashes. Большой OnboardPose CSV и исходные raw CSV остаются локально; их точные download references/hashes включены. Числа предыдущих navigation этапов следует брать из их собственных неизменённых отчётов, не смешивать с угловыми/сенсорными residuals этого аудита.
