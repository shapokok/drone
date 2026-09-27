# NEXT_STAGE_DIAGNOSTIC_REPORT

Дата: 2026-09-25. Этап: анализ сохранённого R7 и подготовка V2, **без нового обучения и без нового Kaggle job**.

## 1. Executive summary

**R7 преимущественно выдаёт почти постоянную внутри окна поправку к GNSS. Убедительного использования IMU для восстановления движения не обнаружено.** Это не один глобальный translation: смещение меняется между окнами по 192 точки. В control среднее собственного окна объясняет 99.9999952% центрированной вариации поправки full; остаток относительно этого среднего — 0.000466519 м RMS. Обнуление IMU меняет outage prediction на 0.00113–0.01965 м RMS, перестановка/разворот — на 0.0000168–0.000223 м, при величине самой поправки 3.07–4.49 м.

Обнаружены реальные проблемы постановки: несовпадающие GNSS/GT origins и проекции; future-gyro matching; некаузальный Transformer; почти повторяющиеся GNSS строки, ошибочно воспринимаемые как новые измерения; отсутствие outage augmentation при обучении R7. Это не доказывает, что IMU бесполезна в принципе, и не позволяет приписать плохой результат одной конкретной причине.

**Подготовлено одно небольшое новое обучение V2:** matched GRU full/gps_only, LSTM baseline, seed 0, 8 epochs, train/validation, causal inputs и residual-motion target. Оно пока не выполнено. Новых pilot checkpoints или pilot метрик нет.

Основание отчёта:

- [PROJECT_HANDOFF.md](PROJECT_HANDOFF.md), [PROTOCOL_FIX_REPORT.md](PROTOCOL_FIX_REPORT.md) — контекст; численные выводы ниже проверены по артефактам.
- R7: `shapok/drone-nav-full-run`, version 7, output **352644733**. Clean diagnostic: `shapok/drone-nav-clean-outages-r7`, version 1, output **352725062**; 60 evaluations, 10 сценариев, 6 методов, один neural seed 0.
- Исходные [summary.csv](outputs/diagnostic_clean_outages/summary.csv), [paired_differences.csv](outputs/diagnostic_clean_outages/paired_differences.csv), [status.json](outputs/diagnostic_clean_outages/status.json) сохранены. Их результаты не заменены текущими локальными проверками.
- Текущий Git: `main`, `766575a07886e335c1d6cac664c5e140550942aa`; tracked diff пуст. Новые audit/diagnostic/pilot файлы untracked; это подготовленные файлы, не новый commit R7.
- Скачаны только небольшие журналы и необходимые R7 arrays/checkpoint; изображения и весь датасет не загружались. Секреты не входят в отчёт/ноутбук.

Статусы доказательств: **результат** = сохранённые predictions/CSV либо явно обозначенный локальный пересчёт; **код** = реализованное поведение, но не свидетельство обучения; **не проверено** = отсутствующие данные или невыполненный запуск. Все сведения о V2 ниже имеют статус «подготовлено кодом», кроме перечисленных тестов.

## 2. GNSS ↔ GT frame analysis

**Текущий код R7.** `src/data/prepare.py:86` выбирает gyro через следующий timestamp; `:108` переводит lat/lon в локальную equirectangular аппроксимацию: X east, Y north, радиус 6378137 м. `:133` читает `GroundTruthAGL.csv`, связывает время через `imgid` с GPS, вычитает первый GT XYZ. `:165` задаёт GPS lat0/lon0 по первой GPS строке, но alt0 по первому GT Z. Таким образом, горизонтальные начала GPS и GT независимы, вертикальное начало общее. Это не строго общий ENU/UTM frame.

Фактические origins: GPS lat0=47.3843571°, lon0=8.5451784°; первый GT UTM=(465666.057548, 5247973.646622, 469.019496) м. Первый processed GNSS=(0,0,−4.109496), GT=(0,0,0) м. Проекция первого GPS в UTM32N даёт (465670.7068474,5247978.0337591) м. В опубликованных GPS UTM столбцах начальная разность GPS−GT равна (4.649299,4.386816,−4.109492) м. Независимое центрирование удаляет эту горизонтальную разность, а затем смешивает локальные east/north и UTM grid axes. Приближённая grid convergence здесь −0.334712°; это не эмпирический yaw из таблицы ниже.

Ground truth относится к фотограмметрической траектории камеры; GPS использован при реконструкции, поэтому эталон не полностью независим от GNSS. В статье датасета описаны WGS84 GNSS и UTM32N ground truth, а также body→camera offset (75.66,29.68,−32.27) мм. Его норма 0.087445 м; это не измеренный GNSS antenna→camera lever arm и не объяснение метрового расхождения. Источник: [Majdik et al., Zurich Urban MAV dataset](https://rpg.ifi.uzh.ch/docs/IJRR17_Majdik.pdf), разделы 4.1/4.3 и Figure 4; [страница датасета](https://rpg.ifi.uzh.ch/zurichmavdataset.html).

Локальный `raw/readme.txt` помечает GPS altitude как MSL, но его описание масштабов lat/lon/alt не совпадает с экспортом CSV: фактически это градусы и метры. Точный datum/geoid для GT Z отдельно не установлен. `GroundTruthAGL` в имени нельзя интерпретировать как доказательство одинаковой высотной системы. Camera/IMU attitude conventions и GNSS antenna extrinsic здесь не восстановлены; lever-arm correction не применялась. Гравитация не вычиталась при диагностике координат/нейросети.

**Протокол вычислений.** Train индексы IMU `[0,18935)`; validation common `[18935,21623)`, 2688 точек; без test. GPS held на IMU grid и линейно интерполированный GT — ровно протокол исходного diagnostic. Transform `y = scale · R · x + translation`; SE2 включает yaw и независимый vertical offset, SE3 — полноценный rigid rotation без scale. RMSE3D=`sqrt(mean(dx²+dy²+dz²))`; H=`sqrt(mean(dx²+dy²))`; V=`sqrt(mean(dz²))`, все в метрах, без дополнительного выравнивания после указанного transform. Это full-GNSS frame diagnostic, не outage результат.

В таблице rounded значения; полная точность, матрицы R, углы и translations сохранены в **канонических** [frame_float64/alignment_metrics.csv](outputs/next_stage_diagnostics/results/frame_float64/alignment_metrics.csv) и [alignment_transforms.json](outputs/next_stage_diagnostics/results/frame_float64/alignment_transforms.json). Transform fit по train и оценка на validation:

| Transform | RMSE3D, м | H, м | V, м | translation XYZ, м | yaw, ° | scale |
| --- | --- | --- | --- | --- | --- | --- |
| raw | 11.309157 | 10.218456 | 4.845636 | (0.000000, 0.000000, 0.000000) | 0.000000 | 1.000000000 |
| translation | 8.226895 | 6.599039 | 4.912686 | (5.822398, 4.102730, 0.410163) | 0.000000 | 1.000000000 |
| se2_vertical | 8.074807 | 6.408434 | 4.912686 | (5.448133, 4.114201, 0.410163) | -0.159348 | 1.000000000 |
| se3 | 8.908264 | 6.508051 | 6.082964 | (5.308788, 3.960185, 3.010783) | -0.177319 | 1.000000000 |
| similarity_diagnostic | 8.926587 | 6.524421 | 6.092281 | (5.324020, 4.421222, 3.034424) | -0.177319 | 0.996577831 |

Validation-fitted **oracle**: fit и score на тех же 2688 точках. Это нижняя диагностическая оценка доступного выравнивания, **не validation score независимого метода и не предлагаемая калибровка**:

| Oracle transform | RMSE3D, м | H, м | V, м | translation XYZ, м | yaw, ° | scale |
| --- | --- | --- | --- | --- | --- | --- |
| translation | 7.057263 | 5.164867 | 4.809273 | (4.073906, 7.819480, -0.592524) | 0.000000 | 1.000000000 |
| se2_vertical | 7.056488 | 5.163809 | 4.809273 | (3.956774, 7.690692, -0.592524) | -0.073310 | 1.000000000 |
| se3 | 6.972255 | 5.234930 | 4.605198 | (4.081569, 7.745850, -4.292897) | -0.038504 | 1.000000000 |
| similarity_diagnostic | 6.969604 | 5.240164 | 4.595221 | (3.845037, 7.960855, -4.293318) | -0.038504 | 0.997651625 |

Полные углы SE3 rotations: train-fit 1.356045°, validation-oracle 2.025633°. Scale diagnostic около 0.997 не доказывает ошибку масштаба датчика: полёт слабо возбуждает все 3D направления, rigid/similarity могут поглощать GNSS drift. На validation train-fit SE3/similarity хуже простого SE2; нет основания автоматически переносить fitted scale/roll/pitch в preprocessing.

Отдельная **физическая перепроекция без какого-либо fit**: восстановить lat/lon из processed offsets в float64, спроецировать WGS84→UTM32N, вычесть тот же GT origin из GNSS и GT. Validation RMSE3D **7.758867**, H **6.059689**, V **4.845636** м вместо raw **11.309157** м. Среднее изменение GNSS относительно R7: **(5.042310,4.824489,0)** м, std=(0.412605,0.044627,0) м. Это показывает существенный frame effect; исходные R7 arrays и predictions не заменялись.

Есть и непостоянная ошибка. Средний требуемый GT−GNSS offset меняется с train (5.822398,4.102730,0.410163) до validation (4.073906,7.819480,−0.592524) м. В 60-секундных validation bins средний X offset 2.186940–8.218891 м, Y 4.953248–12.449208 м, Z −3.091538–0.867092 м. Поэтому один постоянный translation не описывает всю запись. Источник: [bias_60s_bins.json](outputs/next_stage_diagnostics/results/frame_float64/bias_60s_bins.json).

Дополнительная несогласованность экспорта: raw OnboardGPS, перепроецированный в UTM, и `GroundTruthAGL` столбцы `x_gps/y_gps/z_gps` по одинаковому imgid не всегда совпадают. Для 2165 train+val совпавших строк median norm=0.428329 м, P90=1.369199 м, max=16.473105 м; 393 строки отличаются >1 м. Первая точка совпадает с точностью 0.000320 м. Источник: [gps_export_consistency.json](outputs/next_stage_diagnostics/results/gps_export_consistency.json). Возможны иные timestamps/обработка экспортированного GPS, но причина не установлена. Эти столбцы не подменяли OnboardGPS или GT.

**Ответ о поправке 3–4.5 м:** frame/origin mismatch имеет подходящий метровый масштаб и создаёт задачу статической GNSS→GT коррекции. Однако learned mean full=(2.172282,1.307129,0.104550) м не равен ни требуемому mean offset, ни физическому transform. Поэтому подтверждён существенный вклад неверного frame, но не доказано, что сеть точно выучила одно геометрическое преобразование или что оно объясняет весь residual.

Техническое уточнение: первые frame файлы непосредственно в `results/` были рассчитаны с float32 inverse projection и **заменены как источник выводов** подкаталогом `results/frame_float64/`. Старые файлы оставлены для истории, смешивать их нельзя. Именно float64 таблицы использованы выше; повторный пересчёт всех их RMSE по сохранённым series/transforms дал максимальное расхождение 0.0. Это уточнение локальной диагностики, не изменение R7.

![GNSS/GT discrepancy](outputs/next_stage_diagnostics/results/gnss_gt_frame_bias.png)

Рисунок: GT−GNSS по XYZ до и после физической перепроекции без fit, train/validation; источник `frame_float64/train_validation_frame_series.npz`, скрипт `scripts/plot_next_stage_diagnostics.py`.

## 3. Learned neural correction

Определение: `correction(t)=prediction(t)−held_GNSS(t)`. Никакого GT в вычислении поправки нет. Ниже control, все 2688 точек, seed 0; `std` — population std (`ddof=0`), не uncertainty по seeds и не confidence interval. XYZ в метрах.

| Метод | mean XYZ | std XYZ | median XYZ | min XYZ | max XYZ | RMS norm |
| --- | --- | --- | --- | --- | --- | --- |
| fusion_full | (2.172282, 1.307129, 0.104550) | (1.676790, 1.299447, 0.197209) | (3.007690, 1.911613, 0.062805) | (-0.295624, -0.483583, -0.100353) | (3.776596, 2.565216, 0.704998) | 3.313217 |
| fusion_gps_only | (2.180675, 1.287913, 0.121128) | (1.651319, 1.314744, 0.211115) | (2.983479, 1.893826, 0.064640) | (-0.221626, -0.504478, -0.092900) | (3.777603, 2.568985, 0.745988) | 3.305860 |
| fusion_full_zero_imu | (2.168115, 1.309735, 0.100981) | (1.681817, 1.296538, 0.198567) | (3.011204, 1.915359, 0.061671) | (-0.325188, -0.481771, -0.110882) | (3.776566, 2.565445, 0.718766) | 3.312895 |

Для всех трёх методов, всех 10 сценариев и областей all/inside/outside те же mean/std/median/min/max, lag-1 и smoothness сохранены в [correction_statistics.json](outputs/next_stage_diagnostics/results/correction_statistics.json). Основная outage поправка full:

| Эпизод / длительность | mean XYZ, м | std XYZ, м | RMS norm, м |
| --- | --- | --- | --- |
| e1 / 10 с | (3.718873, 2.510856, 0.070751) | (0.007389, 0.004832, 0.007193) | 4.487711 |
| e1 / 30 с | (3.713964, 2.507425, 0.069753) | (0.025726, 0.019674, 0.010752) | 4.481823 |
| e1 / 60 с | (3.604150, 2.414434, 0.060272) | (0.219422, 0.184840, 0.023494) | 4.348090 |
| e2 / 10 с | (3.417101, 2.260679, 0.030029) | (0.346287, 0.298246, 0.012241) | 4.122762 |
| e2 / 30 с | (3.587896, 2.410673, 0.031407) | (0.199814, 0.167267, 0.014298) | 4.330524 |
| e2 / 60 с | (3.240460, 2.157705, 0.020069) | (0.739658, 0.554090, 0.029179) | 4.001454 |
| e3 / 10 с | (2.521285, 1.585676, -0.024169) | (0.611771, 0.439088, 0.012596) | 3.072304 |
| e3 / 30 с | (2.536547, 1.618566, -0.033652) | (1.208482, 0.823151, 0.033346) | 3.345755 |
| e3 / 60 с | (2.151176, 1.368822, -0.011935) | (1.581334, 1.056514, 0.009748) | 3.180929 |

**Window effect сильнее, чем глобальная smoothness.** В control lag-1 Pearson full по XYZ=(0.999631,0.999705,0.999007). Но средний шаг внутри окна всего **0.000141955 м**, а на границе окна **0.497950465 м**. Относительно собственного среднего окна RMS residual равен 0.000466519 м для full, 0.001795400 для gps_only и 0.004011967 для zero_IMU. Глобальная константа оставляет 2.130512 м RMS для full. Значит, это почти постоянное **смещение на окно**, меняющееся между окнами, а не один глобальный offset и не плавно интегрируемая IMU траектория. Доказательство: [window_constant_correction.json](outputs/next_stage_diagnostics/results/window_constant_correction.json).

На e3/60 внутри outage std(X,Y)=(1.5813,1.0565) м может ошибочно выглядеть как динамика. Но средний внутривоконный шаг около 0.000127 м, а граничный — 1.42028 м: большая часть изменения снова обусловлена переходом к следующему окну.

![Correction by window](outputs/next_stage_diagnostics/results/correction_window_structure.png)

Рисунок: три XYZ correction traces для control, серые вертикали — границы 192-point windows. Исходные saved predictions; вычислений attention здесь нет.

**Зависимости.** В control full correlation X/Y correction с time = −0.8981/−0.9257, с held Y = +0.9040/+0.9268, с held X = −0.8535/−0.8653. С accel X около +0.386, с компонентами gyro |r|≤0.0326. На e3/60 по всей записи correlation XYZ с availability=(0.1077,0.0672,0.3444), с outage age=(−0.2883,−0.2258,−0.2788). Только внутри этого outage time и age дают одинаковые r=(−0.7770,−0.7771,0.6920); это коллинеарные величины. Внутри outage availability константа, correlation не определена (`null`), а не ноль. Полный источник: [correlations.json](outputs/next_stage_diagnostics/results/correlations.json).

Эти корреляции не устанавливают причинность: time, position, window и конкретный маршрут связаны. Для проверки простой формы коррекции отдельно выполнен описательный OLS fit по первым 7 validation windows и score по последним 7. Это surrogate анализа выхода замороженной сети, не обучение navigation model и не оценка на test. Для full held-out correction RMSE: constant 3.94886 м; affine(position) 2.55757 м; quadratic(position) 2.05219 м; linear(time) 3.04945 м; linear(IMU) 3.94609 м; position+time 2.45113 м; position+time+IMU 2.44089 м. Все held-out centered R² отрицательны (для affine −1.860, quadratic −0.841). Хороший in-sample fit не переносится на вторую половину. Источник: [surrogate_fits.csv](outputs/next_stage_diagnostics/results/surrogate_fits.csv).

Вывод: зависимая от контекста GNSS поправка на окно подтверждена; глобальный affine/rotation механизм не установлен. Интерпретация «IMU-assisted motion» этими данными не поддержана.

## 4. IMU sensitivity

Сначала сравнение **сохранённых Kaggle predictions**. `RMS difference = sqrt(mean(||p_full−p_other||²))`, без alignment. full и gps_only — разные обученные checkpoints; их разница не является чистой интервенцией IMU. zero_IMU — чувствительность одного и того же full checkpoint.

| Эпизод / длительность | RMS correction full, м | RMS(full−gps_only), м | RMS(full−zero), м | zero / correction |
| --- | --- | --- | --- | --- |
| e1 / 10 с | 4.487711 | 0.003655 | 0.001133 | 0.0252% |
| e1 / 30 с | 4.481823 | 0.004026 | 0.001451 | 0.0324% |
| e1 / 60 с | 4.348090 | 0.012743 | 0.004188 | 0.0963% |
| e2 / 10 с | 4.122762 | 0.015665 | 0.005992 | 0.1453% |
| e2 / 30 с | 4.330524 | 0.015270 | 0.003725 | 0.0860% |
| e2 / 60 с | 4.001454 | 0.055015 | 0.012318 | 0.3078% |
| e3 / 10 с | 3.072304 | 0.092541 | 0.009572 | 0.3116% |
| e3 / 30 с | 3.345755 | 0.040933 | 0.010341 | 0.3091% |
| e3 / 60 с | 3.180929 | 0.043613 | 0.019648 | 0.6177% |

В control RMS(full−gps_only)=0.063388 м; RMS(full−zero)=0.010981 м. По девяти сценариям **вне outage** диапазоны соответственно 0.052031–0.071730 м и 0.010540–0.012009 м. Внутри outage effect zero на 2–3 порядка меньше поправки, shuffle/reverse примерно на 4–5 порядков. Полная статистика RMS/mean/median/max/ratio: [saved_prediction_differences.csv](outputs/next_stage_diagnostics/results/saved_prediction_differences.csv); временные XYZ traces — `results/val_*_series.npz`.

**Дополнительные локальные forwards выполнены:** full checkpoint seed 0, CPU PyTorch 2.8.0, eval/inference_mode, неизменённые параметры. SHA256 `8adacc2e7c16ed7d828ba4d49fab87dda10eeb908026b23f839d974a82a8140f`. Для каждого из 10 сценариев: CPU reference full; все 6 IMU каналов zero; совместная перестановка шестикомпонентных строк внутри каждого 192-point окна (seed 1701); разворот порядка строк внутри каждого окна. GT не модифицировал входы, GPS/masks/windows одинаковые. Всего 40 фиксированных forward traces, **ни одного optimizer step**.

RMS изменения prediction внутри outage относительно full **на том же CPU**:

| Эпизод / длительность | zero, м | shuffle, м | reverse, м | saved GPU−CPU ref, м |
| --- | --- | --- | --- | --- |
| e1 / 10 с | 0.001132556 | 0.000016809 | 0.000025273 | 0.000002204 |
| e1 / 30 с | 0.001451011 | 0.000023558 | 0.000025016 | 0.000003654 |
| e1 / 60 с | 0.004187836 | 0.000032328 | 0.000029578 | 0.000003157 |
| e2 / 10 с | 0.005991777 | 0.000128261 | 0.000138379 | 0.000002168 |
| e2 / 30 с | 0.003724913 | 0.000048375 | 0.000045442 | 0.000002170 |
| e2 / 60 с | 0.012317527 | 0.000117308 | 0.000122252 | 0.000002008 |
| e3 / 10 с | 0.009572235 | 0.000200634 | 0.000222619 | 0.000002548 |
| e3 / 30 с | 0.010340321 | 0.000140297 | 0.000135540 | 0.000002088 |
| e3 / 60 с | 0.019647823 | 0.000119851 | 0.000112497 | 0.000002014 |

В control zero/shuffle/reverse дают 0.010980952 / 0.000179579 / 0.000173552 м RMS. Вне outage ranges: zero 0.010539751–0.012008619 м, shuffle 0.000134214–0.000190183 м, reverse 0.000133759–0.000183734 м. Максимальная абсолютная перемена outage RMSE3D по 9 эпизодам: zero 0.007332357 м; shuffle 0.000008809 м; reverse 0.000066822 м. CPU/GPU difference ≈2–4 микрометра RMS; микроскопические изменения нельзя трактовать как надёжную практическую пользу.

Артефакты: [imu_sensitivity.csv](outputs/next_stage_diagnostics/results/imu_sensitivity.csv), [sensitivity_manifest.json](outputs/next_stage_diagnostics/results/sensitivity_manifest.json), `results/val_*_sensitivity.npz` с predictions, timestamps, masks и permutation indices. Это sensitivity tests фиксированного checkpoint, **не causal importance и не абляции с переобучением**. Низкая чувствительность при этих probes — факт; универсальная ненужность IMU или гарантированный эффект нового обучения — не установленный вывод.

## 5. Causality/window leakage analysis

В R7 `src/models/fusion_transformer.py:45` и `:98` self/cross-attention не передают causal mask. Diagnostic собирает 14 неперекрывающихся окон по 192 точки; по каждой prediction-time проверено, есть ли в её окне разрешённый token после окончания outage. Дополнительно проверен timestamp исходного GNSS fix, чтобы отличить свежий post-outage GNSS от stale hold. В этих 9 эпизодах обе доли совпали.

Ниже номера окон/индексов 0-based относительно common validation; timestamps — processed секунды. Перечень **всех отдельных exposed timestamps**, а не только границ, сохранён в [window_exposure_manifest.json](outputs/next_stage_diagnostics/results/window_exposure_manifest.json).

| Эпизод / длительность | Exposed N / outage N | Доля | Окно | Первый…последний exposed timestamp, с | Без post-GNSS N |
| --- | --- | --- | --- | --- | --- |
| e1 / 10 с | 18 / 100 | 18.0000% | [4] | 1984.070753 … 1985.771762 | 82 |
| e1 / 30 с | 26 / 300 | 8.6667% | [5] | 2003.278753 … 2005.778753 | 274 |
| e1 / 60 с | 134 / 600 | 22.3333% | [6] | 2022.483752 … 2035.788752 | 466 |
| e2 / 10 с | 35 / 100 | 35.0000% | [6] | 2022.483752 … 2025.885753 | 65 |
| e2 / 30 с | 43 / 300 | 14.3333% | [7] | 2041.691753 … 2045.893754 | 257 |
| e2 / 60 с | 151 / 600 | 25.1667% | [8] | 2060.901753 … 2075.906792 | 449 |
| e3 / 10 с | 51 / 100 | 51.0000% | [8] | 2060.901753 … 2065.903754 | 49 |
| e3 / 30 с | 59 / 300 | 19.6667% | [9] | 2080.108753 … 2085.910765 | 241 |
| e3 / 60 с | 167 / 600 | 27.8333% | [10] | 2099.315753 … 2115.923752 | 433 |

Пересчёт сохранённых predictions **только на unexposed outage timestamps**, одинаковая subset mask для всех шести методов. Ниже RMSE3D, м, один seed; полные H/V, число точек и exposed/all subsets также сохранены:

| Эпизод / длительность | full | gps_only | zero_IMU | held | CV | LSTM |
| --- | --- | --- | --- | --- | --- | --- |
| e1 / 10 с | 8.937054 | 8.935492 | 8.937967 | 10.345579 | 8.917724 | 7.981429 |
| e1 / 30 с | 17.736417 | 17.737277 | 17.736798 | 18.224358 | 8.638616 | 17.455118 |
| e1 / 60 с | 30.400274 | 30.401858 | 30.400446 | 30.424752 | 11.957779 | 30.360353 |
| e2 / 10 с | 6.824075 | 6.826117 | 6.822759 | 8.390628 | 10.424122 | 7.724182 |
| e2 / 30 с | 10.454826 | 10.456565 | 10.453951 | 11.375347 | 14.115614 | 11.233682 |
| e2 / 60 с | 20.693735 | 20.694775 | 20.693169 | 20.639057 | 17.105028 | 21.268388 |
| e3 / 10 с | 9.034229 | 9.066897 | 9.034637 | 12.319091 | 14.427855 | 8.352729 |
| e3 / 30 с | 12.199391 | 12.204627 | 12.199229 | 13.521798 | 16.117889 | 12.525704 |
| e3 / 60 с | 23.006742 | 23.000075 | 23.006081 | 23.030617 | 19.924549 | 23.540023 |

Источник: [window_exposure_metrics.csv](outputs/next_stage_diagnostics/results/window_exposure_metrics.csv). Даже на этих subsets CV лучше full во всех трёх 60-секундных эпизодах. full/gps_only остаются близки.

**Предел вывода:** это измерение потенциального доступа к future GNSS и ошибки на другой subset, а не парный counterfactual «тот же timestamp с/без будущего GNSS». Из разницы all/exposed/unexposed RMSE нельзя вычислить причинную выгоду future context. Неисключённые future IMU/context внутри окна остаются; результат на unexposed subset не превращает R7 в causal модель. Дополнительный маскированный inference для изоляции future-GNSS эффекта здесь не выполнялся.

## 6. Timestamp/synchronization findings

Фактические RawAccel/RawGyro/OnboardGPS/GroundTruthAGL файлы из dataset version 2 побитно воспроизвели **все шесть** R7 processed arrays в памяти, включая timestamps. Поэтому диагностика source problems относится к R7, а не к другой загрузке. Файлы/хеши: [raw/extracted_receipt.json](outputs/next_stage_diagnostics/raw/extracted_receipt.json), [r7_inputs/receipt.json](outputs/next_stage_diagnostics/r7_inputs/receipt.json); расчёты: [synchronization.json](outputs/next_stage_diagnostics/results/synchronization.json), `scripts/analyze_next_stage.py:96`.

| Split | IMU N | Будущий gyro N / доля | Median/max gyro lead, с | Median backward gyro age, с | dt min/median/max, с | dt>0.15 N | Elapsed−0.1×intervals, с |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 18935 | 15815 / 83.5226% | 0.097988 / 1.098011 | 0.001022 | 0.004875/0.100000/1.099012 | 12 | 5.162847 |
| validation | 2705 | 2704 / 99.9630% | 0.094994 / 1.097000 | 0.005007 | 0.097997/0.100000/1.100000 | 4 | 1.591996 |

Здесь validation — исходные 2705 samples `[18935,21640)`, в отличие от common evaluation 2688. R7 `searchsorted(t_gyro,t_accel)` без `side='right'−1` выбирает следующий gyro почти везде на validation. Причинный backward matching уменьшает median age до 5.007 мс; max age на validation 20.011 мс. Нельзя заменять этот lead выдуманным постоянным sensor delay: сначала нужен корректный join; физическая latency отдельно не измерена.

Шесть обратных GPS шагов (индекс строки i → i+1):

| i | t[i], с | t[i+1], с | dt, с |
| --- | --- | --- | --- |
| 1589 | 59.991087 | 59.990803 | -0.000284 |
| 23839 | 803.889335 | 803.881891 | -0.007444 |
| 25639 | 863.887603 | 863.883127 | -0.004476 |
| 27439 | 923.889891 | 923.884394 | -0.005497 |
| 29239 | 983.893561 | 983.893179 | -0.000382 |
| 38215 | 1283.899186 | 1283.898937 | -0.000249 |

Все шесть находятся в train; максимальный backward step 0.007444 с. На train+validation stable-sort и backward lookup меняют 3 выбранные GPS строки/позиции относительно несортированного R7. Малое число изменённых строк не делает unsorted searchsorted корректным.

Частота строк и частота изменения координат:

| Split | GPS rows | Row rate, Hz | Coordinate-change events | Change rate, Hz | Median change interval, с | fix_type / satellites |
| --- | --- | --- | --- | --- | --- | --- |
| train | 56826 | 29.929717 | 9472 | 4.989069 | 0.199856 | {'3': 56826} / 5.0–10.0 |
| validation | 8115 | 29.822050 | 1353 | 4.970334 | 0.199843 | {'3': 8115} / 6.0–8.0 |

Таким образом ~30 Hz — частота экспортированных строк, а изменения координат происходят примерно на 5 Hz. Это **proxy**, не доказанная частота физических receiver fixes: отдельного receiver measurement epoch в файле нет, одинаковые реальные fixes могли бы иметь одинаковые координаты. V2 должен явно называть age возрастом coordinate-change proxy. Поля velocity из OnboardGPS не используются как скрытый IMU input.

Конкретные preprocessing изменения подготовлены отдельно в `src/pilot_v2/data.py:20`, `:80`:

1. Пиновать raw hashes; сохранить исходную accel timeline и split boundaries. Stable-sort gyro/GPS, дедуплицировать равные timestamps детерминированно; не сортировать accel молча при нарушении порядка.
2. Gyro выбирать только `source_time <= accel_time`, сохранить age/gap; без past sample завершать проверку ошибкой, не подставлять будущее. Использовать реальные dt; не заполнять секундные gaps как будто прошло 0.1 с.
3. Перепроецировать GPS WGS84→UTM32N и использовать одно общее GNSS-derived XYZ origin для GPS и GT. Не fitted transform; не invent geoid/antenna offset.
4. Для GNSS создавать причинные coordinate-change events при `fix_type>=3`; передавать availability/new-fix/fix-age; запрещать source fixes из outage intervals и stale recovery. Сохранять source IDs/timestamps.
5. GT интерполировать только для label/score, не как вход или reset. Учесть native ~1 Hz: interpolated 10 Hz samples не независимые высокочастотные ground-truth наблюдения.

Полный dataset на диске не пересоздавался. В памяти проверены планы train/validation для нового кода; это не запуск обучения.

## 7. What is actually wrong with R7

Критические ограничения для научного заявления об IMU-assisted navigation:

- **Результат:** full/gps_only/zero_IMU близки; фиксированная full модель почти нечувствительна к временному порядку IMU. Основной выход — кусочно постоянное window смещение GNSS.
- **Код + raw evidence:** несовместимые horizontal origins/projections; метрический выигрыш над raw GNSS может отражать исправление координатной постановки, а не восстановление dynamics.
- **Код + raw evidence:** future gyro lookup; noncausal Transformer; 8.67–51% outage timestamps в зависимости от сценария имеют post-outage GNSS внутри окна.
- **Код/артефакты R7:** сеть обучалась без outages, а residual head возвращает held-GNSS + поправку. Эта цель допускает выучить GNSS→GT bias вместо инерциального движения.
- **Протокол:** один flight, temporal train/val, эти validation episodes уже много раз просмотрены. Новый pilot не даст независимой оценки generalization; 9 эпизодов (3 onsets×3 вложенных duration) не 9 независимых flights. Один seed не позволяет оценить seed variance.
- **GT/датасет:** camera position, приблизительно 1 Hz; не подтверждены GT vertical datum, antenna lever arm и receiver fix epochs. GNSS использован при построении photogrammetric GT. Это ограничивает заявления о метрической sensor-fusion точности.

При этом clean diagnostic корректно сохраняет общие masks/timestamps и paired методы; его старые 152 catalogued artifacts неизменны. Ошибки R7 не делают выдуманными diagnostic числа: они меняют их научную интерпретацию. Нельзя задним числом объявить их результатами исправленного causal pipeline.

Для ориентира — **исходные**, all-outage 3D RMSE из завершённого Kaggle diagnostic (не новые subsetting/projection результаты):

| Сценарий | full, м | gps_only, м | held, м | CV, м |
| --- | --- | --- | --- | --- |
| control | 9.805785 | 9.820753 | 11.309157 | 11.309157 |
| e1 / 10 с | 9.380186 | 9.378845 | 10.707360 | 8.725929 |
| e1 / 30 с | 19.299684 | 19.300247 | 19.692283 | 8.927374 |
| e1 / 60 с | 38.456743 | 38.456590 | 38.429762 | 15.730551 |
| e2 / 10 с | 6.543743 | 6.547768 | 8.111872 | 10.818932 |
| e2 / 30 с | 12.504412 | 12.504036 | 13.149368 | 14.711135 |
| e2 / 60 с | 29.891514 | 29.879720 | 29.699189 | 19.394333 |
| e3 / 10 с | 8.570183 | 8.635606 | 11.222937 | 14.803920 |
| e3 / 30 с | 15.517631 | 15.514042 | 16.402230 | 16.913458 |
| e3 / 60 с | 32.848653 | 32.829842 | 32.932161 | 24.129290 |

Допустимо утверждать: на этих фиксированных validation scenarios текущий checkpoint не демонстрирует практически значимого IMU вклада; на всех трёх 60 s эпизодах CV лучше full. Недопустимо: «IMU вообще не нужна», «Transformer лучше/хуже любого EKF», «преимущество FusionNav доказано», «V2 исправит результат». EKF в данном clean diagnostic не оценивался.

## 8. Proposed V2 scientific formulation

Цель pilot — проверить **добавочную предсказательную пользу IMU для относительного движения камеры во время outage** при одинаковом GNSS контексте. Это пока не полноценный attitude/bias-aware inertial filter и не восстановленная физическая калибровка всех сенсоров.

Общий nominal frame — UTM32N E/N плюс исходная высота, один origin от первого разрешённого GNSS. Horizontal frame согласован физической проекцией; точная vertical datum/extrinsic calibration остаётся открытой. Constant datum/origin bias исключается из relative-motion labels, но не объявляется физически исправленным. Camera vs IMU motion включает небольшой rotation-dependent lever arm; без extrinsics нельзя его точно снять. Если цель статьи требует сертифицированной позиции IMU/антенны, эти metadata нужны до окончательных научных выводов.

Два варианта target:

| Target | Плюсы | Основной риск на этом датасете |
| --- | --- | --- |
| A: `GT(t) − last_valid_GNSS` | Простой position residual, непосредственно абсолютная ошибка | Снова содержит GNSS anchor/frame bias; может дать привлекательные цифры статической коррекцией и не требовать IMU |
| B: relative displacement / residual velocity during outage | Constant GT origin и anchor GNSS error не являются задачей обучения; проверяет перенос движения от legal fix time | GT~1 Hz не подтверждает мгновенное 10 Hz acceleration/velocity; интеграция может дрейфовать; требует строгих dt/anchor conventions |

**Рекомендация: B.** Model head — residual velocity в UTM axes относительно причинного GNSS CV prior. Интегрировать по actual dt только во время outage; loss на `GT(t)−GT(last_legal_source_time)` около одного раза в реальную секунду и в конце episode. Это displacement supervision, не finite-difference noisy 10 Hz velocity labels. Позиционный прогноз = observed GNSS anchor + predicted relative motion; GT anchor используется только в loss/reference, никогда для инициализации состояния. Константный GNSS bias остаётся во вторичной raw absolute error.

Входы full/gps_only одинаковые за исключением восьми IMU-branch features. Общая GNSS/time ветвь: legal fix displacement XYZ, causal trailing-5s GNSS velocity XYZ, proxy fix age/60, link availability, new-fix flag, actual dt (10 features). IMU: accel+gyro 6 каналов, backward gyro age, dt-gap flag (8). Accel и gyro остаются в sensor frame; масштабирование mean/std fit только по train. Гравитация не вычитается с помощью GT/onboard attitude. В gps_only все 8 IMU features обнуляются внутри модели; размеры и параметры сохраняются.

Нет абсолютного XYZ, абсолютного времени полёта, onboard pose, GT и derived GT features среди model inputs. Actual dt есть в обеих моделях. В loss/outputs абсолютные GPS/GT позиции необходимы для anchors и оценки, но не передаются recurrent encoder. Первая dt outage измеряется от последнего legal **source timestamp**, а не от предыдущей grid строки. CV prior заморожен на pre-onset значении, не использует future fixes. При восстановлении GNSS позиция возвращается к разрешённому fix; recurrent state продолжает идти вперёд.

## 9. Minimal pilot experiment

**Подготовлен, не обучен.** Конфиг [v2_motion_pilot.json](configs/v2_motion_pilot.json); заранее фиксированные scenarios/hashes — [v2_motion_pilot_manifest.json](configs/v2_motion_pilot_manifest.json).

| Свойство | Подготовленная настройка |
| --- | --- |
| Модели | GRU full и gps_only: по **6755** parameters; LSTM full: **8867** |
| Архитектура | IMU Linear(8,16)+Tanh; GNSS Linear(10,16)+Tanh; concat32; unidirectional GRU/LSTM hidden32; Linear(32,3) residual velocity |
| Начальный head | Нулевой; до обучения соответствует CV motion. В causal тесте head специально ненулевой, чтобы проверка не была тривиальной |
| Runs/seeds | 3 модели × seed 0; ни один run не выполнен |
| Train/validation/test | Train `[0,18935)`; common val `[18935,21623)`; test не используется |
| Train windows | 1024 samples, stride512; 35 windows/epoch, перекрытие только внутри train |
| Outages | Один random 10/30/60 s blackout/window; warmup30s, recovery≥5s; одинаковые windows/masks/order у full/gps_only, RNG seed+10000+epoch |
| Оптимизация | 8 epochs, AdamW lr=0.001, weight_decay=0.0001, batch4, clip1, Huber delta1m |
| Loss | Integrated relative displacement on outage labels около1Hz; без GT state reset |
| Checkpoint | Последняя фиксированная epoch8; без best-validation выбора, early stopping или перебора seeds |
| Evaluation | Control + те же 9 fixed clean 10/30/60 s outages; непрерывный recurrent state на 2688 точках, reset только в начале сценария |
| Методы в summary | Три trained variants + sensitivity full_zero_IMU + held + CV = ожидаемые 60 rows |
| Primary metric | RMSE3D относительного движения внутри outage, метры; GT last-source anchor только в target; control для этой метрики N/A |
| Secondary metrics | Raw absolute RMSE3D/H/V и last unavailable point error; без alignment; полная precision сохраняется |
| Сохранение | Config/manifest/scaler, source snapshot/hashes, environment, final checkpoints, train history/mask manifest, prediction NPZ/summary/status |

Primary score использует все outage timestamps интерполированного GT; это временное усреднение, не 100/300/600 независимых измерений. Train loss выбирает около1Hz labels. Внутри каждого сценария held/CV/full/gps_only используют одинаковые source restrictions, маски и timestamps. R7 и V2 изменяют preprocessing, target и архитектуру одновременно: сравнение с R7 — контроль направления, **не изолированный architecture ablation**. Главный парный эксперимент — full vs gps_only внутри V2.

Почему компактная recurrent сеть: причинность явная, state переносится через весь outage, нет окно-зависимого доступа к будущему. Имя FusionNav V2 обозначает новый pilot fusion predictor, не доказанное улучшение исходной Transformer архитектуры. LSTM — вспомогательный baseline с другим parameter count; matched comparison проводится между двумя GRU.

Критерий перехода к 5 seeds: устойчивое практически значимое paired улучшение full относительно gps_only на нескольких заранее фиксированных эпизодах, особенно 30/60 s; проверить CV, raw absolute error и чувствительность full к IMU. Обязательный процент не задан. Выигрыш на одном эпизоде, только после отбора лучшего checkpoint или только по абсолютной offset-sensitive метрике недостаточен. При отсутствии повторяемого выигрыша не запускать большой sweep автоматически.

**Проверки подготовки:** 17 unit tests прошли локально и те же 17 — из встроенного notebook пакета в отдельном temporary directory. Проверены causal prefixes/state carry с ненулевым head, train-only scaler, отсутствие GT/absolute-position input, source restrictions, anchor dt, equality zero-head/CV, совпадение full/gps_only parameter counts, детерминированность augmentation и gate до любого I/O. В памяти проверены все 280 training window plans (35×8, без модели/optimizer) и 10 validation scenarios; timestamps/mask hashes совпали с fixed diagnostic manifest. Источник: [preparation_verification.json](outputs/next_stage_diagnostics/results/preparation_verification.json).

**Граница проверки:** локально нет pyproj; установок не было. Для проверки raw→features/labels в памяти явно подставлена аналитическая UTM diagnostic projection. Production `pyproj.Transformer` путь ещё не исполнен; notebook проверяет наличие pyproj и raw hashes до обучения. Эта замена не выдаётся за полный production run. Notebook outputs пусты, execution_count=null, RUN_TRAINING=False; готовность означает подготовку и перечисленные проверки, не успешное обучение на Kaggle.

Команда только для последующего отдельно разрешённого запуска (сейчас не выполнялась):

```bash
PYTHONPATH=src python3 -B -m pilot_v2.run   --config configs/v2_motion_pilot.json   --raw-dir /path/to/pinned/raw-csv-directory   --out-dir outputs/v2_motion_pilot   --run-training
```

На Kaggle следующий **один** notebook: [kaggle_v2_motion_pilot.ipynb](notebooks/kaggle_v2_motion_pilot.ipynb). Он содержит свой source/config/test snapshot, не требует GitHub pull или API credentials. Подключить dataset version2; зависимости numpy/pandas/torch/pyproj должны уже быть доступны. После отдельной команды пользователя разрешается сменить RUN_TRAINING на True и выполнить Save & Run All. Автоматической установки, отправки notebook или старта jobs на этом этапе нет.

## 10. Files prepared

| Файл/каталог | Назначение / статус |
| --- | --- |
| `NEXT_STAGE_DIAGNOSTIC_REPORT.md` | Этот новый отчёт |
| `outputs/next_stage_diagnostics/r7_inputs/` | Пинованные 6 arrays, meta и full seed0 checkpoint; receipt hashes |
| `outputs/next_stage_diagnostics/raw/` | Четыре небольших CSV + readme/MATLAB metadata; download/extraction receipts; не полный image dataset |
| `outputs/next_stage_diagnostics/results/frame_float64/` | Канонические frame metrics/transforms/series/60s bins |
| `results/correction_statistics.json`, `correlations.json`, `surrogate_fits.csv`, `window_constant_correction.json` | Статистика и проверка формы correction |
| `results/saved_prediction_differences.csv`, `val_*_series.npz` | full−gps_only/full−zero и correction по времени; scopes/masks |
| `results/imu_sensitivity.csv`, `sensitivity_manifest.json`, `val_*_sensitivity.npz` | Выполненные local fixed-checkpoint probes; seed/permutations |
| `results/window_exposure_metrics.csv`, `window_exposure_manifest.json` | Exact exposed timestamps и метрики subsets всех 6 методов |
| `results/synchronization.json`, `gps_export_consistency.json` | Raw sensor sync/frequency и расхождение экспортов |
| `results/correction_window_structure.png`, `gnss_gt_frame_bias.png` | Два новых рисунка по сохранённым данным, не attention analysis |
| `results/analysis_status.json`, `preparation_verification.json`, `artifact_index.json` | Статус выполненной диагностики, проверки и hashes/канонические источники |
| `src/diagnostics/next_stage.py`, `scripts/analyze_next_stage.py` | Код локальной диагностики/фиксированных forwards; не training runner |
| `scripts/plot_next_stage_diagnostics.py` | Построение двух рисунков и exposure manifest из saved arrays |
| `src/pilot_v2/{data,model,run}.py` | Отдельный новый V2 pipeline, training guard |
| `configs/v2_motion_pilot.json`, `v2_motion_pilot_manifest.json` | Pilot contract и fixed validation scenarios |
| `tests/test_next_stage_v2.py` | 17 deterministic tests |
| `scripts/build_v2_pilot_notebook.py` | Builder self-contained notebook; сам не обучает и не отправляет Kaggle jobs |
| `notebooks/kaggle_v2_motion_pilot.ipynb` | Один подготовленный следующий notebook, outputs пусты |

Пути `results/...` в этой таблице относительно `outputs/next_stage_diagnostics/`. Полные hashes/список файлов — [artifact_index.json](outputs/next_stage_diagnostics/results/artifact_index.json). Анализатор отказывается писать в уже существующий results каталог; его не следует запускать поверх сохранённого результата. Для повторения потребуется новый каталог вывода; новый raw preprocessing для training запускается только отдельно разрешённым pilot.

Старые clean-outage summary/paired/status и predictions не менялись: повторно проверены все 152 hashes исходного artifact manifest. Все 8 загруженных R7 input hashes совпадают. Старые tracked source/config/notebooks/weights/results не редактировались; новое находится в отдельных файлах. Подготовленный V2 code не должен описываться как версия, которой был обучен R7.

## 11. What still requires user approval/run

**Только следующий запуск обучения:** отдельная команда пользователя на один V2 pilot. Причина ожидания — прямое ограничение текущего запроса «подготовь, но не запускай pilot», а не дополнительный skill/approval requirement. Здесь не требуется ещё один sweep, пересоздание всего dataset или обучение пяти seeds. Пока нет результатов V2 и нельзя утверждать, что IMU contribution улучшился.

Что остаётся реально неизвестным: receiver fix epoch/latency; точный vertical datum/geoid для GT; GNSS antenna↔camera/IMU extrinsics и conventions; причина несовпадения GPS export внутри GroundTruthAGL с OnboardGPS; независимая траектория/полёт для generalization. Это не отсутствующие summary/predictions: необходимые R7 и clean diagnostic артефакты доступны и проверены. Эти metadata ограничивают окончательные научные заявления; relative-motion pilot можно использовать как явно ограниченную проверку гипотезы, не как подтверждение полной физической калибровки.

**A. Current R7 mainly learns: static GNSS correction — conditioned on each window.** Доказательства: 99.9999952% variance correction full объясняет среднее своего окна; RMS внутривоконного остатка 0.000466519 м; shuffle/reverse IMU effect ≪1 мм; gps_only практически воспроизводит full. Это не одна глобальная константа и не доказанный rigid transform. Выражение dynamic GNSS correction возможно только в слабом смысле изменения offset между окнами; IMU-assisted motion не подтверждено.

**B. Нужно ли новое обучение? YES**, если проверяется IMU-assisted motion в исправленной causal постановке. Нет необходимости заново обучать R7 для диагноза: сохранённых результатов и выполненных probes уже достаточно. Новое обучение не гарантирует выигрыша.

**C. Что меняется:** общая UTM проекция/origin, backward sensor sync и actual dt, causal recurrent state, GNSS proxy fix-age/availability, train outages, отсутствие absolute position/time input, relative-motion loss вместо absolute GNSS→GT correction, одинаковая GRU full/gps_only, фиксированный final checkpoint и один seed. Это отдельный pilot, не косметическое исправление старых метрик.

**D. Следующий единственный Kaggle notebook:** `notebooks/kaggle_v2_motion_pilot.ipynb`. **Подготовлен; не запущен.**
