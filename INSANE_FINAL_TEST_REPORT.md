# INSANE FINAL TEST REPORT — остановлен на preflight

**Финальная оценка не выполнена.** Единственный разрешённый Kaggle job завершился технической ошибкой до inference. Повторный job не запускался; проверки не обходились и код/параметры после отправки не менялись.

## Статус и выполненный объём

- Kernel: `shapok/drone-nav-insane-final-test`, version **1**, output run ID **353128919**.
- Фактически выполнено: **0 model evaluations, 0 baseline evaluations, 0 trainings, 0 optimizer steps**.
- Планировалось: 48 model×scenario +24 baseline×scenario, всего72; шесть моделей на8 сценариях.
- Локально прошли33 tests и точная проверка notebook input с совместимостью шести checkpoints. На Kaggle выполнение остановилось в input preflight, **до remote unit tests и до test runner**.
- Полноценные оценки на Mac не выполнялись. Численных test-результатов нет.

## Точный блокер

В notebook cell `In [3]`, `src/insane_final_test/run.py:48`:

```python
assert json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True), 'Test scenario/support/hash mismatch'
```

Ошибка: `AssertionError: Test scenario/support/hash mismatch`.

До этой строки прошли сверки зафиксированных файлов source/config/scalers/checkpoints/raw/calibration и SHA256 файла test manifest. Затем сравнивался весь заново сформированный manifest, включая scenarios, point/mask hashes и inventory, с локально зафиксированным JSON.

**Конкретное несовпавшее поле не установлено:** реализация остановилась без экспорта actual manifest или структурированного diff. Сообщение само по себе не доказывает изменения raw data, outage-масок или GT timestamps. Разницу floating-point/inventory между средами можно рассматривать только как неподтверждённую гипотезу, не как найденную причину. Никакого ослабления равенства или исправления после сбоя не выполнялось.

Доказательства: `kaggle_run_log.json`, читаемая копия `kaggle_run_log.txt`, `failure_execution_evidence.json`, `submission.json`, source snapshot. Kaggle API перечислил105 output files подготовки, но ни одного пути `outputs/insane_final_test/`: научный runner не начал запись результатов. Файлы в этой папке созданы локально для фиксации сбоя; `status.json` явно помечает это происхождение.

## Зафиксированный протокол, ещё не оценённый

Все шесть ранее выбранных на validation best checkpoints сохранены побайтно: adaptive_full и adaptive_gps_only, seeds0/1/2. Основная модель — full; исключения seeds/ensemble нет. Scalers остались train-only, config и архитектура не менялись. Damped-CV τ=5 с. ESKF not_ready, без численных оценок.

Test: только mars_6/mars_7. Chronological/support-only selector проверен против исходного validation-алгоритма. History20s, recovery10s, onset stride60s, durations10/30/60s, прежние gaps/causal features/native GT timestamps. Зафиксированы6 outages и2controls; unsupported=[]; manifest создан до любых predictions/baseline metrics.

| Полёт | Onset, с от IMU epoch | Outage, с | Исходных GT точек внутри outage |
|---|---:|---:|---:|
| mars_6 |20.36518530845642|10|43|
| mars_6 |20.36518530845642|30|127|
| mars_6 |20.36518530845642|60|258|
| mars_7 |21.3|10|47|
| mars_7 |21.3|30|122|
| mars_7 |21.3|60|241|

Это **метаданные поддержки**, не ошибки моделей. Controls содержат392 и393 исходных GT timestamps соответственно. Длительности не менялись, полёты не склеивались, padding не считался измерениями.

## A. Подтвердилось ли преимущество full над gps_only на test?

**Проверить не удалось.** Ни один test-прогноз не выполнен. Для10/30/60с нет full/gps_only RMSE, mean/SD, paired difference или win count. Нули в количестве выполненных оценок не являются нулевыми ошибками моделей. Validation-результаты не подставляются вместо test.

## B. Сравнение с Held/CV/Damped-CV по каждому полёту

**Недоступно для обоих полётов и всех длительностей:** baseline evaluation не началась. τ=5с фиксировано, но test-ошибка Damped-CV неизвестна.

## C. Научные выводы и ограничения

Этот job не добавляет эмпирических свидетельств в пользу или против IMU/adaptive-механизма. Прежние validation-выводы остаются validation-выводами; переноса на отложенные полёты пока не показано. Результаты проверки байтов/локальных tests не заменяют научную оценку.

Не проводится изолированный трёхseed-контроль ungated, gate не является доказанным детектором остановки, ESKF не оценивался. Нельзя заявлять generalization, превосходство над INS/EKF, новизну или готовность к Q2 по этому незавершённому этапу.

## Сохранено и отсутствует

Сохранены FINAL_EVALUATION_LOCK, test manifest, config/scalers, исходные hashes/provenance, шесть неизменных checkpoints, source snapshot, notebook, логи, upload/submission receipts и классификация сбоя.

**Не получены и не сфабрикованы:** summary.csv, paired_differences.csv, duration/per-flight/per-seed test summaries, predictions/GT/masks/g(t), gate figure, фактическое remote hardware environment, test-metric verification. `independent_verification.json` сообщает not_run_no_predictions. Remote regenerated actual manifest не был экспортирован, поэтому точное несовпавшее поле неизвестно.

**STOP.** Автоматического исправления, повторного test или следующей версии нет. Исполняемые исходники и notebook оставлены в отправленном состоянии для разбора ошибки; запускать их повторно в рамках этого разрешения нельзя.
