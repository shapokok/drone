# Минимальное исправление manifest preflight

Первый job (353128919) остановился до inference на равенстве сериализованного целого JSON. Локально regenerated manifest совпадал полностью; исходный failed job actual/diff не сохранил. Один явно разрешённый повтор сохранил следующие реальные Kaggle отличия:

| Preflight | Поле | Locked expected | Kaggle actual | Абсолютная разница | Решение |
|---|---|---|---|---|---|
| check_01 | /inventory/mars_6/projection_error_m | 1.116003289780565e-09 | 1.1160175006352802e-09 | 1.4210854715202004e-14 | accepted |
| check_01 | /inventory/mars_7/projection_error_m | 1.1317560222323664e-09 | 1.116038816917353e-09 | 1.5717205315013416e-11 | accepted |
| check_02 | /inventory/mars_6/projection_error_m | 1.116003289780565e-09 | 1.1160175006352802e-09 | 1.4210854715202004e-14 | accepted |
| check_02 | /inventory/mars_7/projection_error_m | 1.1317560222323664e-09 | 1.116038816917353e-09 | 1.5717205315013416e-11 | accepted |

Это диагностический остаток float64 ECEF→ENU projection check. Точное равенство этого вычисляемого поля было хрупким; сохранённый diff подтверждает источник расхождения в повторном выполнении. Остальные поля, включая все научно определяющие поля, совпали строго. Допуск был зафиксирован до запуска: atol=1e-8м, rtol=0. Ни timestamps/маски, ни сами координаты, ни алгоритм проекции не изменялись.

`FINAL_EVALUATION_LOCK.json` и все закреплённые исходники/сценарии/checkpoints/scalers/raw input остались побайтно прежними. Patch подключает отдельную функцию preflight к неизменному runner; snapshot и hashes patch закреплены в `PREFLIGHT_PATCH_MANIFEST.json`. Новая функция сохраняет actual manifest и field-level diff до assert. Source hash failure, scientific field mismatch, NaN/Inf или diagnostic difference вне допуска останавливают исполнение до inference.

41 локальный test и41 Kaggle test прошли. Добавленные8 tests проверяют точное совпадение, допустимый/недопустимый round-off, строгость IDs/counts/timestamps/масок/source hashes и запись диагностики при ошибке. Test-only runner выполнил48 model evaluations и24 baseline evaluations; обучение и дополнительные seed отсутствуют. Предыдущий job не успел выполнить ни одной оценки, поэтому это первое получение test-метрик, не повтор после просмотра результата.

Подробные свидетельства: `preflight_evidence/check_01/` и `check_02/`, `PREFLIGHT_REPAIR_VERIFICATION.json`, исходный lock, source snapshot, logs и submission/download receipts. Строка retries=0 в неизменном runner status относится к внутренним повторам runner; общая история включает ранее проваленный job и один отдельно разрешённый технический повтор (Kaggle version2).

**STOP. Дополнительные jobs/изменения научного протокола не выполнялись.**
