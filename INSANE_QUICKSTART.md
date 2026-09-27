# INSANE pilot — запуск и артефакты

**Выполнен:** [Kaggle version 3](https://www.kaggle.com/code/shapok/drone-nav-insane-pilot), output ID `353114895`. 18 tests пройдены; результаты скачаны и независимо пересчитаны. См. [INSANE_PILOT_REPORT.md](INSANE_PILOT_REPORT.md). Устойчивый IMU benefit не подтверждён; повторный запуск не требуется.

Готовый notebook: [notebooks/kaggle_insane_pilot.ipynb](notebooks/kaggle_insane_pilot.ipynb). Код, tests и protocol manifests встроены с SHA256: GitHub main не используется. Финальным сообщением пользователь разрешил **один** приватный Kaggle pilot, поэтому `RUN_TRAINING=True`. На Mac выполнялись только подготовка и небольшие проверки.

1. В Kaggle подключить приватный dataset **shapok/insane-navigation-pilot-data-v1, version 2**, GPU, internet off. Единственная настройка пути: `DATA_ROOT` (по умолчанию `/kaggle/input`, вложенный mount определяется по pinned files). Для режима одних проверок установить `RUN_TRAINING=False`.
В dataset используется `insane_payload.b64`: notebook декодирует исходный ZIP и проверяет SHA256; это сохраняет вложенные калибровки побайтно.

2. Notebook: проверка hashes/schema/ENU/масок → causal preparation → unit tests → два обучения seed0 → fixed validation → экспорт. Любая ошибка preflight/tests блокирует обучение. Уже выполненный pilot автоматически повторять нельзя; существующие outputs не перезаписываются.
3. Скачать весь `outputs/insane_pilot/` и `insane_launch/`. Отчёт создаётся автоматически: `INSANE_PILOT_REPORT.md`; основные файлы — `summary.csv`, `paired_differences.csv`, `training_history.csv`, `status.json`, checkpoints, native predictions, manifests/scalers/environment/source snapshot.

**Split:** train `mars_1/2/3`; validation `mars_4/5`; test `mars_6/7` зарезервирован и **не включён в Kaggle inputs**. `outdoor_1` — только loader smoke. 9 fixed outages (10/30/60s) +2controls; 540 одинаковых training episodes для максимум30epochs, patience5. Full/gps_only имеют по7515параметров. Подробности: [INSANE_ARCHITECTURE.md](INSANE_ARCHITECTURE.md), [configs/insane](configs/insane).

Held-GNSS и causal CV готовы. ESKF math tests предусмотрены, но real-data ESKF **not_ready**: в supplied calibration нет плеча ordinary GNSS к IMU и достоверной heading initialization из разрешённых входов. Его отсутствие/расходимость не доказывает преимущества сети. GT — PX4 IMU reference из dual RTK/магнитометра; GT attitude не используется во входах. Оцениваются исходные GT timestamps, не размноженные интерполяцией наблюдения.

Локальные проверки: `python3 -B -m unittest discover -s tests/insane -v`. Повторная загрузка разрешённых публичных источников: `python3 -B scripts/insane/download.py outdoor_1 mars_1 mars_2 mars_3 mars_4 mars_5 mars_6 mars_7` (существующие hashes проверяются). Источники/лицензия: `data/insane/metadata/`, download manifest: `configs/insane/download_manifest.json`. Данные приватны; дополнительная публикация не разрешена.

**После одного pilot — STOP.** Ни test metrics, ни дополнительных seeds, ни смены датасета/метрики ради выигрыша full.
