"""Package the isolated diagnostic for manual Kaggle upload; never launch jobs.

Run from anywhere: python3 -B scripts/build_diagnostic_notebook.py
Use --check to verify that the committed/prepared notebook embeds current files.
Only the new diagnostic notebook is generated; kaggle_train.ipynb is untouched.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks/kaggle_diagnostic_clean_outages.ipynb"
FILES = [
    "src/diagnostics/__init__.py", "src/diagnostics/clean_outages.py",
    "src/diagnostics/run_clean_outages.py", "src/models/fusion_transformer.py",
    "src/models/fusion_lstm.py", "tests/test_clean_outages.py",
    "configs/diagnostic_clean_outages.json", "configs/diagnostic_clean_outages_manifest.json",
]


def cell(kind, source, cell_id):
    result = {"cell_type": kind, "id": cell_id, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        ast.parse(source)
        result.update(execution_count=None, outputs=[])
    return result


def build():
    files = {relative: (ROOT / relative).read_text() for relative in FILES}
    hashes = {relative: hashlib.sha256(text.encode()).hexdigest() for relative, text in files.items()}
    cells = [cell("markdown", """# R7: диагностика чистых отключений GNSS на validation

**Этот notebook подготовлен для ручного Save & Run All. Он не выполнялся при подготовке.**
Обучения, настройки параметров и test-оценки здесь нет. Используются сохранённые
R7 seed=0: full, отдельно обученная gps_only, FusionLSTM. Сравнение с Held-GNSS,
причинной экстраполяцией постоянной скорости и отдельным zero-IMU probing full checkpoint.
Transformer остаётся **некаузальной оконной моделью**, без добавления causal masks.

Добавьте сохранённый output notebook
[shapok/drone-nav-full-run, версия 7, output ID 352644733](https://www.kaggle.com/code/shapok/drone-nav-full-run?scriptVersionId=352644733)
как input этого нового notebook. Нужны его `drone/data/processed/` и `outputs/checkpoints/`.
Версия проверяется по SHA-256 всех десяти файлов; другой export будет отклонён.
Исходный сырой датасет добавлять не требуется. Код и конфигурация уже встроены:
GitHub, Интернет, API-ключи и установка зависимостей не нужны.

GPU можно включить; CPU также поддерживается. Нужны имеющиеся в Kaggle NumPy и PyTorch
с `torch.load(weights_only=True)`. Сохранённые R7 данные только читаются из `/kaggle/input`.
Запустите **Save Version → Save & Run All** в новом notebook. Для повторного полного запуска
используйте свежую сессию: существующий каталог результатов намеренно не перезаписывается.
""", "purpose"), cell("code", """from pathlib import Path
import sys

CODE_ROOT = Path('/kaggle/working/diagnostic_clean_code')
INPUT_ROOT = Path('/kaggle/input')
OUTPUT_DIR = Path('/kaggle/working/outputs/diagnostic_clean_outages')
# Обычно оставьте None: поиск только среди добавленных inputs, затем строгая сверка хешей.
R7_ROOT = None
""", "settings")]
    bootstrap = ("import hashlib\n\nPACKAGED_FILES = " + repr(files)
                 + "\nPACKAGED_SHA256 = " + repr(hashes) + "\n\n" + """for relative, source in PACKAGED_FILES.items():
    payload = source.encode('utf-8')
    assert hashlib.sha256(payload).hexdigest() == PACKAGED_SHA256[relative]
    destination = CODE_ROOT / relative
    if destination.exists():
        if destination.read_bytes() != payload:
            raise RuntimeError(f'Existing diagnostic code differs: {destination}; use a fresh session')
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
sys.path.insert(0, str(CODE_ROOT / 'src'))
CONFIG_PATH = CODE_ROOT / 'configs/diagnostic_clean_outages.json'
print(f'Packaged {len(PACKAGED_FILES)} verified source/config files into {CODE_ROOT}')
""")
    cells.extend([
        cell("code", bootstrap, "embedded-code"),
        cell("markdown", """## Проверки до inference

Unit/smoke tests используют только маленькие синтетические данные и случайные уменьшенные
модели. Их численные значения не являются экспериментальными результатами R7.
Следующая ячейка сверяет настоящий input, общий validation-диапазон и заранее
зафиксированные сценарии; при несовпадении запуск останавливается.
""", "checks-note"),
        cell("code", """import subprocess
subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover',
                '-s', str(CODE_ROOT / 'tests'), '-p', 'test_clean_outages.py', '-v'],
               cwd=CODE_ROOT, check=True)
""", "synthetic-tests"),
        cell("code", """import json
import numpy as np
from diagnostics.run_clean_outages import discover_r7, verify_r7, prepare_validation
from diagnostics.clean_outages import make_scenarios

config = json.loads(CONFIG_PATH.read_text())
if R7_ROOT is None:
    R7_ROOT, verified = discover_r7(INPUT_ROOT, config)
else:
    R7_ROOT = Path(R7_ROOT)
    verified = verify_r7(R7_ROOT, config)
prepared = prepare_validation(R7_ROOT, config)
scenarios = make_scenarios(prepared['t'], config['protocol'])
locked = json.loads((CONFIG_PATH.parent / 'diagnostic_clean_outages_manifest.json').read_text())
assert scenarios == locked['scenarios'], 'Locked scenarios changed'
assert hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest() == locked['config_sha256']
assert hashlib.sha256(prepared['t'].astype('<f8').tobytes()).hexdigest() == locked['common_timestamps_float64_le_sha256']
print('Verified R7 input:', R7_ROOT)
print(json.dumps(prepared['support'], indent=2))
for scenario in scenarios:
    print(scenario['scenario_id'], 'onset=', scenario.get('onset_s'),
          'cutoff=', scenario.get('requested_cutoff_s'),
          'unavailable points=', scenario['n_unavailable'])
assert not OUTPUT_DIR.exists(), f'Results directory exists: {OUTPUT_DIR}; use a fresh session'
""", "r7-preflight"),
        cell("markdown", """## Оценка сохранённых моделей

1 контроль + 9 отдельных эпизодов (3 начала × 10/30/60 секунд), 6 строк на сценарий.
Основная ошибка — raw RMSE на недоступных отсчётах, **без выравнивания**. Контроль использует
весь общий validation-отрезок. Первые две существующие оси считаются горизонтальными,
третья — вертикальной; согласование координат R7 ещё требует отдельной проверки.
Протокол и baseline скорости зафиксированы до просмотра ошибок. Результат без преимущества
full также допустим. Один seed и эпизоды одного полёта не дают межseed/межполётной статистики.
""", "evaluation-note"),
        cell("code", """from diagnostics.run_clean_outages import run
result_dir = run(CONFIG_PATH, input_root=INPUT_ROOT, r7_root=R7_ROOT, out_dir=OUTPUT_DIR)
print('Diagnostic artifacts:', result_dir)
""", "manual-evaluation"),
        cell("code", r"""import csv
from IPython.display import Markdown, display
status = json.loads((OUTPUT_DIR / 'status.json').read_text())
assert status['status'] == 'complete' and status['evaluation_rows'] == 60
with (OUTPUT_DIR / 'paired_differences.csv').open() as stream:
    pairs = list(csv.DictReader(stream))
header = '| Scenario | full − gps_only, m | full − held, m | full − CV, m | zero IMU − full, m |'
table = [header, '|---|---:|---:|---:|---:|']
for row in pairs:
    values = [row[key] for key in ('full_minus_fusion_gps_only_rmse_3d_m',
              'full_minus_held_gnss_rmse_3d_m', 'full_minus_constant_velocity_gnss_rmse_3d_m',
              'sensitivity_zero_imu_minus_full_rmse_3d_m')]
    table.append('| ' + row['scenario_id'] + ' | ' + ' | '.join(f'{float(v):.6f}' for v in values) + ' |')
display(Markdown('\n'.join(table)))
print(json.dumps(status, indent=2))
print('For analysis, download outputs/diagnostic_clean_outages/ in full, including prediction.npz bundles.')
""", "saved-results"),
        cell("markdown", """## Что передать для анализа

Скачайте весь `outputs/diagnostic_clean_outages/` и output/log этой версии notebook.
В каталоге будут `summary.csv` (60 уникальных evaluation-строк),
`paired_differences.csv` (10 сценариев), manifest/config/provenance/status,
snapshot исполненного кода и файлы каждого сценария с масками и происхождением GNSS.
Каждый из 60 `prediction.npz` содержит pred, GT, timestamps, scenario ID,
метаданные checkpoint, ошибки по всей временной линии и `prediction − held-GNSS`.
NPZ читается с `allow_pickle=False`. Положительные/отрицательные разности в таблице —
разности ошибок, а не доверительные интервалы. Zero-IMU — чувствительность фиксированного
checkpoint; gps_only — другой, ранее обученный checkpoint.

Исходный `results.csv`, checkpoints и notebook R7 не изменяются. Основной вопрос анализа:
есть ли преимущество full над отдельно обученной gps_only, удержанием координат
и причинной экстраполяцией при действительно закрытых измерениях?
""", "handoff")])
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                         "language_info": {"name": "python", "version": "3.12"}},
            "nbformat": 4, "nbformat_minor": 5}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = build()
    if args.check:
        if not TARGET.exists() or json.loads(TARGET.read_text()) != expected:
            raise SystemExit("Diagnostic notebook is stale; rebuild it")
        print("Notebook matches source/config; all code cells parse; outputs are empty")
        return
    if TARGET.exists():
        previous = json.loads(TARGET.read_text())
        if any(c.get("outputs") or c.get("execution_count") is not None for c in previous["cells"]):
            raise SystemExit("Refusing to overwrite an executed diagnostic notebook")
    TARGET.write_text(json.dumps(expected, indent=1, ensure_ascii=False) + "\n")
    print(TARGET)


if __name__ == "__main__":
    main()
