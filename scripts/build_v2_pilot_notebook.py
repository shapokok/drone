"""Build only. Does not submit Kaggle jobs, train, or install packages."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FILES=['src/diagnostics/__init__.py','src/diagnostics/clean_outages.py','src/diagnostics/next_stage.py',
       'src/pilot_v2/__init__.py','src/pilot_v2/data.py','src/pilot_v2/model.py','src/pilot_v2/run.py',
       'tests/test_next_stage_v2.py','configs/v2_motion_pilot.json','configs/v2_motion_pilot_manifest.json']


def cell(kind,text,name):
    result={'cell_type':kind,'id':name,'metadata':{},'source':text.splitlines(keepends=True)}
    if kind=='code':
        ast.parse(text);result.update(outputs=[],execution_count=None)
    return result


def build():
    sources={p:(ROOT/p).read_text() for p in FILES}
    bootstrap='import hashlib\nSOURCES = '+repr(sources)+'\nHASHES = '+repr({p:hashlib.sha256(s.encode()).hexdigest() for p,s in sources.items()})+'''
for name,source in SOURCES.items():
    payload=source.encode();assert hashlib.sha256(payload).hexdigest()==HASHES[name]
    path=CODE_ROOT/name
    if path.exists() and path.read_bytes()!=payload:raise RuntimeError('Use a fresh session: '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():path.write_bytes(payload)
sys.path.insert(0,str(CODE_ROOT/'src'))
'''
    cells=[cell('markdown','''# FusionNav V2: causal relative-motion pilot — PREPARED, NOT TRAINED

Запускать только после отдельного указания пользователя. `RUN_TRAINING=False` по умолчанию.
Подключите dataset **mrisdal/zurich-urban-micro-aerial-vehicle, version 2**: нужны только
RawAccel.csv, RawGyro.csv, OnboardGPS.csv и GroundTruthAGL.csv; хеши будут проверены.
Входы R7/checkpoints для нового обучения не используются. Никакие пакеты не устанавливаются.
Нужны NumPy, pandas, PyTorch и pyproj в окружении Kaggle. GPU необязателен.

Один seed, три маленькие модели: одинаковые GRU full/gps_only (нулевая IMU-ветвь),
LSTM full; 8 фиксированных epochs. Контроль + 9 заранее заданных validation outages.
Baselines: held-GNSS и причинный constant-velocity. Test не оценивается.
Primary target/metric — относительное движение камеры от последнего разрешённого fix,
а не абсолютная поправка GNSS→GT. Absolute raw error сохраняется отдельно.

Единые UTM32N оси и общий origin; gyro только из прошлого; dt из timestamps.
Позиция не подгоняется по GT. Точный высотный datum GT и camera/antenna extrinsic остаются
ограничениями: motion-labels устраняют постоянное начальное смещение только в метрике/loss,
не передавая GT модели. Это исследовательский pilot, не физически сертифицированный INS.
''','purpose'),
cell('code',"""from pathlib import Path
import sys
RUN_TRAINING = False  # Change only after a separate explicit user command to run the pilot.
CODE_ROOT = Path('/kaggle/working/v2_motion_pilot_code')
OUT_DIR = Path('/kaggle/working/outputs/v2_motion_pilot')
RAW_DIR = None
""",'settings'),cell('code',bootstrap,'embedded-code'),
cell('code',"""import importlib.util
import subprocess
required=['numpy','pandas','torch','pyproj']
missing=[p for p in required if importlib.util.find_spec(p) is None]
if missing:raise RuntimeError(f'Required packages absent: {missing}; no installation was attempted')
subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(CODE_ROOT/'tests'),
                '-p','test_next_stage_v2.py','-v'],cwd=CODE_ROOT,check=True)
""",'dependency-and-unit-checks'),
cell('code',"""import json
config_path=CODE_ROOT/'configs/v2_motion_pilot.json'
config=json.loads(config_path.read_text())
if RAW_DIR is None:
    candidates=[]
    for p in Path('/kaggle/input').rglob('RawAccel.csv'):
        root=p.parent
        if all((root/n).is_file() for n in config['raw_sha256']):
            if all(hashlib.sha256((root/n).read_bytes()).hexdigest()==h for n,h in config['raw_sha256'].items()):
                candidates.append(root)
    if len(candidates)!=1:raise RuntimeError(f'Expected exactly one matching raw dataset mount, found {len(candidates)}; set RAW_DIR explicitly')
    RAW_DIR=candidates[0]
print('Pinned raw input:',RAW_DIR)
print('Prepared only:',not RUN_TRAINING)
print('Plan: 3 models, seed 0, 8 epochs, 10 fixed validation scenarios, no test')
""",'raw-input-preflight'),
cell('code',"""from pilot_v2.run import execute
if RUN_TRAINING:
    execute(config_path,RAW_DIR,OUT_DIR,run_training=True)
else:
    print('TRAINING NOT STARTED. Wait for the separate user command before enabling RUN_TRAINING.')
""",'explicit-training-gate'),
cell('markdown','''## После разрешённого запуска

Верните весь `outputs/v2_motion_pilot/`: config/manifest, normalization, training history,
checkpoints, summary.csv, status.json и предсказания всех методов/сценариев.
Сравните full и gps_only по paired motion RMSE на 30/60 с, absolute raw error отдельно,
и sensitivity того же full checkpoint к нулевому IMU. Сопоставьте с held/CV.
Улучшение должно повторяться на нескольких фиксированных эпизодах и быть практически
существенным; обязательный процент не задан. Переход к 5 seeds не автоматический.
R7 и прежние diagnostic outputs не перезаписываются.
''','handoff')]
    return {'cells':cells,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},
                                    'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}


if __name__=='__main__':
    notebook=build();path=ROOT/'notebooks/kaggle_v2_motion_pilot.ipynb'
    if path.exists():
        old=json.loads(path.read_text())
        if any(c.get('outputs') or c.get('execution_count') is not None for c in old['cells']):
            raise RuntimeError('Refusing to replace executed notebook')
    path.write_text(json.dumps(notebook,indent=1,ensure_ascii=False)+'\n');print(path)
