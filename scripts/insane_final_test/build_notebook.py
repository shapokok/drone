"""Build a self-contained pinned evaluation notebook; no submission."""
from pathlib import Path
import json,hashlib,ast
ROOT=Path(__file__).resolve().parents[2]

def cell(kind,text,identity):
    d={'cell_type':kind,'id':identity,'metadata':{},'source':text.splitlines(keepends=True)}
    if kind=='code':ast.parse(text);d.update(execution_count=None,outputs=[])
    return d

def main():
    lock=json.loads((ROOT/'configs/insane_final_test/FINAL_EVALUATION_LOCK.json').read_text())
    paths=list(lock['evaluation_source_sha256'])+['configs/insane_final_test/FINAL_EVALUATION_LOCK.json','configs/insane_final_test/test_manifest.json']
    sources={p:(ROOT/p).read_text() for p in paths};hashes={p:hashlib.sha256(s.encode()).hexdigest() for p,s in sources.items()}
    settings="""from pathlib import Path
import sys,json,hashlib
ALLOW_FINAL_TEST = True  # user explicitly authorized one frozen test-only job
DATA_ROOT = Path('/kaggle/input')
CODE_ROOT = Path('/kaggle/working/insane_final_test_code')
OUT_DIR = Path('/kaggle/working/outputs/insane_final_test')
LAUNCH_DIR = Path('/kaggle/working/insane_final_test_launch')
LAUNCH_DIR.mkdir(parents=True,exist_ok=True)
"""
    bootstrap='SOURCES = '+repr(sources)+'\nSOURCE_HASHES = '+repr(hashes)+'''
for relative,source in SOURCES.items():
    content=source.encode()
    if hashlib.sha256(content).hexdigest()!=SOURCE_HASHES[relative]:raise RuntimeError('Embedded source mismatch')
    dest=CODE_ROOT/relative;dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() and dest.read_bytes()!=content:raise RuntimeError('Source differs; STOP, no retry')
    if not dest.exists():dest.write_bytes(content)
sys.path.insert(0,str(CODE_ROOT/'src'))
'''
    transport=json.loads((ROOT/'outputs/insane_final_test_preparation/transport.json').read_text())
    mount="""import base64,io,zipfile,importlib.util
missing=[n for n in ['numpy','pandas','torch','scipy','matplotlib'] if importlib.util.find_spec(n) is None]
if missing:raise RuntimeError('Missing dependencies; STOP: '+str(missing))
payloads=list(DATA_ROOT.rglob('final_test_payload.b64'))
if len(payloads)!=1:raise RuntimeError('Expected exactly one locked final test input')
payload=base64.b64decode(payloads[0].read_bytes(),validate=True)
if hashlib.sha256(payload).hexdigest()!=PAYLOAD_HASH:raise RuntimeError('Transport hash mismatch')
EXTRACT_ROOT=Path('/kaggle/working/insane_final_test_inputs')
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    if sum(i.file_size for i in archive.infolist())>100_000_000:raise RuntimeError('Unexpected input size')
    for name in archive.namelist():
        if Path(name).is_absolute() or '..' in Path(name).parts:raise RuntimeError('Unsafe archive path')
    archive.extractall(EXTRACT_ROOT)
BUNDLE=EXTRACT_ROOT/'bundle'
from insane_final_test.run import preflight
checks=preflight(BUNDLE,CODE_ROOT,load_models=True)[-2]
# All shared support files must also match the original downloaded calibration manifest.
download=json.loads((BUNDLE/'frozen/download_manifest.json').read_text())
for relative,info in download['support_files'].items():
    if hashlib.sha256((BUNDLE/'raw'/relative).read_bytes()).hexdigest()!=info['sha256']:raise RuntimeError('Original calibration hash mismatch')
(LAUNCH_DIR/'preflight.json').write_text(json.dumps(checks,indent=2)+'\\n')
print(checks)
""".replace('PAYLOAD_HASH',repr(transport['payload_sha256']))
    tests="""import subprocess
for suite in ['insane','insane_adaptive','insane_final_test']:
    result=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(CODE_ROOT/'tests'/suite),'-v'],cwd=CODE_ROOT,capture_output=True,text=True)
    log=result.stdout+'\\n'+result.stderr;(LAUNCH_DIR/(suite+'_tests.log')).write_text(log);print(log)
    if result.returncode:raise RuntimeError('Mandatory tests failed; STOP, no evaluation/retry')
"""
    run="""from insane_final_test.run import execute
if ALLOW_FINAL_TEST:
    execute(BUNDLE,CODE_ROOT,OUT_DIR,allow_test=True)
else:
    print('Preflight/tests only; final test gate is false.')
"""
    purpose='''# INSANE: one final test-only evaluation

All six validation-selected checkpoints; seeds0/1/2; main adaptive_full, matched adaptive_gps_only.
Test mars_6/mars_7 only. Fixed scalers/config, τ=5s; no training, tuning, ensemble or retry.
Inputs → pinned manifest/checkpoint/hash checks → 33 mandatory tests → single evaluation → independent saved-array verification → reports.
Source and FINAL_EVALUATION_LOCK are embedded; no checkout of main. STOP after this run regardless of results.
'''
    nb={'cells':[cell('markdown',purpose,'purpose'),cell('code',settings,'settings'),cell('code',bootstrap,'pinned-code'),cell('code',mount,'preflight'),cell('code',tests,'mandatory-tests'),cell('code',run,'final-test-once')],
      'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
    dest=ROOT/'notebooks/kaggle_insane_final_test.ipynb'
    if dest.exists():raise RuntimeError('Pinned final notebook already exists')
    dest.write_text(json.dumps(nb,indent=1,ensure_ascii=False)+'\n');print(str(dest))

if __name__=='__main__':main()
