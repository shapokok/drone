"""Build pinned self-contained notebook; does not upload or run it."""
from pathlib import Path
import ast,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from insane.run import code_version

def cell(kind,text,name):
    d={'cell_type':kind,'id':name,'metadata':{},'source':text.splitlines(keepends=True)}
    if kind=='code':ast.parse(text);d.update(execution_count=None,outputs=[])
    return d

def main():
    version,hashes=code_version(ROOT);sources={p:(ROOT/p).read_text() for p in hashes}
    transport=json.loads((ROOT/'outputs/insane_preparation/transport.json').read_text())
    bootstrap='SOURCES = '+repr(sources)+'\nSOURCE_HASHES = '+repr(hashes)+'''
import hashlib
for relative,source in SOURCES.items():
    content=source.encode()
    if hashlib.sha256(content).hexdigest()!=SOURCE_HASHES[relative]:raise RuntimeError('Embedded source mismatch')
    dest=CODE_ROOT/relative;dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() and dest.read_bytes()!=content:raise RuntimeError('Use a fresh session; source differs')
    if not dest.exists():dest.write_bytes(content)
sys.path.insert(0,str(CODE_ROOT/'src'))
'''
    settings="""from pathlib import Path
import sys
# The user explicitly authorized one private Kaggle pilot after successful tests.
RUN_TRAINING = True
DATA_ROOT = Path('/kaggle/input')  # one setting; discover the unique pinned bundle in current/legacy Kaggle layouts
CODE_ROOT = Path('/kaggle/working/insane_code')
OUT_DIR = Path('/kaggle/working/outputs/insane_pilot')
LAUNCH_DIR = Path('/kaggle/working/insane_launch')
LAUNCH_DIR.mkdir(parents=True,exist_ok=True)
"""
    mount="""import zipfile,json,importlib.util,base64,io
required=['numpy','pandas','torch','scipy']
missing=[n for n in required if importlib.util.find_spec(n) is None]
if missing:raise RuntimeError('Missing dependencies (no install attempted): '+str(missing))
if not DATA_ROOT.exists():raise FileNotFoundError('Attach the private INSANE data package and set DATA_ROOT')
payloads=list(DATA_ROOT.rglob('insane_payload.b64'))
roots=[]
if payloads:
    if len(payloads)!=1:raise RuntimeError('Ambiguous pinned payload')
    payload=base64.b64decode(payloads[0].read_bytes(),validate=True)
    if hashlib.sha256(payload).hexdigest()!='41d462d2f4e0d2f4f41989175c45a9ce63064bd29e3ec42ab0fd499c27b63408':raise RuntimeError('Transport hash mismatch')
    extracted=Path('/kaggle/working/insane_inputs')
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        if sum(i.file_size for i in z.infolist())>100_000_000:raise RuntimeError('Unexpected bundle size')
        for n in z.namelist():
            if Path(n).is_absolute() or '..' in Path(n).parts:raise RuntimeError('Unsafe archive path')
        z.extractall(extracted)
    roots=[p.parent.parent for p in extracted.rglob('px4_imu.csv') if p.parent.name=='mars_1']
else:
    roots=[p.parent.parent for p in DATA_ROOT.rglob('px4_imu.csv') if p.parent.name=='mars_1']
if not roots:
    archives=list(DATA_ROOT.rglob('insane_data.zip'))
    if len(archives)!=1:raise RuntimeError('Expected one INSANE input archive')
    extracted=Path('/kaggle/working/insane_inputs')
    with zipfile.ZipFile(archives[0]) as z:
        if sum(i.file_size for i in z.infolist())>100_000_000:raise RuntimeError('Unexpected bundle size')
        for n in z.namelist():
            if Path(n).is_absolute() or '..' in Path(n).parts:raise RuntimeError('Unsafe archive path')
        z.extractall(extracted)
    roots=[p.parent.parent for p in extracted.rglob('px4_imu.csv') if p.parent.name=='mars_1']
if len(roots)!=1:raise RuntimeError('Ambiguous data mount')
RAW_ROOT=roots[0]
from insane.run import preflight,code_version
checks=preflight(RAW_ROOT,CODE_ROOT/'configs/insane')[-1]
(LAUNCH_DIR/'preflight.json').write_text(json.dumps(checks,indent=2)+'\\n')
print(checks)
"""
    mount=mount.replace('41d462d2f4e0d2f4f41989175c45a9ce63064bd29e3ec42ab0fd499c27b63408',transport['payload_sha256'])
    tests="""import subprocess
result=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(CODE_ROOT/'tests/insane'),'-p','test_*.py','-v'],cwd=CODE_ROOT,capture_output=True,text=True)
log=result.stdout+'\\n'+result.stderr
(LAUNCH_DIR/'unit_tests.log').write_text(log)
print(log)
if result.returncode:raise RuntimeError('Tests failed; training is blocked')
"""
    run="""from insane.run import execute
if RUN_TRAINING:
    execute(RAW_ROOT,CODE_ROOT/'configs/insane',OUT_DIR,CODE_ROOT,run_training=True,require_cuda=True)
else:
    print('Preflight/tests only. Training gate is false.')
"""
    cells=[cell('markdown',f'''# FusionNav–INSANE: one frozen pilot

Private research run, seed0, full + matched gps_only, <=30epochs, patience5.
Train mars_1/2/3; validation mars_4/5; test mars_6/7 is reserved and absent from the input bundle.
Code version `{version}` is embedded; no checkout of a moving branch, no Zurich checkpoint.
Native ground_truth_8hz timestamps, clean10/30/60s outages, held/CV baselines.
ESKF is synthetically tested but real-data not_ready; it cannot substantiate a neural advantage.
The final user instruction explicitly authorized this one run; RUN_TRAINING is enabled.
No extra seeds, test evaluation, automatic retries, or changed metrics after this pilot.
''','purpose'),cell('code',settings,'settings'),cell('code',bootstrap,'pinned-code'),
       cell('code',mount,'verify-inputs-and-prepare'),cell('code',tests,'mandatory-tests'),cell('code',run,'train-evaluate-export'),
       cell('markdown','''Results and automatic report are in `/kaggle/working/outputs/insane_pilot/`.
Download all outputs. The selected checkpoints use the predefined validation early-stopping criterion.
`status.json` records actual epochs; `summary.csv` retains unsupported ESKF rows explicitly.
STOP after this pilot. Results do not authorize further seeds or changes to the protocol.''','stop')]
    nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
    path=ROOT/'notebooks/kaggle_insane_pilot.ipynb'
    if path.exists() and any(c.get('outputs') for c in json.loads(path.read_text())['cells']):raise RuntimeError('Never overwrite executed notebook')
    path.write_text(json.dumps(nb,indent=1,ensure_ascii=False)+'\n')
    print(json.dumps({'notebook':str(path),'code_version':version,'bytes':path.stat().st_size},indent=2))

if __name__=='__main__':main()
