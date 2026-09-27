"""Embed an immutable adaptive experiment; never submit from this script."""
from pathlib import Path
import json,hashlib,ast,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from insane_adaptive.run import code_version

def cell(kind,source,identity):
    result={'cell_type':kind,'id':identity,'metadata':{},'source':source.splitlines(keepends=True)}
    if kind=='code':ast.parse(source);result.update(execution_count=None,outputs=[])
    return result

def main():
    version,hashes=code_version(ROOT);sources={p:(ROOT/p).read_text() for p in hashes}
    old=json.loads((ROOT/'notebooks/kaggle_insane_pilot.ipynb').read_text());by_id={c['id']:''.join(c['source']) for c in old['cells']}
    settings=by_id['settings'].replace('one private Kaggle pilot','one private Kaggle adaptive job with six trainings').replace('insane_code','insane_adaptive_code').replace('outputs/insane_pilot','outputs/insane_adaptive').replace('insane_launch','insane_adaptive_launch')
    bootstrap='SOURCES = '+repr(sources)+'\nSOURCE_HASHES = '+repr(hashes)+'\n'+by_id['pinned-code'].split('import hashlib\n',1)[1]
    bootstrap='import hashlib\n'+bootstrap
    mount=by_id['verify-inputs-and-prepare'].replace("['numpy','pandas','torch','scipy']","['numpy','pandas','torch','scipy','matplotlib']")
    mount=mount.replace('from insane.run import preflight,code_version','from insane_adaptive.run import preflight,code_version')
    mount=mount.replace("preflight(RAW_ROOT,CODE_ROOT/'configs/insane')","preflight(RAW_ROOT,CODE_ROOT)")
    # Payload must still be version2, exactly the same five-flight private bundle.
    tests="""import subprocess
for suite in ['insane','insane_adaptive']:
    result=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(CODE_ROOT/'tests'/suite),'-p','test_*.py','-v'],cwd=CODE_ROOT,capture_output=True,text=True)
    log=result.stdout+'\\n'+result.stderr
    (LAUNCH_DIR/(suite+'_tests.log')).write_text(log)
    print(log)
    if result.returncode:raise RuntimeError('Tests failed; STOP, no training/retry')
"""
    run="""from insane_adaptive.run import execute
if RUN_TRAINING:
    execute(RAW_ROOT,CODE_ROOT,OUT_DIR,run_training=True)
else:
    print('Only mandatory preflight/tests; no training or evaluation.')
"""
    cells=[cell('markdown',f'''# INSANE adaptive motion: one frozen six-training job

Train mars_1–3, validation mars_4–5. Test mars_6–7 absent and reserved.
Adaptive full/gps_only, seeds0/1/2. Original encoder + sigmoid scalar gate.
Code SHA256 `{version}`. All source/config/test bytes embedded; no main checkout.
Train-only Damped-CV selection precedes validation metrics; original fixed masks/GT points.
RUN_TRAINING=True is explicitly authorized. On any technical failure STOP, no retry.
''','purpose'),cell('code',settings,'settings'),cell('code',bootstrap,'pinned-code'),cell('code',mount,'verify-inputs-and-prepare'),cell('code',tests,'mandatory-tests'),cell('code',run,'train-evaluate-export'),cell('markdown','''Download `outputs/insane_adaptive/` and `insane_adaptive_launch/`.
Includes automatic INSANE_ADAPTIVE_REPORT.md, METHODS_DRAFT.md and every gate trace.
STOP after six scheduled trainings. No test, additional seeds, retries or next version.''','stop')]
    nb={'cells':cells,'metadata':old['metadata'],'nbformat':4,'nbformat_minor':5};p=ROOT/'notebooks/kaggle_insane_adaptive.ipynb'
    if (ROOT/'outputs/insane_adaptive_launch/submission_attempt.json').exists():raise RuntimeError('Already submitted; notebook frozen')
    p.write_text(json.dumps(nb,indent=1,ensure_ascii=False)+'\n');print(json.dumps({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'code_version':version},indent=2))

if __name__=='__main__':main()
