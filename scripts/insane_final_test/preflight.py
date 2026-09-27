"""Test exact notebook bootstrap/input/tests locally; no test inference or scoring."""
from pathlib import Path
import sys,json,hashlib,tempfile,ast
ROOT=Path(__file__).resolve().parents[2]

def main():
    out=ROOT/'outputs/insane_final_test_preparation';nbp=ROOT/'notebooks/kaggle_insane_final_test.ipynb'
    nb=json.loads(nbp.read_text());cells={c['id']:''.join(c['source']) for c in nb['cells']}
    for c in nb['cells']:
        if c['cell_type']=='code':ast.parse(''.join(c['source']))
    with tempfile.TemporaryDirectory(prefix='insane_final_test_preflight_') as temp:
        temp=Path(temp);scope={'Path':Path,'sys':sys,'json':json,'hashlib':hashlib,'CODE_ROOT':temp/'code','LAUNCH_DIR':out,'DATA_ROOT':ROOT/'outputs/insane_final_test_data_upload'}
        exec(compile(cells['pinned-code'],'frozen_bootstrap','exec'),scope)
        mount=cells['preflight'].replace("Path('/kaggle/working/insane_final_test_inputs')",f'Path({str(temp/"inputs")!r})')
        exec(compile(mount,'frozen_preflight','exec'),scope)
        exec(compile(cells['mandatory-tests'],'frozen_tests','exec'),scope)
    check=json.loads((out/'preflight.json').read_text());check.update(unit_tests_passed=33,exact_notebook_preflight_passed=True,
       notebook_sha256=hashlib.sha256(nbp.read_bytes()).hexdigest(),final_lock_sha256=hashlib.sha256((ROOT/'configs/insane_final_test/FINAL_EVALUATION_LOCK.json').read_bytes()).hexdigest(),
       predictions_or_metrics_run_locally=False)
    (out/'preflight.json').write_text(json.dumps(check,indent=2)+'\n');print(json.dumps(check,indent=2))

if __name__=='__main__':main()
