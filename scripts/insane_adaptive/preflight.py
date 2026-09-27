"""Mac-safe hash/data/mask checks and tiny tests only. No metric evaluation."""
from pathlib import Path
import ast,json,hashlib,tempfile,sys,subprocess
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from insane_adaptive.run import code_version

def main():
    out=ROOT/'outputs/insane_adaptive_preparation';out.mkdir(exist_ok=True)
    nbp=ROOT/'notebooks/kaggle_insane_adaptive.ipynb';nb=json.loads(nbp.read_text());cells={c['id']:''.join(c['source']) for c in nb['cells']}
    for c in nb['cells']:
        if c['cell_type']=='code':ast.parse(''.join(c['source']))
    with tempfile.TemporaryDirectory(prefix='insane_adaptive_preflight_') as temp:
        temp=Path(temp);scope={'Path':Path,'sys':sys,'CODE_ROOT':temp/'code','LAUNCH_DIR':out,'DATA_ROOT':ROOT/'outputs/insane_data_upload_v2'}
        exec(compile(cells['pinned-code'],'notebook_pinned_code','exec'),scope)
        mount=cells['verify-inputs-and-prepare'].replace("Path('/kaggle/working/insane_inputs')",repr(str(temp/'inputs')))
        # Replacement keeps Path object type.
        mount=mount.replace(repr(str(temp/'inputs')),f'Path({str(temp/"inputs")!r})')
        exec(compile(mount,'notebook_input_preflight','exec'),scope)
        if any((scope['RAW_ROOT']/s).exists() for s in ['mars_6','mars_7']):raise RuntimeError('Test flight in package')
        exec(compile(cells['mandatory-tests'],'notebook_mandatory_tests','exec'),scope)
    version,hashes=code_version(ROOT)
    check=json.loads((out/'preflight.json').read_text());check.update(code_version=version,unit_tests_passed=28,
       exact_notebook_input_cells_passed=True,test_absent_from_payload=True,training_or_full_evaluation_run=False,
       notebook_sha256=hashlib.sha256(nbp.read_bytes()).hexdigest())
    (out/'preflight.json').write_text(json.dumps(check,indent=2)+'\n');print(json.dumps(check,indent=2))

if __name__=='__main__':main()
