"""Pin the minimal preflight patch without changing any original lock/source/input."""
from pathlib import Path
import hashlib,json,ast
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    original=ROOT/'configs/insane_final_test/FINAL_EVALUATION_LOCK.json';lock=json.loads(original.read_text())
    for rel,h in lock['evaluation_source_sha256'].items():assert sha(ROOT/rel)==h,rel
    paths=[str(p.relative_to(ROOT)) for base in ['src/insane_final_test_patch','tests/insane_final_test_patch'] for p in sorted((ROOT/base).glob('*.py'))]
    patch={'authorization':'One user-authorized retry of final test after preflight repair; no protocol changes',
        'prior_failed_run_id':'353128919','original_lock_sha256':sha(original),'original_test_manifest_sha256':lock['test_manifest_sha256'],
        'patch_source_sha256':{p:sha(ROOT/p) for p in paths},
        'scope':'Runtime replacement of run.preflight only; all original locked scientific source files byte-identical',
        'tolerance':{'field':'inventory/{mars_6,mars_7}/projection_error_m','atol_m':1e-8,'rtol':0.0},
        'strict_fields':'every other manifest field, all source/input/checkpoint/config/scaler hashes',
        'unchanged_checkpoint_sha256':lock['checkpoint_sha256']}
    pp=ROOT/'configs/insane_final_test_patch/PREFLIGHT_PATCH_MANIFEST.json'
    if pp.exists():raise RuntimeError('Patch already frozen')
    pp.write_text(json.dumps(patch,indent=2)+'\n')
    old=json.loads((ROOT/'notebooks/kaggle_insane_final_test.ipynb').read_text());by={c['id']:''.join(c['source']) for c in old['cells']}
    allpaths=list(lock['evaluation_source_sha256'])+paths+['configs/insane_final_test/FINAL_EVALUATION_LOCK.json','configs/insane_final_test/test_manifest.json','configs/insane_final_test_patch/PREFLIGHT_PATCH_MANIFEST.json']
    sources={p:(ROOT/p).read_text() for p in allpaths};hashes={p:sha(ROOT/p) for p in allpaths}
    bootstrap='SOURCES = '+repr(sources)+'\nSOURCE_HASHES = '+repr(hashes)+'\n'+by['pinned-code'].split('for relative,source in SOURCES.items():',1)[1]
    bootstrap=bootstrap.replace('\n\n    content=', '\n    content=',1)
    # Preserve original loop body; only rebuild its literal prefix.
    bootstrap='SOURCES = '+repr(sources)+'\nSOURCE_HASHES = '+repr(hashes)+'\nfor relative,source in SOURCES.items():'+by['pinned-code'].split('for relative,source in SOURCES.items():',1)[1]
    settings=by['settings'].replace('outputs/insane_final_test','outputs/insane_final_test_retry').replace('insane_final_test_launch','insane_final_test_retry_launch')
    mount=by['preflight'].replace('from insane_final_test.run import preflight','from insane_final_test_patch.preflight import preflight')
    mount=mount.replace('preflight(BUNDLE,CODE_ROOT,load_models=True)','preflight(BUNDLE,CODE_ROOT,load_models=True,diagnostics_root=LAUNCH_DIR/\'preflight_evidence\')')
    tests=by['mandatory-tests'].replace("['insane','insane_adaptive','insane_final_test']","['insane','insane_adaptive','insane_final_test','insane_final_test_patch']")
    run="""import functools,shutil
import insane_final_test.run as runner
from insane_final_test_patch.preflight import preflight as patched_preflight
# Only the diagnostic preflight is replaced. Model/data/scoring/runner bytes stay locked.
runner.preflight=functools.partial(patched_preflight,diagnostics_root=LAUNCH_DIR/'preflight_evidence')
if ALLOW_FINAL_TEST:
    try:
        runner.execute(BUNDLE,CODE_ROOT,OUT_DIR,allow_test=True)
    finally:
        if OUT_DIR.exists():
            pp=CODE_ROOT/'configs/insane_final_test_patch/PREFLIGHT_PATCH_MANIFEST.json'
            shutil.copyfile(pp,OUT_DIR/'PREFLIGHT_PATCH_MANIFEST.json')
            patch=json.loads(pp.read_text())
            for rel in patch['patch_source_sha256']:
                dst=OUT_DIR/'preflight_patch_snapshot'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(CODE_ROOT/rel,dst)
            shutil.copytree(LAUNCH_DIR/'preflight_evidence',OUT_DIR/'preflight_evidence')
else:
    print('Preflight/tests only, no test inference.')
"""
    new={'purpose':'''# One authorized final-test retry: manifest preflight repair only

Original FINAL_EVALUATION_LOCK, scenarios, native timestamps/masks, checkpoints, scalers, tau5s and seeds0/1/2 are unchanged.
Only inventory projection-error roundoff gets atol1e-8m, rtol0; all other fields/hashes strict.
Actual manifest and field-level diff are saved before assertions. Fatal source/scenario/mask/timestamp mismatch blocks inference.
41 mandatory local/remote tests. One evaluation only after passing Kaggle preflight/tests; no training or further retry.
''','settings':settings,'pinned-code':bootstrap,'preflight':mount,'mandatory-tests':tests,'final-test-once':run}
    for c in old['cells']:
        c['source']=new[c['id']].splitlines(keepends=True)
        if c['cell_type']=='code':ast.parse(new[c['id']]);c['execution_count']=None;c['outputs']=[]
    dest=ROOT/'notebooks/kaggle_insane_final_test_retry.ipynb'
    if dest.exists():raise RuntimeError('Retry notebook already exists')
    dest.write_text(json.dumps(old,indent=1,ensure_ascii=False)+'\n')
    print(json.dumps({'original_lock_sha256':sha(original),'patch_manifest_sha256':sha(pp),'notebook_sha256':sha(dest)},indent=2))
if __name__=='__main__':main()
