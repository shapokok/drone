"""Only replaces brittle preflight. Original scientific source/lock are untouched."""
from pathlib import Path
import json,traceback
import torch
from insane.data import sha
from insane_adaptive.model import AdaptiveFusionNav
from insane_final_test.protocol import prepare_test
from insane_final_test.run import state_hash
from .comparison import compare_and_record,persist

def preflight(bundle,root,load_models=True,diagnostics_root=None):
    bundle=Path(bundle);root=Path(root)
    diagnostics_root=Path(diagnostics_root) if diagnostics_root else root.parent/'insane_final_test_retry_launch/preflight_evidence'
    diagnostics_root.mkdir(parents=True,exist_ok=True)
    number=1
    while (diagnostics_root/f'check_{number:02d}').exists():number+=1
    folder=diagnostics_root/f'check_{number:02d}';folder.mkdir()
    actual={'status':'not_regenerated_yet'};hash_checks=[]
    # Initial evidence exists even if a read/parse/source check fails before regeneration.
    persist(folder,actual,{'status':'pending','fields_differing':[],'hash_checks':hash_checks})
    try:
        lock=json.loads((root/'configs/insane_final_test/FINAL_EVALUATION_LOCK.json').read_text())
        patch=json.loads((root/'configs/insane_final_test_patch/PREFLIGHT_PATCH_MANIFEST.json').read_text())
        checks=[('FINAL_EVALUATION_LOCK',root/'configs/insane_final_test/FINAL_EVALUATION_LOCK.json',patch['original_lock_sha256'])]
        checks += [('source:'+p,root/p,h) for p,h in lock['evaluation_source_sha256'].items()]
        checks += [('patch_source:'+p,root/p,h) for p,h in patch['patch_source_sha256'].items()]
        checks += [('frozen:'+p,bundle/p,h) for p,h in lock['frozen_files_sha256'].items()]
        checks += [('raw:'+p,bundle/'raw'/p,h) for p,h in lock['raw_files_sha256'].items()]
        checks += [('test_manifest',root/'configs/insane_final_test/test_manifest.json',lock['test_manifest_sha256'])]
        for name,path,expected_hash in checks:
            value=sha(path) if path.is_file() else None
            hash_checks.append({'path':name,'expected_sha256':expected_hash,'actual_sha256':value,'matched':value==expected_hash})
        if any(not x['matched'] for x in hash_checks):
            persist(folder,{'status':'not_regenerated','reason':'locked source/input hash mismatch'},
               {'status':'fail','fields_differing':[],'hash_checks':hash_checks,'reason':'STOP before using mismatched source/input'})
            raise ValueError('Source/input SHA256 mismatch; STOP before regeneration/inference')
        cfg=json.loads((bundle/'frozen/config.json').read_text());scaler=json.loads((bundle/'frozen/scalers.json').read_text())
        manifest=json.loads((bundle/'frozen/download_manifest.json').read_text())
        expected=json.loads((root/'configs/insane_final_test/test_manifest.json').read_text())
        actual,cache=prepare_test(bundle/'raw',cfg,scaler,manifest)
        diff=compare_and_record(expected,actual,folder,hash_checks)
        # Every assertion below is preceded by the persisted actual manifest + diff.
        assert scaler['fit_flights']==['mars_1','mars_2','mars_3'] and lock['tau_s']==5
        assert lock['seeds']==[0,1,2] and lock['test_flights']==['mars_6','mars_7']
        assert json.loads((bundle/'frozen/damping_lock.json').read_text())['tau_s']==5
        for name in ['mars_1','mars_2','mars_3','mars_4','mars_5','outdoor_1']:
            if (bundle/'raw'/name).exists():raise ValueError('Non-test raw flight in final evaluation input')
        models={};states={}
        if load_models:
            for method in lock['learned_methods']:
                for seed in lock['seeds']:
                    name=f'{method}_seed{seed}_best.pt';p=bundle/'frozen/checkpoints'/name
                    assert sha(p)==lock['checkpoint_sha256'][name]
                    m=AdaptiveFusionNav(method,cfg['hidden']);state=torch.load(p,map_location='cpu',weights_only=True)
                    m.load_state_dict(state,strict=True);m.eval();assert sum(v.numel() for v in m.parameters())==7548
                    models[(method,seed)]=m;states[name]=state_hash(m)
        checks={'status':'pass','checkpoint_compatibility_checked':load_models,'checkpoints':len(models),'all_hashes_pass':True,
          'test_manifest_exact_match':len(diff['fields_differing'])==0,'test_manifest_scientific_fields_exact_match':True,
          'tolerated_inventory_differences':diff['tolerated_inventory_difference_count'],
          'manifest_diagnostics_directory':str(folder),'train_scaler_reused_without_refit':True,'tau_s':5,'test_flights':['mars_6','mars_7'],
          'scenario_count':len(actual['scenarios']),'unsupported':actual['unsupported'],'model_predictions_computed':False,'baseline_metrics_computed':False,
          'projection_max_error_m':max(i['projection_error_m'] for i in actual['inventory'].values())}
        return cfg,scaler,actual,cache,models,states,checks,lock
    except Exception:
        (folder/'failure_traceback.txt').write_text(traceback.format_exc())
        # Preserve field-level diagnostics already written; make regeneration failures explicit.
        current=json.loads((folder/'manifest_field_diff.json').read_text())
        if current.get('status')=='pending':
            persist(folder,actual,{'status':'error_before_comparison','fields_differing':[],'hash_checks':hash_checks,'traceback_file':'failure_traceback.txt'})
        raise
