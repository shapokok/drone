"""Strict scientific manifest comparison with one diagnostic numerical exception."""
from pathlib import Path
import json,math

# Projection residual is diagnostic float64 ECEF/ENU arithmetic, not data or a metric.
# Epochs, origins, support bounds, scenario timestamps and all counts/hashes stay exact.
PROJECTION_ATOL_M=1e-8
PROJECTION_RTOL=0.0

def compare(expected,actual):
    # JSON normalization removes Python tuple/list representation differences only.
    expected=json.loads(json.dumps(expected));actual=json.loads(json.dumps(actual));diff=[]
    def visit(e,a,path):
        pointer='/'+ '/'.join(str(p) for p in path)
        tolerant=(len(path)==3 and path[0]=='inventory' and path[1] in ['mars_6','mars_7'] and path[2]=='projection_error_m')
        if tolerant:
            numeric=type(e) in (int,float) and type(a) in (int,float)
            finite=numeric and math.isfinite(e) and math.isfinite(a)
            delta=abs(a-e) if finite else None
            accepted=bool(finite and delta<=PROJECTION_ATOL_M)
            if not accepted or e!=a:
                diff.append({'path':pointer,'expected':e,'actual':a,'comparison':'diagnostic_inventory_numeric',
                    'absolute_difference':delta,'atol':PROJECTION_ATOL_M,'rtol':PROJECTION_RTOL,'accepted':accepted})
            return
        if type(e)!=type(a):
            diff.append({'path':pointer,'expected':e,'actual':a,'comparison':'strict_type','accepted':False});return
        if isinstance(e,dict):
            for k in sorted(set(e)|set(a)):
                if k not in e or k not in a:diff.append({'path':pointer+'/'+k,'expected':e.get(k),'actual':a.get(k),'comparison':'strict_key_presence','accepted':False})
                else:visit(e[k],a[k],path+[k])
        elif isinstance(e,list):
            if len(e)!=len(a):diff.append({'path':pointer,'expected_count':len(e),'actual_count':len(a),'comparison':'strict_length','accepted':False})
            for i,(ee,aa) in enumerate(zip(e,a)):visit(ee,aa,path+[i])
        elif (isinstance(e,float) and (not math.isfinite(e) or not math.isfinite(a))) or e!=a:
            diff.append({'path':pointer,'expected':e,'actual':a,'comparison':'strict_value','accepted':False})
    visit(expected,actual,[])
    return {'status':'pass' if all(r['accepted'] for r in diff) else 'fail','fields_differing':diff,
        'strict_mismatch_count':sum(not r['accepted'] for r in diff),'tolerated_inventory_difference_count':sum(r['accepted'] for r in diff),
        'policy':{'scientific_fields':'strict: all scenarios/counts/IDs/masks/native timestamp hashes/epochs/origins/segments/source hashes',
                  'only_numeric_exception':'/inventory/{mars_6,mars_7}/projection_error_m','atol_m':PROJECTION_ATOL_M,'rtol':PROJECTION_RTOL}}

def safe_json(value):
    if isinstance(value,float) and not math.isfinite(value):return {'nonfinite_value':repr(value)}
    if isinstance(value,dict):return {k:safe_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [safe_json(v) for v in value]
    return value

def persist(folder,actual,diff):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    for name,value in [('actual_regenerated_manifest.json',actual),('manifest_field_diff.json',diff)]:
        (folder/name).write_text(json.dumps(safe_json(value),indent=2,allow_nan=False)+'\n')

def compare_and_record(expected,actual,folder,hash_checks=None):
    diff=compare(expected,actual)
    diff['hash_checks']=hash_checks or []
    if any(not x['matched'] for x in diff['hash_checks']):diff['status']='fail'
    persist(folder,actual,diff)
    if diff['status']!='pass':raise ValueError('Strict manifest/source mismatch; saved structured diff; STOP before inference')
    return diff
