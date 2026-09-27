"""Package the completed, verified experiment without raw inputs or credentials."""
from pathlib import Path
import json,hashlib,zipfile
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/insane_adaptive'

def main():
    assert json.loads((OUT/'independent_verification.json').read_text())['status']=='pass'
    files=set()
    for name in ['INSANE_ADAPTIVE_REPORT.md','METHODS_DRAFT.md','notebooks/kaggle_insane_adaptive.ipynb']:
        files.add(ROOT/name)
    for base in ['src/insane_adaptive','tests/insane_adaptive','scripts/insane_adaptive','configs/insane_adaptive','outputs/insane_adaptive','outputs/insane_adaptive_preparation','outputs/insane_adaptive_launch']:
        for p in (ROOT/base).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.pyc','.zip'] and p.name!='artifact_manifest.json':files.add(p)
    for name in ['data/insane/LICENSE.txt','data/insane/metadata/LICENSE.txt']:
        p=ROOT/name
        if p.exists():files.add(p)
    manifest=[]
    for p in sorted(files):
        assert p.name not in ['kaggle.json','.env'] and p.suffix!='.b64'
        b=p.read_bytes();manifest.append({'path':str(p.relative_to(ROOT)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
    m=OUT/'artifact_manifest.json';m.write_text(json.dumps({'experiment':'one six-training adaptive job, no test','files':manifest},indent=2)+'\n');files.add(m)
    dest=ROOT/'INSANE_ADAPTIVE_HANDOFF.zip'
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):z.write(p,p.relative_to(ROOT))
    with zipfile.ZipFile(dest) as z:assert z.testzip() is None
    print(json.dumps({'path':str(dest),'files':len(files),'bytes':dest.stat().st_size,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()},indent=2))

if __name__=='__main__':main()
