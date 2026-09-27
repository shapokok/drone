"""Prepare the specifically authorized private Kaggle inputs; excludes test flights."""
from pathlib import Path
import hashlib,json,zipfile,shutil,base64,io
ROOT=Path(__file__).resolve().parents[2]

def main():
    root=ROOT/'data/insane';dest=ROOT/'outputs/insane_data_upload';dest.mkdir(exist_ok=True)
    if (dest/'upload_receipt.json').exists():raise RuntimeError('Private package already uploaded; preserve that version')
    m=json.loads((ROOT/'configs/insane/download_manifest.json').read_text());split=json.loads((ROOT/'configs/insane/splits.json').read_text())
    names=set(m['support_files'])
    for seq in split['train']+split['validation']:
        for file in m['sequences'][seq]['files']:names.add(file['path'])
    for name in names:
        assert not any(seq in Path(name).parts for seq in split['test_reserved']+split['loader_smoke_only'])
    files={n:{'bytes':(root/n).stat().st_size,'sha256':hashlib.sha256((root/n).read_bytes()).hexdigest()} for n in sorted(names)}
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
        for name in files:z.write(root/name,'insane_data/'+name)
    payload=buffer.getvalue()
    (dest/'insane_payload.b64').write_bytes(base64.b64encode(payload))
    (ROOT/'outputs/insane_preparation/transport.json').write_text(json.dumps({'payload_sha256':hashlib.sha256(payload).hexdigest(),'encoding':'base64 of ZIP; extract once'},indent=2)+'\n')
    shutil.copyfile(root/'metadata/LICENSE.txt',dest/'LICENSE.txt')
    (dest/'source_manifest.json').write_text(json.dumps({'private':True,'flights':split['train']+split['validation'],
       'excluded_test':split['test_reserved'],'files':files},indent=2)+'\n')
    (dest/'README.md').write_text('''# INSANE minimal private research subset

Source: https://www.aau.at/en/smart-systems-technologies/control-of-networked-systems/datasets/insane-dataset/
Authors tools pinned at 9a1c8c0fdd195f2d869fff292f2ce5b273c5a03d.
Brommer et al., INSANE: Cross-Domain UAV Data Sets with Increased Number of Sensors for developing Advanced and Novel Estimators, arXiv:2210.09114.
License is **BSD-2 with additional conditions including no commercial use**: retain the supplied full LICENSE.txt and all attribution. The Kaggle license category Other does not replace those terms.
Only px4_imu.csv, ordinary px4_gps.csv, ground_truth_8hz.csv, stream metadata and preprocessed calibration/reference code are included. No cameras, RTK input streams, test flights or Zurich data/checkpoints. Private research only; not a public redistribution.
''')
    metadata={'title':'INSANE Navigation Pilot Data V1','id':'shapok/insane-navigation-pilot-data-v1','licenses':[{'name':'other'}],
       'description':(dest/'README.md').read_text()}
    (dest/'dataset-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({'private_encoded_payload_bytes':(dest/'insane_payload.b64').stat().st_size,'uncompressed_bytes':sum(f['bytes'] for f in files.values()),'files':len(files),'test_uploaded':False},indent=2))

if __name__=='__main__':main()
