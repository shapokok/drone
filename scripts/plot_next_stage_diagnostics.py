"""Plot saved diagnostics only; no inference or training."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/next_stage_diagnostics/results'
OLD=ROOT/'outputs/diagnostic_clean_outages'

fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
z=np.load(OUT/'val_control_series.npz',allow_pickle=False)
t=z['timestamps']
for j,name in enumerate(['X / East','Y / North','Z / Up']):
    for method,label in [('fusion_full','full'),('fusion_gps_only','gps_only'),('fusion_full_zero_imu','full, zero IMU')]:
        axes[j].plot(t,z[method+'_correction'][:,j],label=label,linewidth=1)
    for boundary in t[192::192]:axes[j].axvline(boundary,color='grey',alpha=.25,linewidth=.7)
    axes[j].set_ylabel(name+' correction [m]');axes[j].grid(alpha=.2)
axes[0].legend();axes[0].set_title('Saved R7 control: correction is nearly constant within each 192-sample window')
axes[-1].set_xlabel('Processed timestamp [s]');fig.tight_layout()
fig.savefig(OUT/'correction_window_structure.png',dpi=160);plt.close(fig)

fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
z=np.load(OUT/'frame_float64/train_validation_frame_series.npz',allow_pickle=False)
t=z['timestamps'];raw=z['gt']-z['gps_r7'];physical=z['gt']-z['gps_common_utm']
for j,name in enumerate(['X / East','Y / North','Z / Up']):
    axes[j].plot(t,raw[:,j],label='GT minus R7 GNSS',linewidth=.7)
    axes[j].plot(t,physical[:,j],label='GT minus common UTM GNSS (no fit)',linewidth=.7)
    axes[j].axvline(t[18935],color='black',linestyle='--',label='train / validation boundary')
    axes[j].set_ylabel(name+' difference [m]');axes[j].grid(alpha=.2)
axes[0].legend();axes[0].set_title('Coordinate mismatch is systematic but not the entire time-varying GNSS error')
axes[-1].set_xlabel('Processed timestamp [s]');fig.tight_layout()
fig.savefig(OUT/'gnss_gt_frame_bias.png',dpi=160);plt.close(fig)

manifest=[]
for s in json.loads((OLD/'protocol_manifest.json').read_text())['scenarios'][1:]:
    z=np.load(OUT/(s['scenario_id']+'_series.npz'),allow_pickle=False)
    idx=np.flatnonzero(z['potential_post_gnss'])
    manifest.append({'scenario_id':s['scenario_id'],'exposed_indices_common_grid':idx.tolist(),
                     'exposed_timestamps_s':z['timestamps'][idx].tolist(),
                     'outage_n':int((~z['availability']).sum()),
                     'no_postoutage_GNSS_n':int((~z['availability']&~z['potential_post_gnss']).sum()),
                     'window_ids_containing_postoutage_tokens':sorted(set((idx//192).tolist()))})
(OUT/'window_exposure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Saved two figures and exact exposure timestamps')
