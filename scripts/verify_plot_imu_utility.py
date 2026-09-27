"""Independent saved-result checks and plots; no new prediction runs."""
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/imu_utility_diagnostic'
rows=list(csv.DictReader((OUT/'summary.csv').open()));manifest=json.loads((OUT/'manifest.json').read_text())
assert len(rows)==54 and len({(r['scenario_id'],r['method']) for r in rows})==54
maxdiff=0.
for r in rows:
 d=np.load(OUT/r['scenario_id']/(r['method']+'.npz'),allow_pickle=False);p=d['pred'];gt=d['gt'];target=d['motion_target'];t=d['timestamps']
 assert np.isfinite(p).all() and np.isfinite(gt).all() and len(t)==int(r['n_outage'])
 for prefix,e in [('motion_',p-d['anchor']-target),('raw_',p-gt)]:
  vals={'rmse_3d_m':np.sqrt(np.mean(np.sum(e*e,axis=1))),'rmse_horizontal_m':np.sqrt(np.mean(np.sum(e[:,:2]**2,axis=1))),
        'rmse_vertical_m':np.sqrt(np.mean(e[:,2]**2)),'final_3d_m':np.linalg.norm(e[-1]),'final_horizontal_m':np.linalg.norm(e[-1,:2]),'final_abs_vertical_m':abs(e[-1,2])}
  for k,v in vals.items():maxdiff=max(maxdiff,abs(float(r[prefix+k])-v))
 if r['method']!='constant_velocity_gnss':
  rot=d['orientation'];np.testing.assert_allclose(np.einsum('nji,njk->nik',rot,rot),np.broadcast_to(np.eye(3),rot.shape),atol=1e-12)
 assert hashlib.sha256(d['full_common_availability_mask'].astype('uint8').tobytes()).hexdigest()==next(s['mask_uint8_sha256'] for s in manifest['scenarios'] if s['scenario_id']==r['scenario_id'])
assert maxdiff==0.
for path,h in manifest['source_files'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h
v2=ROOT/'outputs/v2_motion_pilot';v2idx=json.loads((v2/'artifact_manifest.json').read_text())
for path,h in v2idx['files'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h['sha256'],path
cfg=OUT/'config.json';assert hashlib.sha256(cfg.read_bytes()).hexdigest()==manifest['config_sha256']
check={'status':'passed','summary_rows':54,'duplicate_keys':0,'nonfinite':False,'max_recomputed_metric_discrepancy_m':maxdiff,
       'v2_artifact_hashes_unchanged':len(v2idx['files']),'frozen_protocol_unchanged':True,'SO3_orthogonality_tolerance':1e-12,
       'masks_match_all_fixed_scenarios':True,'primary_method':'dr_zero_bias','synthetic_tests':9,'no_training_no_kaggle_no_test':True}
(OUT/'verification.json').write_text(json.dumps(check,indent=2)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
summary=list(csv.DictReader((OUT/'duration_summary.csv').open()))
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
names={'constant_velocity_gnss':'GNSS CV','dr_zero_bias':'DR zero bias (primary)','dr_mean_bias':'DR mean bias','dr_median_bias':'DR median bias','cv_acceleration_residual':'CV + accel residual','horizontal_dr_zero_bias':'Horizontal DR'}
for method,label in names.items():
 values=sorted([r for r in summary if r['method']==method],key=lambda r:int(r['duration_s']))
 for ax,k in zip(axes,['mean_episode_motion_rmse_3d_m','mean_episode_motion_horizontal_m']):
  ax.plot([int(r['duration_s']) for r in values],[float(r[k]) for r in values],marker='o',label=label)
for ax,title in zip(axes,['3D relative-motion RMSE','Horizontal relative-motion RMSE']):
 ax.set_yscale('log');ax.set_xticks([10,30,60]);ax.set_xlabel('Outage duration [s]');ax.set_ylabel('Mean of 3 onset RMSEs [m], log scale');ax.set_title(title);ax.grid(True,alpha=.3)
axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'drift_vs_duration.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True)
for e,ax in zip([1,2,3],axes):
 for method in ['constant_velocity_gnss','dr_zero_bias','dr_mean_bias']:
  d=np.load(OUT/f'val_e{e}_d60s'/(method+'.npz'));error=np.linalg.norm(d['pred']-d['anchor']-d['motion_target'],axis=1)
  onset=next(s['onset_s'] for s in manifest['scenarios'] if s['scenario_id']==f'val_e{e}_d60s')
  ax.plot(d['timestamps']-onset,error,label=names[method])
 ax.set_yscale('log');ax.set_ylabel('3D motion error [m]');ax.set_title(f'e{e}: saved 60 s trajectory, no alignment');ax.grid(True,alpha=.3);ax.legend(fontsize=8)
axes[-1].set_xlabel('Time since outage onset [s]');fig.tight_layout();fig.savefig(OUT/'error_over_outage.png',dpi=170);plt.close(fig)
print(json.dumps(check,indent=2))
