"""Read saved forensic CSV only; no new sensor or navigation computation."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/drone-next-stage-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'outputs/imu_calibration_audit'
t=pd.read_csv(p/'orientation_disagreement_over_time.csv')
w=pd.read_csv(p/'orientation_window_summary.csv')
labels={'raw_zero':'Raw gyro, zero offset','raw_train_mean_sensitivity':'Raw gyro, train-mean sensitivity','onboard_omega_to_onboard_q':'Onboard Omega vs onboard q'}
colors={'raw_zero':'#b2182b','raw_train_mean_sensitivity':'#2166ac','onboard_omega_to_onboard_q':'#762a83'}
fig,axs=plt.subplots(1,3,figsize=(15,4.5))
for method in ['raw_zero','raw_train_mean_sensitivity']:
 d=t[t.method==method].copy();d['elapsed_round']=d.elapsed_s.round().astype(int)
 g=d.groupby('elapsed_round').error_deg
 lo=g.quantile(.05);hi=g.quantile(.95);mid=g.median()
 axs[0].plot(mid.index,mid,color=colors[method],label=labels[method]);axs[0].fill_between(mid.index,lo,hi,color=colors[method],alpha=.18)
axs[0].set(xlabel='Elapsed time (s)',ylabel='Gravity-direction disagreement (deg)',title='63 train windows: median, P05-P95')
for method in labels:
 d=w[w.method==method]
 axs[1].plot(d.start_s,d.final_error_deg,'.-',ms=3,lw=.8,label=labels[method],color=colors[method])
axs[1].axvline(1336.1542017,color='gray',ls='--',lw=1)
axs[1].set(xlabel='Window start: board clock (s)',ylabel='Final angular disagreement (deg)',title='All 30s windows retained; no navigation')
lag=pd.read_csv(p/'raw_onboard_lag_profiles.csv')
axs[2].plot(lag.lag_s,lag.corr_raw_z_vs_negative_onboard_z,'o-',label='raw Z vs -onboard Z')
axs[2].plot(lag.lag_s,lag.mean_abs_xy_pair_correlation,'o-',label='mean |XY pair correlation|')
axs[2].set(xlabel='Lag: onboard(t) vs raw(t - lag), s',ylabel='Pearson correlation',title='Offline 10Hz grid; no delay adopted')
for ax in axs:ax.grid(alpha=.2);ax.legend(fontsize=7)
fig.tight_layout();fig.savefig(p/'sensor_consistency_diagnostics.png',dpi=180);fig.savefig(p/'sensor_consistency_diagnostics.svg');plt.close(fig)
print('Saved sensor_consistency_diagnostics.png/.svg')
