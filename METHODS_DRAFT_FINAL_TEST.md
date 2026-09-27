# METHODS_DRAFT — final test stage (attempted, not executed)

This separate version preserves the previous METHODS_DRAFT.md unchanged. The preceding six adaptive trainings and their validation results remain documented there; no additional model training occurred.

## Locked final-test protocol

All six validation-selected best checkpoints of adaptive_full/adaptive_gps_only, seeds0/1/2, were frozen by SHA256 before inference. The existing causal GRU architecture, gate, input preprocessing, native GT scoring, actual-dt integration and train-only scalers were retained. Damped-CV τ remained5s. The planned test set was only mars_6/mars_7, using existing raw files and calibration hashes in a separate private Kaggle input. Neither seed selection nor ensemble nor test tuning was allowed.

The selector reused the validation chronological/support algorithm:20s permitted history,10s recovery,60s onset stride and10/30/60s outages, with unchanged gap limits. It produced one onset per test flight (six outages) plus two full-GNSS controls. No scenario was chosen by movement or prediction error. Primary was to remain native-timestamp relative-motion RMSE3D within outage, with absolute3D/H/V, final unavailable and recovery errors. All methods were to use identical points/masks, no GT attitude/RTK/onboard pose inputs or true-state resets. Episode means were to be summarized within seed then mean/sample SD across seeds, separately by flight; controls would have N/A outage metrics. Validation/test would not be pooled.

## What actually executed

One Kaggle job was submitted: `shapok/drone-nav-insane-final-test`, version1, output run ID353128919. Local33tests and exact-package checkpoint compatibility checks passed. On Kaggle, source/input hashes and the saved manifest hash were checked, but the regenerated test-manifest JSON failed exact equality in preflight before remote tests, checkpoint loading in that cell, model inference, baselines or scientific output export. Thus0model and0baseline evaluations completed. No test metric was produced, no weight updated, and no retry occurred.

The exact differing manifest field is not known because the failed code did not export actual/diff. A platform-dependent numeric difference is an unverified hypothesis only. The submitted source and notebook remain unchanged; checks were not weakened. The failure is documented rather than described as completed held-out evaluation.

## Implications

There is no new held-out evidence for generalization or IMU benefit from this attempt. Existing validation estimates cannot substitute for test. No claim about an isolated learned-gate effect, stop detection, INS/EKF superiority, statistical significance or Q2 readiness is justified. ESKF remains not_ready. The final-test stage stops pending a separately authorized resolution; this document does not authorize a retry.
