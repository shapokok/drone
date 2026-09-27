# Saved experiment summaries

These are byte-identical copies of selected local artifacts, not new runs or
recomputed metrics. `artifact_manifest.json` records original paths, sizes and
SHA256. Full artifacts remain in local `outputs/` and private handoff archives.

| Directory | Stage | Interpretation |
| --- | --- | --- |
| [insane_pilot](insane_pilot/) | Initial validation pilot, seed 0 | Separate architecture; do not pool with adaptive seeds. |
| [insane_adaptive](insane_adaptive/) | Adaptive validation, seeds 0/1/2 | Train mars_1–3; validation mars_4–5. |
| [insane_final_test](insane_final_test/) | Failed first final-test attempt | Preflight failure before inference; no test metrics. |
| [insane_final_test_retry](insane_final_test_retry/) | Completed authorized final-test retry | Fixed checkpoints; mars_6–7; Kaggle v2, run 353132214. |

Each completed stage includes `summary.csv`, `paired_differences.csv`, saved
aggregate tables, status and verification evidence. Training stages also include
`training_history.csv`; the final test performed no training. Gate visualizations
are saved in the adaptive and successful final-test directories. Final preflight
field-level diffs and regenerated manifests are in
`insane_final_test_retry/preflight_evidence/`.

The primary metric is relative-motion RMSE3D within the outage, in metres.
Read the corresponding root report for exact aggregation and limitations.
Standard deviations across seeds are not confidence intervals; seeds are not
independent flights. ESKF is not ready and has no valid comparison scores.

Raw sensor/GT time series, per-timestamp predictions, weights and full source
snapshots are intentionally excluded. Executed source/configuration hashes are
preserved; code and fixed configurations are versioned in the repository.
The historical `source_git_commit` identifies the base used when creating the
execution snapshots; use their recorded SHA256 values to identify actual code.
Reports retain original local paths so they remain compatible with handoff ZIPs.
