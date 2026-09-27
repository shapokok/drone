# RESULTS_DRAFT — final test unavailable

The single final evaluation job (`shapok/drone-nav-insane-final-test`, version1, run ID353128919) failed during test-manifest preflight. Completed model evaluations:0; baseline evaluations:0. No full/gps_only, Held, CV or Damped-CV test metrics were calculated. All10/30/60s comparisons on both mars_6/mars_7 are unavailable, not zero.

| Test flight | Outages | full/gps_only | Held/CV/Damped-CV |
|---|---|---|---|
| mars_6 |10/30/60s|Not evaluated|Not evaluated|
| mars_7 |10/30/60s|Not evaluated|Not evaluated|

The error was `AssertionError: Test scenario/support/hash mismatch` at exact equality of locally locked and remotely regenerated manifests. The precise differing field was not recorded. This technical failure neither confirms nor refutes the scientific hypothesis. Validation findings remain separate and are not reported here as test results. There are no test seed means, SDs, paired comparisons, gate traces or trajectory metrics to interpret. No training, tuning, changed parameters or retry followed the failure. See INSANE_FINAL_TEST_REPORT.md for evidence and limitations.
