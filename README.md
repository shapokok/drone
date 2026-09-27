# Intelligent 3D Drone Navigation (IMU + GNSS fusion)

The latest completed stage is the locked **INSANE adaptive final test** on
`mars_6` and `mars_7`: Kaggle version 2, run `353132214`. No further training,
seeds, or test runs are scheduled. The earlier version failed preflight before
inference; its records are retained separately.

Start with:

- [Final test report](INSANE_FINAL_TEST_REPORT_RETRY.md),
  [methods](METHODS_DRAFT_FINAL_TEST_RETRY.md), and
  [results draft](RESULTS_DRAFT_FINAL_TEST_RETRY.md).
- [Preflight repair and verification](INSANE_PREFLIGHT_FIX_REPORT.md).
- [Published result tables and provenance](results/README.md).
- [Adaptive validation report](INSANE_ADAPTIVE_REPORT.md) and
  [initial INSANE pilot report](INSANE_PILOT_REPORT.md).
- [Executed final notebook](notebooks/kaggle_insane_final_test_retry.ipynb),
  [evaluation lock](configs/insane_final_test/FINAL_EVALUATION_LOCK.json), and
  [separate technical preflight patch](src/insane_final_test_patch/).

On the final test, adaptive full has lower relative-motion RMSE3D than matched
GPS-only in 17/18 episode-seed pairs, but its seed-mean error is higher than
Held-GNSS, CV, and Damped-CV on both flights at every outage duration. This does
not establish an overall advantage over the simple baselines or generalization.

The repository contains source, tests, fixed manifests, notebooks, reports, and
compact result tables. Raw sensor data, credentials, checkpoints, prediction
arrays, full local outputs, and handoff archives are excluded from Git. Reports
retain original local `outputs/...` references; portable copies of selected
artifacts are indexed in `results/`. Checkpoints and private datasets must be
obtained separately to reproduce inference. Historical notebooks preserve their
executed settings and are **not instructions to start another run**.

## Historical Zurich prototype

The documentation below describes the original Zurich implementation. Its
commands and proposed experiments are historical and do not supersede the
locked INSANE protocol or the completed-run reports above.

Reference implementation for *Intelligent 3D navigation of drones based on
inertial and satellite data* — a dual-branch cross-attention fusion
Transformer for IMU+GPS trajectory estimation, benchmarked against a
classical strapdown-INS/EKF baseline, with a GPS-denied robustness study.

```
src/
  data/prepare.py        Zurich Urban MAV raw CSVs -> synced npy arrays
  data/dataset.py         windowing, GPS-outage augmentation, temporal split
  baselines/ekf.py         strapdown INS + loosely-coupled EKF baseline
  baselines/tune_ekf.py     grid-search EKF noise params on val, same
                              tuning budget the neural models get from
                              early stopping -- run before evaluate.py
  models/fusion_transformer.py   proposed dual-branch fusion model
  models/fusion_lstm.py           same idea, LSTM backbone (ablation arm)
  metrics.py                ATE / RPE (Umeyama-aligned) / per-axis RMSE
  train.py                    training loop, resume + results.csv logging
  evaluate.py                  outage-rate sweep + metrics over a checkpoint
  xai.py                        branch-ablation probing (causal contribution
                                  of IMU vs GPS) for Figure 3
  report.py                     builds paper tables/figures from results.csv
```

## 0. Smoke test first (no GPU, no dataset needed)

```bash
python src/metrics.py
python src/baselines/ekf.py
python src/models/fusion_transformer.py
python src/models/fusion_lstm.py
python src/data/dataset.py
```

Each runs on synthetic data and prints an `ok` line in a few seconds.
Catches shape bugs before touching Kaggle.

## 1. Data

Dataset: [Zurich Urban Micro Aerial Vehicle](https://www.kaggle.com/datasets/mrisdal/zurich-urban-micro-aerial-vehicle)
— 2 km urban low-altitude flight, time-synced `RawAccel.csv`,
`RawGyro.csv`, `OnboardGPS.csv`, `GroundTruthAGL.csv`.

**The exact column names/sampling rates are not confirmed from public docs
alone.** Before trusting `prepare.py`'s parser, run it in inspect mode
against the real files (first thing to do on Kaggle, see section 4):

```bash
python src/data/prepare.py --inspect --raw /kaggle/input/zurich-urban-micro-aerial-vehicle
```

This only prints headers/dtypes/inferred sampling rates — no parsing
assumptions are trusted until this output is checked by hand. Once
confirmed:

```bash
python src/data/prepare.py --raw /kaggle/input/zurich-urban-micro-aerial-vehicle --out data/processed
```

Upload `data/processed/` as a Kaggle Dataset once re-running prep inside
every training notebook wastes GPU quota for nothing.

## 2. Training

```bash
# proposed model, one seed
python src/train.py --model fusion_transformer --seed 0

# ablation arm: LSTM backbone instead of Transformer
python src/train.py --model fusion_lstm --seed 0

# baseline EKF has no training loop — evaluate.py runs it directly
python src/evaluate.py --model ekf --outage_rates 0 0.1 0.3 0.5 0.7

# 5 seeds for the headline table
python src/train.py --model fusion_transformer --seeds 0 1 2 3 4
```

Every finished run appends a row to `outputs/results.csv`. Re-running the
same command skips whatever is already there (`--force` overrides), so a
12-hour Kaggle timeout costs one run, not the batch.

## 3. Protocol (fixed — do not vary between models)

* IMU window feeds the fusion model at full IMU rate; GPS is resampled to
  the same grid with an explicit availability mask channel (not silently
  interpolated away).
* Position target is local ENU, origin at the first ground-truth sample.
* Split is by flight-time segment, not random rows — a single continuous
  trajectory leaks across nearby windows if split randomly.
* GPS-denied evaluation: synthetic outage masks at {0, 10, 30, 50, 70}%
  dropout, same masks reused across all compared models for a fair fight.
* ATE and RPE computed after Umeyama alignment to ground truth (standard
  SLAM/odometry convention) — not raw unaligned RMSE.
* 5 seeds per configuration, reported as mean ± std.
* The EKF baseline gets the same tuning budget as the proposed model:
  its noise parameters are grid-searched on the validation split
  (`baselines/tune_ekf.py`) rather than left on hand-picked defaults —
  comparing a tuned model against an untuned baseline is the first
  thing a reviewer would flag.

## 4. Kaggle rules that actually matter

1. **First Kaggle run is `--inspect`, not training.** Confirms the real
   CSV schema before `prepare.py` is trusted with actual parsing.
2. **Debug with GPU off.** CPU sessions do not consume quota.
3. **Use Save & Run All (commit), not the interactive session** — it
   keeps running after you close the laptop.
4. **Chain notebooks**: `00_prepare` (inspect + parse) → its output
   becomes the input dataset of `01_train` → checkpoints feed
   `02_evaluate`/figures. Nothing recomputed.
5. **Enable internet** (needs phone verification) so the notebook can
   `git clone` this repo instead of pasting cells by hand.
6. Keep `outputs/results.csv`, `outputs/predictions/`, and
   `outputs/checkpoints/` in the notebook output.

## 5. Paper deliverables

Tables: (1) main accuracy — FusionNav vs EKF vs raw-GPS vs IMU-only dead
reckoning; (2) GPS-denied robustness across outage rates; (3) ablation
(no cross-attention / single-branch / LSTM backbone); (4) mean±std over 5
seeds. Figures: (1) trajectory overlay; (2) error growth during a single
GPS outage; (3) attention-weight shift toward IMU as GPS drops; (4) ATE
distribution across test segments/seeds. Each table/figure pair carries
distinct information — no figure just re-plots a table.

**Note for Limitations section (paper, not code):** the Zurich MAV ground
truth is not motion-capture-grade — it is itself camera-position-derived,
so treat reported ATE/RPE as relative-quality comparisons between models,
not absolute error against a true reference.
