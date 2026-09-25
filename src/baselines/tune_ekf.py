"""Grid search over EKF noise parameters on the VALIDATION split, so the
classical baseline gets the same tuning budget as the proposed model --
the neural models are model-selected via early stopping on val loss;
this is the equivalent for a filter with no gradient to descend. The
shipped q_accel/q_gyro/r_gps defaults in ekf.py were never characterized
against this sensor's real noise, only placeholders.

Selects by ATE at full GPS availability on val; writes the winning
combo to ekf_tuned_params.json, which evaluate.py loads for every Table
1/2/3 `ekf` row (both the `full` and `no_bias` ablation reuse the same
tuned noise params -- only estimate_bias differs between them).
"""
import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from baselines.ekf import run_ekf  # noqa: E402
from data.dataset import temporal_split  # noqa: E402
from metrics import trajectory_metrics  # noqa: E402
from train import load_processed  # noqa: E402

Q_ACCEL_GRID = [0.01, 0.05, 0.1, 0.5, 1.0]
Q_GYRO_GRID = [0.001, 0.005, 0.01, 0.05]
R_GPS_GRID = [0.5, 1.0, 2.0, 5.0, 10.0]


def tune(processed_dir, out_path):
    imu, gps_held, gt_pos = load_processed(processed_dir)
    tr, va, te = temporal_split(len(imu))

    dt = 1.0 / 50.0
    meta_path = Path(processed_dir) / "meta.json"
    if meta_path.exists():
        dt = 1.0 / json.loads(meta_path.read_text())["imu_rate_hz"]

    accel_va, gyro_va = imu[va, :3], imu[va, 3:]
    gps_va = gps_held[va]
    avail_va = np.ones(len(gps_va), dtype=bool)  # tune at full GPS availability
    gt_va = gt_pos[va]

    combos = list(product(Q_ACCEL_GRID, Q_GYRO_GRID, R_GPS_GRID))
    best = None
    for q_accel, q_gyro, r_gps in combos:
        pred = run_ekf(accel_va, gyro_va, gps_va, avail_va, dt, gps_pos0=gt_va[0],
                        q_accel=q_accel, q_gyro=q_gyro, r_gps=r_gps)
        ate = trajectory_metrics(pred, gt_va, align=False)["ATE"]
        if best is None or ate < best["ate"]:
            best = {"q_accel": q_accel, "q_gyro": q_gyro, "r_gps": r_gps, "ate": ate}

    print(f"tuned EKF on {len(combos)} combos | best: {best}")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(
        {"q_accel": best["q_accel"], "q_gyro": best["q_gyro"], "r_gps": best["r_gps"],
         "val_ate": best["ate"], "n_combos_searched": len(combos)}, indent=2))
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default="data/processed")
    ap.add_argument("--out", default="outputs/ekf_tuned_params.json")
    args = ap.parse_args()
    tune(args.processed_dir, args.out)
