"""Zurich Urban MAV raw CSVs -> synced numpy arrays.

The dataset's public docs describe the *shape* of each file (timestamp in
the first column, then sensor readings) but not the exact header names,
so this module has two modes:

  --inspect   prints real headers/dtypes/sampling rates for every CSV
              under --raw. Run this FIRST against the actual Kaggle
              input before trusting anything below. No parsing
              assumptions are applied in this mode.

  (default)   parses RawAccel.csv, RawGyro.csv, OnboardGPS.csv,
              GroundTruthAGL.csv using the column-name heuristics in
              `_find_col`, and fails loudly with the available columns
              listed if a heuristic can't find what it needs -- rerun
              --inspect and adjust `COLUMN_HINTS` below rather than
              guessing blind.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# first-match-wins substrings (lowercased) used to locate each logical
# column inside a raw CSV whose exact header text is unconfirmed.
COLUMN_HINTS = {
    "timestamp": ["timestamp", "time_ms", "time", "t"],
    "accel_x": ["accx", "acc_x", "ax", "accel_x"],
    "accel_y": ["accy", "acc_y", "ay", "accel_y"],
    "accel_z": ["accz", "acc_z", "az", "accel_z"],
    "gyro_x": ["gyrox", "gyro_x", "gx", "gyro_x"],
    "gyro_y": ["gyroy", "gyro_y", "gy", "gyro_y"],
    "gyro_z": ["gyroz", "gyro_z", "gz", "gyro_z"],
    "lat": ["lat", "latitude"],
    "lon": ["lon", "lng", "longitude"],
    "alt": ["alt", "altitude", "height"],
}


def _find_col(columns, key):
    lower = {c: c.lower() for c in columns}
    for hint in COLUMN_HINTS[key]:
        for orig, low in lower.items():
            if hint in low:
                return orig
    raise KeyError(
        f"could not find a column for '{key}' among {list(columns)}. "
        f"Run --inspect and add the real header text to COLUMN_HINTS['{key}']."
    )


def inspect(raw_dir):
    raw_dir = Path(raw_dir)
    csvs = sorted(raw_dir.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"no CSVs found under {raw_dir} -- check the Kaggle input path")

    for path in csvs:
        df = pd.read_csv(path, nrows=2000)
        print(f"\n=== {path.relative_to(raw_dir)} ===")
        print("columns:", list(df.columns))
        print("dtypes:\n", df.dtypes)
        if len(df) > 1:
            t = df.iloc[:, 0].to_numpy(dtype=float)
            dt = np.median(np.diff(t))
            unit_guess = "ms" if dt > 1 else "s"
            rate_hz = 1000.0 / dt if unit_guess == "ms" else 1.0 / dt
            print(f"first-column median step: {dt:.4f} (guessed unit={unit_guess}) "
                  f"-> ~{rate_hz:.2f} Hz")
        print(df.head(3))


def latlon_alt_to_enu(lat, lon, alt, lat0, lon0, alt0):
    """Equirectangular approximation -- adequate at the ~2 km scale of a
    single urban flight, not meant for anything larger.
    """
    R = 6378137.0
    lat0_r = np.radians(lat0)
    east = np.radians(lon - lon0) * R * np.cos(lat0_r)
    north = np.radians(lat - lat0) * R
    up = alt - alt0
    return np.stack([east, north, up], axis=1)


def load_imu(raw_dir):
    accel = pd.read_csv(raw_dir / "RawAccel.csv")
    gyro = pd.read_csv(raw_dir / "RawGyro.csv")

    t_a = accel[_find_col(accel.columns, "timestamp")].to_numpy(dtype=np.float64)
    t_g = gyro[_find_col(gyro.columns, "timestamp")].to_numpy(dtype=np.float64)

    acc_cols = [_find_col(accel.columns, k) for k in ("accel_x", "accel_y", "accel_z")]
    gyro_cols = [_find_col(gyro.columns, k) for k in ("gyro_x", "gyro_y", "gyro_z")]

    acc_vals = accel[acc_cols].to_numpy(dtype=np.float64)
    gyro_idx = np.searchsorted(t_g, t_a).clip(max=len(t_g) - 1)
    gyro_vals = gyro[gyro_cols].to_numpy(dtype=np.float64)[gyro_idx]

    imu = np.concatenate([acc_vals, gyro_vals], axis=1)
    return t_a, imu


def load_gps(raw_dir, lat0, lon0, alt0):
    gps = pd.read_csv(raw_dir / "OnboardGPS.csv")
    t = gps[_find_col(gps.columns, "timestamp")].to_numpy(dtype=np.float64)
    lat = gps[_find_col(gps.columns, "lat")].to_numpy(dtype=np.float64)
    lon = gps[_find_col(gps.columns, "lon")].to_numpy(dtype=np.float64)
    alt = gps[_find_col(gps.columns, "alt")].to_numpy(dtype=np.float64)
    pos = latlon_alt_to_enu(lat, lon, alt, lat0, lon0, alt0)
    return t, pos


def load_ground_truth(raw_dir, lat0, lon0, alt0):
    gt = pd.read_csv(raw_dir / "GroundTruthAGL.csv")
    t = gt[_find_col(gt.columns, "timestamp")].to_numpy(dtype=np.float64)
    lat = gt[_find_col(gt.columns, "lat")].to_numpy(dtype=np.float64)
    lon = gt[_find_col(gt.columns, "lon")].to_numpy(dtype=np.float64)
    alt = gt[_find_col(gt.columns, "alt")].to_numpy(dtype=np.float64)
    pos = latlon_alt_to_enu(lat, lon, alt, lat0, lon0, alt0)
    return t, pos


def prepare(raw_dir, out_dir):
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt = pd.read_csv(raw_dir / "GroundTruthAGL.csv")
    lat0 = gt[_find_col(gt.columns, "lat")].iloc[0]
    lon0 = gt[_find_col(gt.columns, "lon")].iloc[0]
    alt0 = gt[_find_col(gt.columns, "alt")].iloc[0]

    t_imu, imu = load_imu(raw_dir)
    t_gps, gps_pos = load_gps(raw_dir, lat0, lon0, alt0)
    t_gt, gt_pos = load_ground_truth(raw_dir, lat0, lon0, alt0)

    np.save(out_dir / "t_imu.npy", t_imu)
    np.save(out_dir / "imu.npy", imu.astype(np.float32))
    np.save(out_dir / "t_gps.npy", t_gps)
    np.save(out_dir / "gps_pos.npy", gps_pos.astype(np.float32))
    np.save(out_dir / "t_gt.npy", t_gt)
    np.save(out_dir / "gt_pos.npy", gt_pos.astype(np.float32))

    meta = {
        "origin_lat": float(lat0), "origin_lon": float(lon0), "origin_alt": float(alt0),
        "n_imu": int(len(t_imu)), "n_gps": int(len(t_gps)), "n_gt": int(len(t_gt)),
        "imu_rate_hz": float(1.0 / np.median(np.diff(t_imu))),
        "gps_rate_hz": float(1.0 / np.median(np.diff(t_gps))),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--inspect", action="store_true",
                     help="print real headers/dtypes/rates, parse nothing")
    args = ap.parse_args()

    if args.inspect:
        inspect(args.raw)
    else:
        prepare(args.raw, args.out)
