"""Zurich Urban MAV raw CSVs -> synced numpy arrays.

Schema confirmed against the real dataset via a Kaggle --inspect run
(2026-09-24), not guessed from public docs. Departures from what the
docs implied:

  - Timestamps are in the "Timpstemp" column (a typo in the dataset
    itself, present in every log file) and are MICROSECONDS.
  - RawAccel.csv / RawGyro.csv share identical bare column names
    ('x','y','z', ...) -- accel vs gyro is which FILE you read, not the
    column name. True raw IMU, but only ~10 Hz (27050 rows / ~2714 s
    flight), not the higher rate the docs' phrasing suggested.
  - OnboardGPS.csv is ~30 Hz (one row per captured image, 81169 rows),
    with lat/lon/alt already in plain decimal degrees / meters in this
    export (the readme's "1E7 degrees" / "millimeters" scaling does
    NOT apply to this CSV -- verified against real sample values).
  - GroundTruthAGL.csv is sparse (~1 Hz, 2708 rows), keyed by `imgid`
    (an image ID) rather than a timestamp, and its x_gt/y_gt/z_gt are
    already WGS84 / UTM zone 32N METERS -- not lat/lon, no projection
    needed. Its timestamp is recovered by joining `imgid` against
    OnboardGPS.csv (which carries both imgid and Timpstemp).
  - The platform is tethered (OnboardPose.csv has Tether_force /
    Tether_angle columns), which is why the flight runs ~45 minutes --
    long for a battery MAV, unremarkable for a tethered one. Not used
    in the pipeline, but worth noting in the paper's setup section.

Still: run --inspect against whatever's actually mounted before
trusting this, in case a future dataset version changes headers.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

COLUMN_HINTS = {
    "timestamp": ["timpstemp", "timestamp", "time_ms", "time"],
    "x": ["x"], "y": ["y"], "z": ["z"],
    "lat": ["lat", "latitude"],
    "lon": ["lon", "lng", "longitude"],
    "alt": ["alt", "altitude", "height"],
    "imgid": ["imgid"],
}


def _clean_columns(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def _find_col(columns, key):
    lower = {c: c.lower() for c in columns}
    for hint in COLUMN_HINTS[key]:
        for orig, low in lower.items():
            if low == hint or low.startswith(hint):
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
        df = _clean_columns(pd.read_csv(path, nrows=2000))
        print(f"\n=== {path.relative_to(raw_dir)} ===")
        print("columns:", list(df.columns))
        if "Timpstemp" in df.columns and len(df) > 1:
            t = df["Timpstemp"].to_numpy(dtype=float)
            dt_us = np.median(np.diff(t))
            print(f"Timpstemp median step: {dt_us:.1f} us -> ~{1e6/dt_us:.2f} Hz")
        print(df.head(3))


def load_imu(raw_dir):
    """RawAccel.csv + RawGyro.csv, nearest-timestamp merged. Both share
    bare x/y/z column names -- read separately, never by column name
    across files. Returns t_imu in SECONDS (source is microseconds).
    """
    accel = _clean_columns(pd.read_csv(raw_dir / "RawAccel.csv"))
    gyro = _clean_columns(pd.read_csv(raw_dir / "RawGyro.csv"))

    t_a = accel[_find_col(accel.columns, "timestamp")].to_numpy(dtype=np.float64) * 1e-6
    t_g = gyro[_find_col(gyro.columns, "timestamp")].to_numpy(dtype=np.float64) * 1e-6

    acc_cols = [_find_col(accel.columns, k) for k in ("x", "y", "z")]
    gyro_cols = [_find_col(gyro.columns, k) for k in ("x", "y", "z")]

    acc_vals = accel[acc_cols].to_numpy(dtype=np.float64)
    gyro_idx = np.searchsorted(t_g, t_a).clip(max=len(t_g) - 1)
    gyro_vals = gyro[gyro_cols].to_numpy(dtype=np.float64)[gyro_idx]

    imu = np.concatenate([acc_vals, gyro_vals], axis=1)
    return t_a, imu


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


def load_gps(raw_dir, lat0, lon0, alt0):
    """OnboardGPS.csv: lat/lon/alt already in plain degrees/meters in
    this export -- see module docstring. Returns t in SECONDS.
    """
    gps = _clean_columns(pd.read_csv(raw_dir / "OnboardGPS.csv"))
    t = gps[_find_col(gps.columns, "timestamp")].to_numpy(dtype=np.float64) * 1e-6
    lat = gps[_find_col(gps.columns, "lat")].to_numpy(dtype=np.float64)
    lon = gps[_find_col(gps.columns, "lon")].to_numpy(dtype=np.float64)
    alt = gps[_find_col(gps.columns, "alt")].to_numpy(dtype=np.float64)
    pos = latlon_alt_to_enu(lat, lon, alt, lat0, lon0, alt0)
    return t, pos, gps


def load_ground_truth(raw_dir, gps_df):
    """GroundTruthAGL.csv is imgid-keyed and UTM32N-metric already (no
    projection needed); its timestamp comes from joining imgid against
    OnboardGPS.csv, which is dense enough (~30 Hz, one row/image) to
    cover every ground-truth imgid. Returns (t_gt seconds, gt_pos_enu,
    origin dict) where gt_pos_enu is relative to the FIRST ground-truth
    sample (which also anchors the GPS ENU origin -- both signals share
    one physical origin point).
    """
    gt = _clean_columns(pd.read_csv(raw_dir / "GroundTruthAGL.csv"))
    imgid_col = _find_col(gt.columns, "imgid")
    x_gt, y_gt, z_gt = gt["x_gt"].to_numpy(np.float64), gt["y_gt"].to_numpy(np.float64), gt["z_gt"].to_numpy(np.float64)

    ts_col = _find_col(gps_df.columns, "timestamp")
    imgid_gps_col = _find_col(gps_df.columns, "imgid")
    ts_by_imgid = gps_df.set_index(imgid_gps_col)[ts_col]

    t_gt_us = gt[imgid_col].map(ts_by_imgid).to_numpy(dtype=np.float64)
    valid = ~np.isnan(t_gt_us)
    if (~valid).any():
        print(f"dropping {(~valid).sum()} ground-truth rows with no matching OnboardGPS imgid")

    t_gt = t_gt_us[valid] * 1e-6
    order = np.argsort(t_gt)
    t_gt = t_gt[order]

    x0, y0, z0 = x_gt[valid][order][0], y_gt[valid][order][0], z_gt[valid][order][0]
    gt_pos = np.stack([x_gt[valid][order] - x0, y_gt[valid][order] - y0, z_gt[valid][order] - z0], axis=1)

    first_imgid = int(gt[imgid_col][valid].to_numpy()[order][0])
    lat_col, lon_col = _find_col(gps_df.columns, "lat"), _find_col(gps_df.columns, "lon")
    origin_row = gps_df[gps_df[imgid_gps_col] == first_imgid].iloc[0]
    origin = {"lat0": float(origin_row[lat_col]), "lon0": float(origin_row[lon_col]), "alt0": float(z0),
              "utm_x0": float(x0), "utm_y0": float(y0), "utm_z0": float(z0)}
    return t_gt, gt_pos.astype(np.float64), origin


def prepare(raw_dir, out_dir):
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_imu, imu = load_imu(raw_dir)

    gps_raw = _clean_columns(pd.read_csv(raw_dir / "OnboardGPS.csv"))
    t_gt, gt_pos, origin = load_ground_truth(raw_dir, gps_raw)
    t_gps, gps_pos, _ = load_gps(raw_dir, origin["lat0"], origin["lon0"], origin["alt0"])

    np.save(out_dir / "t_imu.npy", t_imu)
    np.save(out_dir / "imu.npy", imu.astype(np.float32))
    np.save(out_dir / "t_gps.npy", t_gps)
    np.save(out_dir / "gps_pos.npy", gps_pos.astype(np.float32))
    np.save(out_dir / "t_gt.npy", t_gt)
    np.save(out_dir / "gt_pos.npy", gt_pos.astype(np.float32))

    meta = {
        **origin,
        "n_imu": int(len(t_imu)), "n_gps": int(len(t_gps)), "n_gt": int(len(t_gt)),
        "imu_rate_hz": float(1.0 / np.median(np.diff(t_imu))),
        "gps_rate_hz": float(1.0 / np.median(np.diff(t_gps))),
        "gt_rate_hz": float(1.0 / np.median(np.diff(t_gt))),
        "flight_duration_s": float(t_imu[-1] - t_imu[0]),
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
