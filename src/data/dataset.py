"""Resample GPS onto the IMU grid, split a single continuous flight by
time (not by window, to avoid leakage), window, and inject synthetic
GPS-outage blocks for the robustness sweep.
"""
import numpy as np
import torch
from torch.utils.data import Dataset


def outage_rng(outage_rate):
    """Deterministic, shared across models/figures: the same injected
    outage blocks whichever model (or which report.py figure) is using
    this outage_rate, so comparisons stay apples-to-apples.
    """
    return np.random.default_rng(int(round(outage_rate * 100000)) + 1)


def resample_gps_to_grid(t_imu, t_gps, gps_pos):
    """Hold-last-value resample of low-rate GPS onto the high-rate IMU
    timeline. Returns (gps_held (N,3), native_mask (N,) bool), where
    native_mask[i] is True exactly at the IMU sample a real GPS reading
    lands on.
    """
    idx = np.searchsorted(t_gps, t_imu, side="right") - 1
    idx = np.clip(idx, 0, len(t_gps) - 1)
    gps_held = gps_pos[idx]

    native_idx = np.clip(np.searchsorted(t_imu, t_gps), 0, len(t_imu) - 1)
    native_mask = np.zeros(len(t_imu), dtype=bool)
    native_mask[native_idx] = True
    return gps_held, native_mask


def temporal_split(n, train_frac=0.7, val_frac=0.1):
    """Contiguous index ranges on the RAW (pre-windowing) timeline. A
    single continuous trajectory leaks across nearby windows if split at
    the window level instead.
    """
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return slice(0, n_train), slice(n_train, n_train + n_val), slice(n_train + n_val, n)


def apply_block_outage(n, gps_held, target_rate, rng, block_len_range=(50, 400)):
    """Inject contiguous GPS-outage blocks until roughly `target_rate` of
    timesteps are unavailable. Baseline (target_rate=0) is "fully
    available": normal operation holds a fix that is at most one native
    GPS period stale (~0.5 s here), which is available for navigation
    purposes -- outages model a sustained loss of signal (urban canyon,
    jamming), not the ordinary inter-fix gap. During an injected outage
    the held GPS value is frozen at the outage's first timestep (mimics a
    real signal loss, not just relabeling an already-held sample).
    """
    avail = np.ones(n, dtype=bool)
    gps_out = gps_held.copy()
    if target_rate <= 0:
        return avail, gps_out

    guard = 0
    while (1.0 - avail.mean()) < target_rate and guard < 10000:
        guard += 1
        start = int(rng.integers(0, n))
        length = int(rng.integers(*block_len_range))
        end = min(n, start + length)
        gps_out[start:end] = gps_out[start]
        avail[start:end] = False
    return avail, gps_out


def make_windows(imu, gps_held, avail_mask, gt_pos, window=192, stride=96):
    n = len(imu)
    starts = list(range(0, n - window + 1, stride))
    if not starts:
        raise ValueError(f"sequence too short ({n}) for window={window}")
    idx = np.array(starts)

    def gather(arr):
        return np.stack([arr[s:s + window] for s in idx])

    return {
        "imu": gather(imu).astype(np.float32),
        "gps": gather(gps_held).astype(np.float32),
        "gps_mask": gather(avail_mask)[..., None].astype(np.float32),
        "gt_pos": gather(gt_pos).astype(np.float32),
    }


class TrajectoryWindowDataset(Dataset):
    """Thin torch wrapper around the dict produced by make_windows()."""

    def __init__(self, windows):
        self.w = windows
        self.n = len(windows["imu"])

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {k: torch.from_numpy(v[i]) for k, v in self.w.items()}


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    n_imu = 6000
    t_imu = np.arange(n_imu) * 0.02          # 50 Hz IMU
    t_gps = np.arange(0, n_imu * 0.02, 0.5)  # 2 Hz GPS

    gt_pos = np.stack([0.5 * t_imu, 0.1 * t_imu ** 1.2, np.sin(t_imu / 5)], axis=1)
    gps_pos = gt_pos[np.searchsorted(t_imu, t_gps).clip(max=n_imu - 1)] + rng.normal(0, 1, (len(t_gps), 3))
    imu = rng.normal(0, 0.1, (n_imu, 6))

    gps_held, native_mask = resample_gps_to_grid(t_imu, t_gps, gps_pos)
    assert gps_held.shape == (n_imu, 3)
    assert 0 < native_mask.mean() < 0.2, f"native GPS rate implausible: {native_mask.mean()}"

    tr, va, te = temporal_split(n_imu)
    assert tr.stop == va.start and va.stop == te.start and te.stop == n_imu

    n_tr = tr.stop - tr.start
    avail, gps_out = apply_block_outage(n_tr, gps_held[tr], target_rate=0.3, rng=rng)
    achieved = 1.0 - avail.mean()
    assert achieved >= 0.3 - 0.05, f"outage injection undershot target: {achieved}"

    avail0, gps_out0 = apply_block_outage(n_tr, gps_held[tr], target_rate=0.0, rng=rng)
    assert avail0.all(), "target_rate=0 should mean fully available (no injected outage)"

    windows = make_windows(imu[tr], gps_out, avail, gt_pos[tr], window=192, stride=96)
    ds = TrajectoryWindowDataset(windows)
    assert len(ds) > 0
    sample = ds[0]
    assert sample["imu"].shape == (192, 6)
    assert sample["gps_mask"].shape == (192, 1)

    from torch.utils.data import DataLoader
    batch = next(iter(DataLoader(ds, batch_size=8, shuffle=True)))
    assert batch["gt_pos"].shape == (min(8, len(ds)), 192, 3)

    print(f"dataset ok | native_gps_rate={native_mask.mean():.3f} "
          f"outage_achieved={achieved:.3f} n_windows={len(ds)}")
