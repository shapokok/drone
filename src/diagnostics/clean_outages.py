"""Clean outages on the existing R7 coordinate/time representation.

No training, GT-dependent input construction, alignment, or filesystem writes.
Availability is a link-level flag, NOT a claim that a new fix arrived. The
separate update mask and source indices make that distinction auditable.
"""
import hashlib

import numpy as np


def _timeline(t):
    t = np.asarray(t, dtype=np.float64)
    if t.ndim != 1 or not len(t) or not np.isfinite(t).all() or (np.diff(t) <= 0).any():
        raise ValueError("timestamps must be finite, nonempty and strictly increasing")
    return t


def false_runs(available):
    """Half-open index intervals in the FINAL mask, independent of block order."""
    missing = ~np.asarray(available, dtype=bool)
    edges = np.diff(np.r_[False, missing, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1).tolist(),
                    np.flatnonzero(edges == -1).tolist()))


def availability_from_blocks(n, blocks):
    available = np.ones(n, dtype=bool)
    for start, stop in blocks:
        if not (0 <= start < stop <= n):
            raise ValueError("blocks must be nonempty half-open intervals within the record")
        available[start:stop] = False
    return available


def mask_sha256(available):
    """Hash uint8 availability bytes, one byte per common-grid sample."""
    return hashlib.sha256(np.asarray(available, dtype=np.uint8).tobytes()).hexdigest()


def common_validation_indices(t_imu, t_gt, train_fraction, val_fraction, window):
    """Split first, restrict to original GT support, then trim incomplete windows.

    All methods use these SAME indices. No GT extrapolation, no concatenation
    across holes, no carry-over from train/test. Windows start at common[0].
    """
    t = _timeline(t_imu)
    g = _timeline(t_gt)
    if window < 1 or not (0 < train_fraction < 1 and 0 < val_fraction < 1 - train_fraction):
        raise ValueError("invalid split/window")
    start = int(len(t) * train_fraction)
    stop = start + int(len(t) * val_fraction)
    raw = np.arange(start, stop, dtype=np.int64)
    supported = raw[(t[raw] >= g[0]) & (t[raw] <= g[-1])]
    if len(supported) and (np.diff(supported) != 1).any():
        raise ValueError("noncontiguous supported validation region")
    n = len(supported) // window * window
    if not n:
        raise ValueError("validation has no complete supported window")
    common = supported[:n]
    return common, {
        "split": "validation", "original_split_indices_half_open": [start, stop],
        "original_split_n": len(raw), "gt_support_s": [float(g[0]), float(g[-1])],
        "excluded_outside_gt_support_n": len(raw) - len(supported),
        "discarded_incomplete_window_n": len(supported) - n,
        "common_indices_half_open": [int(common[0]), int(common[-1]) + 1],
        "common_n": n, "window": window, "stride": window, "window_count": n // window,
        "common_first_timestamp_s": float(t[common[0]]),
        "common_last_timestamp_s": float(t[common[-1]]),
    }


def make_scenarios(t, protocol):
    """Choose starts from timestamps only; never inspect positions/errors.

    Three fractions of the admissible onset interval are snapped forward to a
    grid point. Every duration uses the same starts, in separate full-trace
    scenarios. [onset, cutoff) is unavailable. RNG is explicitly unused.
    """
    t = _timeline(t)
    durations = protocol["durations_s"]
    fractions = protocol["start_fractions"]
    warmup, recovery = protocol["warmup_s"], protocol["recovery_s"]
    if min(durations) <= 0 or warmup <= 0 or recovery <= 0:
        raise ValueError("positive durations, warmup and recovery required")
    if any(not 0 <= f <= 1 for f in fractions):
        raise ValueError("start fractions must be in [0, 1]")
    lo, hi = t[0] + warmup, t[-1] - recovery - max(durations)
    if hi <= lo:
        raise ValueError("validation too short for the fixed protocol")
    starts = [int(np.searchsorted(t, lo + f * (hi - lo), side="left")) for f in fractions]
    if len(set(starts)) != len(starts):
        raise ValueError("not enough distinct starts; do not silently change protocol")
    scenarios = [{"scenario_id": "val_control", "split": "validation",
                  "episode_id": None, "requested_duration_s": 0,
                  "source_blocked_intervals_s": [], "unavailable_index_runs": [],
                  "n_unavailable": 0, "metric_scope": "whole_common_trace"}]
    for episode, start in enumerate(starts, 1):
        for duration in durations:
            onset = float(t[start])
            cutoff = onset + duration
            stop = int(np.searchsorted(t, cutoff, side="left"))
            if start == 0 or stop >= len(t) or t[-1] - t[stop] < recovery:
                raise ValueError("episode lacks warmup/recovery; protocol cannot be used")
            scenarios.append({
                "scenario_id": f"val_e{episode}_d{duration:g}s", "split": "validation",
                "episode_id": episode, "requested_duration_s": duration,
                "target_onset_s": float(lo + fractions[episode - 1] * (hi - lo)),
                "onset_s": onset, "requested_cutoff_s": cutoff,
                "last_unavailable_s": float(t[stop - 1]),
                "first_link_available_after_s": float(t[stop]),
                "observed_mask_interval_s": float(t[stop] - t[start]),
                "unavailable_sample_span_s": float(t[stop - 1] - t[start]),
                "available_history_span_s": float(t[start] - t[0]),
                "recovery_span_s": float(t[-1] - t[stop]),
                "source_blocked_intervals_s": [[onset, cutoff]],
                "unavailable_index_runs": [[start, stop]], "n_unavailable": stop - start,
                "metric_scope": "outage_only",
            })
    for scenario in scenarios:
        available = availability_from_blocks(len(t), scenario["unavailable_index_runs"])
        scenario.update(n_common=len(t), mask_uint8_sha256=mask_sha256(available),
                        generator_seed=protocol["generator_seed"], rng_used=False)
    return scenarios


def clean_gnss(t, candidates, source_indices, source_timestamps, available,
               blocked_intervals=None):
    """Sequentially accept only legal candidate fixes after forming final mask.

    Candidates are the unchanged R7 GPS resampling. Source timestamps/indices
    refer to exported processed GPS rows, NOT verified physical receiver fixes.
    During a false run, keep the last accepted value from BEFORE its onset.
    Also reject a buffered candidate stamped inside a denied time interval on
    recovery. No candidate at the first masked sample is accepted. The first
    grid sample must have a legal fix; this protocol provides explicit warmup.
    """
    t = _timeline(t)
    candidates = np.asarray(candidates)
    ids = np.asarray(source_indices, dtype=np.int64)
    st = np.asarray(source_timestamps, dtype=np.float64)
    available = np.asarray(available, dtype=bool)
    if (candidates.shape != (len(t), 3) or ids.shape != t.shape
            or st.shape != t.shape or available.shape != t.shape):
        raise ValueError("GNSS inputs have incompatible shapes")
    if not np.isfinite(candidates).all() or not np.isfinite(st).all() or (ids < 0).any():
        raise ValueError("invalid candidate fixes")
    if (st > t).any():
        raise ValueError("R7 resampling selected a future GPS source; refusing silent repair")
    if blocked_intervals is None:
        blocked_intervals = [(t[a], t[b] if b < len(t) else np.inf)
                             for a, b in false_runs(available)]
    forbidden_source = np.zeros(len(t), dtype=bool)
    denied_grid = np.zeros(len(t), dtype=bool)
    for a, b in blocked_intervals:
        if not a < b:
            raise ValueError("invalid denied time interval")
        forbidden_source |= (st >= a) & (st < b)
        denied_grid |= (t >= a) & (t < b)
    if not np.array_equal(denied_grid, ~available):
        raise ValueError("source denial intervals must match the final grid mask")
    held = np.empty_like(candidates)
    held_ids = np.empty(len(t), dtype=np.int64)
    held_st = np.empty(len(t), dtype=np.float64)
    accepted_at = np.empty(len(t), dtype=np.int64)
    updates = np.zeros(len(t), dtype=bool)
    last = None
    for i in range(len(t)):
        if available[i] and not forbidden_source[i]:
            if last is not None and (ids[i] < ids[last] or st[i] < st[last]):
                raise ValueError("selected source order reversed; refusing silent resampling change")
            if last is None or ids[i] != ids[last]:
                last = i
                updates[i] = True
            elif not np.array_equal(candidates[i], candidates[last]) or st[i] != st[last]:
                raise ValueError("same source index has inconsistent timestamp/position")
        if last is None:
            raise ValueError("no legal pre-outage fix; provide available GNSS history, never GT")
        held[i], held_ids[i], held_st[i], accepted_at[i] = candidates[last], ids[last], st[last], last
    return {"held_gnss": held, "availability_mask": available.copy(),
            "gnss_update_mask": updates, "held_source_index": held_ids,
            "held_source_timestamp_s": held_st, "accepted_at_grid_index": accepted_at,
            "fix_age_s": t - held_st,
            "rejected_buffered_source_mask": available & forbidden_source}


def constant_velocity_gnss(t, clean, history_s=5.0, min_span_s=1.0):
    """Frozen causal least-squares XYZ velocity from accepted pre-onset fixes.

    Unique source rows within the trailing history_s (source time), at least
    two distinct timestamps spanning min_span_s. Otherwise v=0. Anchor at the
    last legal position AND its source time, extrapolate only in mask=False.
    Link recovery returns to held-GNSS; no fit to GT or later samples.
    """
    t = _timeline(t)
    if history_s <= 0 or min_span_s <= 0:
        raise ValueError("positive velocity history/span required")
    held = clean["held_gnss"]
    st = clean["held_source_timestamp_s"]
    update = clean["gnss_update_mask"]
    pred = held.astype(np.float64).copy()
    records = []
    for a, b in false_runs(clean["availability_mask"]):
        if a == 0:
            raise ValueError("CV requires a legal pre-onset anchor")
        hist = np.flatnonzero(update & (np.arange(len(t)) < a)
                              & (st >= t[a] - history_s) & (st < t[a]))
        # An exported stream can contain multiple rows with the same timestamp.
        # Keep the last accepted row per timestamp; do not overweight duplicates.
        if len(hist):
            hist = hist[np.r_[np.diff(st[hist]) != 0, True]]
        velocity = np.zeros(3, dtype=np.float64)
        span = float(st[hist[-1]] - st[hist[0]]) if len(hist) else 0.0
        fallback = len(hist) < 2 or span < min_span_s
        if not fallback:
            x = st[hist] - st[hist].mean()
            y = held[hist].astype(np.float64)
            velocity = (x[:, None] * (y - y.mean(axis=0))).sum(axis=0) / (x @ x)
        pred[a:b] = held[a - 1] + (t[a:b] - st[a - 1])[:, None] * velocity
        records.append({"indices_half_open": [a, b], "velocity_m_s": velocity.tolist(),
                        "history_source_indices": clean["held_source_index"][hist].tolist(),
                        "history_grid_indices": hist.tolist(), "history_span_s": span,
                        "history_source_timestamps_s": st[hist].tolist(),
                        "anchor_source_index": int(clean["held_source_index"][a - 1]),
                        "anchor_source_timestamp_s": float(st[a - 1]),
                        "anchor_position_m": held[a - 1].tolist(),
                        "zero_velocity_fallback": bool(fallback)})
    return pred, records


def raw_errors(pred, gt):
    pred, gt = np.asarray(pred, dtype=np.float64), np.asarray(gt, dtype=np.float64)
    if pred.shape != gt.shape or pred.ndim != 2 or pred.shape[1] != 3:
        raise ValueError("pred and GT must be matching N x 3 arrays")
    if not np.isfinite(pred).all() or not np.isfinite(gt).all():
        raise ValueError("nonfinite prediction/GT")
    delta = pred - gt
    return {"error_xyz_m": delta, "error_3d_m": np.linalg.norm(delta, axis=1),
            "error_horizontal_m": np.linalg.norm(delta[:, :2], axis=1),
            "error_vertical_signed_m": delta[:, 2]}


def raw_metrics(pred, gt, available):
    """Sample-weighted RAW errors; no rigid alignment, fit, SD, or CI.

    All unavailable points for an outage scenario (one episode here); the whole
    common trace for control. Final errors always mean last unavailable sample,
    and are None for control. Horizontal=existing first two project axes.
    """
    errors = raw_errors(pred, gt)
    available = np.asarray(available, dtype=bool)
    if available.shape != (len(pred),):
        raise ValueError("metric mask shape mismatch")
    missing = np.flatnonzero(~available)
    chosen = missing if len(missing) else np.arange(len(pred))
    if not len(chosen):
        raise ValueError("empty metric support")
    d = errors["error_xyz_m"][chosen]
    row = {"metric_scope": "outage_only" if len(missing) else "whole_common_trace",
           "n_evaluated": len(chosen), "rmse_3d_m": float(np.sqrt(np.mean(np.sum(d * d, axis=1)))),
           "rmse_horizontal_m": float(np.sqrt(np.mean(np.sum(d[:, :2] ** 2, axis=1)))),
           "rmse_vertical_m": float(np.sqrt(np.mean(d[:, 2] ** 2))),
           "final_unavailable_error_3d_m": None, "final_unavailable_error_horizontal_m": None,
           "final_unavailable_abs_vertical_m": None}
    if len(missing):
        last = missing[-1]
        row.update(final_unavailable_error_3d_m=float(errors["error_3d_m"][last]),
                   final_unavailable_error_horizontal_m=float(errors["error_horizontal_m"][last]),
                   final_unavailable_abs_vertical_m=float(abs(errors["error_vertical_signed_m"][last])))
    return row
