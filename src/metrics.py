"""ATE / RPE trajectory metrics, Umeyama-aligned (TUM/SLAM convention).

All functions take/return numpy arrays. Position arrays are (T, 3) in a
local ENU frame. No GPU, no external deps beyond numpy.
"""
import numpy as np


def umeyama_alignment(source, target, with_scale=False):
    """Least-squares rigid (optionally similarity) alignment: R, t, s such
    that s * R @ source[i] + t ~= target[i]. Horn/Umeyama closed form.
    """
    assert source.shape == target.shape
    n, dim = source.shape

    mu_src = source.mean(axis=0)
    mu_tgt = target.mean(axis=0)
    src_c = source - mu_src
    tgt_c = target - mu_tgt

    cov = (tgt_c.T @ src_c) / n
    u, d, vt = np.linalg.svd(cov)

    s_sign = np.ones(dim)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_sign[-1] = -1

    r = u @ np.diag(s_sign) @ vt

    if with_scale:
        var_src = (src_c ** 2).sum() / n
        scale = (d * s_sign).sum() / var_src
    else:
        scale = 1.0

    t = mu_tgt - scale * r @ mu_src
    return r, t, scale


def apply_alignment(positions, r, t, scale):
    return scale * (positions @ r.T) + t


def absolute_trajectory_error(pred, gt, align=True, with_scale=False):
    """ATE: RMSE of per-point translational error after alignment.

    pred, gt: (T, 3) arrays, same length, temporally corresponding.
    Returns (ate_rmse, aligned_pred).
    """
    assert pred.shape == gt.shape, f"{pred.shape} vs {gt.shape}"
    if align:
        r, t, s = umeyama_alignment(pred, gt, with_scale=with_scale)
        aligned = apply_alignment(pred, r, t, s)
    else:
        aligned = pred
    err = np.linalg.norm(aligned - gt, axis=1)
    return float(np.sqrt((err ** 2).mean())), aligned


def relative_pose_error(pred, gt, delta):
    """RPE: RMSE of translational drift over a fixed-length window,
    computed on ALIGNED trajectories (call after ATE alignment, or pass
    already-aligned pred).

    delta: number of timesteps between the two ends of each segment.
    Returns rpe_rmse (float). len(pred) must exceed delta.
    """
    assert pred.shape == gt.shape
    n = len(pred)
    if n <= delta:
        raise ValueError(f"trajectory too short ({n}) for delta={delta}")
    pred_delta = pred[delta:] - pred[:-delta]
    gt_delta = gt[delta:] - gt[:-delta]
    err = np.linalg.norm(pred_delta - gt_delta, axis=1)
    return float(np.sqrt((err ** 2).mean()))


def per_axis_rmse(pred, gt):
    """RMSE per ENU axis, no alignment (diagnostic, not the headline metric)."""
    assert pred.shape == gt.shape
    err = pred - gt
    return {
        axis: float(np.sqrt((err[:, i] ** 2).mean()))
        for i, axis in enumerate(("E", "N", "U"))
    }


def trajectory_metrics(pred, gt, rpe_delta=10, align=True):
    """Convenience bundle used by evaluate.py / report.py."""
    ate, aligned = absolute_trajectory_error(pred, gt, align=align)
    rpe = relative_pose_error(aligned, gt, delta=rpe_delta)
    axis_rmse = per_axis_rmse(aligned, gt)
    return {"ATE": ate, "RPE": rpe, **{f"RMSE_{k}": v for k, v in axis_rmse.items()}}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    t = np.linspace(0, 20, 400)
    gt = np.stack([t, 0.5 * t, 2.0 * np.sin(t / 3)], axis=1)

    # rotated/translated/noisy "prediction" of the same trajectory
    theta = 0.15
    rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])
    noisy = gt @ rot.T + np.array([5.0, -2.0, 0.3]) + rng.normal(0, 0.05, gt.shape)

    m = trajectory_metrics(noisy, gt, rpe_delta=10)
    assert m["ATE"] < 0.2, m  # alignment should remove the rigid transform, leave only noise
    assert all(np.isfinite(v) for v in m.values())

    ate_unaligned, _ = absolute_trajectory_error(noisy, gt, align=False)
    assert ate_unaligned > m["ATE"], "alignment should reduce error vs raw"

    print(f"metrics ok | aligned ATE={m['ATE']:.4f} RPE={m['RPE']:.4f} "
          f"unaligned ATE={ate_unaligned:.4f}")
