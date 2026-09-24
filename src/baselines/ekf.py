"""Loosely-coupled strapdown INS + EKF baseline (multiplicative EKF / MEKF).

State: position (3, ENU), velocity (3, ENU), attitude, accel bias (3),
gyro bias (3). Attitude is NOT stored as an accumulated small-angle
vector -- that is only a valid rotation for infinitesimal angles, and
naive Euler-integrating it over a real flight (minutes, real turns)
makes the "rotation" matrix non-orthogonal and the filter diverge
without bound, especially through an extended GPS outage with nothing
to correct it. The nominal attitude is instead a proper rotation matrix
R (body->ENU), propagated via the matrix exponential (Rodrigues'
formula); only the KALMAN CORRECTION each step is small-angle -- folded
into R via right-multiplication, then implicitly reset to zero. This is
the standard MEKF pattern, not a novel contribution: the credible
classical baseline a reviewer expects.
"""
import warnings

import numpy as np

# Apple Accelerate's BLAS raises spurious "invalid value encountered in
# matmul" warnings on some matrix sizes even when the result is finite and
# correct (macOS-only quirk; OpenBLAS/MKL on Kaggle's Linux boxes don't do
# this). Silenced here rather than in every call site.
warnings.filterwarnings("ignore", message=".*encountered in matmul.*")

IDX_POS = slice(0, 3)
IDX_VEL = slice(3, 6)
IDX_ATT = slice(6, 9)   # error-state only: small-angle correction, radians
IDX_ABIAS = slice(9, 12)
IDX_GBIAS = slice(12, 15)
STATE_DIM = 15

GRAVITY = np.array([0.0, 0.0, -9.80665])


def _skew(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def _so3_exp(w):
    """Rodrigues' formula: exponential map from a rotation vector to SO(3).
    Used for BOTH the nominal attitude's kinematic propagation and for
    folding a Kalman correction into it -- never accumulated as a raw
    vector, which is only valid infinitesimally.
    """
    theta = np.linalg.norm(w)
    K = _skew(w)
    if theta < 1e-8:
        return np.eye(3) + K
    axis_K = K / theta
    return np.eye(3) + np.sin(theta) * axis_K + (1 - np.cos(theta)) * (axis_K @ axis_K)


class StrapdownEKF:
    def __init__(self, q_accel=0.05, q_gyro=0.01, q_bias=1e-6, r_gps=2.0, estimate_bias=True):
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.R = np.eye(3)  # nominal attitude, body -> ENU
        self.abias = np.zeros(3)
        self.gbias = np.zeros(3)
        self.P = np.eye(STATE_DIM) * 1e-3
        self.q_accel = q_accel   # accel noise density (m/s^2 / sqrt(Hz))
        self.q_gyro = q_gyro     # gyro noise density (rad/s / sqrt(Hz))
        self.q_bias = q_bias     # bias random-walk
        self.R_gps = np.eye(3) * r_gps ** 2
        # ablation: bias states are held at zero (never estimated), the
        # rest of the filter is unchanged -- isolates what online bias
        # correction buys over a fixed zero-bias strapdown/EKF.
        self.estimate_bias = estimate_bias

    def set_initial_position(self, pos0):
        self.pos = np.array(pos0, dtype=float)

    def predict(self, accel_body, gyro_body, dt):
        """One strapdown mechanization step. accel_body/gyro_body: (3,) raw IMU."""
        f = accel_body - self.abias
        accel_enu = self.R @ f + GRAVITY
        omega = gyro_body - self.gbias

        self.pos = self.pos + self.vel * dt + 0.5 * accel_enu * dt ** 2
        self.vel = self.vel + accel_enu * dt
        self.R = self.R @ _so3_exp(omega * dt)

        F = np.eye(STATE_DIM)
        F[IDX_POS, IDX_VEL] = np.eye(3) * dt
        F[IDX_VEL, IDX_ATT] = -self.R @ _skew(f) * dt
        F[IDX_VEL, IDX_ABIAS] = -self.R * dt
        F[IDX_ATT, IDX_GBIAS] = -np.eye(3) * dt

        Q = np.zeros((STATE_DIM, STATE_DIM))
        Q[IDX_VEL, IDX_VEL] = np.eye(3) * (self.q_accel ** 2) * dt
        Q[IDX_ATT, IDX_ATT] = np.eye(3) * (self.q_gyro ** 2) * dt
        Q[IDX_ABIAS, IDX_ABIAS] = np.eye(3) * self.q_bias * dt
        Q[IDX_GBIAS, IDX_GBIAS] = np.eye(3) * self.q_bias * dt

        self.P = F @ self.P @ F.T + Q
        if not self.estimate_bias:
            self.abias[:] = 0.0
            self.gbias[:] = 0.0

    def update_gps(self, gps_pos):
        H = np.zeros((3, STATE_DIM))
        H[:, IDX_POS] = np.eye(3)
        y = gps_pos - self.pos
        S = H @ self.P @ H.T + self.R_gps
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ y

        self.pos = self.pos + dx[IDX_POS]
        self.vel = self.vel + dx[IDX_VEL]
        self.R = self.R @ _so3_exp(dx[IDX_ATT])  # fold small-angle correction in, then it's gone
        self.abias = self.abias + dx[IDX_ABIAS]
        self.gbias = self.gbias + dx[IDX_GBIAS]

        self.P = (np.eye(STATE_DIM) - K @ H) @ self.P
        if not self.estimate_bias:
            self.abias[:] = 0.0
            self.gbias[:] = 0.0

    @property
    def position(self):
        return self.pos.copy()


def run_ekf(accel, gyro, gps, gps_mask, dt, gps_pos0=None, estimate_bias=True):
    """Run the filter over a full sequence.

    accel, gyro: (T, 3) body-frame IMU at the common time grid.
    gps: (T, 3) ENU GPS position, valid only where gps_mask[t] is True
         (values elsewhere are ignored, e.g. NaN or stale-hold).
    dt: scalar or (T,) timestep(s).
    estimate_bias: False runs the "no bias term" ablation (Table 3).
    Returns pred_pos: (T, 3).
    """
    T = len(accel)
    dt_arr = np.full(T, dt) if np.isscalar(dt) else dt

    ekf = StrapdownEKF(estimate_bias=estimate_bias)
    first_valid = int(np.argmax(gps_mask)) if gps_mask.any() else 0
    ekf.set_initial_position(gps_pos0 if gps_pos0 is not None else gps[first_valid])

    out = np.zeros((T, 3))
    for t in range(T):
        ekf.predict(accel[t], gyro[t], dt_arr[t])
        if gps_mask[t]:
            ekf.update_gps(gps[t])
        out[t] = ekf.position
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T, dt = 600, 0.02  # 12 s at 50 Hz
    t = np.arange(T) * dt

    true_pos = np.stack([2.0 * t, 0.3 * t ** 1.5, 0.5 * np.sin(t)], axis=1)
    true_vel = np.gradient(true_pos, dt, axis=0)
    true_accel_enu = np.gradient(true_vel, dt, axis=0) - GRAVITY

    accel_body = true_accel_enu + rng.normal(0, 0.02, (T, 3))
    gyro_body = rng.normal(0, 0.005, (T, 3))  # near-zero true angular rate here

    gps_mask = np.zeros(T, dtype=bool)
    gps_mask[::25] = True  # ~2 Hz GPS on a 50 Hz IMU grid
    gps = true_pos + rng.normal(0, 1.0, (T, 3))

    pred = run_ekf(accel_body, gyro_body, gps, gps_mask, dt, gps_pos0=true_pos[0])
    pred_nobias = run_ekf(accel_body, gyro_body, gps, gps_mask, dt, gps_pos0=true_pos[0],
                           estimate_bias=False)

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from metrics import absolute_trajectory_error

    ate, _ = absolute_trajectory_error(pred, true_pos, align=False)
    ate_nobias, _ = absolute_trajectory_error(pred_nobias, true_pos, align=False)
    assert np.isfinite(ate) and ate < 5.0, f"EKF diverged, ATE={ate}"
    assert np.isfinite(ate_nobias)

    # A real regression target for today's bug: a long GPS-outage stretch
    # (no correction at all) must NOT blow up. Naive small-angle attitude
    # accumulation diverged to hundreds of km here; a proper SO(3)
    # propagation stays within a physically plausible dead-reckoning drift.
    gps_mask_outage = np.zeros(T, dtype=bool)
    gps_mask_outage[:50] = True  # a short lock-on, then a long outage
    pred_outage = run_ekf(accel_body, gyro_body, gps, gps_mask_outage, dt, gps_pos0=true_pos[0])
    drift = np.linalg.norm(pred_outage[-1] - true_pos[-1])
    assert np.isfinite(drift) and drift < 1000.0, (
        f"EKF diverged under a long outage: final drift={drift:.1f} m "
        f"(physically implausible for a {T*dt:.0f}s / {np.linalg.norm(true_pos[-1]):.0f}m-scale run)"
    )

    print(f"ekf ok | ATE={ate:.4f} m (bias-estimating) / {ate_nobias:.4f} m (no-bias ablation) "
          f"over {T} steps, {gps_mask.sum()} GPS updates | outage-drift={drift:.2f} m")
