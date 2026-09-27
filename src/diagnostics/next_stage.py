"""Numerical diagnostics only; no training and no changes to the R7 pipeline."""
import numpy as np


def rmse_xyz(pred, target):
    e = np.asarray(pred, float) - np.asarray(target, float)
    if not len(e):
        return {k: None for k in ('rmse_3d_m', 'rmse_horizontal_m', 'rmse_vertical_m')}
    return {'rmse_3d_m': float(np.sqrt(np.mean(np.sum(e*e, axis=1)))),
            'rmse_horizontal_m': float(np.sqrt(np.mean(np.sum(e[:, :2]**2, axis=1)))),
            'rmse_vertical_m': float(np.sqrt(np.mean(e[:, 2]**2)))}


def fit_frame(x, y, kind):
    """Return column-vector convention y = scale * R @ x + translation.

    caller must explicitly identify fitting/evaluation split. A validation fit
    is an in-sample oracle diagnostic, never an independent evaluation.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    mx, my = x.mean(0), y.mean(0)
    rotation, scale = np.eye(3), 1.
    if kind == 'raw':
        translation = np.zeros(3)
    else:
        if kind != 'translation':
            dimensions = 2 if kind == 'se2_vertical' else 3
            a, b = (x-mx)[:, :dimensions], (y-my)[:, :dimensions]
            u, singular, vt = np.linalg.svd(np.einsum('ni,nj->ij',a,b))
            sign = np.ones(dimensions)
            sign[-1] = np.linalg.det(vt.T @ u.T)
            rotation[:dimensions, :dimensions] = vt.T @ np.diag(sign) @ u.T
            if kind == 'similarity_diagnostic':
                scale = float((singular*sign).sum() / np.sum(a*a))
            elif kind not in ('se2_vertical', 'se3'):
                raise ValueError(kind)
        translation = my-scale*(rotation @ mx)
    return {'kind': kind, 'rotation': rotation.tolist(), 'translation_m': translation.tolist(),
            'scale': scale, 'yaw_deg': float(np.degrees(np.arctan2(rotation[1,0],rotation[0,0]))),
            'rotation_angle_deg': float(np.degrees(np.arccos(np.clip((np.trace(rotation)-1)/2,-1,1))))}


def apply_frame(x, fitted):
    return fitted['scale'] * np.einsum('ni,ji->nj',np.asarray(x),np.array(fitted['rotation'])) + fitted['translation_m']


def utm32_wgs84(lat_deg, lon_deg):
    """WGS84 transverse Mercator series, zone 32N, restricted to Zurich.

    This analytic diagnostic is cross-checked against the dataset's UTM GPS
    columns. V2 notebook uses pyproj for production projection instead.
    """
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    a, f, k = 6378137., 1/298.257223563, .9996
    e2 = f*(2-f)
    ep2 = e2/(1-e2)
    n = a/np.sqrt(1-e2*np.sin(lat)**2)
    t = np.tan(lat)**2
    c = ep2*np.cos(lat)**2
    aa = np.cos(lat)*(lon-np.deg2rad(9.))
    m = a*((1-e2/4-3*e2**2/64-5*e2**3/256)*lat
           -(3*e2/8+3*e2**2/32+45*e2**3/1024)*np.sin(2*lat)
           +(15*e2**2/256+45*e2**3/1024)*np.sin(4*lat)
           -35*e2**3/3072*np.sin(6*lat))
    east = 500000+k*n*(aa+(1-t+c)*aa**3/6+(5-18*t+t*t+72*c-58*ep2)*aa**5/120)
    north = k*(m+n*np.tan(lat)*(aa*aa/2+(5-t+9*c+4*c*c)*aa**4/24
                            +(61-58*t+t*t+600*c-330*ep2)*aa**6/720))
    return np.column_stack((east, north))


def correction_stats(correction, t, chosen, window=192):
    c = np.asarray(correction, float)
    z = c[chosen]
    if not len(z):
        return {'n': 0}
    consecutive = chosen[1:] & chosen[:-1]
    d = np.diff(c, axis=0)[consecutive]
    dt = np.diff(t)[consecutive]
    lag = []
    for axis in range(3):
        x, y = c[:-1, axis][consecutive], c[1:, axis][consecutive]
        lag.append(float(np.corrcoef(x, y)[0,1]) if len(x)>1 and x.std()>1e-12 and y.std()>1e-12 else None)
    boundary = (np.arange(1,len(t)) % window == 0) & consecutive
    within = (np.arange(1,len(t)) % window != 0) & consecutive
    norm = np.linalg.norm(z,axis=1)
    delta_norm = np.linalg.norm(np.diff(c,axis=0),axis=1)
    return {'n': len(z), 'mean_xyz_m':z.mean(0).tolist(), 'std_xyz_m_ddof0':z.std(0).tolist(),
            'median_xyz_m':np.median(z,axis=0).tolist(), 'min_xyz_m':z.min(0).tolist(),
            'max_xyz_m':z.max(0).tolist(), 'rms_norm_m':float(np.sqrt(np.mean(norm**2))),
            'centered_rms_m':float(np.sqrt(np.mean(np.sum((z-z.mean(0))**2,axis=1)))),
            'lag1_pearson_xyz':lag,
            'adjacent_increment_rms_m':float(np.sqrt(np.mean(np.sum(d*d,axis=1)))) if len(d) else None,
            'adjacent_derivative_rms_m_s':float(np.sqrt(np.mean(np.sum((d/dt[:,None])**2,axis=1)))) if len(d) else None,
            'mean_window_boundary_jump_m':float(delta_norm[boundary].mean()) if boundary.any() else None,
            'mean_within_window_jump_m':float(delta_norm[within].mean()) if within.any() else None}


def pearson_features(correction, features, chosen):
    result={}
    for name, feature in features.items():
        a=np.asarray(feature)[chosen]
        result[name]=[]
        for axis in range(3):
            b=correction[chosen,axis]
            r=float(np.corrcoef(a,b)[0,1]) if len(a)>2 and np.std(a)>1e-12 and np.std(b)>1e-12 else None
            result[name].append(r)
    return result


def future_gnss_exposure(t, available, source_t, cutoff, window=192):
    """Potential post-recovery attention, not a causal effect measurement."""
    available=np.asarray(available,bool)
    exposed=np.zeros(len(t),bool)
    fresh=np.zeros(len(t),bool)
    for a in range(0,len(t),window):
        b=min(a+window,len(t))
        for i in range(a,b):
            if not available[i]:
                candidates=(np.arange(a,b)>i)&available[a:b]&(t[a:b]>=cutoff)
                exposed[i]=candidates.any()
                fresh[i]=(candidates&(source_t[a:b]>=cutoff)).any()
    return exposed,fresh


def permute_imu_windows(imu, window, mode, seed):
    """Paired accel+gyro row perturbation; no GNSS/GT input to this function."""
    x=np.asarray(imu).reshape(-1,window,6).copy()
    rng=np.random.default_rng(seed)
    indices=[]
    for w in range(len(x)):
        p=rng.permutation(window) if mode=='shuffle' else np.arange(window)[::-1]
        indices.append(p)
        x[w]=x[w,p]
    return x.reshape(-1,6), np.array(indices)
