"""Station instruments and the error model (shared, published).

  1. bilinear interpolation to the station position
  2. a station-specific operator M = diag(eta) @ R: counting efficiency eta
     per channel and row-stochastic broadening R with a station-specific width
  3. Gaussian errors in natural-log space with white, AR(1) and
     per-deployment bias terms; for counts the bias is shared by every channel
     and time of one deployment
  4. a detection limit and a quality flag

Only the error realisation and its seed stay on the authoring side.
"""

import numpy as np

import scenarios as SC

N_BINS = 12
N_T = SC.N_OBS
DETECT_LIMIT = 4.0           # cm-3 post-operator
REPR_FLOOR_N = 0.06
REPR_FLOOR_V = 0.05
LOG_FLOOR_N = 0.05           # cm-3 before a log
LOG_FLOOR_V = 0.001          # ug m-3 before a log

INSTRUMENTS = {
    "S1": dict(side=0.11, d50=1.95, k=0.27, ev=1.00, en=1.05),
    "S2": dict(side=0.17, d50=2.35, k=0.31, ev=1.12, en=1.18),
    "S3": dict(side=0.08, d50=1.75, k=0.24, ev=0.94, en=0.92),
    "S4": dict(side=0.15, d50=2.20, k=0.33, ev=1.22, en=1.02),
    "S5": dict(side=0.10, d50=1.85, k=0.25, ev=1.02, en=1.12),
    "S6": dict(side=0.13, d50=2.05, k=0.29, ev=1.06, en=0.98),
    "W1": dict(side=0.19, d50=2.45, k=0.30, ev=1.16, en=1.22),
    "W2": dict(side=0.09, d50=1.70, k=0.23, ev=0.90, en=0.88),
}

ERR_VAPOUR = dict(sw=0.06, ss=0.07, rho=0.55, sb=0.065)
ERR_COUNTS = dict(sw=0.075, ss=0.065, rho=0.65, sb=0.08)


def bin_centres():
    e = 1.5 * (15.0 / 1.5) ** (np.arange(N_BINS + 1) / N_BINS)
    return np.sqrt(e[:-1] * e[1:])


def broadening(side):
    R = np.zeros((N_BINS, N_BINS))
    for j in range(N_BINS):
        R[j, j] = 1.0
        if j > 0:
            R[j, j - 1] = side
        if j < N_BINS - 1:
            R[j, j + 1] = side
    return R / R.sum(axis=1, keepdims=True)


def efficiency(d50, k):
    return 1.0 / (1.0 + np.exp(-(bin_centres() - d50) / k))


def operator(sid):
    ins = INSTRUMENTS[sid]
    return efficiency(ins["d50"], ins["k"])[:, None] * broadening(ins["side"])


def ar1(n, rho):
    i = np.arange(n)
    return rho ** np.abs(i[:, None] - i[None, :])


def vapour_block(sid):
    e = ERR_VAPOUR
    r = e["sw"] ** 2 * np.eye(N_T) + e["ss"] ** 2 * ar1(N_T, e["rho"]) \
        + e["sb"] ** 2 * np.ones((N_T, N_T))
    return r * INSTRUMENTS[sid]["ev"] ** 2


def counts_block(sid):
    e = ERR_COUNTS
    tt = e["sw"] ** 2 * np.eye(N_T) + e["ss"] ** 2 * ar1(N_T, e["rho"])
    r = np.kron(tt, np.eye(N_BINS)) + e["sb"] ** 2 * np.ones((N_T * N_BINS,) * 2)
    return r * INSTRUMENTS[sid]["en"] ** 2


def bilinear(x_c, y_c, sx, sy):
    dx = x_c[1] - x_c[0]; dy = y_c[1] - y_c[0]
    fi = (sx - x_c[0]) / dx; fj = (sy - y_c[0]) / dy
    i0 = int(np.clip(np.floor(fi), 0, len(x_c) - 2))
    j0 = int(np.clip(np.floor(fj), 0, len(y_c) - 2))
    a = np.clip(fi - i0, 0.0, 1.0); b = np.clip(fj - j0, 0.0, 1.0)
    return (j0, i0), np.array([[(1 - b) * (1 - a), (1 - b) * a],
                               [b * (1 - a), b * a]])


def sample(field, jj_ii, w):
    """Bilinear sample of the trailing two axes of a field."""
    j0, i0 = jj_ii
    return (field[..., j0:j0 + 2, i0:i0 + 2] * w).sum(axis=(-2, -1))


def add_errors(vapour, counts, sid, rng):
    Lv = np.linalg.cholesky(vapour_block(sid))
    v = vapour * np.exp(Lv @ rng.standard_normal(N_T))
    Ln = np.linalg.cholesky(counts_block(sid))
    en = (Ln @ rng.standard_normal(N_T * N_BINS)).reshape(N_T, N_BINS)
    qc = (counts >= DETECT_LIMIT).astype(np.int8)
    n = np.where(qc > 0, counts * np.exp(en), 0.0)
    return v, n, qc


def sigma_vapour(sid):
    e = ERR_VAPOUR
    m = (e["sw"] ** 2 + e["ss"] ** 2 + e["sb"] ** 2) * INSTRUMENTS[sid]["ev"] ** 2
    return float(np.sqrt(m + REPR_FLOOR_V ** 2))


def sigma_counts(sid):
    e = ERR_COUNTS
    m = (e["sw"] ** 2 + e["ss"] ** 2 + e["sb"] ** 2) * INSTRUMENTS[sid]["en"] ** 2
    return float(np.sqrt(m + REPR_FLOOR_N ** 2))
