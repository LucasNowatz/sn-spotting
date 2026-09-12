"""Station observation operator and error model (shared definition).

  1. bilinear interpolation of the field to the station position
  2. a station-specific sizing operator M_s = diag(eta_s) @ R_s, where R_s is
     a row-stochastic broadening matrix with a station-specific side weight
     and eta_s is that station's size-dependent counting efficiency
  3. seeded, time-correlated log-space errors with a per-deployment bias term,
     for vapour (41 x 41 blocks) and for counts (41*12 x 41*12 blocks)
  4. a detection limit and a quality flag for sparsely populated channels

Operators, error blocks and every constant here are published.  Only the
noise realisation and its seed stay in authoring/provenance.
"""

import numpy as np

N_BINS = 12
N_T = 41
DETECT_LIMIT = 5.0           # cm-3 post-operator, below this a channel is flagged
REPR_FLOOR_N = 0.06          # representation-error floor, counts (log)
REPR_FLOOR_V = 0.05          # representation-error floor, vapour (log)

# per-station instrument characteristics: (broadening side weight, D50 nm,
# efficiency steepness nm, vapour error scale, count error scale)
INSTRUMENTS = {
    "S1": dict(side=0.12, d50=2.00, k=0.28, ev=1.00, en=1.00),
    "S2": dict(side=0.16, d50=2.30, k=0.30, ev=1.10, en=1.15),
    "S3": dict(side=0.09, d50=1.80, k=0.25, ev=0.95, en=0.95),
    "S4": dict(side=0.14, d50=2.15, k=0.32, ev=1.20, en=1.05),
    "S5": dict(side=0.10, d50=1.90, k=0.26, ev=1.00, en=1.10),
    "W1": dict(side=0.18, d50=2.40, k=0.30, ev=1.15, en=1.20),
    "W2": dict(side=0.08, d50=1.75, k=0.24, ev=0.90, en=0.90),
}

# error model in natural-log units: R = sw^2 I + ss^2 AR1(rho) + sb^2 11^T
# over the 41 record times (vapour), and the same time structure Kronecker
# the identity over channels plus a bias shared by every channel and time of
# one deployment (counts)
ERR_VAPOUR = dict(sw=0.07, ss=0.06, rho=0.60, sb=0.06)
ERR_COUNTS = dict(sw=0.08, ss=0.06, rho=0.60, sb=0.07)


def bin_edges():
    return 1.5 * (15.0 / 1.5) ** (np.arange(N_BINS + 1) / N_BINS)


def bin_centres():
    e = bin_edges()
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
    D = bin_centres()
    return 1.0 / (1.0 + np.exp(-(D - d50) / k))


def sizing_operator(sid):
    ins = INSTRUMENTS[sid]
    return efficiency(ins["d50"], ins["k"])[:, None] * broadening(ins["side"])


def ar1(n, rho):
    i = np.arange(n)
    return rho ** np.abs(i[:, None] - i[None, :])


def vapour_block(sid):
    e = ERR_VAPOUR
    r = (e["sw"] ** 2) * np.eye(N_T) + (e["ss"] ** 2) * ar1(N_T, e["rho"]) \
        + (e["sb"] ** 2) * np.ones((N_T, N_T))
    return r * INSTRUMENTS[sid]["ev"] ** 2


def counts_block(sid):
    """(41*12) x (41*12) block, index = time * 12 + channel."""
    e = ERR_COUNTS
    tt = (e["sw"] ** 2) * np.eye(N_T) + (e["ss"] ** 2) * ar1(N_T, e["rho"])
    r = np.kron(tt, np.eye(N_BINS)) + (e["sb"] ** 2) * np.ones(
        (N_T * N_BINS, N_T * N_BINS))
    return r * INSTRUMENTS[sid]["en"] ** 2


def bilinear_weights(x_c, y_c, sx, sy):
    dx = x_c[1] - x_c[0]
    dy = y_c[1] - y_c[0]
    fi = (sx - x_c[0]) / dx
    fj = (sy - y_c[0]) / dy
    i0 = int(np.clip(np.floor(fi), 0, len(x_c) - 2))
    j0 = int(np.clip(np.floor(fj), 0, len(y_c) - 2))
    a = np.clip(fi - i0, 0.0, 1.0)
    b = np.clip(fj - j0, 0.0, 1.0)
    return (j0, i0), np.array([[(1 - b) * (1 - a), (1 - b) * a],
                               [b * (1 - a), b * a]])


def sample_field(field, jj_ii, w):
    j0, i0 = jj_ii
    if field.ndim == 2:
        return float((field[j0:j0 + 2, i0:i0 + 2] * w).sum())
    return (field[:, j0:j0 + 2, i0:i0 + 2] * w[None]).sum(axis=(1, 2))


def apply_operator(n_true, sid):
    """Station sizing operator applied to (time, bin) true counts."""
    return n_true @ sizing_operator(sid).T


def add_noise(vapour, counts, sid, rng):
    """Seeded correlated log-space errors.  Returns (vapour, counts, qc)."""
    Lv = np.linalg.cholesky(vapour_block(sid))
    ev = Lv @ rng.standard_normal(N_T)
    v = vapour * np.exp(ev)
    Ln = np.linalg.cholesky(counts_block(sid))
    en = (Ln @ rng.standard_normal(N_T * N_BINS)).reshape(N_T, N_BINS)
    qc = (counts >= DETECT_LIMIT).astype(np.int8)
    n = np.where(qc > 0, counts * np.exp(en), 0.0)
    return v, n, qc


def sigma_vapour_log(sid):
    """Marginal 1-sigma in log space, measurement plus representation."""
    e = ERR_VAPOUR
    m = (e["sw"] ** 2 + e["ss"] ** 2 + e["sb"] ** 2) * INSTRUMENTS[sid]["ev"] ** 2
    return float(np.sqrt(m + REPR_FLOOR_V ** 2))


def sigma_counts_log(sid):
    e = ERR_COUNTS
    m = (e["sw"] ** 2 + e["ss"] ** 2 + e["sb"] ** 2) * INSTRUMENTS[sid]["en"] ** 2
    return float(np.sqrt(m + REPR_FLOOR_N ** 2))
