"""The published Gaussian prior on the twelve log-parameters."""

import numpy as np

PARAM_IDS = ["log_Kh", "log_we", "log_tauv", "log_Yorg", "log_betaorg",
             "log_Heff", "log_taup", "log_sA", "log_sB", "log_sC",
             "log_s_event", "log_q_event"]
PHYSICAL = ["kh", "we", "tauv", "yorg", "betaorg", "heff", "taup",
            "sA", "sB", "sC", "s_ev", "q_ev"]
UNITS = ["m2 s-1", "m s-1", "h", "1", "nm h-1 per ug m-3", "kJ mol-1", "h",
         "1", "1", "1", "1", "1"]
NOMINAL = np.array([6000.0, 0.007, 2.2, 1.0, 0.9, 65.0, 14.0,
                    1.0, 1.0, 1.0, 1.0, 1.0])
MU = np.log(NOMINAL)
SIGMA = np.array([0.35, 0.45, 0.40, 0.30, 0.40, 0.30, 0.45,
                  0.35, 0.35, 0.35, 0.35, 0.30])
LOWER_PHYS = np.array([1800.0, 0.001, 0.6, 0.3, 0.25, 22.0, 3.0,
                       0.3, 0.3, 0.3, 0.35, 0.35])
UPPER_PHYS = np.array([18000.0, 0.03, 8.0, 3.0, 3.2, 150.0, 50.0,
                       3.3, 3.3, 3.3, 2.9, 2.9])
LOWER = np.log(LOWER_PHYS)
UPPER = np.log(UPPER_PHYS)


def correlation():
    c = np.eye(len(PARAM_IDS))
    for a in (7, 8, 9):
        for b in (7, 8, 9):
            if a != b:
                c[a, b] = 0.40
    c[2, 3] = c[3, 2] = -0.30
    c[4, 5] = c[5, 4] = 0.30
    return c


def covariance():
    return (SIGMA[:, None] * SIGMA[None, :]) * correlation()


def to_physical(theta):
    return {k: float(np.exp(v)) for k, v in zip(PHYSICAL, np.asarray(theta))}


def as_json():
    return dict(
        param_ids=PARAM_IDS, physical_names=PHYSICAL, units=UNITS,
        nominal=[float(x) for x in NOMINAL],
        mu=[float(x) for x in MU], sigma=[float(x) for x in SIGMA],
        lower=[float(x) for x in LOWER], upper=[float(x) for x in UPPER],
        correlation=[[float(x) for x in r] for r in correlation()],
        covariance=[[float(x) for x in r] for r in covariance()],
        note=("Natural logarithms of the physical value in the stated unit, "
              "or of a dimensionless multiplier. Gaussian prior with mean mu "
              "and the given (non-diagonal) covariance; bounds are hard "
              "limits in the same coordinates. The truth is one draw."))
