"""The published Gaussian prior on the twelve log-parameters.

Every number here is written to environment/data/priors.json.  The prior is
in natural-log coordinates of the physical values (or of the dimensionless
multipliers), with a non-diagonal covariance: the three regional source
strengths share a nucleation-parameterisation calibration term, the yield
scale and the vapour lifetime come from one chamber-derived fit and are
anticorrelated, and the growth coefficient and its temperature sensitivity
were fitted together.  An inversion that treats B as diagonal gets a
different, wrong posterior.
"""

import numpy as np

PARAM_IDS = ["log_Kh", "log_we", "log_tauv", "log_Yorg", "log_betaorg",
             "log_Heff", "log_taup", "log_sA", "log_sB", "log_sC",
             "log_s_event", "log_q_event"]
PHYSICAL = ["kh", "we", "tauv", "yorg", "betaorg", "heff", "taup",
            "sA", "sB", "sC", "s_ev", "q_ev"]
UNITS = ["m2 s-1", "m s-1", "h", "1", "nm h-1 per ug m-3", "kJ mol-1", "h",
         "1", "1", "1", "1", "1"]
NOMINAL = np.array([6500.0, 0.006, 2.5, 1.0, 1.0, 70.0, 12.0,
                    1.0, 1.0, 1.0, 1.0, 1.0])
MU = np.log(NOMINAL)
SIGMA = np.array([0.35, 0.45, 0.40, 0.30, 0.40, 0.30, 0.45,
                  0.35, 0.35, 0.35, 0.35, 0.30])
# hard bounds on the physical values
LOWER_PHYS = np.array([2000.0, 0.001, 0.7, 0.3, 0.3, 25.0, 3.0,
                       0.3, 0.3, 0.3, 0.35, 0.35])
UPPER_PHYS = np.array([20000.0, 0.03, 8.0, 3.0, 3.5, 160.0, 48.0,
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
    c = correlation()
    return (SIGMA[:, None] * SIGMA[None, :]) * c


def to_physical(theta):
    return {k: float(np.exp(v)) for k, v in zip(PHYSICAL, np.asarray(theta))}


def to_theta(p):
    return np.array([np.log(p[k]) for k in PHYSICAL])


def as_json():
    return dict(
        param_ids=PARAM_IDS,
        physical_names=PHYSICAL,
        units=UNITS,
        nominal=[float(x) for x in NOMINAL],
        mu=[float(x) for x in MU],
        sigma=[float(x) for x in SIGMA],
        lower=[float(x) for x in LOWER],
        upper=[float(x) for x in UPPER],
        correlation=[[float(x) for x in row] for row in correlation()],
        covariance=[[float(x) for x in row] for row in covariance()],
        note=("All parameters are natural logarithms of the physical value in "
              "the stated unit, or of a dimensionless multiplier. The prior is "
              "Gaussian in these log coordinates with mean mu and covariance "
              "'covariance'; it is not diagonal. Bounds are hard limits in the "
              "same log coordinates. The truth is one draw from this prior."),
    )
