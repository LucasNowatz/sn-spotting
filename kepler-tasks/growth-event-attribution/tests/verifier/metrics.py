"""Scored metrics for the growth-event verifier (sealed).

Every metric compares against truth produced by the seeded generator at
2.5 km, never against anything the reference solution computed.
Uncertainties combine the published error model with a representation-error
floor, so a correct solution on the public 5 km grid is not penalised for not
being the generator.
"""

import numpy as np

COUNT_FLOOR = 5.0          # cm-3, metric floor, published
N_BINS = 12
Z90 = 1.6448536269514722


def log_nrmse(pred, true, sigma):
    """Uncertainty-normalised RMSE in natural-log space."""
    r = (np.log(np.maximum(pred, 1e-9)) - np.log(np.maximum(true, 1e-9))) \
        / np.maximum(sigma, 1e-12)
    return float(np.sqrt(np.mean(r ** 2)))


def log_rmse_counts(pred, true, sigma, mask):
    """Weighted RMSE in natural-log count space over valid channels."""
    if not mask.any():
        return np.inf
    r = (np.log(np.maximum(pred, 0.0) + COUNT_FLOOR)
         - np.log(np.maximum(true, 0.0) + COUNT_FLOOR)) / np.maximum(sigma, 1e-12)
    return float(np.sqrt(np.mean(r[mask] ** 2)))


def count_weighted_diameter(counts, centres):
    w = np.maximum(counts, 0.0)
    tot = w.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.exp((w * np.log(centres)).sum(axis=-1) / np.where(tot > 0, tot, 1))
    return np.where(tot > 0, d, np.nan)


def diameter_mae(pred, true, centres, min_total):
    dp = count_weighted_diameter(pred, centres)
    dt = count_weighted_diameter(true, centres)
    m = (true.sum(axis=-1) >= min_total) & np.isfinite(dp) & np.isfinite(dt)
    if not m.any():
        return np.inf, 0
    return float(np.mean(np.abs(dp[m] - dt[m]))), int(m.sum())


def age_mae(pred, true):
    m = np.isfinite(true) & np.isfinite(pred)
    if not m.any():
        return np.inf, 0
    return float(np.mean(np.abs(pred[m] - true[m]))), int(m.sum())


def source_l1(pred, true):
    m = np.isfinite(true).all(axis=1) & np.isfinite(pred).all(axis=1)
    if not m.any():
        return np.inf, 0
    return float(np.mean(np.abs(pred[m] - true[m]).sum(axis=1))), int(m.sum())


def coverage(y_log, mu_log, sd, z=Z90):
    return float(np.mean(np.abs(y_log - mu_log) <= z * sd))


def gaussian_log_score(y_log, mu_log, sd):
    sd = np.asarray(sd, float)
    return float(np.mean(0.5 * np.log(2.0 * np.pi * sd ** 2)
                         + (y_log - mu_log) ** 2 / (2.0 * sd ** 2)))
