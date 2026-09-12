"""Scored metrics (sealed).  Truth comes from the 2 km generator."""
import numpy as np

COUNT_FLOOR = 4.0
Z90 = 1.6448536269514722


def log_nrmse(pred, true, sigma):
    r = (np.log(np.maximum(pred, 1e-9)) - np.log(np.maximum(true, 1e-9))) / np.maximum(sigma, 1e-12)
    return float(np.sqrt(np.mean(r ** 2)))


def log_rmse_counts(pred, true, sigma, mask):
    if not mask.any():
        return np.inf
    r = (np.log(np.maximum(pred, 0.0) + COUNT_FLOOR) - np.log(np.maximum(true, 0.0) + COUNT_FLOOR)) \
        / np.maximum(sigma, 1e-12)
    return float(np.sqrt(np.mean(r[mask] ** 2)))


def weighted_diameter(counts, centres):
    w = np.maximum(counts, 0.0); tot = w.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.exp((w * np.log(centres)).sum(axis=-1) / np.where(tot > 0, tot, 1))
    return np.where(tot > 0, d, np.nan)


def diameter_mae(pred, true, centres, min_total):
    dp, dt = weighted_diameter(pred, centres), weighted_diameter(true, centres)
    m = (true.sum(axis=-1) >= min_total) & np.isfinite(dp) & np.isfinite(dt)
    if not m.any():
        return np.inf, 0
    return float(np.mean(np.abs(dp[m] - dt[m]))), int(m.sum())


def coverage(y, mu, sd, z=Z90):
    return float(np.mean(np.abs(y - mu) <= z * sd))


def log_score(y, mu, sd):
    sd = np.asarray(sd, float)
    return float(np.mean(0.5 * np.log(2.0 * np.pi * sd ** 2) + (y - mu) ** 2 / (2.0 * sd ** 2)))
