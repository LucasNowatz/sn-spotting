#!/usr/bin/env python3
"""Whitened Gauss-Newton / Levenberg-Marquardt inversion (reference).

Phi(theta) = 0.5 (y - h(theta))^T R^-1 (y - h(theta))
           + 0.5 (theta - mu)^T B^-1 (theta - mu)

in log-parameter coordinates, with y the natural-log vapour and count
observations, R the block-diagonal published error covariance restricted to
the usable channels, and B the published non-diagonal prior covariance.
Whitening every block with its Cholesky factor turns the objective into a
plain least-squares problem whose Gauss-Newton Hessian J^T J is the inverse
posterior covariance.

Forward runs are farmed out over (episode, parameter) pairs, so one full
Jacobian over twelve parameters costs about thirteen episode-sets of forward
model time divided by the number of workers.
"""

import json
import os
import sys

import numpy as np
from joblib import Parallel, delayed
from netCDF4 import Dataset
from scipy.linalg import cholesky, solve_triangular

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as MO
import observe_ref as OB

PRED_FLOOR = 0.05          # cm-3 floor before taking the log of a prediction
VAP_FLOOR = 0.001          # ug m-3 floor, published in the specification
N_T, N_B = 41, 12
_POOL = None


def pool(n_jobs):
    global _POOL
    if _POOL is None:
        _POOL = Parallel(n_jobs=n_jobs)
    return _POOL


def load_prior(data_dir):
    p = json.load(open(os.path.join(data_dir, "priors.json")))
    return (p["param_ids"], np.array(p["mu"]), np.array(p["covariance"]),
            np.array(p["lower"]), np.array(p["upper"]))


class Observations:
    """Visible station data as whitened log-space blocks."""

    def __init__(self, data_dir, episodes):
        OB.ensure_geometry(data_dir)
        self.blocks = []            # (episode, station, kind, y, Lchol, mask)
        for e in episodes:
            with Dataset(os.path.join(data_dir, "episodes", e,
                                      "observations.nc")) as d:
                sid = [str(s) for s in d["station_id"][:]]
                v = np.array(d["vapour"][:], dtype=float)
                n = np.array(d["counts"][:], dtype=float)
                qc = np.array(d["qc"][:], dtype=int)
                qv = np.array(d["qc_vapour"][:], dtype=int)
            for i, s in enumerate(sid):
                if qv[i, 0] < 0:
                    continue
                Rv = OB.vapour_cov(s)
                self.blocks.append((e, s, "vapour", np.log(v[i]),
                                    cholesky(Rv, lower=True), None))
                m = (qc[i] > 0).ravel()
                Rn = OB.counts_cov(s)[np.ix_(m, m)]
                self.blocks.append((e, s, "counts", np.log(n[i].ravel()[m]),
                                    cholesky(Rn, lower=True), m))
        self.n_obs = sum(len(b[3]) for b in self.blocks)
        self.episodes = list(episodes)

    def whitened(self, pred, kind_filter=None):
        """Whitened residual vector y - h(theta) over the selected blocks.

        pred maps (episode, station) to dict(vapour=(41,), counts=(41,12)).
        """
        parts = []
        for e, s, kind, y, L, m in self.blocks:
            if kind_filter and kind != kind_filter:
                continue
            if kind == "vapour":
                h = np.log(np.maximum(pred[(e, s)]["vapour"], VAP_FLOOR))
            else:
                h = np.log(np.maximum(pred[(e, s)]["counts"].ravel()[m],
                                      PRED_FLOOR))
            parts.append(solve_triangular(L, y - h, lower=True))
        return np.concatenate(parts)


# --------------------------------------------------------------------------
def _predict_episode(e, theta, data_dir, stations, vapour_only):
    OB.ensure_geometry(data_dir)
    case = MO.get_case(data_dir, e)
    p = MO.to_phys(theta)
    out = {}
    if vapour_only:
        snaps = MO.run_vapour(case, p)
        for s in stations:
            out[(e, s)] = dict(vapour=OB.sample_series(snaps, case, s))
    else:
        res = MO.run_full(case, p)
        for s in stations:
            out[(e, s)] = dict(vapour=OB.sample_series(res["c"], case, s),
                               counts=OB.reported_counts(res["n"], case, s))
    return out


def predict_many(thetas, episodes, data_dir, stations, n_jobs,
                 vapour_only=False):
    """Predictions for a list of parameter vectors; returns a list of dicts."""
    jobs = [(i, e) for i in range(len(thetas)) for e in episodes]
    res = pool(n_jobs)(delayed(_predict_episode)(e, thetas[i], data_dir,
                                                 stations, vapour_only)
                       for i, e in jobs)
    out = [dict() for _ in thetas]
    for (i, e), r in zip(jobs, res):
        out[i].update(r)
    return out


class Inversion:
    def __init__(self, data_dir, obs, free, n_jobs=4, vapour_only=False,
                 stations=("S1", "S2", "S3", "S4", "S5"), step=0.02):
        self.data_dir = data_dir
        self.obs = obs
        self.ids, self.mu, self.B, self.lo, self.hi = load_prior(data_dir)
        self.free = np.array(sorted(free), dtype=int)
        self.n_jobs = n_jobs
        self.vapour_only = vapour_only
        self.kind = "vapour" if vapour_only else None
        self.stations = list(stations)
        self.step = step
        self.Lb = cholesky(self.B, lower=True)

    def residual_from_pred(self, theta, pred):
        r1 = self.obs.whitened(pred, self.kind)
        r2 = solve_triangular(self.Lb, theta - self.mu, lower=True)
        return np.concatenate([r1, r2])

    def residual(self, theta):
        pred = predict_many([theta], self.obs.episodes, self.data_dir,
                            self.stations, self.n_jobs, self.vapour_only)[0]
        return self.residual_from_pred(theta, pred)

    def jacobian(self, theta):
        """Stacked whitened Jacobian by forward differences, and r(theta)."""
        nf = len(self.free)
        thetas = [np.array(theta, float)]
        for p in self.free:
            t = np.array(theta, float); t[p] += self.step
            thetas.append(t)
        preds = predict_many(thetas, self.obs.episodes, self.data_dir,
                             self.stations, self.n_jobs, self.vapour_only)
        r0 = self.residual_from_pred(thetas[0], preds[0])
        J = np.zeros((r0.size, nf))
        for i in range(nf):
            ri = self.residual_from_pred(thetas[i + 1], preds[i + 1])
            J[:, i] = (ri - r0) / self.step
        return J, r0

    def fit(self, theta0, max_iter=25, tol=1e-5, verbose=True, log=print):
        th = np.array(theta0, float)
        lam = 1e-3
        r = self.residual(th)
        cost = 0.5 * float(r @ r)
        for it in range(max_iter):
            J, r = self.jacobian(th)
            cost = 0.5 * float(r @ r)
            g = J.T @ r
            H = J.T @ J
            improved, rel = False, 0.0
            for _ in range(10):
                step = np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-12), -g)
                cand = th.copy()
                cand[self.free] = np.clip(th[self.free] + step,
                                          self.lo[self.free], self.hi[self.free])
                rc = self.residual(cand)
                cc = 0.5 * float(rc @ rc)
                if cc < cost:
                    rel = (cost - cc) / max(cost, 1e-12)
                    th, cost = cand, cc
                    lam = max(lam * 0.3, 1e-8)
                    improved = True
                    break
                lam *= 6.0
            if verbose:
                log(f"    iter {it:2d}  cost {cost:.4f}  lambda {lam:.1e}  "
                    + " ".join(f"{self.ids[p][4:]}={th[p]:+.3f}" for p in self.free))
            if not improved or rel < tol:
                break
        return th, cost

    def posterior(self, theta):
        """Gauss-Newton posterior covariance over the full vector, and J."""
        J, r = self.jacobian(theta)
        H = J.T @ J
        C = np.linalg.inv(H)
        cov = np.zeros((len(self.mu), len(self.mu)))
        cov[np.ix_(self.free, self.free)] = C
        return cov, J, r

    def dof_signal(self, cov):
        return float(np.trace(np.eye(len(self.mu)) - cov @ np.linalg.inv(self.B)))
