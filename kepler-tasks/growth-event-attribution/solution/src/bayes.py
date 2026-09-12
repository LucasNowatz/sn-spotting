#!/usr/bin/env python3
"""Whitened Gauss-Newton / Levenberg-Marquardt inversion (reference).

Objective in the prior's log coordinates:
  0.5 (y - h)^T R^-1 (y - h) + 0.5 (theta - mu)^T B^-1 (theta - mu)
with y the log observations, R the published block covariance restricted to
usable channels and B the non-diagonal prior.  Each block is whitened by its
Cholesky factor; the Gauss-Newton Hessian of the whitened residual is the
inverse posterior covariance.
"""

import json
import os
import sys

import numpy as np
from joblib import Parallel, delayed
from netCDF4 import Dataset
from scipy.linalg import cholesky, solve_triangular

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import forward as FW
import network as NW

LOG_FLOOR_N, LOG_FLOOR_V = 0.05, 0.001
_POOL = None


def pool(n_jobs):
    global _POOL
    if _POOL is None:
        _POOL = Parallel(n_jobs=n_jobs)
    return _POOL


def load_prior(data_dir):
    p = json.load(open(os.path.join(data_dir, "prior.json")))
    return p["param_ids"], np.array(p["mu"]), np.array(p["covariance"]), \
        np.array(p["lower"]), np.array(p["upper"])


class Records:
    """Visible station records as whitened log-space blocks."""

    def __init__(self, data_dir, episodes):
        NW.load(data_dir)
        self.blocks = []
        for e in episodes:
            with Dataset(os.path.join(data_dir, "episodes", e, "station_records.nc")) as d:
                sid = [str(s) for s in d["station_id"][:]]
                v = np.array(d["vapour"][:], float); n = np.array(d["counts"][:], float)
                qc = np.array(d["counts_flag"][:], int); qv = np.array(d["vapour_flag"][:], int)
            for i, s in enumerate(sid):
                if qv[i, 0] < 0:
                    continue
                self.blocks.append((e, s, "vapour", np.log(v[i]),
                                    cholesky(NW.vapour_cov(s), lower=True), None))
                m = (qc[i] > 0).ravel()
                self.blocks.append((e, s, "counts", np.log(n[i].ravel()[m]),
                                    cholesky(NW.counts_cov(s)[np.ix_(m, m)], lower=True), m))
        self.n_obs = sum(len(b[3]) for b in self.blocks)
        self.episodes = list(episodes)
        self.stations = sorted({b[1] for b in self.blocks})

    def whitened(self, pred, kind=None):
        parts = []
        for e, s, k, y, L, m in self.blocks:
            if kind and k != kind:
                continue
            if k == "vapour":
                h = np.log(np.maximum(pred[(e, s)]["vapour"], LOG_FLOOR_V))
            else:
                h = np.log(np.maximum(pred[(e, s)]["counts"].ravel()[m], LOG_FLOOR_N))
            parts.append(solve_triangular(L, y - h, lower=True))
        return np.concatenate(parts)


def _predict(e, theta, data_dir, stations, vapour_only):
    NW.load(data_dir)
    ep = FW.episode(data_dir, e)
    p = FW.from_theta(theta)
    out = {}
    if vapour_only:
        c = FW.run_vapour(ep, p)
        for s in stations:
            out[(e, s)] = dict(vapour=NW.at_station(c, ep, s))
    else:
        res = FW.run(ep, p)
        for s in stations:
            out[(e, s)] = dict(vapour=NW.at_station(res["c"], ep, s),
                               counts=NW.reported(res["n"], ep, s))
    return out


def predict_many(thetas, episodes, data_dir, stations, n_jobs, vapour_only=False):
    jobs = [(i, e) for i in range(len(thetas)) for e in episodes]
    res = pool(n_jobs)(delayed(_predict)(e, thetas[i], data_dir, stations, vapour_only)
                       for i, e in jobs)
    out = [dict() for _ in thetas]
    for (i, e), r in zip(jobs, res):
        out[i].update(r)
    return out


class Inversion:
    def __init__(self, data_dir, rec, free, n_jobs=4, vapour_only=False, step=0.02):
        self.data_dir, self.rec = data_dir, rec
        self.ids, self.mu, self.B, self.lo, self.hi = load_prior(data_dir)
        self.free = np.array(sorted(free), dtype=int)
        self.n_jobs, self.vapour_only, self.step = n_jobs, vapour_only, step
        self.kind = "vapour" if vapour_only else None
        self.Lb = cholesky(self.B, lower=True)

    def _residual(self, theta, pred):
        r1 = self.rec.whitened(pred, self.kind)
        r2 = solve_triangular(self.Lb, theta - self.mu, lower=True)
        return np.concatenate([r1, r2])

    def residual(self, theta):
        pred = predict_many([theta], self.rec.episodes, self.data_dir, self.rec.stations,
                            self.n_jobs, self.vapour_only)[0]
        return self._residual(theta, pred)

    def jacobian(self, theta):
        thetas = [np.array(theta, float)]
        for p in self.free:
            t = np.array(theta, float); t[p] += self.step; thetas.append(t)
        preds = predict_many(thetas, self.rec.episodes, self.data_dir, self.rec.stations,
                             self.n_jobs, self.vapour_only)
        r0 = self._residual(thetas[0], preds[0])
        J = np.zeros((r0.size, len(self.free)))
        for i in range(len(self.free)):
            J[:, i] = (self._residual(thetas[i + 1], preds[i + 1]) - r0) / self.step
        return J, r0

    def fit(self, theta0, max_iter=25, tol=1e-5, log=print):
        th = np.array(theta0, float)
        lam = 1e-3
        cost = 0.5 * float(self.residual(th) @ self.residual(th))
        for it in range(max_iter):
            J, r = self.jacobian(th)
            cost = 0.5 * float(r @ r)
            g, H = J.T @ r, J.T @ J
            improved, rel = False, 0.0
            for _ in range(10):
                step = np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-12), -g)
                cand = th.copy()
                cand[self.free] = np.clip(th[self.free] + step, self.lo[self.free], self.hi[self.free])
                rc = self.residual(cand)
                cc = 0.5 * float(rc @ rc)
                if cc < cost:
                    rel = (cost - cc) / max(cost, 1e-12)
                    th, cost, improved = cand, cc, True
                    lam = max(lam * 0.3, 1e-8)
                    break
                lam *= 6.0
            log(f"    iter {it:2d}  cost {cost:.4f}  " +
                " ".join(f"{self.ids[p][4:]}={th[p]:+.3f}" for p in self.free))
            if not improved or rel < tol:
                break
        return th, cost

    def posterior(self, theta):
        J, r = self.jacobian(theta)
        cov = np.zeros((len(self.mu),) * 2)
        cov[np.ix_(self.free, self.free)] = np.linalg.inv(J.T @ J)
        return cov, J, r

    def dof_signal(self, cov):
        return float(np.trace(np.eye(len(self.mu)) - cov @ np.linalg.inv(self.B)))
