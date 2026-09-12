#!/usr/bin/env python3
"""Reference pipeline.

    pipeline.py --data /app/data --out /app/output

1. vapour-only fit of the five parameters the vapour field depends on;
2. joint LM over all twelve from that start and two random prior draws;
3. Gauss-Newton posterior, dof and chi-square;
4. restricted analysis with the regional strengths fixed at the prior mean;
5. counterfactual event statistics (full re-integrations) and the delta
   method for their spread under both posteriors;
6. withheld predictions with predictive spread, fitted values at the
   calibration rows, and a tagged pass for the region shares.
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
from joblib import delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import forward as FW
import network as NW
import bayes as BY

OBSERVED = ["E01", "E02", "E03", "E04", "E05", "E06"]
VAPOUR_FREE = [0, 1, 2, 3, 11]
SOURCE_BASE = [7, 8, 9]
E_GROUP, C_GROUP = [10], [11]
REPR_V, REPR_N = 0.05, 0.06
Z95 = 1.959963984540054
STEP = 0.02


def log(m):
    print(f"[ref {time.strftime('%H:%M:%S')}] {m}", flush=True)


def rows(path):
    return list(csv.DictReader(open(path)))


def cf_thetas(theta):
    th = np.asarray(theta, float)
    t00 = th.copy(); t00[E_GROUP] = 0.0; t00[C_GROUP] = 0.0
    t10 = th.copy(); t10[C_GROUP] = 0.0
    t01 = th.copy(); t01[E_GROUP] = 0.0
    return [t00, t10, t01, th.copy()]


def _q(e, theta, data_dir):
    return FW.event_statistic(FW.run(FW.episode(data_dir, e), FW.from_theta(theta))["n"])


def statistics(thetas, events, data_dir, n_jobs):
    jobs = [(i, e) for i in range(len(thetas)) for e in events]
    res = BY.pool(n_jobs)(delayed(_q)(e, thetas[i], data_dir) for i, e in jobs)
    q = np.zeros(len(thetas))
    for (i, e), v in zip(jobs, res):
        q[i] += v / len(events)
    return q


def decompose(q):
    q00, q10, q01, q11 = q
    return dict(Q00=q00, Q10=q10, Q01=q01, Q11=q11,
                A_E=0.5 * ((q10 - q00) + (q11 - q01)), A_C=0.5 * ((q01 - q00) + (q11 - q10)))


def attribution(theta, events, data_dir, n_jobs):
    th = np.asarray(theta, float)
    stack = cf_thetas(th)
    for p in range(len(th)):
        t = th.copy(); t[p] += STEP; stack += cf_thetas(t)
    q = statistics(stack, events, data_dir, n_jobs).reshape(len(th) + 1, 4)
    base = decompose(q[0])
    g = np.zeros((2, len(th)))
    for p in range(len(th)):
        a = decompose(q[p + 1])
        g[0, p] = (a["A_E"] - base["A_E"]) / STEP
        g[1, p] = (a["A_C"] - base["A_C"]) / STEP
    return base, g


def _tagged(e, theta, data_dir):
    NW.load(data_dir)
    return e, FW.run(FW.episode(data_dir, e), FW.from_theta(theta), tags=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n-jobs", type=int, default=4)
    a = ap.parse_args(argv)
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    NW.load(a.data)
    man = json.load(open(os.path.join(a.data, "dataset_manifest.json")))
    all_ep = [e["episode"] for e in man["episodes"]]
    events = [e["episode"] for e in man["episodes"] if e["event_day"]]
    rec = BY.Records(a.data, OBSERVED)
    ids, mu, B, lo, hi = BY.load_prior(a.data)
    npar = len(mu)
    log(f"{rec.n_obs} usable observations; event days {events}")

    log("stage 1: vapour-only fit")
    th1, _ = BY.Inversion(a.data, rec, VAPOUR_FREE, a.n_jobs, vapour_only=True).fit(mu, 20, log=log)

    log("stage 2: joint fit")
    inv = BY.Inversion(a.data, rec, list(range(npar)), a.n_jobs)
    rng = np.random.default_rng(5)
    Lb = np.linalg.cholesky(B)
    starts = [th1.copy()]
    for _ in range(2):
        s = th1 + 0.6 * (Lb @ rng.standard_normal(npar)); s[VAPOUR_FREE] = th1[VAPOUR_FREE]
        starts.append(np.clip(s, lo, hi))
    best, best_cost = None, np.inf
    for i, s in enumerate(starts):
        log(f"  start {i + 1}")
        th, c = inv.fit(s, 25, log=log)
        log(f"  start {i + 1} cost {c:.4f}")
        if c < best_cost:
            best, best_cost = th, c
    theta = best

    log("stage 3: posterior")
    cov, J, r = inv.posterior(theta)
    sd = np.sqrt(np.diag(cov)); dof = inv.dof_signal(cov)
    chi2 = float(r[:rec.n_obs] @ r[:rec.n_obs])
    pred_cal = BY.predict_many([theta], OBSERVED, a.data, rec.stations, a.n_jobs)[0]
    rv = rec.whitened(pred_cal, "vapour"); chi2_v = float(rv @ rv)
    corr = cov / np.outer(sd, sd)
    log(f"  chi2 {chi2:.1f} (vapour {chi2_v:.1f}); dof {dof:.2f}")
    log("  sd: " + " ".join(f"{k[4:]}={v:.3f}" for k, v in zip(ids, sd)))

    log("stage 4: restricted analysis")
    inv_r = BY.Inversion(a.data, rec, [i for i in range(npar) if i not in SOURCE_BASE], a.n_jobs)
    th_r = theta.copy(); th_r[SOURCE_BASE] = mu[SOURCE_BASE]
    th_r, _ = inv_r.fit(th_r, 15, log=log)
    cov_r, _, _ = inv_r.posterior(th_r)

    log("stage 5: counterfactuals")
    attr, g = attribution(theta, events, a.data, a.n_jobs)
    _, g_r = attribution(th_r, events, a.data, a.n_jobs)
    sd_e = float(np.sqrt(g[0] @ cov @ g[0])); sd_c = float(np.sqrt(g[1] @ cov @ g[1]))
    sd_e_r = float(np.sqrt(g_r[0] @ cov_r @ g_r[0])); sd_c_r = float(np.sqrt(g_r[1] @ cov_r @ g_r[1]))
    log(f"  A_E {attr['A_E']:.2f} +- {sd_e:.2f} (restricted {sd_e_r:.2f}); "
        f"A_C {attr['A_C']:.2f} +- {sd_c:.2f} (restricted {sd_c_r:.2f}); "
        f"corr(s_event, sA) {corr[10, 7]:+.3f}")

    log("stage 6: predictions, fitted values, region shares")
    windex = rows(os.path.join(a.data, "withheld_index.csv"))
    cindex = rows(os.path.join(a.data, "calibration_index.csv"))
    stations = NW.stations()
    thetas = [theta] + [theta + np.eye(npar)[p] * STEP for p in range(npar)]
    preds = BY.predict_many(thetas, all_ep, a.data, stations, a.n_jobs)

    def gather(pred, index):
        v = np.zeros(len(index)); n = np.zeros((len(index), FW.N_BINS))
        for rr in index:
            i = int(rr["row_id"]); key = (rr["episode"], rr["station"]); k = int(rr["time_index"])
            v[i] = pred[key]["vapour"][k]; n[i] = pred[key]["counts"][k]
        return v, n
    v0, n0 = gather(preds[0], windex)
    fv, fn = gather(preds[0], cindex)
    lv = np.log(np.maximum(v0, BY.LOG_FLOOR_V)); ln = np.log(np.maximum(n0, BY.LOG_FLOOR_N))
    Jv = np.zeros((len(windex), npar)); Jn = np.zeros((len(windex), FW.N_BINS, npar))
    for p in range(npar):
        vp, np_ = gather(preds[p + 1], windex)
        Jv[:, p] = (np.log(np.maximum(vp, BY.LOG_FLOOR_V)) - lv) / STEP
        Jn[:, :, p] = (np.log(np.maximum(np_, BY.LOG_FLOOR_N)) - ln) / STEP
    var_v = np.einsum("np,pq,nq->n", Jv, cov, Jv)
    var_n = np.einsum("nbp,pq,nbq->nb", Jn, cov, Jn)
    noise_v = np.array([NW.vapour_cov(rr["station"])[0, 0] for rr in windex])
    noise_n = np.array([NW.counts_cov(rr["station"])[0, 0] for rr in windex])
    sd_v = np.sqrt(var_v + noise_v + REPR_V ** 2)
    sd_n = np.sqrt(var_n + noise_n[:, None] + REPR_N ** 2)

    tagged = dict(BY.pool(a.n_jobs)(delayed(_tagged)(e, theta, a.data) for e in events))
    shares = dict(domain={}, stations={})
    for e in events:
        res = tagged[e]; ep = FW.episode(a.data, e)
        shares["domain"][e] = {r: float(x) for r, x in zip("ABC", FW.domain_shares(res))}
        shares["stations"][e] = {}
        for s in ("W1", "W2"):
            g_tot = NW.at_station(res["n"][FW.Q_FIRST_RECORD:, FW.GROWN_FIRST_BIN:].sum(axis=1), ep, s).mean()
            g_tag = NW.at_station(res["tagged"][FW.Q_FIRST_RECORD:, :, FW.GROWN_FIRST_BIN:].sum(axis=2), ep, s).mean(axis=0)
            shares["stations"][e][s] = {r: float(x) for r, x in zip("ABC", g_tag / g_tot)}

    with open(os.path.join(a.out, "posterior.json"), "w") as f:
        json.dump(dict(param_ids=ids, estimate=theta.tolist(), posterior_sd=sd.tolist(),
                       ci_lower=(theta - Z95 * sd).tolist(), ci_upper=(theta + Z95 * sd).tolist(),
                       posterior_prior_sd_ratio=(sd / np.sqrt(np.diag(B))).tolist(),
                       dof_signal=dof, chi2_calibration=chi2,
                       chi2_by_stream=dict(vapour=chi2_v, counts=chi2 - chi2_v),
                       wall_time_s=time.time() - t0), f, indent=1)
    np.savez(os.path.join(a.out, "predictions.npz"), row_id=np.arange(len(windex)),
             vapour=v0, vapour_log_sd=sd_v, pnsd=n0, pnsd_log_sd=sd_n,
             fit_row_id=np.arange(len(cindex)), fit_vapour=fv, fit_pnsd=fn)
    with open(os.path.join(a.out, "attribution.json"), "w") as f:
        json.dump(dict(units="cm-3", Q00=attr["Q00"], Q10=attr["Q10"], Q01=attr["Q01"],
                       Q11=attr["Q11"], A_E=attr["A_E"], A_C=attr["A_C"], A_E_sd=sd_e,
                       A_C_sd=sd_c, A_E_sd_sources_fixed=sd_e_r, A_C_sd_sources_fixed=sd_c_r,
                       corr_s_event_sA=float(corr[10, 7])), f, indent=1)
    with open(os.path.join(a.out, "region_shares.json"), "w") as f:
        json.dump(shares, f, indent=1)
    log(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
