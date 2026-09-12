#!/usr/bin/env python3
"""Reference workflow for the growth-event benchmark.

    run.py --input-dir /app/data --output-dir /app/output

Stages:
  1. vapour-only fit of the five parameters the vapour field depends on
     (a cheap, well-conditioned start for the joint problem);
  2. joint whitened Levenberg-Marquardt over all twelve log-parameters with
     the full correlated error covariance and the non-diagonal prior, from
     the stage-1 start and two random prior draws;
  3. Gauss-Newton posterior covariance, degrees of freedom for signal and
     the calibration chi-square;
  4. restricted analysis with the three regional source strengths held at
     their prior means;
  5. counterfactual event statistics with the two anomaly groups switched
     on and off in every combination, re-integrated in full, and the delta
     method for their uncertainty under both posteriors;
  6. withheld predictions with latent and total predictive spread, and the
     tagged pass for the lineage diagnostics.
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
from joblib import delayed
from scipy.linalg import solve_triangular

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as MO
import observe_ref as OB
import invert as IV

OBSERVED_EP = ["E01", "E02", "E03", "E04", "E05", "E06"]
SEEN = ["S1", "S2", "S3", "S4", "S5"]
VAPOUR_FREE = [0, 1, 2, 3, 11]
SOURCE_BASE = [7, 8, 9]
GROUP_E = [10]
GROUP_C = [11]
REPR_V, REPR_N = 0.05, 0.06
Z95 = 1.959963984540054
SEED = 11


def log(msg):
    print(f"[ref {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_index(path):
    return list(csv.DictReader(open(path)))


# --------------------------------------------------------------------------
def counterfactual_thetas(theta):
    th = np.asarray(theta, float)
    t00 = th.copy(); t00[GROUP_E] = 0.0; t00[GROUP_C] = 0.0
    t10 = th.copy(); t10[GROUP_C] = 0.0
    t01 = th.copy(); t01[GROUP_E] = 0.0
    return [t00, t10, t01, th.copy()]


def _q_one(e, theta, data_dir):
    case = MO.get_case(data_dir, e)
    res = MO.run_full(case, MO.to_phys(theta))
    return MO.event_statistic(res["n"])


def statistics(thetas, events, data_dir, n_jobs):
    """Event statistic (mean over event episodes) for each theta."""
    jobs = [(i, e) for i in range(len(thetas)) for e in events]
    res = IV.pool(n_jobs)(delayed(_q_one)(e, thetas[i], data_dir)
                          for i, e in jobs)
    q = np.zeros(len(thetas))
    for (i, e), v in zip(jobs, res):
        q[i] += v / len(events)
    return q


def decompose(q):
    q00, q10, q01, q11 = q
    return dict(Q00=q00, Q10=q10, Q01=q01, Q11=q11,
                A_E=0.5 * ((q10 - q00) + (q11 - q01)),
                A_C=0.5 * ((q01 - q00) + (q11 - q10)))


def attribution_with_gradient(theta, events, data_dir, n_jobs, step=0.02):
    th = np.asarray(theta, float)
    stacks = counterfactual_thetas(th)
    for p in range(len(th)):
        t = th.copy(); t[p] += step
        stacks += counterfactual_thetas(t)
    q = statistics(stacks, events, data_dir, n_jobs).reshape(len(th) + 1, 4)
    base = decompose(q[0])
    g = np.zeros((2, len(th)))
    for p in range(len(th)):
        a = decompose(q[p + 1])
        g[0, p] = (a["A_E"] - base["A_E"]) / step
        g[1, p] = (a["A_C"] - base["A_C"]) / step
    return base, g


# --------------------------------------------------------------------------
def _final(e, theta, data_dir):
    OB.ensure_geometry(data_dir)
    return e, MO.run_full(MO.get_case(data_dir, e), MO.to_phys(theta), tags=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--random-starts", type=int, default=2)
    a = ap.parse_args(argv)
    t0 = time.time()
    os.makedirs(a.output_dir, exist_ok=True)
    OB.load_geometry(a.input_dir)
    manifest = json.load(open(os.path.join(a.input_dir, "data_manifest.json")))
    all_ep = [e["episode"] for e in manifest["episodes"]]
    events = [e["episode"] for e in manifest["episodes"] if e["event_day"]]
    obs = IV.Observations(a.input_dir, OBSERVED_EP)
    ids, mu, B, lo, hi = IV.load_prior(a.input_dir)
    npar = len(mu)
    log(f"{obs.n_obs} usable observations, {npar} parameters, "
        f"event episodes {events}")

    # --- stage 1: vapour only --------------------------------------------
    log("stage 1: vapour-only fit")
    inv_v = IV.Inversion(a.input_dir, obs, VAPOUR_FREE, a.n_jobs,
                         vapour_only=True)
    th1, c1 = inv_v.fit(mu, max_iter=20, log=log)
    log(f"  vapour stage cost {c1:.3f}")

    # --- stage 2: joint fit from several starts --------------------------
    log("stage 2: joint fit")
    inv = IV.Inversion(a.input_dir, obs, list(range(npar)), a.n_jobs)
    rng = np.random.default_rng(SEED)
    Lb = np.linalg.cholesky(B)
    starts = [th1.copy()]
    for _ in range(a.random_starts):
        s = th1 + 0.6 * (Lb @ rng.standard_normal(npar))
        s[VAPOUR_FREE] = th1[VAPOUR_FREE]
        starts.append(np.clip(s, lo, hi))
    best, best_cost = None, np.inf
    for i, s in enumerate(starts):
        log(f"  start {i + 1}/{len(starts)}")
        th, c = inv.fit(s, max_iter=25, log=log)
        log(f"  start {i + 1} cost {c:.4f}")
        if c < best_cost:
            best, best_cost = th, c
    theta = best
    log("best: " + " ".join(f"{k[4:]}={v:+.3f}" for k, v in zip(ids, theta)))

    # --- stage 3: posterior ----------------------------------------------
    log("stage 3: posterior covariance")
    cov, J, r = inv.posterior(theta)
    sd = np.sqrt(np.diag(cov))
    dof = inv.dof_signal(cov)
    n_data = obs.n_obs
    chi2 = float(r[:n_data] @ r[:n_data])
    rv = obs.whitened(IV.predict_many([theta], OBSERVED_EP, a.input_dir, SEEN,
                                      a.n_jobs)[0], "vapour")
    chi2_v = float(rv @ rv)
    chi2_n = chi2 - chi2_v
    corr = cov / np.outer(sd, sd)
    log(f"  chi2 {chi2:.1f} (vapour {chi2_v:.1f}, counts {chi2_n:.1f}) over "
        f"{n_data}; dof_signal {dof:.2f}")
    log("  posterior sd: " + " ".join(f"{k[4:]}={v:.3f}" for k, v in zip(ids, sd)))

    # --- stage 4: restricted analysis ------------------------------------
    log("stage 4: restricted analysis, source strengths fixed at the prior mean")
    free_r = [i for i in range(npar) if i not in SOURCE_BASE]
    inv_r = IV.Inversion(a.input_dir, obs, free_r, a.n_jobs)
    th_r = theta.copy(); th_r[SOURCE_BASE] = mu[SOURCE_BASE]
    th_r, c_r = inv_r.fit(th_r, max_iter=15, log=log)
    cov_r, _, _ = inv_r.posterior(th_r)

    # --- stage 5: attribution --------------------------------------------
    log("stage 5: counterfactual event statistics")
    attr, grad = attribution_with_gradient(theta, events, a.input_dir, a.n_jobs)
    ae_sd = float(np.sqrt(grad[0] @ cov @ grad[0]))
    ac_sd = float(np.sqrt(grad[1] @ cov @ grad[1]))
    attr_r, grad_r = attribution_with_gradient(th_r, events, a.input_dir, a.n_jobs)
    ae_sd_r = float(np.sqrt(grad_r[0] @ cov_r @ grad_r[0]))
    ac_sd_r = float(np.sqrt(grad_r[1] @ cov_r @ grad_r[1]))
    log(f"  Q00 {attr['Q00']:.3f} Q10 {attr['Q10']:.3f} Q01 {attr['Q01']:.3f} "
        f"Q11 {attr['Q11']:.3f}")
    log(f"  A_E {attr['A_E']:.3f} +- {ae_sd:.3f} (restricted {ae_sd_r:.3f}); "
        f"A_C {attr['A_C']:.3f} +- {ac_sd:.3f} (restricted {ac_sd_r:.3f})")
    log(f"  corr(s_event, q_event) {corr[10, 11]:+.3f}; "
        f"corr(s_event, sA) {corr[10, 7]:+.3f}")

    # --- stage 6: predictions and lineage --------------------------------
    log("stage 6: withheld predictions, predictive spread and lineage")
    index = load_index(os.path.join(a.input_dir, "prediction_index.csv"))
    n_rows = len(index)
    stations_all = OB.station_ids()
    thetas = [theta] + [theta + np.eye(npar)[p] * 0.02 for p in range(npar)]
    preds = IV.predict_many(thetas, all_ep, a.input_dir, stations_all, a.n_jobs)

    def row_values(pred):
        v = np.zeros(n_rows); n = np.zeros((n_rows, MO.N_BINS))
        for rrow in index:
            i = int(rrow["row_id"]); key = (rrow["episode"], rrow["station"])
            k = int(rrow["time_index"])
            v[i] = pred[key]["vapour"][k]
            n[i] = pred[key]["counts"][k]
        return v, n
    v0, n0 = row_values(preds[0])
    cal_index = load_index(os.path.join(a.input_dir, "calibration_index.csv"))
    fv = np.zeros(len(cal_index)); fn = np.zeros((len(cal_index), MO.N_BINS))
    for rrow in cal_index:
        i = int(rrow["row_id"]); key = (rrow["episode"], rrow["station"])
        k = int(rrow["time_index"])
        fv[i] = preds[0][key]["vapour"][k]
        fn[i] = preds[0][key]["counts"][k]
    lv = np.log(np.maximum(v0, 1e-6)); ln = np.log(np.maximum(n0, IV.PRED_FLOOR))
    Jv = np.zeros((n_rows, npar)); Jn = np.zeros((n_rows, MO.N_BINS, npar))
    for p in range(npar):
        vp, np_ = row_values(preds[p + 1])
        Jv[:, p] = (np.log(np.maximum(vp, 1e-6)) - lv) / 0.02
        Jn[:, :, p] = (np.log(np.maximum(np_, IV.PRED_FLOOR)) - ln) / 0.02
    var_lat_v = np.einsum("np,pq,nq->n", Jv, cov, Jv)
    var_lat_n = np.einsum("nbp,pq,nbq->nb", Jn, cov, Jn)
    noise_v = np.zeros(n_rows); noise_n = np.zeros(n_rows)
    for rrow in index:
        i = int(rrow["row_id"]); s = rrow["station"]
        noise_v[i] = OB.vapour_cov(s)[0, 0]
        noise_n[i] = OB.counts_cov(s)[0, 0]
    sd_v = np.sqrt(var_lat_v + noise_v + REPR_V ** 2)
    sd_n = np.sqrt(var_lat_n + noise_n[:, None] + REPR_N ** 2)

    runs = dict(IV.pool(a.n_jobs)(delayed(_final)(e, theta, a.input_dir)
                                  for e in all_ep))
    cases = {e: MO.get_case(a.input_dir, e) for e in all_ep}
    queries = load_index(os.path.join(a.input_dir, "history_queries.csv"))
    hist = []
    for q in queries:
        case = cases[q["episode"]]; res = runs[q["episode"]]
        k, b = int(q["time_index"]), int(q["diameter_bin"])
        ag = OB.sample_bins(res["ag"], case, q["station"])[k, b]
        tg = np.array([OB.sample_bins(res["tg"][:, s], case, q["station"])[k, b]
                       for s in range(3)])
        tot = tg.sum()
        if tot > 1e-12:
            hist.append((q["query_id"], ag / tot, tg / tot))
        else:
            hist.append((q["query_id"], 0.0, np.full(3, 1 / 3)))

    # --- write --------------------------------------------------------------
    out = a.output_dir
    with open(os.path.join(out, "posterior.json"), "w") as f:
        json.dump(dict(
            param_ids=ids,
            estimate=[float(x) for x in theta],
            posterior_sd=[float(x) for x in sd],
            ci_lower=[float(x) for x in theta - Z95 * sd],
            ci_upper=[float(x) for x in theta + Z95 * sd],
            posterior_prior_sd_ratio=[float(x) for x in sd / np.sqrt(np.diag(B))],
            dof_signal=dof,
            chi2_calibration=chi2,
            chi2_by_stream=dict(vapour=chi2_v, counts=chi2_n),
            n_calibration=int(n_data),
            objective=best_cost,
            wall_time_s=time.time() - t0), f, indent=1)
    np.savez(os.path.join(out, "predictions.npz"), row_id=np.arange(n_rows),
             vapour=v0, vapour_log_sd=sd_v, pnsd=n0, pnsd_log_sd=sd_n,
             fit_row_id=np.arange(len(cal_index)), fit_vapour=fv, fit_pnsd=fn)
    with open(os.path.join(out, "airmass_history.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "mean_particle_age_hr", "source_fraction_A",
                    "source_fraction_B", "source_fraction_C"])
        for qid, age, fr in hist:
            w.writerow([qid, f"{age:.6f}"] + [f"{x:.6f}" for x in fr])
    with open(os.path.join(out, "attribution.json"), "w") as f:
        json.dump(dict(
            units="cm-3",
            Q00=attr["Q00"], Q10=attr["Q10"], Q01=attr["Q01"], Q11=attr["Q11"],
            A_E=attr["A_E"], A_C=attr["A_C"],
            A_E_sd=ae_sd, A_C_sd=ac_sd,
            A_E_sd_sources_fixed=ae_sd_r, A_C_sd_sources_fixed=ac_sd_r,
            corr_s_event_q_event=float(corr[10, 11]),
            corr_s_event_sA=float(corr[10, 7]),
            uncertainty_convention="posterior standard deviation"), f, indent=1)
    log(f"done in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
