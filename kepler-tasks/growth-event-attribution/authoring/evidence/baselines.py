"""Shortcut baselines and adversarial attempts that must all score zero.

Each builds a complete, well-formed submission the cheap way, or perturbs the
oracle's own output in the way a lazy or dishonest attempt would, then the
real gates are applied.  Every one is documented in baselines.md.

    python baselines.py <data_dir> <tests_dir> <oracle_dir> <out_root> [which ...]
"""
import csv
import json
import os
import shutil
import sys

import numpy as np
from netCDF4 import Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
N_BINS = 12


def read_index(p):
    return list(csv.DictReader(open(p)))


def load_observed(data_dir, episodes):
    out = {}
    for e in episodes:
        with Dataset(os.path.join(data_dir, "episodes", e,
                                  "observations.nc")) as d:
            sid = [str(s) for s in d["station_id"][:]]
            v = np.array(d["vapour"][:], float)
            n = np.array(d["counts"][:], float)
            qv = np.array(d["qc_vapour"][:], int)
        for i, s in enumerate(sid):
            if qv[i, 0] > 0:
                out[(e, s)] = (v[i], n[i])
    return out


def station_xy(data_dir):
    xy = {}
    with open(os.path.join(data_dir, "stations.csv")) as f:
        for r in csv.DictReader(f):
            xy[r["station_id"]] = (float(r["x_m"]), float(r["y_m"]))
    return xy


def load_oracle(oracle_dir):
    post = json.load(open(os.path.join(oracle_dir, "posterior.json")))
    pr = dict(np.load(os.path.join(oracle_dir, "predictions.npz")))
    hist = list(csv.DictReader(open(os.path.join(oracle_dir,
                                                 "airmass_history.csv"))))
    att = json.load(open(os.path.join(oracle_dir, "attribution.json")))
    return post, pr, hist, att


def write(out_dir, post, pr, hist_rows, att):
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    json.dump(post, open(os.path.join(out_dir, "posterior.json"), "w"), indent=1)
    np.savez(os.path.join(out_dir, "predictions.npz"), **pr)
    with open(os.path.join(out_dir, "airmass_history.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "mean_particle_age_hr", "source_fraction_A",
                    "source_fraction_B", "source_fraction_C"])
        for r in hist_rows:
            w.writerow(r)
    json.dump(att, open(os.path.join(out_dir, "attribution.json"), "w"), indent=1)


def observed_fit(ctx):
    """Fitted values equal to the observations themselves, the claim of a
    perfect fit, for the shortcut baselines."""
    cal = read_index(os.path.join(ctx["data"], "calibration_index.csv"))
    fv = np.zeros(len(cal)); fn = np.zeros((len(cal), N_BINS))
    for r in cal:
        i = int(r["row_id"]); v, n = ctx["obs"][(r["episode"], r["station"])]
        k = int(r["time_index"])
        fv[i] = max(v[k], 1e-3); fn[i] = np.maximum(n[k], 0.0)
    return fv, fn


def prior_post(data_dir):
    p = json.load(open(os.path.join(data_dir, "priors.json")))
    mu = np.array(p["mu"]); sd = np.array(p["sigma"])
    return dict(param_ids=p["param_ids"], estimate=mu.tolist(),
                posterior_sd=sd.tolist(),
                ci_lower=(mu - 1.96 * sd).tolist(),
                ci_upper=(mu + 1.96 * sd).tolist(),
                posterior_prior_sd_ratio=[1.0] * 12, dof_signal=0.0,
                chi2_calibration=1.0e5,
                chi2_by_stream=dict(vapour=2.0e4, counts=8.0e4))


def plausible_att():
    return dict(units="cm-3", Q00=250.0, Q10=360.0, Q01=310.0, Q11=440.0,
                A_E=120.0, A_C=70.0, A_E_sd=10.0, A_C_sd=8.0,
                A_E_sd_sources_fixed=7.0, A_C_sd_sources_fixed=6.0,
                corr_s_event_q_event=-0.3, corr_s_event_sA=-0.6)


def hist_rows_const(queries, age, frac):
    return [[q["query_id"], f"{age:.6f}"] + [f"{x:.6f}" for x in frac]
            for q in queries]


# --------------------------------------------------------------------------
def climatology(ctx):
    """Mean observed vapour and counts by record across stations, with the
    prior as the posterior and guessed attribution numbers."""
    index, queries, obs, data_dir = ctx["index"], ctx["queries"], ctx["obs"], ctx["data"]
    nt = 41
    vsum = np.zeros(nt); nsum = np.zeros((nt, N_BINS)); cnt = 0
    for (e, s), (v, n) in obs.items():
        vsum += v; nsum += n; cnt += 1
    vm = vsum / cnt; nm = nsum / cnt
    vap = np.array([vm[int(r["time_index"])] for r in index])
    pns = np.array([nm[int(r["time_index"])] for r in index])
    fv, fn = observed_fit(ctx)
    pr = dict(row_id=np.arange(len(index)), vapour=np.maximum(vap, 1e-3),
              vapour_log_sd=np.full(len(index), 0.3), pnsd=pns,
              pnsd_log_sd=np.full((len(index), N_BINS), 0.5),
              fit_vapour=fv, fit_pnsd=fn)
    return prior_post(data_dir), pr, hist_rows_const(queries, 4.0, [1/3]*3), plausible_att()


def nearest_station(ctx):
    index, queries, obs, data_dir = ctx["index"], ctx["queries"], ctx["obs"], ctx["data"]
    xy = station_xy(data_dir)
    seen = sorted({s for (_, s) in obs})
    vap = np.zeros(len(index)); pns = np.zeros((len(index), N_BINS))
    for r in index:
        i = int(r["row_id"]); k = int(r["time_index"])
        tx, ty = xy[r["station"]]
        best, bd = None, np.inf
        for s in seen:
            sx, sy = xy[s]
            d = (sx - tx) ** 2 + (sy - ty) ** 2
            if d < bd and (r["episode"], s) in obs:
                bd, best = d, s
        key = (r["episode"], best) if best else sorted(obs)[0]
        v, n = obs[key]
        vap[i] = v[k]; pns[i] = n[k]
    fv, fn = observed_fit(ctx)
    pr = dict(row_id=np.arange(len(index)), vapour=np.maximum(vap, 1e-3),
              vapour_log_sd=np.full(len(index), 0.3), pnsd=pns,
              pnsd_log_sd=np.full((len(index), N_BINS), 0.5),
              fit_vapour=fv, fit_pnsd=fn)
    return prior_post(data_dir), pr, hist_rows_const(queries, 4.0, [0.5, 0.3, 0.2]), plausible_att()


def zero_width(ctx):
    """The oracle's numbers with every uncertainty collapsed."""
    post, pr, hist, att = load_oracle(ctx["oracle"])
    post = dict(post); pr = dict(pr); att = dict(att)
    sd = np.full(12, 1e-4)
    est = np.array(post["estimate"])
    post["posterior_sd"] = sd.tolist()
    post["ci_lower"] = (est - 1.96 * sd).tolist()
    post["ci_upper"] = (est + 1.96 * sd).tolist()
    post["posterior_prior_sd_ratio"] = (sd / np.array(json.load(open(
        os.path.join(ctx["data"], "priors.json")))["sigma"])).tolist()
    post["dof_signal"] = 12.0
    pr["vapour_log_sd"] = np.full_like(pr["vapour_log_sd"], 1e-3)
    pr["pnsd_log_sd"] = np.full_like(pr["pnsd_log_sd"], 1e-3)
    for k in ("A_E_sd", "A_C_sd"):
        att[k] = 0.05
    att["A_E_sd_sources_fixed"] = 0.04; att["A_C_sd_sources_fixed"] = 0.04
    rows = [[r["query_id"], r["mean_particle_age_hr"], r["source_fraction_A"],
             r["source_fraction_B"], r["source_fraction_C"]] for r in hist]
    return post, pr, rows, att


def hedged(ctx):
    """The oracle's numbers with every uncertainty inflated eightfold."""
    post, pr, hist, att = load_oracle(ctx["oracle"])
    post = dict(post); pr = dict(pr); att = dict(att)
    sd = np.array(post["posterior_sd"]) * 8.0
    sig = np.array(json.load(open(os.path.join(ctx["data"], "priors.json")))["sigma"])
    sd = np.minimum(sd, sig)
    est = np.array(post["estimate"])
    post["posterior_sd"] = sd.tolist()
    post["ci_lower"] = (est - 1.96 * sd).tolist()
    post["ci_upper"] = (est + 1.96 * sd).tolist()
    post["posterior_prior_sd_ratio"] = (sd / sig).tolist()
    pr["vapour_log_sd"] = pr["vapour_log_sd"] * 8.0
    pr["pnsd_log_sd"] = pr["pnsd_log_sd"] * 8.0
    for k in ("A_E_sd", "A_C_sd", "A_E_sd_sources_fixed", "A_C_sd_sources_fixed"):
        att[k] = att[k] * 8.0
    rows = [[r["query_id"], r["mean_particle_age_hr"], r["source_fraction_A"],
             r["source_fraction_B"], r["source_fraction_C"]] for r in hist]
    return post, pr, rows, att


def guessed_attribution(ctx):
    """Prior-only parameters and predictions, but the attribution copied at
    its exact truth values from the sealed file."""
    post, pr, rows, _ = climatology(ctx)
    t = json.load(open(os.path.join(ctx["tests"], "hidden_truth",
                                    "parameters.json")))["attribution"]
    att = plausible_att()
    for k in ("Q00", "Q10", "Q01", "Q11", "A_E", "A_C"):
        att[k] = t[k]
    return post, pr, rows, att


def linearised_attribution(ctx):
    """The oracle's parameters, but the counterfactuals replaced by a
    linearisation around the optimum: A_E and A_C from the gradient alone,
    and Q10, Q01 constructed additively."""
    post, pr, hist, att = load_oracle(ctx["oracle"])
    att = dict(att)
    q00, q11 = att["Q00"], att["Q11"]
    tot = q11 - q00
    # linear split in proportion to the single-switch responses, ignoring the
    # interaction: the same numbers an analyst gets from two one-at-a-time runs
    d_e = att["Q10"] - q00
    d_c = att["Q01"] - q00
    att["A_E"] = tot * d_e / (d_e + d_c)
    att["A_C"] = tot * d_c / (d_e + d_c)
    att["Q10"] = q00 + att["A_E"]
    att["Q01"] = q00 + att["A_C"]
    rows = [[r["query_id"], r["mean_particle_age_hr"], r["source_fraction_A"],
             r["source_fraction_B"], r["source_fraction_C"]] for r in hist]
    return post, pr, rows, att


def false_sensitivity(ctx):
    """Correct analysis but the restricted sd inflated above the full one."""
    post, pr, hist, att = load_oracle(ctx["oracle"])
    att = dict(att)
    att["A_E_sd_sources_fixed"] = att["A_E_sd"] * 1.2
    att["A_C_sd_sources_fixed"] = att["A_C_sd"] * 1.2
    rows = [[r["query_id"], r["mean_particle_age_hr"], r["source_fraction_A"],
             r["source_fraction_B"], r["source_fraction_C"]] for r in hist]
    return post, pr, rows, att


def shuffled_predictions(ctx):
    """The oracle's predictions in a permuted row order."""
    post, pr, hist, att = load_oracle(ctx["oracle"])
    pr = dict(pr)
    rng = np.random.default_rng(3)
    perm = rng.permutation(len(pr["vapour"]))
    for k in ("vapour", "vapour_log_sd", "pnsd", "pnsd_log_sd"):
        pr[k] = pr[k][perm]
    rows = [[r["query_id"], r["mean_particle_age_hr"], r["source_fraction_A"],
             r["source_fraction_B"], r["source_fraction_C"]] for r in hist]
    return post, pr, rows, att


def wrong_operator(ctx):
    """The oracle's parameters and fields, but the withheld stations sampled
    with S1's sizing operator instead of their own."""
    post, pr, hist, att = load_oracle(ctx["oracle"])
    pr = dict(pr)
    with Dataset(os.path.join(ctx["data"], "sizing_operators.nc")) as d:
        ids = [str(s) for s in d["station_id"][:]]
        ops = np.array(d["sizing_operator"][:])
    M = {s: ops[i] for i, s in enumerate(ids)}
    pinv = {s: np.linalg.pinv(M[s]) for s in ids}
    pns = pr["pnsd"].copy()
    for r in ctx["index"]:
        i = int(r["row_id"]); s = r["station"]
        n_true = np.maximum(pinv[s] @ pns[i], 0.0)
        pns[i] = M["S1"] @ n_true
    pr["pnsd"] = pns
    rows = [[r["query_id"], r["mean_particle_age_hr"], r["source_fraction_A"],
             r["source_fraction_B"], r["source_fraction_C"]] for r in hist]
    return post, pr, rows, att


def frozen_ohms_no_anomaly(ctx):
    """An analysis that never lets the event anomalies move: the oracle's
    other parameters, both anomalies at the prior mean, attribution zero."""
    post, pr, hist, att = load_oracle(ctx["oracle"])
    post = dict(post); att = dict(att)
    est = np.array(post["estimate"]); est[10] = 0.0; est[11] = 0.0
    post["estimate"] = est.tolist()
    sd = np.array(post["posterior_sd"])
    post["ci_lower"] = (est - 1.96 * sd).tolist()
    post["ci_upper"] = (est + 1.96 * sd).tolist()
    q = att["Q00"]
    att.update(Q00=q, Q10=q, Q01=q, Q11=q, A_E=0.0, A_C=0.0)
    rows = [[r["query_id"], r["mean_particle_age_hr"], r["source_fraction_A"],
             r["source_fraction_B"], r["source_fraction_C"]] for r in hist]
    return post, pr, rows, att


def diagonal_chi2(ctx):
    """The oracle's analysis, but the calibration chi-square computed with
    only the diagonal of the published covariance, as an inversion that
    ignored the correlations would honestly report."""
    import sys as _sys
    _sys.path.insert(0, os.path.join(HERE, "..", "..", "solution", "src"))
    import model as MO
    import observe_ref as OB
    import invert as IV
    post, pr, hist, att = load_oracle(ctx["oracle"])
    post = dict(post)
    OB.load_geometry(ctx["data"])
    theta = np.array(post["estimate"])
    obs = IV.Observations(ctx["data"], ["E01", "E02", "E03", "E04", "E05", "E06"])
    pred = IV.predict_many([theta], obs.episodes, ctx["data"],
                           ["S1", "S2", "S3", "S4", "S5"], 4)[0]
    chi_v = chi_n = 0.0
    for e, s, kind, y, L, m in obs.blocks:
        if kind == "vapour":
            h = np.log(np.maximum(pred[(e, s)]["vapour"], 1e-6))
            d = np.diag(OB.vapour_cov(s))
            chi_v += float((((y - h) ** 2) / d).sum())
        else:
            h = np.log(np.maximum(pred[(e, s)]["counts"].ravel()[m], IV.PRED_FLOOR))
            d = np.diag(OB.counts_cov(s))[m]
            chi_n += float((((y - h) ** 2) / d).sum())
    post["chi2_calibration"] = chi_v + chi_n
    post["chi2_by_stream"] = dict(vapour=chi_v, counts=chi_n)
    rows = [[r["query_id"], r["mean_particle_age_hr"], r["source_fraction_A"],
             r["source_fraction_B"], r["source_fraction_C"]] for r in hist]
    return post, pr, rows, att


BASELINES = {
    "climatology": climatology,
    "diagonal_chi2": diagonal_chi2,
    "nearest_station": nearest_station,
    "zero_width": zero_width,
    "hedged": hedged,
    "guessed_attribution": guessed_attribution,
    "linearised_attribution": linearised_attribution,
    "false_sensitivity": false_sensitivity,
    "shuffled_predictions": shuffled_predictions,
    "wrong_operator": wrong_operator,
    "no_anomaly": frozen_ohms_no_anomaly,
}


def main():
    data_dir, tests_dir, oracle_dir, out_root = sys.argv[1:5]
    which = sys.argv[5:] or list(BASELINES)
    ctx = dict(data=data_dir, tests=tests_dir, oracle=oracle_dir,
               index=read_index(os.path.join(data_dir, "prediction_index.csv")),
               queries=read_index(os.path.join(data_dir, "history_queries.csv")),
               obs=load_observed(data_dir, ["E01", "E02", "E03", "E04", "E05", "E06"]))
    for name in which:
        out = os.path.join(out_root, name)
        post, pr, rows, att = BASELINES[name](ctx)
        write(out, post, pr, rows, att)
        print(f"wrote {name} -> {out}")


if __name__ == "__main__":
    main()
