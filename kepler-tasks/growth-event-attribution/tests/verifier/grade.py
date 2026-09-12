"""Gate logic for the growth-event verifier (sealed).

Each gate returns a list of failure strings; empty means it passed.  Nothing
is accepted because a JSON file asserts it: withheld predictions are compared
to generator truth, lineage diagnostics to tagged-tracer truth, predictive
spreads to the withheld noisy observations, and the submitted parameter
vector is re-run through a trusted forward model that must reproduce the
submitted predictions, the reported chi-square and the four counterfactual
event statistics.
"""

import csv
import json
import os

import numpy as np

from . import metrics as MT

N_BINS = 12
N_PARAM = 12
Z95 = 1.959963984540054


def load_thresholds(path):
    return json.load(open(path))


def read_index(path):
    return list(csv.DictReader(open(path)))


def _num(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError
    x = float(x)
    if not np.isfinite(x):
        raise ValueError
    return x


def _vec(x, n, name, errs):
    try:
        a = np.asarray(x, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        errs.append(f"{name} is not a numeric array")
        return None
    if a.size != n:
        errs.append(f"{name} has {a.size} entries, expected {n}")
        return None
    if not np.isfinite(a).all():
        errs.append(f"{name} has non-finite entries")
        return None
    return a


def gate_schema(out_dir, index, queries, param_ids, thr, cal_index):
    errs = []
    for f in ("posterior.json", "predictions.npz", "airmass_history.csv",
              "attribution.json"):
        if not os.path.exists(os.path.join(out_dir, f)):
            errs.append(f"missing required artifact /app/output/{f}")
    if errs:
        return errs, None

    # ---- posterior.json
    try:
        post = json.load(open(os.path.join(out_dir, "posterior.json")))
    except (ValueError, OSError) as e:
        return [f"posterior.json is not readable JSON: {e}"], None
    if not isinstance(post, dict):
        return ["posterior.json must contain an object"], None
    if list(post.get("param_ids", [])) != list(param_ids):
        errs.append("posterior.json param_ids must equal the published order")
    P = {}
    for k in ("estimate", "posterior_sd", "ci_lower", "ci_upper",
              "posterior_prior_sd_ratio"):
        P[k] = _vec(post.get(k), N_PARAM, f"posterior.json {k}", errs)
    for k in ("dof_signal", "chi2_calibration"):
        try:
            P[k] = _num(post.get(k))
        except ValueError:
            errs.append(f"posterior.json {k} must be a finite number")
    cs = post.get("chi2_by_stream")
    if not isinstance(cs, dict) or not all(k in cs for k in ("vapour", "counts")):
        errs.append("posterior.json chi2_by_stream must have vapour and counts")
    else:
        try:
            P["chi2_vapour"] = _num(cs["vapour"]); P["chi2_counts"] = _num(cs["counts"])
        except ValueError:
            errs.append("chi2_by_stream entries must be finite numbers")
    if errs:
        return errs, None
    if (P["posterior_sd"] <= 0).any():
        errs.append("posterior_sd must be positive")
    if (P["ci_lower"] >= P["ci_upper"]).any():
        errs.append("ci_lower must be below ci_upper for every parameter")
    if (P["posterior_prior_sd_ratio"] <= 0).any() or \
            (P["posterior_prior_sd_ratio"] > 1.05).any():
        errs.append("posterior_prior_sd_ratio must be positive and not "
                    "exceed the prior")

    # ---- predictions.npz
    try:
        pr = np.load(os.path.join(out_dir, "predictions.npz"))
    except Exception as e:                                  # noqa: BLE001
        return errs + [f"predictions.npz is not readable: {e}"], None
    for k in ("vapour", "vapour_log_sd", "pnsd", "pnsd_log_sd", "fit_vapour",
              "fit_pnsd"):
        if k not in pr:
            errs.append(f"predictions.npz has no {k} array")
    if errs:
        return errs, None
    v = np.asarray(pr["vapour"], float); sv = np.asarray(pr["vapour_log_sd"], float)
    n = np.asarray(pr["pnsd"], float); sn = np.asarray(pr["pnsd_log_sd"], float)
    fv = np.asarray(pr["fit_vapour"], float); fn = np.asarray(pr["fit_pnsd"], float)
    nr, nc = len(index), len(cal_index)
    if v.shape != (nr,) or sv.shape != (nr,):
        errs.append(f"vapour arrays must have shape {(nr,)}")
    if n.shape != (nr, N_BINS) or sn.shape != (nr, N_BINS):
        errs.append(f"pnsd arrays must have shape {(nr, N_BINS)}")
    if fv.shape != (nc,) or fn.shape != (nc, N_BINS):
        errs.append(f"fit_vapour and fit_pnsd must have shapes {(nc,)} and "
                    f"{(nc, N_BINS)}")
    if errs:
        return errs, None
    for name, arr in (("vapour", v), ("vapour_log_sd", sv), ("pnsd", n),
                      ("pnsd_log_sd", sn), ("fit_vapour", fv), ("fit_pnsd", fn)):
        if not np.isfinite(arr).all():
            errs.append(f"predictions {name} contains non-finite values")
    if (v <= 0).any() or (fv <= 0).any():
        errs.append("predicted vapour must be positive")
    if (n < 0).any() or (fn < 0).any():
        errs.append("predicted counts must be non-negative")
    if (sv <= 0).any() or (sn <= 0).any():
        errs.append("predictive log standard deviations must be positive")

    # ---- airmass_history.csv
    rows = read_index(os.path.join(out_dir, "airmass_history.csv"))
    if len(rows) != len(queries):
        errs.append(f"airmass_history.csv has {len(rows)} rows, expected "
                    f"{len(queries)}")
        return errs, None
    by_id = {}
    for r in rows:
        try:
            qid = int(r["query_id"])
            age = float(r["mean_particle_age_hr"])
            fr = np.array([float(r[f"source_fraction_{c}"])
                           for c in ("A", "B", "C")])
        except (TypeError, ValueError, KeyError):
            errs.append("airmass_history.csv has a malformed row")
            return errs, None
        if not np.isfinite(age) or not np.isfinite(fr).all():
            errs.append(f"query {qid} has non-finite values")
            continue
        if age < 0:
            errs.append(f"query {qid} has negative mean particle age")
        if (fr < -1e-6).any():
            errs.append(f"query {qid} has a negative source fraction")
        if abs(fr.sum() - 1.0) > thr["source_sum_tol"]:
            errs.append(f"query {qid} source fractions sum to {fr.sum():.4f}")
        by_id[qid] = (age, fr)
    missing = set(range(len(queries))) - set(by_id)
    if missing:
        errs.append(f"airmass_history.csv is missing query ids "
                    f"{sorted(missing)[:6]}")

    # ---- attribution.json
    try:
        att = json.load(open(os.path.join(out_dir, "attribution.json")))
    except (ValueError, OSError) as e:
        return errs + [f"attribution.json is not readable JSON: {e}"], None
    A = {}
    for k in ("Q00", "Q10", "Q01", "Q11", "A_E", "A_C", "A_E_sd", "A_C_sd",
              "A_E_sd_sources_fixed", "A_C_sd_sources_fixed",
              thr["corr_key"]):
        try:
            A[k] = _num(att.get(k))
        except ValueError:
            errs.append(f"attribution.json {k} must be a finite number")
    if errs:
        return errs, None
    for k in ("A_E_sd", "A_C_sd", "A_E_sd_sources_fixed",
              "A_C_sd_sources_fixed"):
        if A[k] <= 0:
            errs.append(f"attribution.json {k} must be positive")
    if not -1.0 <= A[thr["corr_key"]] <= 1.0:
        errs.append("a correlation must lie in [-1, 1]")
    if errs:
        return errs, None
    return [], dict(post=P, vapour=v, vapour_sd=sv, pnsd=n, pnsd_sd=sn,
                    fit_vapour=fv, fit_pnsd=fn, history=by_id, attr=A)


def gate_bounds(sub, prior):
    errs = []
    lo, hi = np.array(prior["lower"]), np.array(prior["upper"])
    th = sub["post"]["estimate"]
    for i, pid in enumerate(prior["param_ids"]):
        if not lo[i] <= th[i] <= hi[i]:
            errs.append(f"{pid} = {th[i]:.4f} is outside the published bounds "
                        f"[{lo[i]:.3f}, {hi[i]:.3f}]")
    return errs


def truth_arrays(truth, index):
    nv = np.empty(len(index)); nn = np.empty((len(index), N_BINS))
    ov = np.empty(len(index)); on = np.empty((len(index), N_BINS))
    qc = np.empty((len(index), N_BINS), dtype=int)
    st = []
    for r in index:
        i = int(r["row_id"])
        e, s, k = r["episode"], r["station"], int(r["time_index"])
        nv[i] = truth[f"{e}__{s}__vapour"][k]
        nn[i] = truth[f"{e}__{s}__counts"][k]
        ov[i] = truth[f"{e}__{s}__vapour_obs"][k]
        on[i] = truth[f"{e}__{s}__counts_obs"][k]
        qc[i] = truth[f"{e}__{s}__qc"][k]
        st.append(s)
    return nv, nn, ov, on, qc, np.array(st)


def _sigmas(stations, thr):
    sv = np.array([thr["sigma_vapour_log"][s] for s in stations])
    sn = np.array([thr["sigma_counts_log"][s] for s in stations])
    return sv, sn


def gate_vapour(sub, T, thr):
    tv, _, _, _, _, st = T
    sv, _ = _sigmas(st, thr)
    score = MT.log_nrmse(sub["vapour"], tv, sv)
    lim = thr["vapour_nrmse_max"]
    if score > lim:
        return [f"withheld vapour normalised log RMSE {score:.3f} exceeds "
                f"{lim:.3f}"], score
    return [], score


def gate_pnsd(sub, T, thr):
    _, tn, _, _, _, st = T
    _, sn = _sigmas(st, thr)
    mask = tn >= thr["count_valid_min"]
    score = MT.log_rmse_counts(sub["pnsd"], tn, np.repeat(sn[:, None], N_BINS, 1),
                               mask)
    lim = thr["pnsd_logrmse_max"]
    if score > lim:
        return [f"withheld size-distribution log RMSE {score:.3f} exceeds "
                f"{lim:.3f} over {int(mask.sum())} channels"], score
    return [], score


def gate_growth(sub, T, centres, thr):
    _, tn, _, _, _, _ = T
    mae, n = MT.diameter_mae(sub["pnsd"], tn, centres, thr["diam_min_total"])
    lim = thr["diameter_mae_max_nm"]
    if mae > lim:
        return [f"number-weighted diameter MAE {mae:.3f} nm exceeds "
                f"{lim:.3f} nm over {n} samples"], mae
    return [], mae


def gate_history(sub, qtruth, thr):
    errs = []
    ap = np.array([sub["history"][q["query_id"]][0] for q in qtruth])
    at = np.array([q["age_hr"] for q in qtruth])
    fp = np.array([sub["history"][q["query_id"]][1] for q in qtruth])
    ft = np.array([q["frac"] for q in qtruth])
    mae, n1 = MT.age_mae(ap, at)
    l1, n2 = MT.source_l1(fp, ft)
    if mae > thr["age_mae_max_hr"]:
        errs.append(f"mean particle age MAE {mae:.3f} h exceeds "
                    f"{thr['age_mae_max_hr']:.3f} h over {n1} queries")
    if l1 > thr["source_l1_max"]:
        errs.append(f"source-fraction mean L1 error {l1:.3f} exceeds "
                    f"{thr['source_l1_max']:.3f} over {n2} queries")
    return errs, (mae, l1)


def gate_per_episode(sub, T, index, thr):
    rows_by_ep = {}
    for r in index:
        rows_by_ep.setdefault(r["episode"], []).append(int(r["row_id"]))
    tv, tn, _, _, _, st = T
    sv, sn = _sigmas(st, thr)
    passed, detail = [], []
    for e in thr["hard_episodes"]:
        idx = np.array(rows_by_ep.get(e, []), dtype=int)
        if idx.size == 0:
            continue
        nv = MT.log_nrmse(sub["vapour"][idx], tv[idx], sv[idx])
        m = tn[idx] >= thr["count_valid_min"]
        ln = MT.log_rmse_counts(sub["pnsd"][idx], tn[idx],
                                np.repeat(sn[idx][:, None], N_BINS, 1), m)
        ok = nv <= thr["episode_vapour_nrmse_max"] and \
            ln <= thr["episode_pnsd_logrmse_max"]
        detail.append(f"{e}: vapour {nv:.3f}, pnsd {ln:.3f}"
                      f"{' pass' if ok else ' fail'}")
        if ok:
            passed.append(e)
    hard = [e for e in thr["hard_episodes"] if e in rows_by_ep]
    if len(passed) < len(hard):
        failed = sorted(set(hard) - set(passed))
        return [f"every withheld episode must pass on its own; "
                f"{', '.join(failed)} did not ({'; '.join(detail)})"], detail
    return [], detail


def gate_calibration(sub, T, thr):
    """Predictive spreads against the withheld noisy observations."""
    errs = []
    _, tn, ov, on, qc, _ = T
    lv = np.log(ov); mv = np.log(sub["vapour"])
    cov_v = MT.coverage(lv, mv, sub["vapour_sd"])
    ls_v = MT.gaussian_log_score(lv, mv, sub["vapour_sd"])
    m = qc > 0
    ln = np.log(on[m]); mn = np.log(np.maximum(sub["pnsd"][m], 1e-3))
    cov_n = MT.coverage(ln, mn, sub["pnsd_sd"][m])
    ls_n = MT.gaussian_log_score(ln, mn, sub["pnsd_sd"][m])
    lo, hi = thr["coverage_90"]
    for name, c in (("vapour", cov_v), ("counts", cov_n)):
        if not lo <= c <= hi:
            errs.append(f"90% predictive intervals cover {c:.3f} of the "
                        f"withheld {name} observations, outside "
                        f"[{lo:.2f}, {hi:.2f}]")
    if ls_v > thr["log_score_vapour_max"]:
        errs.append(f"mean Gaussian log score for vapour {ls_v:.3f} exceeds "
                    f"{thr['log_score_vapour_max']:.3f}")
    if ls_n > thr["log_score_counts_max"]:
        errs.append(f"mean Gaussian log score for counts {ls_n:.3f} exceeds "
                    f"{thr['log_score_counts_max']:.3f}")
    if np.median(sub["vapour_sd"]) > thr["median_sd_max"] or \
            np.median(sub["pnsd_sd"][m]) > thr["median_sd_max"]:
        errs.append("median predictive log sd is implausibly wide")
    return errs, dict(cov_v=cov_v, cov_n=cov_n, ls_v=ls_v, ls_n=ls_n)


def gate_posterior(sub, truth_theta, prior, thr):
    """Honesty of the posterior: width bands and degrees of freedom."""
    errs = []
    P = sub["post"]
    ids = list(prior["param_ids"])
    ratio = P["posterior_prior_sd_ratio"]
    sd_check = P["posterior_sd"] / np.sqrt(np.diag(np.array(prior["covariance"])))
    if np.abs(ratio - sd_check).max() > 0.02:
        errs.append("posterior_prior_sd_ratio is not posterior_sd divided by "
                    "the prior sd")
    for pid, band in thr["sd_ratio_bands"].items():
        i = ids.index(pid)
        if not band[0] <= ratio[i] <= band[1]:
            errs.append(f"posterior/prior sd ratio for {pid} is {ratio[i]:.3f}, "
                        f"outside [{band[0]:.2f}, {band[1]:.2f}]")
    lo, hi = thr["dof_signal"]
    if not lo <= P["dof_signal"] <= hi:
        errs.append(f"dof_signal = {P['dof_signal']:.2f} is outside "
                    f"[{lo:.1f}, {hi:.1f}]")
    # Point values and credible intervals are not graded against the truth:
    # two correct solvers with different numerical diffusion sit at different
    # points along the same valley, far apart in units of the data-limited
    # posterior width, while predicting the observations equally well.  The
    # count is kept as a diagnostic only.
    inside = int(np.sum((truth_theta >= P["ci_lower"]) &
                        (truth_theta <= P["ci_upper"])))
    return errs, inside


def gate_attribution(sub, truth_attr, thr):
    errs = []
    A = sub["attr"]
    total = A["Q11"] - A["Q00"]
    if abs((A["A_E"] + A["A_C"]) - total) > 1e-4 * max(abs(total), 1.0):
        errs.append("A_E + A_C must equal Q11 - Q00")
    for key in ("A_E", "A_C"):
        tv = truth_attr[key]
        if abs(A[key] - tv) > thr["attr_tol"][key]:
            errs.append(f"{key} = {A[key]:.2f} cm-3 is outside "
                        f"{thr['attr_tol'][key]:.2f} cm-3 of the truth")
        lo, hi = thr["attr_sd_band"][key]
        sd = A[key + "_sd"]
        if not lo <= sd <= hi:
            errs.append(f"{key}_sd = {sd:.2f} cm-3 is outside [{lo:.2f}, "
                        f"{hi:.2f}]; the reported confidence does not match "
                        f"what these data support")
    return errs


def gate_sensitivity(sub, thr):
    errs = []
    A = sub["attr"]
    key = thr["sensitivity_key"]
    sd, sd_r = A[key + "_sd"], A[key + "_sd_sources_fixed"]
    if sd_r >= sd:
        errs.append(f"holding the regional source strengths fixed must "
                    f"reduce the uncertainty in {key}; you report {sd_r:.3f} "
                    f"fixed and {sd:.3f} free")
        return errs
    share = 1.0 - (sd_r / sd) ** 2
    lo, hi = thr["source_variance_share"]
    if not lo <= share <= hi:
        errs.append(f"source-strength uncertainty accounts for {share:.2f} "
                    f"of the variance of {key}, outside [{lo:.2f}, {hi:.2f}]")
    c = A[thr["corr_key"]]
    lo, hi = thr["corr_band"]
    if not lo <= c <= hi:
        errs.append(f"{thr['corr_key']} = {c:.3f} is outside "
                    f"[{lo:.2f}, {hi:.2f}]")
    return errs
