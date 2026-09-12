"""Gate logic (sealed).  Each check returns a list of failure strings."""
import csv
import json
import os

import numpy as np

from . import scores as SC

N_BINS = 12
N_PARAM = 12
REGIONS = ("A", "B", "C")


def read_rows(path):
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
        errs.append(f"{name} is not numeric"); return None
    if a.size != n:
        errs.append(f"{name} has {a.size} entries, expected {n}"); return None
    if not np.isfinite(a).all():
        errs.append(f"{name} has non-finite entries"); return None
    return a


def read_submission(out_dir, windex, cindex, param_ids, events, lim):
    errs = []
    for f in ("posterior.json", "predictions.npz", "attribution.json", "region_shares.json"):
        if not os.path.exists(os.path.join(out_dir, f)):
            errs.append(f"missing artifact /app/output/{f}")
    if errs:
        return errs, None
    try:
        post = json.load(open(os.path.join(out_dir, "posterior.json")))
        att = json.load(open(os.path.join(out_dir, "attribution.json")))
        shr = json.load(open(os.path.join(out_dir, "region_shares.json")))
    except (ValueError, OSError) as e:
        return [f"a JSON artifact is not readable: {e}"], None
    if not all(isinstance(d, dict) for d in (post, att, shr)):
        return ["JSON artifacts must contain objects"], None
    if list(post.get("param_ids", [])) != list(param_ids):
        errs.append("posterior.json param_ids must equal the published order")
    P = {}
    for k in ("estimate", "posterior_sd", "ci_lower", "ci_upper", "posterior_prior_sd_ratio"):
        P[k] = _vec(post.get(k), N_PARAM, "posterior.json " + k, errs)
    for k in ("dof_signal", "chi2_calibration"):
        try:
            P[k] = _num(post.get(k))
        except ValueError:
            errs.append(f"posterior.json {k} must be a finite number")
    cs = post.get("chi2_by_stream")
    if not isinstance(cs, dict) or not all(k in cs for k in ("vapour", "counts")):
        errs.append("posterior.json chi2_by_stream needs vapour and counts")
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
        errs.append("ci_lower must be below ci_upper")
    if (P["posterior_prior_sd_ratio"] <= 0).any() or (P["posterior_prior_sd_ratio"] > 1.05).any():
        errs.append("posterior_prior_sd_ratio must be positive and not exceed the prior")

    try:
        pr = np.load(os.path.join(out_dir, "predictions.npz"))
    except Exception as e:                                  # noqa: BLE001
        return errs + [f"predictions.npz is not readable: {e}"], None
    need = ("vapour", "vapour_log_sd", "pnsd", "pnsd_log_sd", "fit_vapour", "fit_pnsd")
    for k in need:
        if k not in pr:
            errs.append(f"predictions.npz has no {k}")
    if errs:
        return errs, None
    A = {k: np.asarray(pr[k], float) for k in need}
    nr, nc = len(windex), len(cindex)
    shapes = dict(vapour=(nr,), vapour_log_sd=(nr,), pnsd=(nr, N_BINS), pnsd_log_sd=(nr, N_BINS),
                  fit_vapour=(nc,), fit_pnsd=(nc, N_BINS))
    for k, shp in shapes.items():
        if A[k].shape != shp:
            errs.append(f"{k} has shape {A[k].shape}, expected {shp}")
    if errs:
        return errs, None
    for k in need:
        if not np.isfinite(A[k]).all():
            errs.append(f"{k} has non-finite values")
    if (A["vapour"] <= 0).any() or (A["fit_vapour"] <= 0).any():
        errs.append("vapour values must be positive")
    if (A["pnsd"] < 0).any() or (A["fit_pnsd"] < 0).any():
        errs.append("counts must be non-negative")
    if (A["vapour_log_sd"] <= 0).any() or (A["pnsd_log_sd"] <= 0).any():
        errs.append("predictive log sd must be positive")

    T = {}
    for k in ("Q00", "Q10", "Q01", "Q11", "A_E", "A_C", "A_E_sd", "A_C_sd",
              "A_E_sd_sources_fixed", "A_C_sd_sources_fixed", lim["corr_key"]):
        try:
            T[k] = _num(att.get(k))
        except ValueError:
            errs.append(f"attribution.json {k} must be a finite number")
    if errs:
        return errs, None
    for k in ("A_E_sd", "A_C_sd", "A_E_sd_sources_fixed", "A_C_sd_sources_fixed"):
        if T[k] <= 0:
            errs.append(f"attribution.json {k} must be positive")
    if not -1.0 <= T[lim["corr_key"]] <= 1.0:
        errs.append("a correlation must lie in [-1, 1]")

    S = dict(domain={}, stations={})
    try:
        for e in events:
            S["domain"][e] = np.array([_num(shr["domain"][e][r]) for r in REGIONS])
            S["stations"][e] = {}
            for s in lim["share_stations"]:
                S["stations"][e][s] = np.array([_num(shr["stations"][e][s][r]) for r in REGIONS])
    except (KeyError, TypeError, ValueError):
        errs.append("region_shares.json must give finite A, B, C shares for every event day, "
                    "domain-wide and at every withheld station")
        return errs, None
    for e in events:
        vecs = [S["domain"][e]] + list(S["stations"][e].values())
        for v in vecs:
            if (v < -1e-6).any() or v.sum() > 1.0 + lim["share_sum_tol"]:
                errs.append(f"region shares for {e} must be non-negative and sum to at most 1")
                break
    if errs:
        return errs, None
    return [], dict(post=P, pred=A, attr=T, shares=S)


def bounds(sub, prior):
    lo, hi = np.array(prior["lower"]), np.array(prior["upper"])
    th = sub["post"]["estimate"]
    return [f"{pid} = {th[i]:.4f} is outside [{lo[i]:.3f}, {hi[i]:.3f}]"
            for i, pid in enumerate(prior["param_ids"]) if not lo[i] <= th[i] <= hi[i]]


def truth_arrays(truth, windex):
    n = len(windex)
    tv = np.empty(n); tn = np.empty((n, N_BINS)); ov = np.empty(n); on = np.empty((n, N_BINS))
    qc = np.empty((n, N_BINS), dtype=int); st = []
    for r in windex:
        i = int(r["row_id"]); e, s, k = r["episode"], r["station"], int(r["time_index"])
        tv[i] = truth[f"{e}__{s}__vapour"][k]; tn[i] = truth[f"{e}__{s}__counts"][k]
        ov[i] = truth[f"{e}__{s}__vapour_obs"][k]; on[i] = truth[f"{e}__{s}__counts_obs"][k]
        qc[i] = truth[f"{e}__{s}__flag"][k]; st.append(s)
    return dict(tv=tv, tn=tn, ov=ov, on=on, qc=qc, st=np.array(st))


def _sig(st, lim):
    return (np.array([lim["sigma_vapour_log"][s] for s in st]),
            np.array([lim["sigma_counts_log"][s] for s in st]))


def vapour(sub, T, lim):
    sv, _ = _sig(T["st"], lim)
    s = SC.log_nrmse(sub["pred"]["vapour"], T["tv"], sv)
    return ([f"withheld vapour log NRMSE {s:.3f} exceeds {lim['vapour_max']:.3f}"]
            if s > lim["vapour_max"] else []), s


def pnsd(sub, T, lim):
    _, sn = _sig(T["st"], lim)
    m = T["tn"] >= lim["count_valid_min"]
    s = SC.log_rmse_counts(sub["pred"]["pnsd"], T["tn"], np.repeat(sn[:, None], N_BINS, 1), m)
    return ([f"withheld size-distribution log RMSE {s:.3f} exceeds {lim['pnsd_max']:.3f}"]
            if s > lim["pnsd_max"] else []), s


def growth(sub, T, centres, lim):
    mae, n = SC.diameter_mae(sub["pred"]["pnsd"], T["tn"], centres, lim["diam_min_total"])
    return ([f"number-weighted diameter MAE {mae:.4f} nm exceeds {lim['diameter_max_nm']:.4f}"]
            if mae > lim["diameter_max_nm"] else []), mae


def per_episode(sub, T, windex, lim):
    by = {}
    for r in windex:
        by.setdefault(r["episode"], []).append(int(r["row_id"]))
    sv, sn = _sig(T["st"], lim)
    detail, failed = [], []
    for e in lim["hard_episodes"]:
        idx = np.array(by.get(e, []), dtype=int)
        if idx.size == 0:
            continue
        nv = SC.log_nrmse(sub["pred"]["vapour"][idx], T["tv"][idx], sv[idx])
        m = T["tn"][idx] >= lim["count_valid_min"]
        ln = SC.log_rmse_counts(sub["pred"]["pnsd"][idx], T["tn"][idx],
                                np.repeat(sn[idx][:, None], N_BINS, 1), m)
        ok = nv <= lim["episode_vapour_max"] and ln <= lim["episode_pnsd_max"]
        detail.append(f"{e}: vapour {nv:.3f}, pnsd {ln:.3f}{' pass' if ok else ' FAIL'}")
        if not ok:
            failed.append(e)
    return ([f"each withheld episode must pass on its own; {', '.join(failed)} did not "
             f"({'; '.join(detail)})"] if failed else []), detail


def region_shares(sub, truth, truth_json, events, lim):
    errs = []
    dom = np.array([np.abs(sub["shares"]["domain"][e] - np.array(truth_json["domain_region_shares"][e])).sum()
                    for e in events])
    sta = []
    for e in events:
        for s in lim["share_stations"]:
            sta.append(np.abs(sub["shares"]["stations"][e][s] - truth[f"{e}__{s}__shares"]).sum())
    sta = np.array(sta)
    if dom.mean() > lim["domain_share_l1_max"]:
        errs.append(f"domain region shares: mean L1 error {dom.mean():.3f} exceeds "
                    f"{lim['domain_share_l1_max']:.3f}")
    if sta.mean() > lim["station_share_l1_max"]:
        errs.append(f"withheld-station region shares: mean L1 error {sta.mean():.3f} exceeds "
                    f"{lim['station_share_l1_max']:.3f}")
    return errs, (float(dom.mean()), float(sta.mean()))


def calibration(sub, T, lim):
    errs = []
    P = sub["pred"]
    lv, mv = np.log(T["ov"]), np.log(P["vapour"])
    cov_v = SC.coverage(lv, mv, P["vapour_log_sd"]); ls_v = SC.log_score(lv, mv, P["vapour_log_sd"])
    m = T["qc"] > 0
    ln, mn = np.log(T["on"][m]), np.log(np.maximum(P["pnsd"][m], 1e-3))
    cov_n = SC.coverage(ln, mn, P["pnsd_log_sd"][m]); ls_n = SC.log_score(ln, mn, P["pnsd_log_sd"][m])
    for name, c, band in (("vapour", cov_v, lim["coverage_vapour"]), ("counts", cov_n, lim["coverage_counts"])):
        if not band[0] <= c <= band[1]:
            errs.append(f"90% intervals cover {c:.3f} of withheld {name} observations, outside "
                        f"[{band[0]:.2f}, {band[1]:.2f}]")
    if ls_v > lim["log_score_vapour_max"]:
        errs.append(f"vapour log score {ls_v:.3f} exceeds {lim['log_score_vapour_max']:.3f}")
    if ls_n > lim["log_score_counts_max"]:
        errs.append(f"counts log score {ls_n:.3f} exceeds {lim['log_score_counts_max']:.3f}")
    if np.median(P["vapour_log_sd"]) > lim["median_sd_max"] or np.median(P["pnsd_log_sd"][m]) > lim["median_sd_max"]:
        errs.append("median predictive log sd is implausibly wide")
    return errs, dict(cov_v=cov_v, cov_n=cov_n, ls_v=ls_v, ls_n=ls_n)


def posterior(sub, prior, lim):
    errs = []
    P = sub["post"]; ids = list(prior["param_ids"])
    ratio = P["posterior_prior_sd_ratio"]
    if np.abs(ratio - P["posterior_sd"] / np.sqrt(np.diag(np.array(prior["covariance"])))).max() > 0.02:
        errs.append("posterior_prior_sd_ratio is not posterior_sd over the prior sd")
    for pid, band in lim["sd_ratio_bands"].items():
        i = ids.index(pid)
        if not band[0] <= ratio[i] <= band[1]:
            errs.append(f"posterior/prior sd ratio for {pid} is {ratio[i]:.4f}, outside "
                        f"[{band[0]:.4f}, {band[1]:.4f}]")
    lo, hi = lim["dof_signal"]
    if not lo <= P["dof_signal"] <= hi:
        errs.append(f"dof_signal {P['dof_signal']:.2f} is outside [{lo:.2f}, {hi:.2f}]")
    return errs


def attribution(sub, truth_attr, lim):
    errs = []
    A = sub["attr"]
    total = A["Q11"] - A["Q00"]
    if abs((A["A_E"] + A["A_C"]) - total) > 1e-4 * max(abs(total), 1.0):
        errs.append("A_E + A_C must equal Q11 - Q00")
    for k in ("A_E", "A_C"):
        if abs(A[k] - truth_attr[k]) > lim["attr_tol"][k]:
            errs.append(f"{k} = {A[k]:.2f} cm-3 is outside {lim['attr_tol'][k]:.2f} cm-3 of the truth")
        lo, hi = lim["attr_sd_band"][k]
        if not lo <= A[k + "_sd"] <= hi:
            errs.append(f"{k}_sd = {A[k + '_sd']:.2f} is outside [{lo:.2f}, {hi:.2f}]")
    return errs


def sensitivity(sub, lim):
    errs = []
    A = sub["attr"]; key = lim["sensitivity_key"]
    sd, sd_r = A[key + "_sd"], A[key + "_sd_sources_fixed"]
    if sd_r >= sd:
        return [f"fixing the regional strengths must reduce the spread of {key}; "
                f"reported {sd_r:.3f} fixed and {sd:.3f} free"]
    share = 1.0 - (sd_r / sd) ** 2
    lo, hi = lim["source_variance_share"]
    if not lo <= share <= hi:
        errs.append(f"source-strength variance share of {key} is {share:.2f}, outside [{lo:.2f}, {hi:.2f}]")
    c = A[lim["corr_key"]]; lo, hi = lim["corr_band"]
    if not lo <= c <= hi:
        errs.append(f"{lim['corr_key']} = {c:.3f} is outside [{lo:.2f}, {hi:.2f}]")
    return errs
