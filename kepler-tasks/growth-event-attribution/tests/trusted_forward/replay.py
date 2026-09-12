"""Trusted forward re-run of the submitted parameter vector (sealed).

Recomputed here from the submitted estimate alone, with the generator's own
solver on the public 5 km grid and meteorology regenerated analytically, so
no gridded input is baked into the verifier image:

  * the withheld vapour and size-distribution predictions and the fitted
    values at the calibration rows, which must agree with the submitted ones
    inside an envelope loose enough for any correct discretisation and tight
    enough to reject numbers unrelated to the estimate;
  * the four counterfactual event statistics, which must agree with the
    submitted ones (an attribution asserted without the coupled re-runs
    behind it disagrees).

Separately, the calibration chi-square is recomputed from the *submitted*
fitted values with the full published covariance, which is pure arithmetic on
the agent's own numbers and so carries a tight tolerance: a fit that used a
diagonal covariance and reports the chi-square it computed disagrees.
"""

import os
import sys

import numpy as np
from netCDF4 import Dataset
from scipy.linalg import cholesky, solve_triangular

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import driver as D                                          # noqa: E402
import episodes as E                                        # noqa: E402
import observe as O                                         # noqa: E402
import prior as PR                                          # noqa: E402
import truth_model as M                                     # noqa: E402

COUNT_FLOOR = 5.0
PRED_FLOOR = 0.05          # cm-3, published log floor for counts
VAP_FLOOR = 0.001          # ug m-3, published log floor for vapour
OBSERVED = ["E01", "E02", "E03", "E04", "E05", "E06"]
SEEN = ["S1", "S2", "S3", "S4", "S5"]


def _theta_to_p(theta):
    return PR.to_physical(np.asarray(theta, float))


def _rows(index):
    want = {}
    for r in index:
        want.setdefault(r["episode"], []).append(
            (int(r["row_id"]), r["station"], int(r["time_index"])))
    return want


def _gather(runs, want, n_rows):
    tv = np.zeros(n_rows); tn = np.zeros((n_rows, 12))
    for e, rows in want.items():
        s = runs[e][0]
        for row, sid, k in rows:
            tv[row] = s[sid]["vapour"][k]
            tn[row] = s[sid]["counts"][k]
    return tv, tn


def _discrepancy(sub_v, sub_n, tv, tn, stations, thr):
    sv = np.array([thr["sigma_vapour_log"][s] for s in stations])
    sn = np.array([thr["sigma_counts_log"][s] for s in stations])
    dv = float(np.sqrt(np.mean(((np.log(np.maximum(sub_v, VAP_FLOOR))
                                 - np.log(np.maximum(tv, VAP_FLOOR))) / sv) ** 2)))
    m = tn >= thr["count_valid_min"]
    if m.any():
        dl = float(np.sqrt(np.mean(
            ((np.log(np.maximum(sub_n, 0.0) + COUNT_FLOOR)
              - np.log(tn + COUNT_FLOOR))
             / np.repeat(sn[:, None], 12, 1))[m] ** 2)))
    else:
        dl = np.inf
    return dv, dl


def chi2_from_fit(sub, cal_index, data_dir):
    """Calibration chi-square from the submitted fitted values, full R."""
    by_key = {}
    for r in cal_index:
        by_key.setdefault((r["episode"], r["station"]), []).append(
            (int(r["time_index"]), int(r["row_id"])))
    chi2_v = chi2_n = 0.0
    for e in OBSERVED:
        with Dataset(os.path.join(data_dir, "episodes", e,
                                  "observations.nc")) as d:
            sid = [str(x) for x in d["station_id"][:]]
            v = np.array(d["vapour"][:], float)
            n = np.array(d["counts"][:], float)
            qc = np.array(d["qc"][:], int)
            qv = np.array(d["qc_vapour"][:], int)
        for i, sname in enumerate(sid):
            if qv[i, 0] < 0:
                continue
            rows = sorted(by_key[(e, sname)])
            idx = np.array([rid for _, rid in rows])
            hv = np.log(np.maximum(sub["fit_vapour"][idx], VAP_FLOOR))
            Lv = cholesky(O.vapour_block(sname), lower=True)
            rv = solve_triangular(Lv, np.log(v[i]) - hv, lower=True)
            chi2_v += float(rv @ rv)
            mm = (qc[i] > 0).ravel()
            hn = np.log(np.maximum(sub["fit_pnsd"][idx].ravel()[mm], PRED_FLOOR))
            Ln = cholesky(O.counts_block(sname)[np.ix_(mm, mm)], lower=True)
            rn = solve_triangular(Ln, np.log(n[i].ravel()[mm]) - hn, lower=True)
            chi2_n += float(rn @ rn)
    return chi2_v + chi2_n, chi2_v


def check(sub, index, cal_index, data_dir, thr):
    errs = []
    theta = np.asarray(sub["post"]["estimate"], float)

    # ---- chi-square from the submitted fitted values ---------------------
    chi2_fit, chi2_fit_v = chi2_from_fit(sub, cal_index, data_dir)
    rep = sub["post"]["chi2_calibration"]
    if abs(rep - chi2_fit) > thr["chi2_report_tol"]:
        errs.append(f"chi2_calibration = {rep:.1f} but recomputing it from "
                    f"your fitted values with the full published covariance "
                    f"gives {chi2_fit:.1f}; a diagonal covariance gives a "
                    f"different number")
    if abs(sub["post"]["chi2_vapour"] - chi2_fit_v) > thr["chi2_stream_report_tol"]:
        errs.append(f"chi2_by_stream vapour = {sub['post']['chi2_vapour']:.1f} "
                    f"but recomputation from your fitted values gives "
                    f"{chi2_fit_v:.1f}")
    lo, hi = thr["chi2_band"]
    if not lo <= chi2_fit <= hi:
        errs.append(f"the calibration chi-square of your fitted values is "
                    f"{chi2_fit:.1f}, outside [{lo:.0f}, {hi:.0f}]; the fit "
                    f"does not explain the visible observations")

    # ---- trusted forward at the submitted estimate -----------------------
    want = _rows(index)
    want_cal = _rows(cal_index)
    episodes = sorted(set(want) | set(want_cal))
    st_by_ep = {e: sorted({s for _, s, _ in want.get(e, [])}
                          | {s for _, s, _ in want_cal.get(e, [])})
                for e in episodes}
    p = _theta_to_p(theta)
    runs = {}
    for e in episodes:
        ep, res = D.run(e, p)
        runs[e] = (D.sample(res, ep, st_by_ep[e]), res)

    tv, tn = _gather(runs, want, len(index))
    dv, dl = _discrepancy(sub["vapour"], sub["pnsd"], tv, tn,
                          [r["station"] for r in index], thr)
    if dv > thr["consistency_vapour_max"]:
        errs.append(f"the reported parameters give withheld vapour that "
                    f"differs from the submitted predictions by {dv:.2f} "
                    f"sigma, above the {thr['consistency_vapour_max']:.2f} "
                    f"envelope")
    if dl > thr["consistency_pnsd_max"]:
        errs.append(f"the reported parameters give withheld size "
                    f"distributions that differ from the submitted "
                    f"predictions by {dl:.2f} sigma, above the "
                    f"{thr['consistency_pnsd_max']:.2f} envelope")
    fv, fn = _gather(runs, want_cal, len(cal_index))
    fdv, fdl = _discrepancy(sub["fit_vapour"], sub["fit_pnsd"], fv, fn,
                            [r["station"] for r in cal_index], thr)
    if fdv > thr["consistency_fit_vapour_max"]:
        errs.append(f"the reported parameters give fitted vapour that "
                    f"differs from the submitted fitted values by {fdv:.2f} "
                    f"sigma, above the {thr['consistency_fit_vapour_max']:.2f} "
                    f"envelope")
    if fdl > thr["consistency_fit_pnsd_max"]:
        errs.append(f"the reported parameters give fitted size distributions "
                    f"that differ from the submitted fitted values by "
                    f"{fdl:.2f} sigma, above the "
                    f"{thr['consistency_fit_pnsd_max']:.2f} envelope")

    # chi-square of the trusted forward itself, as information only
    chi2_trusted, _ = chi2_from_fit(dict(fit_vapour=fv, fit_pnsd=fn),
                                    cal_index, data_dir)

    # ---- counterfactual event statistics ---------------------------------
    cfs = dict(Q00=theta.copy(), Q10=theta.copy(), Q01=theta.copy(),
               Q11=theta.copy())
    cfs["Q00"][10] = 0.0; cfs["Q00"][11] = 0.0
    cfs["Q10"][11] = 0.0
    cfs["Q01"][10] = 0.0
    q = {}
    for k, t in cfs.items():
        pk = _theta_to_p(t)
        vals = []
        for e in E.EVENT_EPISODES:
            if k == "Q11":
                vals.append(M.grown_number_statistic(runs[e][1]["n"]))
            else:
                _, res = D.run(e, pk)
                vals.append(M.grown_number_statistic(res["n"]))
        q[k] = float(np.mean(vals))
    for k in ("Q00", "Q10", "Q01", "Q11"):
        if abs(sub["attr"][k] - q[k]) > thr["burden_tol"]:
            errs.append(f"{k} = {sub['attr'][k]:.3f} cm-3 but an independent "
                        f"simulation from your own estimate gives {q[k]:.3f} "
                        f"cm-3; the counterfactual was not run correctly")
    return errs, dict(dv=dv, dl=dl, fdv=fdv, fdl=fdl, chi2_fit=chi2_fit,
                      chi2_fit_v=chi2_fit_v, chi2_trusted=chi2_trusted, q=q)
