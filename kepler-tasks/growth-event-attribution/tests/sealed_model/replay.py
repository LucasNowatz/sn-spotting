"""Sealed re-run of the submitted estimate.

Recomputed with the generator's own solver on the public 4 km grid, from the
submitted estimate alone (meteorology is regenerated analytically):

  * withheld predictions and fitted values, held to the submitted ones inside
    an envelope loose enough for any correct discretisation and tight enough
    to reject numbers unrelated to the estimate;
  * the four counterfactual event statistics.

Separately, the calibration chi-square is recomputed from the *submitted*
fitted values with the full published covariance, which is arithmetic on the
agent's own numbers and carries a tight tolerance.
"""
import os
import sys

import numpy as np
from netCDF4 import Dataset
from scipy.linalg import cholesky, solve_triangular

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import driver as D                                          # noqa: E402
import scenarios as SC                                      # noqa: E402
import simulator as SM                                      # noqa: E402
import instruments as IN                                    # noqa: E402
import prior as PR                                          # noqa: E402

COUNT_FLOOR = 4.0
OBSERVED = ["E01", "E02", "E03", "E04", "E05", "E06"]


def _rows(index):
    want = {}
    for r in index:
        want.setdefault(r["episode"], []).append((int(r["row_id"]), r["station"], int(r["time_index"])))
    return want


def _gather(runs, want, n):
    tv = np.zeros(n); tn = np.zeros((n, 12))
    for e, rr in want.items():
        s = runs[e][0]
        for row, sid, k in rr:
            tv[row] = s[sid]["vapour"][k]; tn[row] = s[sid]["counts"][k]
    return tv, tn


def _disc(sv_, sn_, tv, tn, stations, lim):
    sv = np.array([lim["sigma_vapour_log"][s] for s in stations])
    sn = np.array([lim["sigma_counts_log"][s] for s in stations])
    dv = float(np.sqrt(np.mean(((np.log(np.maximum(sv_, IN.LOG_FLOOR_V))
                                 - np.log(np.maximum(tv, IN.LOG_FLOOR_V))) / sv) ** 2)))
    m = tn >= lim["count_valid_min"]
    dl = float(np.sqrt(np.mean(((np.log(np.maximum(sn_, 0.0) + COUNT_FLOOR) - np.log(tn + COUNT_FLOOR))
                                / np.repeat(sn[:, None], 12, 1))[m] ** 2))) if m.any() else np.inf
    return dv, dl


def chi2_from_fit(fit_v, fit_n, cindex, data_dir):
    by = {}
    for r in cindex:
        by.setdefault((r["episode"], r["station"]), []).append((int(r["time_index"]), int(r["row_id"])))
    cv = cn = 0.0
    for e in OBSERVED:
        with Dataset(os.path.join(data_dir, "episodes", e, "station_records.nc")) as d:
            sid = [str(x) for x in d["station_id"][:]]
            v = np.array(d["vapour"][:], float); n = np.array(d["counts"][:], float)
            qc = np.array(d["counts_flag"][:], int); qv = np.array(d["vapour_flag"][:], int)
        for i, s in enumerate(sid):
            if qv[i, 0] < 0:
                continue
            idx = np.array([rid for _, rid in sorted(by[(e, s)])])
            hv = np.log(np.maximum(fit_v[idx], IN.LOG_FLOOR_V))
            rv = solve_triangular(cholesky(IN.vapour_block(s), lower=True), np.log(v[i]) - hv, lower=True)
            cv += float(rv @ rv)
            mm = (qc[i] > 0).ravel()
            hn = np.log(np.maximum(fit_n[idx].ravel()[mm], IN.LOG_FLOOR_N))
            rn = solve_triangular(cholesky(IN.counts_block(s)[np.ix_(mm, mm)], lower=True),
                                  np.log(n[i].ravel()[mm]) - hn, lower=True)
            cn += float(rn @ rn)
    return cv + cn, cv


def check(sub, windex, cindex, data_dir, lim):
    errs = []
    theta = np.asarray(sub["post"]["estimate"], float)
    P = sub["pred"]

    chi2_fit, chi2_fit_v = chi2_from_fit(P["fit_vapour"], P["fit_pnsd"], cindex, data_dir)
    rep = sub["post"]["chi2_calibration"]
    if abs(rep - chi2_fit) > lim["chi2_report_tol"]:
        errs.append(f"chi2_calibration = {rep:.1f} but recomputing it from your fitted values with "
                    f"the full published covariance gives {chi2_fit:.1f}")
    if abs(sub["post"]["chi2_vapour"] - chi2_fit_v) > lim["chi2_stream_report_tol"]:
        errs.append(f"chi2_by_stream vapour = {sub['post']['chi2_vapour']:.1f} but recomputation "
                    f"gives {chi2_fit_v:.1f}")
    if chi2_fit > lim["chi2_max"]:
        errs.append(f"the calibration chi-square of your fitted values, {chi2_fit:.1f}, exceeds "
                    f"{lim['chi2_max']:.0f}; the fit does not explain the visible observations")

    want, want_c = _rows(windex), _rows(cindex)
    episodes = sorted(set(want) | set(want_c))
    st = {e: sorted({s for _, s, _ in want.get(e, [])} | {s for _, s, _ in want_c.get(e, [])})
          for e in episodes}
    p = PR.to_physical(theta)
    runs = {}
    for e in episodes:
        sc, res = D.run(e, p)
        runs[e] = (D.sample_network(res, sc, st[e]), res)
    tv, tn = _gather(runs, want, len(windex))
    dv, dl = _disc(P["vapour"], P["pnsd"], tv, tn, [r["station"] for r in windex], lim)
    fv, fn = _gather(runs, want_c, len(cindex))
    fdv, fdl = _disc(P["fit_vapour"], P["fit_pnsd"], fv, fn, [r["station"] for r in cindex], lim)
    for name, val, key in (("withheld vapour", dv, "consistency_vapour_max"),
                           ("withheld size distributions", dl, "consistency_pnsd_max"),
                           ("fitted vapour", fdv, "consistency_vapour_max"),
                           ("fitted size distributions", fdl, "consistency_pnsd_max")):
        if val > lim[key]:
            errs.append(f"a sealed re-run of your estimate gives {name} that differ from what "
                        f"you submitted by {val:.2f} sigma, above the {lim[key]:.2f} envelope")

    cfs = dict(Q00=theta.copy(), Q10=theta.copy(), Q01=theta.copy(), Q11=theta.copy())
    cfs["Q00"][10] = 0.0; cfs["Q00"][11] = 0.0; cfs["Q10"][11] = 0.0; cfs["Q01"][10] = 0.0
    q = {}
    for k, t in cfs.items():
        pk = PR.to_physical(t); vals = []
        for e in SC.EVENT_EPISODES:
            vals.append(SM.event_statistic(runs[e][1]["n"]) if k == "Q11"
                        else SM.event_statistic(D.run(e, pk)[1]["n"]))
        q[k] = float(np.mean(vals))
    for k in ("Q00", "Q10", "Q01", "Q11"):
        if abs(sub["attr"][k] - q[k]) > lim["statistic_tol"]:
            errs.append(f"{k} = {sub['attr'][k]:.3f} cm-3 but a sealed re-run from your own estimate "
                        f"gives {q[k]:.3f} cm-3; the counterfactual was not run correctly")
    return errs, dict(dv=dv, dl=dl, fdv=fdv, fdl=fdl, chi2_fit=chi2_fit, chi2_fit_v=chi2_fit_v, q=q)
