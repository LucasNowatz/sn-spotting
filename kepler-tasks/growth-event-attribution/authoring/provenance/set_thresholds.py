"""Set the verifier thresholds from measurement, not by hand.

Each limit has to clear two bars at once: comfortably above what a correct
solution achieves (the measured floor and the oracle's own score), and
comfortably below what the cheapest shortcut achieves.  This script reads
the achievable floor, the oracle's scores (with the trusted-forward replay)
and every baseline's scores, applies the documented rules, and refuses to
write a threshold file if any accuracy limit cannot satisfy both bars.

    python set_thresholds.py <oracle_dir> <baseline_root>
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import score_submission as SS                               # noqa: E402

# Each accuracy limit is the larger of a multiple of the achievable floor and
# a multiple of the oracle's own score, and must stay below a fraction of the
# nearest baseline that has to be caught by that metric.  The wrong-operator
# attempt keeps the oracle's vapour and lineage numbers by construction and is
# caught by the diameter gate, so it enters only that metric's baseline set.
SHORTCUTS = ("climatology", "nearest_station")
RULES = {
    "vapour":   dict(floor=4.0, oracle=2.5, headroom=0.40, base=SHORTCUTS),
    "pnsd":     dict(floor=1.8, oracle=1.8, headroom=0.40, base=SHORTCUTS),
    "diameter": dict(floor=4.0, oracle=2.5, headroom=0.70,
                     base=SHORTCUTS + ("wrong_operator",)),
    "age":      dict(floor=3.0, oracle=2.5, headroom=0.45, base=SHORTCUTS),
    "source":   dict(floor=3.0, oracle=2.5, headroom=0.45, base=SHORTCUTS),
}
LIMIT_KEY = dict(vapour="vapour_nrmse_max", pnsd="pnsd_logrmse_max",
                 diameter="diameter_mae_max_nm", age="age_mae_max_hr",
                 source="source_l1_max")
FLOOR_KEY = dict(vapour="vapour_nrmse", pnsd="pnsd_logrmse",
                 diameter="diameter_mae", age="age_mae", source="source_l1")
EPISODE_SLACK = 1.6
CONSISTENCY_SLACK = 2.0


def main():
    oracle_dir, base_root = sys.argv[1], sys.argv[2]
    thr = json.load(open(os.path.join(ROOT, "tests", "thresholds.json")))
    ctx = SS.context()
    floor = json.load(open(os.path.join(HERE, "floor.json")))
    oracle = SS.score(oracle_dir, ctx, replay=True)
    base = {}
    for name in sorted(os.listdir(base_root)):
        base[name] = SS.score(os.path.join(base_root, name), ctx,
                              replay=(name in ("linearised_attribution",
                                               "wrong_operator")))
    report, ok = [], True

    # --- accuracy limits ---------------------------------------------------
    for name, rule in RULES.items():
        f = floor[FLOOR_KEY[name]]
        o = oracle[name]
        nearest = min(base[b][name] for b in rule["base"]
                      if b in base and np.isfinite(base[b][name]))
        want = max(rule["floor"] * f, rule["oracle"] * o)
        ceiling = rule["headroom"] * nearest
        if want > ceiling:
            ok = False
            note = f"CANNOT SEPARATE: needs {want:.4g}, nearest baseline {nearest:.4g}"
        else:
            note = f"floor {f:.4g}, oracle {o:.4g}, nearest baseline {nearest:.4g}"
        thr[LIMIT_KEY[name]] = float(want)
        report.append((name, want, note))
    thr["episode_vapour_nrmse_max"] = float(thr["vapour_nrmse_max"] * EPISODE_SLACK)
    thr["episode_pnsd_logrmse_max"] = float(thr["pnsd_logrmse_max"] * EPISODE_SLACK)
    thr["consistency_vapour_max"] = float(max(
        thr["vapour_nrmse_max"] * CONSISTENCY_SLACK, 6.0 * oracle["replay"]["dv"]))
    thr["consistency_pnsd_max"] = float(max(
        thr["pnsd_logrmse_max"] * CONSISTENCY_SLACK, 6.0 * oracle["replay"]["dl"]))

    # --- predictive calibration -------------------------------------------
    cal = oracle["calibration"]
    thr["coverage_90"] = [0.80, 0.97]
    thr["log_score_vapour_max"] = float(cal["ls_v"] + 0.25)
    thr["log_score_counts_max"] = float(cal["ls_n"] + 0.25)
    pr = np.load(os.path.join(oracle_dir, "predictions.npz"))
    thr["median_sd_max"] = float(4.0 * max(np.median(pr["vapour_log_sd"]),
                                           np.median(pr["pnsd_log_sd"])))

    # --- posterior honesty -------------------------------------------------
    ratio = np.array(oracle["sd_ratio"])
    ids = ctx["prior"]["param_ids"]
    order = np.argsort(ratio)
    chosen = [order[0], order[-1], order[-2]]
    bands = {}
    for i in chosen:
        r = float(ratio[i])
        bands[ids[i]] = [float(max(r / 3.0, 1e-4)), float(min(3.0 * r, 1.05))]
    thr["sd_ratio_bands"] = bands
    thr["dof_signal"] = [float(oracle["dof"] - 1.5), float(min(oracle["dof"] + 1.2, 12.0))]

    # --- attribution --------------------------------------------------------
    a = oracle["attr"]
    ta = ctx["params"]["attribution"]
    tol = {}
    for k in ("A_E", "A_C"):
        err = abs(a[k + "_err"])
        fl = abs(floor["q_public"][k] - floor["q_truth"][k])
        tol[k] = float(max(3.0 * err, 4.0 * fl, 0.06 * abs(ta[k])))
    thr["attr_tol"] = tol
    thr["attr_sd_band"] = {k: [float(0.5 * a[k + "_sd"]), float(2.0 * a[k + "_sd"])]
                           for k in ("A_E", "A_C")}
    key = thr["sensitivity_key"]
    share = a["share_E"] if key == "A_E" else a["share_C"]
    thr["source_variance_share"] = [float(max(share - 0.20, 0.05)),
                                    float(min(share + 0.15, 0.97))]
    c = a["corr"]
    thr["corr_band"] = [float(max(c - 0.22, -1.0)), float(min(c + 0.22, 1.0))]

    # --- replay agreement ----------------------------------------------------
    rp = oracle["replay"]
    # The reported chi-square is checked against a recomputation from the
    # submitted fitted values, which is pure arithmetic on the agent's own
    # numbers, so the tolerance is one per cent; the band on that value only
    # has to reject a fit that does not explain the data, so it is open below
    # and 1.5 times the oracle above (a correct solver at the truth sits at
    # 1.11 times the oracle).
    thr["chi2_report_tol"] = float(0.01 * rp["chi2_fit"])
    thr["chi2_stream_report_tol"] = float(0.01 * rp["chi2_fit"])
    thr["chi2_band"] = [0.0, float(1.50 * rp["chi2_fit"])]
    thr["consistency_fit_vapour_max"] = thr["consistency_vapour_max"]
    thr["consistency_fit_pnsd_max"] = thr["consistency_pnsd_max"]
    dq = max(abs(v) for v in rp["dq"].values())
    fq = max(abs(floor["q_public"][k] - floor["q_truth"][k])
             for k in ("Q00", "Q10", "Q01", "Q11"))
    thr["burden_tol"] = float(max(4.0 * dq, 2.0 * fq))
    thr["_comment"] = ("every limit set by authoring/provenance/set_thresholds.py "
                       "from the measured floor, the oracle and the baselines; "
                       "see authoring/provenance/CALIBRATION.md")

    print(f"{'metric':10s} {'limit':>9s}  note")
    for name, ch, note in report:
        print(f"{name:10s} {ch:9.4g}  {note}")
    if not ok:
        print("\nREFUSING to write thresholds: an accuracy limit cannot "
              "separate the oracle from the baselines")
        return 1
    json.dump(thr, open(os.path.join(ROOT, "tests", "thresholds.json"), "w"),
              indent=2)
    json.dump(dict(oracle=oracle, baselines=base),
              open(os.path.join(HERE, "threshold_inputs.json"), "w"),
              indent=1, default=float)
    print("\nthresholds.json written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
