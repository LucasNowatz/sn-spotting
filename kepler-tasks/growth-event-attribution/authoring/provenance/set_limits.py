"""Set every verifier limit from measurement.

Accuracy limits are the larger of a multiple of the achievable floor and a
multiple of the oracle's own score, and must stay below a fraction of the
nearest shortcut that the metric is responsible for catching; the script
refuses to write limits.json if any accuracy limit cannot separate the two.

    python set_limits.py <oracle_dir> <baseline_root>
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import score_submission as SS                               # noqa: E402

SHORTCUTS = ("climatology", "nearest_station")
RULES = {
    "vapour":         dict(key="vapour_max", floor=4.0, oracle=2.5, headroom=0.40, base=SHORTCUTS),
    "pnsd":           dict(key="pnsd_max", floor=1.8, oracle=1.8, headroom=0.40, base=SHORTCUTS),
    "diameter":       dict(key="diameter_max_nm", floor=3.0, oracle=2.5, headroom=0.70,
                           base=SHORTCUTS + ("wrong_operator",)),
    "shares_domain":  dict(key="domain_share_l1_max", floor=3.0, oracle=2.5, headroom=0.45, base=SHORTCUTS),
    "shares_station": dict(key="station_share_l1_max", floor=3.0, oracle=2.5, headroom=0.45, base=SHORTCUTS),
}
FLOOR_KEY = dict(vapour="vapour", pnsd="pnsd", diameter="diameter",
                 shares_domain="domain_share_l1", shares_station="station_share_l1")


def main():
    oracle_dir, base_root = sys.argv[1], sys.argv[2]
    lim = json.load(open(os.path.join(ROOT, "tests", "limits.json")))
    ctx = SS.context()
    floor = json.load(open(os.path.join(HERE, "floor.json")))
    # a correct solver with a different limiter, or on a finer grid, must not
    # be failed for that difference: the floor is the worst of the measured ones
    for name in ("floor_mc.json", "floor_fine.json"):
        alt = json.load(open(os.path.join(HERE, name)))
        for k in ("vapour", "pnsd", "diameter", "domain_share_l1", "station_share_l1", "chi2_truth"):
            floor[k] = max(floor[k], alt[k])
        for k in floor["q_public"]:
            if abs(alt["q_public"][k] - alt["q_truth"][k]) > abs(floor["q_public"][k] - floor["q_truth"][k]):
                floor["q_public"][k] = alt["q_public"][k]
    oracle = SS.score(oracle_dir, ctx, replay=True)
    base = {n: SS.score(os.path.join(base_root, n), ctx, replay=(n in ("linearised", "wrong_operator")))
            for n in sorted(os.listdir(base_root))}
    ok = True
    print(f"{'metric':16s} {'limit':>9s}  note")
    for name, rule in RULES.items():
        f, o = floor[FLOOR_KEY[name]], oracle[name]
        nearest = min(base[b][name] for b in rule["base"] if b in base and np.isfinite(base[b][name]))
        want = max(rule["floor"] * f, rule["oracle"] * o)
        if want > rule["headroom"] * nearest:
            ok = False; note = f"CANNOT SEPARATE: needs {want:.4g}, nearest {nearest:.4g}"
        else:
            note = f"floor {f:.4g}, oracle {o:.4g}, nearest baseline {nearest:.4g}"
        lim[rule["key"]] = float(want)
        print(f"{name:16s} {want:9.4g}  {note}")
    lim["episode_vapour_max"] = 1.6 * lim["vapour_max"]
    lim["episode_pnsd_max"] = 1.6 * lim["pnsd_max"]
    lim["consistency_vapour_max"] = 2.0 * lim["vapour_max"]
    lim["consistency_pnsd_max"] = 2.0 * lim["pnsd_max"]
    cal = oracle["calibration"]
    lim["coverage_vapour"] = [float(min(0.80, cal["cov_v"] - 0.05)), 0.97]
    lim["coverage_counts"] = [float(min(0.80, cal["cov_n"] - 0.05)), 0.97]
    lim["log_score_vapour_max"] = float(cal["ls_v"] + 0.25)
    lim["log_score_counts_max"] = float(cal["ls_n"] + 0.25)
    pr = np.load(os.path.join(oracle_dir, "predictions.npz"))
    lim["median_sd_max"] = float(4.0 * max(np.median(pr["vapour_log_sd"]), np.median(pr["pnsd_log_sd"])))
    ratio = np.array(oracle["sd_ratio"]); ids = ctx["prior"]["param_ids"]
    order = np.argsort(ratio)
    lim["sd_ratio_bands"] = {ids[i]: [float(max(ratio[i] / 3.0, 1e-4)), float(min(3.0 * ratio[i], 1.05))]
                             for i in (order[0], order[-1], order[-2])}
    lim["dof_signal"] = [float(oracle["dof"] - 1.5), float(min(oracle["dof"] + 1.2, 12.0))]
    a = oracle["attr"]; ta = ctx["tj"]["attribution"]
    lim["attr_tol"] = {k: float(max(2.0 * abs(a[k + "_err"]),
                                    4.0 * abs(floor["q_public"][k] - floor["q_truth"][k]),
                                    0.25 * abs(ta[k]))) for k in ("A_E", "A_C")}
    lim["attr_sd_band"] = {k: [float(0.5 * a[k + "_sd"]), float(2.0 * a[k + "_sd"])] for k in ("A_E", "A_C")}
    share = a["share_E"] if lim["sensitivity_key"] == "A_E" else a["share_C"]
    lim["source_variance_share"] = [float(max(share - 0.20, 0.05)), float(min(share + 0.15, 0.97))]
    lim["corr_band"] = [float(max(a["corr"] - 0.22, -1.0)), float(min(a["corr"] + 0.22, 1.0))]
    rp = oracle["replay"]
    lim["chi2_report_tol"] = float(0.01 * rp["chi2_fit"])
    lim["chi2_stream_report_tol"] = float(0.01 * rp["chi2_fit"])
    lim["chi2_max"] = float(max(1.5 * rp["chi2_fit"], 1.3 * floor["chi2_truth"]))
    dq = max(abs(v) for v in rp["dq"].values())
    fq = max(abs(floor["q_public"][k] - floor["q_truth"][k]) for k in ("Q00", "Q10", "Q01", "Q11"))
    lim["statistic_tol"] = float(max(4.0 * dq, 1.5 * fq, 0.01 * ta["Q11"]))
    lim["_comment"] = "set by authoring/provenance/set_limits.py from measurement; see CALIBRATION.md"
    if not ok:
        print("\nREFUSING to write limits: an accuracy limit cannot separate oracle from baselines")
        return 1
    json.dump(lim, open(os.path.join(ROOT, "tests", "limits.json"), "w"), indent=2)
    json.dump(dict(oracle=oracle, baselines=base), open(os.path.join(HERE, "limit_inputs.json"), "w"),
              indent=1, default=float)
    print("\nlimits.json written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
