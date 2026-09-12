"""Achievable floor: the reference solver at the true parameters on the
public 4 km grid against the generator's own run.  Sets every limit.

    python calibrate.py [n_jobs] [dt_seconds] [koren|mc|fine]

With "mc" the generator's own solver is run with the monotonized-central
limiter instead of the reference solver, and with "fine" on a 2 km grid;
both measure how far a correct solver that differs from the generator lands
from the truth, and every limit allows for the worst of them.
"""
import csv
import json
import os
import sys

import numpy as np
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "solution", "src"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import forward as FW                                        # noqa: E402
import network as NW                                        # noqa: E402
import bayes as BY                                          # noqa: E402
from gates import checks as G                               # noqa: E402
from gates import scores as SC                              # noqa: E402

ENV = os.path.join(ROOT, "environment", "data")
SEALED = os.path.join(ROOT, "tests", "sealed")
OBSERVED = ["E01", "E02", "E03", "E04", "E05", "E06"]


SCHEME = "koren"


def run_ep(e, p, dt, tags):
    if SCHEME in ("mc", "fine"):
        sys.path.insert(0, os.path.join(ROOT, "tests", "sealed_model"))
        import dynamics as DY
        import driver as D
        if SCHEME == "mc":
            DY.LIMITER = "mc"
            sc, res = D.run(e, p, dt=dt, tags=tags)
        else:
            sc, res = D.run(e, p, nx=140, ny=100, dt=30.0, tags=tags)
        return e, sc, res
    ep = FW.Episode(os.path.join(ENV, "episodes", e))
    return e, ep, FW.run(ep, p, dt=dt, tags=tags)


def run_q(e, p, dt):
    return FW.event_statistic(run_ep(e, p, dt, False)[2]["n"])


def main():
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    dt = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
    global SCHEME
    SCHEME = sys.argv[3] if len(sys.argv) > 3 else "koren"
    tj = json.load(open(os.path.join(SEALED, "truth.json")))
    theta = np.array(tj["theta_true"]); p = FW.from_theta(theta)
    NW.load(ENV)
    truth = np.load(os.path.join(SEALED, "network_truth.npz"))
    windex = list(csv.DictReader(open(os.path.join(ENV, "withheld_index.csv"))))
    lim = json.load(open(os.path.join(ROOT, "tests", "limits.json")))
    man = json.load(open(os.path.join(ENV, "dataset_manifest.json")))
    events = man["event_statistic"]["event_days"]
    eps = [e["episode"] for e in man["episodes"]]

    runs = {e: (ep, r) for e, ep, r in Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(run_ep)(e, p, dt, e in events) for e in eps)}
    nv = np.zeros(len(windex)); nn = np.zeros((len(windex), 12))
    for r in windex:
        i = int(r["row_id"]); ep, res = runs[r["episode"]]; k = int(r["time_index"])
        nv[i] = NW.at_station(res["c"], ep, r["station"])[k]
        nn[i] = NW.reported(res["n"], ep, r["station"])[k]
    T = G.truth_arrays(truth, windex)
    sv = np.array([lim["sigma_vapour_log"][s] for s in T["st"]])
    sn = np.array([lim["sigma_counts_log"][s] for s in T["st"]])
    mask = T["tn"] >= lim["count_valid_min"]
    out = dict(dt=dt,
               vapour=SC.log_nrmse(nv, T["tv"], sv),
               pnsd=SC.log_rmse_counts(nn, T["tn"], np.repeat(sn[:, None], 12, 1), mask),
               diameter=SC.diameter_mae(nn, T["tn"], FW.centres(), lim["diam_min_total"])[0])
    by = {}
    for r in windex:
        by.setdefault(r["episode"], []).append(int(r["row_id"]))
    out["per_episode"] = {}
    for e, idx in by.items():
        idx = np.array(idx); m = T["tn"][idx] >= lim["count_valid_min"]
        out["per_episode"][e] = [SC.log_nrmse(nv[idx], T["tv"][idx], sv[idx]),
                                 SC.log_rmse_counts(nn[idx], T["tn"][idx], np.repeat(sn[idx][:, None], 12, 1), m)]
    # region shares
    dom, sta = [], []
    for e in events:
        ep, res = runs[e]
        dom.append(np.abs(FW.domain_shares(res) - np.array(tj["domain_region_shares"][e])).sum())
        for s in ("W1", "W2"):
            g_tot = NW.at_station(res["n"][FW.Q_FIRST_RECORD:, FW.GROWN_FIRST_BIN:].sum(axis=1), ep, s).mean()
            g_tag = NW.at_station(res["tagged"][FW.Q_FIRST_RECORD:, :, FW.GROWN_FIRST_BIN:].sum(axis=2), ep, s).mean(axis=0)
            sta.append(np.abs(g_tag / g_tot - truth[f"{e}__{s}__shares"]).sum())
    out["domain_share_l1"] = float(np.mean(dom)); out["station_share_l1"] = float(np.mean(sta))
    # counterfactuals on the public grid
    cf = {k: theta.copy() for k in ("Q00", "Q10", "Q01", "Q11")}
    cf["Q00"][10] = 0.0; cf["Q00"][11] = 0.0; cf["Q10"][11] = 0.0; cf["Q01"][10] = 0.0
    jobs = [(k, e) for k in cf for e in events]
    qq = Parallel(n_jobs=n_jobs)(delayed(run_q)(e, FW.from_theta(cf[k]), dt) for k, e in jobs)
    q = {k: 0.0 for k in cf}
    for (k, e), v in zip(jobs, qq):
        q[k] += v / len(events)
    q["A_E"] = 0.5 * ((q["Q10"] - q["Q00"]) + (q["Q11"] - q["Q01"]))
    q["A_C"] = 0.5 * ((q["Q01"] - q["Q00"]) + (q["Q11"] - q["Q10"]))
    out["q_public"] = q; out["q_truth"] = tj["attribution"]
    # chi-square at the truth
    rec = BY.Records(ENV, OBSERVED)
    pred = {}
    for e in OBSERVED:
        ep, res = runs[e]
        for s in rec.stations:
            pred[(e, s)] = dict(vapour=NW.at_station(res["c"], ep, s), counts=NW.reported(res["n"], ep, s))
    rw = rec.whitened(pred); out["chi2_truth"] = float(rw @ rw); out["n_obs"] = rec.n_obs

    print(f"\nfloor at the truth, dt = {dt:.0f} s, scheme {SCHEME}")
    for k in ("vapour", "pnsd", "diameter", "domain_share_l1", "station_share_l1", "chi2_truth"):
        print(f"  {k:18s} {out[k]:.4f}")
    for e in sorted(out["per_episode"]):
        print(f"  {e}: {out['per_episode'][e][0]:.3f} {out['per_episode'][e][1]:.3f}")
    for k in ("Q00", "Q10", "Q01", "Q11", "A_E", "A_C"):
        print(f"  {k}: {q[k]:9.3f} truth {tj['attribution'][k]:9.3f} diff {q[k] - tj['attribution'][k]:+.3f}")
    name = ("floor.json" if dt == 60.0 else f"floor_dt{int(dt)}.json") if SCHEME == "koren" else f"floor_{SCHEME}.json"
    json.dump(out, open(os.path.join(HERE, name), "w"), indent=2)


if __name__ == "__main__":
    main()
