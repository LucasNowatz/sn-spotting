"""Measure the achievable error floor, which sets every verifier threshold.

Runs the reference forward model on the public 5 km grid at the true
parameters and compares it with the 2.5 km generator truth at the withheld
rows, the lineage queries and the four counterfactual event statistics.  A
correct solver cannot do better than this, so the thresholds are set from it
rather than from anything that makes the oracle pass.

    python calibrate.py [n_jobs] [dt_seconds]
"""
import csv
import json
import os
import sys

import numpy as np
from joblib import Parallel, delayed
from scipy.linalg import cholesky, solve_triangular

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "solution", "src"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import model as MO                                          # noqa: E402
import observe_ref as OB                                    # noqa: E402
import invert as IV                                         # noqa: E402
from verifier import metrics as MT                          # noqa: E402
from verifier import grade as G                             # noqa: E402

ENV = os.path.join(ROOT, "environment", "data")
HID = os.path.join(ROOT, "tests", "hidden_truth")
OBSERVED = ["E01", "E02", "E03", "E04", "E05", "E06"]
DT = 60.0


def run_ep(e, p, dt):
    case = MO.Case(os.path.join(ENV, "episodes", e))
    return e, case, MO.run_full(case, p, dt=dt, tags=True)


def run_q(e, p, dt):
    case = MO.Case(os.path.join(ENV, "episodes", e))
    return MO.event_statistic(MO.run_full(case, p, dt=dt)["n"])


def main():
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    dt = float(sys.argv[2]) if len(sys.argv) > 2 else DT
    par = json.load(open(os.path.join(HID, "parameters.json")))
    theta = np.array(par["theta_true"])
    p = MO.to_phys(theta)
    OB.load_geometry(ENV)
    truth = np.load(os.path.join(HID, "station_truth.npz"))
    index = list(csv.DictReader(open(os.path.join(ENV, "prediction_index.csv"))))
    queries = json.load(open(os.path.join(HID, "query_truth.json")))
    thr = json.load(open(os.path.join(ROOT, "tests", "thresholds.json")))
    manifest = json.load(open(os.path.join(ENV, "data_manifest.json")))
    events = manifest["split"]["event_episodes"]
    eps = [e["episode"] for e in manifest["episodes"]]

    out = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(run_ep)(e, p, dt) for e in eps)
    runs = {e: (c, r) for e, c, r in out}

    nv = np.zeros(len(index)); nn = np.zeros((len(index), 12))
    for r in index:
        i = int(r["row_id"]); e = r["episode"]; s = r["station"]
        k = int(r["time_index"])
        case, res = runs[e]
        nv[i] = OB.sample_series(res["c"], case, s)[k]
        nn[i] = OB.reported_counts(res["n"], case, s)[k]
    T = G.truth_arrays(truth, index)
    tv, tn, _, _, _, st = T
    sv = np.array([thr["sigma_vapour_log"][s] for s in st])
    sn = np.array([thr["sigma_counts_log"][s] for s in st])
    v_nrmse = MT.log_nrmse(nv, tv, sv)
    mask = tn >= thr["count_valid_min"]
    p_lrmse = MT.log_rmse_counts(nn, tn, np.repeat(sn[:, None], 12, 1), mask)
    d_mae, n_d = MT.diameter_mae(nn, tn, MO.centres(), thr["diam_min_total"])

    ap, at, fp, ft = [], [], [], []
    for q in queries:
        case, res = runs[q["episode"]]
        k, b = q["time_index"], q["diameter_bin"]
        ag = OB.sample_bins(res["ag"], case, q["station"])[k, b]
        tg = np.array([OB.sample_bins(res["tg"][:, s], case,
                                      q["station"])[k, b] for s in range(3)])
        tot = tg.sum()
        ap.append(ag / tot if tot > 1e-12 else np.nan)
        fp.append(tg / tot if tot > 1e-12 else np.full(3, np.nan))
        at.append(q["age_hr"]); ft.append(q["frac"])
    a_mae, _ = MT.age_mae(np.array(ap), np.array(at))
    s_l1, _ = MT.source_l1(np.array(fp), np.array(ft))

    per_ep = {}
    rows_by_ep = {}
    for r in index:
        rows_by_ep.setdefault(r["episode"], []).append(int(r["row_id"]))
    for e, idx in rows_by_ep.items():
        idx = np.array(idx)
        m2 = tn[idx] >= thr["count_valid_min"]
        per_ep[e] = (MT.log_nrmse(nv[idx], tv[idx], sv[idx]),
                     MT.log_rmse_counts(nn[idx], tn[idx],
                                        np.repeat(sn[idx][:, None], 12, 1), m2))

    # counterfactual statistics on the public grid at the true parameters
    cf = dict(Q00=theta.copy(), Q10=theta.copy(), Q01=theta.copy(),
              Q11=theta.copy())
    cf["Q00"][10] = 0.0; cf["Q00"][11] = 0.0
    cf["Q10"][11] = 0.0; cf["Q01"][10] = 0.0
    jobs = [(k, e) for k in cf for e in events]
    qq = Parallel(n_jobs=n_jobs)(delayed(run_q)(e, MO.to_phys(cf[k]), dt)
                                 for k, e in jobs)
    q = {k: 0.0 for k in cf}
    for (k, e), v in zip(jobs, qq):
        q[k] += v / len(events)
    q["A_E"] = 0.5 * ((q["Q10"] - q["Q00"]) + (q["Q11"] - q["Q01"]))
    q["A_C"] = 0.5 * ((q["Q01"] - q["Q00"]) + (q["Q11"] - q["Q10"]))
    ta = par["attribution"]

    # chi-square at the truth with the full covariance
    obs = IV.Observations(ENV, OBSERVED)
    pred = {}
    for e in OBSERVED:
        case, res = runs[e]
        for s in ("S1", "S2", "S3", "S4", "S5"):
            pred[(e, s)] = dict(vapour=OB.sample_series(res["c"], case, s),
                                counts=OB.reported_counts(res["n"], case, s))
    rw = obs.whitened(pred)
    chi2 = float(rw @ rw)

    print(f"\nachievable floor, reference solver at the true parameters, "
          f"dt = {dt:.0f} s")
    print(f"  withheld vapour log NRMSE    {v_nrmse:.4f}")
    print(f"  withheld PNSD log-RMSE       {p_lrmse:.4f}  "
          f"({int(mask.sum())} channels)")
    print(f"  diameter MAE                 {d_mae:.4f} nm ({n_d} samples)")
    print(f"  mean particle age MAE        {a_mae:.4f} h")
    print(f"  source-fraction mean L1      {s_l1:.4f}")
    print(f"  chi2 at the truth            {chi2:.1f} over {obs.n_obs} obs")
    print("  per episode (vapour NRMSE, PNSD log-RMSE):")
    for e in sorted(per_ep):
        print(f"    {e}: {per_ep[e][0]:.4f}  {per_ep[e][1]:.4f}")
    print("  event statistics, 5 km vs 2.5 km truth:")
    for k in ("Q00", "Q10", "Q01", "Q11", "A_E", "A_C"):
        print(f"    {k}: {q[k]:9.3f}  truth {ta[k]:9.3f}  "
              f"diff {q[k] - ta[k]:+8.3f}")
    json.dump(dict(dt=dt, vapour_nrmse=v_nrmse, pnsd_logrmse=p_lrmse,
                   diameter_mae=d_mae, age_mae=a_mae, source_l1=s_l1,
                   chi2_truth=chi2, n_obs=obs.n_obs,
                   per_episode={k: list(v) for k, v in per_ep.items()},
                   q_public=q, q_truth=ta),
              open(os.path.join(HERE, "floor.json" if dt == 60.0 else
                                f"floor_dt{int(dt)}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
