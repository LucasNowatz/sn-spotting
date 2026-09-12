"""Seeded generator (authoring side only).

Draws the truth from the prior, runs every episode on the published 4 km
grid with a 30 s step (the reference is a separate implementation at 60 s)
carrying tagged number per source region, samples the network through the
published instruments, adds one draw of the published correlated errors,
runs the counterfactual event statistics, and writes the public dataset and
the sealed truth.

    python build_dataset.py [n_jobs]
"""
import json
import os
import sys

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import driver as D
import scenarios as SC
import simulator as SM
import instruments as IN
import prior as PR
import writers as W

MASTER_SEED = 20260703
NOISE_OFFSET = 9109
FINE_NX, FINE_NY, FINE_DT = 70, 50, 30.0

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENV = os.path.join(ROOT, "environment", "data")
SEALED = os.path.join(ROOT, "tests", "sealed")
WITHHELD_STATIONS = {"W1", "W2"}
WITHHELD_EPISODES = {"E07", "E08"}
SCREEN = dict(max_abs_z=1.8, min_s_anom=0.40, min_q_anom=0.30, max_kh=11000.0)


def select_truth():
    rng = np.random.default_rng(MASTER_SEED)
    L = np.linalg.cholesky(PR.covariance())
    for k in range(10000):
        th = PR.MU + L @ rng.standard_normal(len(PR.MU))
        z = (th - PR.MU) / PR.SIGMA
        if np.abs(z).max() > SCREEN["max_abs_z"]:
            continue
        if np.any(th < PR.LOWER) or np.any(th > PR.UPPER) or np.exp(th[0]) > SCREEN["max_kh"]:
            continue
        if th[10] < SCREEN["min_s_anom"] or th[11] < SCREEN["min_q_anom"]:
            continue
        return th, k
    raise RuntimeError("no prior draw satisfied the screening rules")


def run_one(eid, p):
    sc, res = D.run(eid, p, nx=FINE_NX, ny=FINE_NY, dt=FINE_DT, tags=True)
    s = D.sample_network(res, sc)
    out = {}
    for sid, d in s.items():
        out[sid] = dict(vapour=d["vapour"], counts=d["counts"], n_true=d["n_true"],
                        shares=D.station_region_shares(d))
    return eid, out, SM.event_statistic(res["n"]), SM.region_shares_domain(res)


def run_q(eid, p):
    _, res = D.run(eid, p, nx=FINE_NX, ny=FINE_NY, dt=FINE_DT)
    return SM.event_statistic(res["n"])


def counterfactuals(theta):
    th = np.asarray(theta, float)
    t00 = th.copy(); t00[10] = 0.0; t00[11] = 0.0
    t10 = th.copy(); t10[11] = 0.0
    t01 = th.copy(); t01[10] = 0.0
    return dict(Q00=t00, Q10=t10, Q01=t01, Q11=th.copy())


def main():
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    os.makedirs(os.path.join(ENV, "episodes"), exist_ok=True)
    os.makedirs(SEALED, exist_ok=True)
    theta, draw = select_truth()
    p_true = PR.to_physical(theta)
    print("accepted prior draw", draw)
    print("true parameters:", {k: round(v, 5) for k, v in p_true.items()})

    ids = list(SC.EPISODES)
    res = Parallel(n_jobs=n_jobs, verbose=5)(delayed(run_one)(e, p_true) for e in ids)
    q_true = {e: q for e, _, q, _ in res}
    shares_dom = {e: [float(x) for x in sh] for e, _, _, sh in res if e in SC.EVENT_EPISODES}
    results = {e: r for e, r, _, _ in res}

    cf = counterfactuals(theta)
    jobs = [(k, e) for k in ("Q00", "Q10", "Q01") for e in SC.EVENT_EPISODES]
    out = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(run_q)(e, PR.to_physical(cf[k])) for k, e in jobs)
    qcf = {"Q11": float(np.mean([q_true[e] for e in SC.EVENT_EPISODES]))}
    for (k, e), q in zip(jobs, out):
        qcf.setdefault(k, []).append(q)
    for k in ("Q00", "Q10", "Q01"):
        qcf[k] = float(np.mean(qcf[k]))
    qcf["A_E"] = 0.5 * ((qcf["Q10"] - qcf["Q00"]) + (qcf["Q11"] - qcf["Q01"]))
    qcf["A_C"] = 0.5 * ((qcf["Q01"] - qcf["Q00"]) + (qcf["Q11"] - qcf["Q10"]))
    print("attribution truth:", {k: round(v, 3) for k, v in qcf.items()})

    times = SC.record_times()
    station_ids = [s["id"] for s in SC.STATIONS]
    rng = np.random.default_rng(MASTER_SEED + NOISE_OFFSET)
    index, sealed = [], {}
    for eid in ids:
        d = os.path.join(ENV, "episodes", eid)
        os.makedirs(d, exist_ok=True)
        sc = D.build(eid, SC.NX_PUB, SC.NY_PUB)
        W.write_meteorology(os.path.join(d, "meteorology.nc"), sc)
        W.write_sources(os.path.join(d, "sources.nc"), sc)
        obs = {}
        for sid in station_ids:
            r = results[eid][sid]
            v, n, qc = IN.add_errors(r["vapour"], r["counts"], sid, rng)
            obs[sid] = dict(vapour=v, counts=n, qc=qc)
            sealed[f"{eid}__{sid}__vapour"] = r["vapour"]
            sealed[f"{eid}__{sid}__counts"] = r["counts"]
            sealed[f"{eid}__{sid}__vapour_obs"] = v
            sealed[f"{eid}__{sid}__counts_obs"] = n
            sealed[f"{eid}__{sid}__flag"] = qc
            sealed[f"{eid}__{sid}__shares"] = r["shares"]
        withheld = set(WITHHELD_STATIONS)
        if eid in WITHHELD_EPISODES:
            withheld = set(station_ids)
        W.write_records(os.path.join(d, "station_records.nc"), obs, station_ids, times, withheld)
        index.append(dict(episode=eid, status=SC.EPISODES[eid]["status"],
                          event_day=bool(SC.EPISODES[eid]["event"]),
                          regime=SC.EPISODES[eid]["note"],
                          observed_stations=sorted(set(station_ids) - withheld),
                          files={f: W.sha256(os.path.join(d, f)) for f in
                                 ("meteorology.nc", "sources.nc", "station_records.nc")}))
        print(f"  wrote {eid}")

    np.savez_compressed(os.path.join(SEALED, "network_truth.npz"), **sealed)
    with open(os.path.join(SEALED, "truth.json"), "w") as f:
        json.dump(dict(theta_true=[float(x) for x in theta], param_ids=PR.PARAM_IDS,
                       physical=p_true, prior_draw_index=draw, master_seed=MASTER_SEED,
                       noise_seed=MASTER_SEED + NOISE_OFFSET,
                       generator_grid=[FINE_NX, FINE_NY], generator_dt_s=FINE_DT,
                       event_statistic_per_episode=q_true,
                       domain_region_shares=shares_dom, attribution=qcf), f, indent=2)
    W.write_network(os.path.join(ENV, "network.csv"))
    W.write_bins(os.path.join(ENV, "diameter_bins.csv"))
    W.write_instruments(os.path.join(ENV, "instruments.nc"), station_ids)
    W.write_error_model(os.path.join(ENV, "error_model.nc"), station_ids)
    with open(os.path.join(ENV, "prior.json"), "w") as f:
        json.dump(PR.as_json(), f, indent=1)
    with open(os.path.join(ENV, "episodes.json"), "w") as f:
        json.dump(index, f, indent=2)
    print("generator finished")


if __name__ == "__main__":
    main()
