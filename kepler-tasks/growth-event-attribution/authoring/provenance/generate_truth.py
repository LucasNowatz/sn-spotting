"""Seeded generator for the growth-event benchmark (authoring side only).

Draws the true parameter vector from the published prior, runs the forward
model at 2.5 km with a 30 s step for all eight episodes carrying source-tagged
particle number and a number-age moment, samples the seven stations through
the published station operators, adds seeded correlated log-space errors, and
writes:

  environment/data/...   the public observing system, with W1/W2 masked in
                         every episode and every station masked in E07/E08
  tests/hidden_truth/... the sealed noiseless station truth, the noisy values
                         of every withheld observation, the lineage
                         diagnostics, the event statistic and its
                         counterfactuals, and the true parameter vector

    python generate_truth.py [n_jobs]
"""
import json
import os
import sys

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import driver as D
import episodes as E
import observe as O
import prior as PR
import truth_model as M
import writers as W

MASTER_SEED = 20260912
NOISE_OFFSET = 4711
FINE_NX, FINE_NY, FINE_DT = 120, 96, 30.0
PUB_NX, PUB_NY = 60, 48

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENV = os.path.join(ROOT, "environment", "data")
HID = os.path.join(ROOT, "tests", "hidden_truth")

WITHHELD_STATIONS = {"W1", "W2"}
WITHHELD_EPISODES = {"E07", "E08"}

# Screening applied to the seeded sequence of prior draws.  Disclosed in
# general terms in the model specification; the accepted index is not.
SCREEN = dict(max_abs_z=1.5, min_abs_anom=0.20, max_kh=12000.0)


def select_truth():
    rng = np.random.default_rng(MASTER_SEED)
    L = np.linalg.cholesky(PR.covariance())
    for k in range(10000):
        th = PR.MU + L @ rng.standard_normal(len(PR.MU))
        z = (th - PR.MU) / PR.SIGMA
        if np.abs(z).max() > SCREEN["max_abs_z"]:
            continue
        if np.any(th < PR.LOWER) or np.any(th > PR.UPPER):
            continue
        if np.exp(th[0]) > SCREEN["max_kh"]:
            continue
        a, b = th[10], th[11]
        if np.sign(a) != np.sign(b) or a <= 0.0:
            continue
        if min(abs(a), abs(b)) < SCREEN["min_abs_anom"]:
            continue
        return th, k
    raise RuntimeError("no prior draw satisfied the screening rules")


def run_one(eid, p):
    """Fine-grid tagged run, sampled at every station, plus the statistic."""
    ep, out = D.run(eid, p, nx=FINE_NX, ny=FINE_NY, dt=FINE_DT, tags=True)
    s = D.sample(out, ep, tags=True)
    res = {}
    for sid, d in s.items():
        n_npf = d["tagged"].sum(axis=1)                 # (time, bin)
        with np.errstate(invalid="ignore", divide="ignore"):
            age = np.where(n_npf > 1e-12, d["age_moment"] / n_npf, np.nan)
            frac = np.where(n_npf[:, None, :] > 1e-12,
                            d["tagged"] / n_npf[:, None, :], np.nan)
        res[sid] = dict(vapour=d["vapour"], counts=d["counts"],
                        n_true=d["n_true"], npf_number=n_npf,
                        age_hr=age, frac=frac)
    return eid, res, M.grown_number_statistic(out["n"])


def run_q(eid, p):
    """Fine-grid untagged run for one counterfactual statistic."""
    ep, out = D.run(eid, p, nx=FINE_NX, ny=FINE_NY, dt=FINE_DT, tags=False)
    return eid, M.grown_number_statistic(out["n"])


def counterfactuals(theta):
    th = np.asarray(theta, float)
    t00 = th.copy(); t00[10] = 0.0; t00[11] = 0.0
    t10 = th.copy(); t10[11] = 0.0
    t01 = th.copy(); t01[10] = 0.0
    return dict(Q00=t00, Q10=t10, Q01=t01, Q11=th.copy())


def main():
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    os.makedirs(ENV, exist_ok=True)
    os.makedirs(os.path.join(ENV, "episodes"), exist_ok=True)
    os.makedirs(HID, exist_ok=True)
    theta, draw = select_truth()
    p_true = PR.to_physical(theta)
    print("accepted prior draw", draw)
    print("true parameters:", {k: round(v, 5) for k, v in p_true.items()})

    ids = list(E.EPISODES)
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(run_one)(e, p_true) for e in ids)
    q_true = {e: q for e, _, q in results}
    results = {e: r for e, r, _ in results}

    # counterfactual event statistics on the fine grid
    cf = counterfactuals(theta)
    jobs = [(k, e) for k in ("Q00", "Q10", "Q01") for e in E.EVENT_EPISODES]
    out = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(run_q)(e, PR.to_physical(cf[k])) for k, e in jobs)
    qcf = {"Q11": float(np.mean([q_true[e] for e in E.EVENT_EPISODES]))}
    for (k, e), (_, q) in zip(jobs, out):
        qcf.setdefault(k, []).append(q)
    for k in ("Q00", "Q10", "Q01"):
        qcf[k] = float(np.mean(qcf[k]))
    qcf["A_E"] = 0.5 * ((qcf["Q10"] - qcf["Q00"]) + (qcf["Q11"] - qcf["Q01"]))
    qcf["A_C"] = 0.5 * ((qcf["Q01"] - qcf["Q00"]) + (qcf["Q11"] - qcf["Q10"]))
    print("event statistic and attribution:", {k: round(v, 4) for k, v in qcf.items()})

    times = E.rec_times()
    station_ids = [s["id"] for s in E.STATIONS]
    rng = np.random.default_rng(MASTER_SEED + NOISE_OFFSET)
    manifest_cases = []
    noisy = {}

    for eid in ids:
        d = os.path.join(ENV, "episodes", eid)
        os.makedirs(d, exist_ok=True)
        pub = D.build_episode(eid, PUB_NX, PUB_NY)
        W.write_met(os.path.join(d, "met.nc"), pub)
        W.write_forcing(os.path.join(d, "forcing.nc"), pub)

        obs = {}
        for sid in station_ids:
            r = results[eid][sid]
            v, n, qc = O.add_noise(r["vapour"], r["counts"], sid, rng)
            obs[sid] = dict(vapour=v, counts=n, qc=qc)
            noisy[f"{eid}__{sid}__vapour_obs"] = v
            noisy[f"{eid}__{sid}__counts_obs"] = n
            noisy[f"{eid}__{sid}__qc"] = qc
        withheld = set(WITHHELD_STATIONS)
        if eid in WITHHELD_EPISODES:
            withheld = set(station_ids)
        W.write_observations(os.path.join(d, "observations.nc"), obs,
                             station_ids, times, withheld)
        manifest_cases.append(dict(
            episode=eid, status=E.EPISODES[eid]["status"],
            event_day=bool(E.EPISODES[eid]["event"]),
            regime=E.EPISODES[eid]["note"],
            observed_stations=sorted(set(station_ids) - withheld),
            files={f: W.sha256(os.path.join(d, f))
                   for f in ("met.nc", "forcing.nc", "observations.nc")}))
        print(f"  wrote {eid}: {len(station_ids) - len(withheld)} observed stations")

    np.savez_compressed(
        os.path.join(HID, "station_truth.npz"),
        **{f"{eid}__{sid}__{k}": results[eid][sid][k]
           for eid in ids for sid in station_ids
           for k in ("vapour", "counts", "n_true", "npf_number", "age_hr",
                     "frac")},
        **noisy)
    with open(os.path.join(HID, "parameters.json"), "w") as f:
        json.dump(dict(theta_true=[float(x) for x in theta],
                       param_ids=PR.PARAM_IDS, physical=p_true,
                       prior_draw_index=draw, master_seed=MASTER_SEED,
                       noise_seed=MASTER_SEED + NOISE_OFFSET,
                       generator_grid=[FINE_NX, FINE_NY],
                       generator_dt_s=FINE_DT,
                       event_statistic_per_episode=q_true,
                       attribution=qcf), f, indent=2)

    W.write_stations(os.path.join(ENV, "stations.csv"))
    W.write_size_bins(os.path.join(ENV, "size_bins.csv"))
    W.write_operators(os.path.join(ENV, "sizing_operators.nc"), station_ids)
    W.write_error_covariance(os.path.join(ENV, "error_covariance.nc"),
                             station_ids)
    with open(os.path.join(ENV, "priors.json"), "w") as f:
        json.dump(PR.as_json(), f, indent=1)
    with open(os.path.join(ENV, "episode_index.json"), "w") as f:
        json.dump(manifest_cases, f, indent=2)
    print("generator finished")


if __name__ == "__main__":
    main()
