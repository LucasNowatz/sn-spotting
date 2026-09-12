"""Build prediction_index.csv, history_queries.csv and data_manifest.json.

The history queries are deliberately chosen at times when a station is seeing
a mixture of source regions rather than one, so the diagnostic cannot be
answered by naming the nearest upwind region.
"""
import csv
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import episodes as E
import observe as O
import truth_model as M

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENV = os.path.join(ROOT, "environment", "data")
HID = os.path.join(ROOT, "tests", "hidden_truth")

WITHHELD_STATIONS = ["W1", "W2"]
WITHHELD_EPISODES = ["E07", "E08"]
OBSERVED_EPISODES = ["E01", "E02", "E03", "E04", "E05", "E06"]
ALL_STATIONS = [s["id"] for s in E.STATIONS]
N_QUERIES = 80
MIX_MAX = 0.75
MIN_COUNT = 25.0


def prediction_rows():
    rows = []
    for e in OBSERVED_EPISODES:
        for s in WITHHELD_STATIONS:
            for k in range(E.N_OBS):
                rows.append((e, s, k))
    for e in WITHHELD_EPISODES:
        for s in ALL_STATIONS:
            for k in range(E.N_OBS):
                rows.append((e, s, k))
    return rows


def calibration_rows():
    rows = []
    for e in OBSERVED_EPISODES:
        for s in ["S1", "S2", "S3", "S4", "S5"]:
            for k in range(E.N_OBS):
                rows.append((e, s, k))
    return rows


def pick_queries(truth):
    cands = []
    for e in list(E.EPISODES):
        for s in ALL_STATIONS:
            n = truth[f"{e}__{s}__n_true"]
            npf = truth[f"{e}__{s}__npf_number"]
            fr = truth[f"{e}__{s}__frac"]
            age = truth[f"{e}__{s}__age_hr"]
            for k in range(n.shape[0]):
                for b in range(M.N_BINS):
                    f = fr[k, :, b]
                    if not np.all(np.isfinite(f)):
                        continue
                    if n[k, b] < MIN_COUNT or npf[k, b] < MIN_COUNT * 0.5:
                        continue
                    if not np.isfinite(age[k, b]):
                        continue
                    if f.max() > MIX_MAX:
                        continue
                    cands.append((e, s, k, b, float(f.max()), float(age[k, b])))
    return cands


def choose_spread(cands, n_want, seed=4242):
    rng = np.random.default_rng(seed)
    by_key = {}
    for c in cands:
        by_key.setdefault((c[0], c[1]), []).append(c)
    keys = sorted(by_key)
    out = []
    while len(out) < n_want and keys:
        for key in list(keys):
            pool = by_key[key]
            if not pool:
                keys.remove(key)
                continue
            pool.sort(key=lambda c: c[4])
            take = pool.pop(rng.integers(0, min(6, len(pool))))
            out.append(take)
            if len(out) >= n_want:
                break
    return out


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    truth = np.load(os.path.join(HID, "station_truth.npz"))
    rows = prediction_rows()
    with open(os.path.join(ENV, "prediction_index.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "episode", "station", "time_index"])
        for i, (e, s, k) in enumerate(rows):
            w.writerow([i, e, s, k])
    print(f"prediction_index.csv: {len(rows)} rows")
    crows = calibration_rows()
    with open(os.path.join(ENV, "calibration_index.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "episode", "station", "time_index"])
        for i, (e, s, k) in enumerate(crows):
            w.writerow([i, e, s, k])
    print(f"calibration_index.csv: {len(crows)} rows")

    cands = pick_queries(truth)
    picked = choose_spread(cands, N_QUERIES)
    with open(os.path.join(ENV, "history_queries.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "episode", "station", "time_index",
                    "diameter_bin"])
        for i, c in enumerate(picked):
            w.writerow([i, c[0], c[1], c[2], c[3]])
    mx = np.array([c[4] for c in picked])
    ages = np.array([c[5] for c in picked])
    print(f"history_queries.csv: {len(picked)} rows from {len(cands)} "
          f"candidates; dominant source fraction {mx.min():.2f}-{mx.max():.2f}"
          f", age {ages.min():.2f}-{ages.max():.2f} h")

    with open(os.path.join(HID, "query_truth.json"), "w") as f:
        json.dump([dict(query_id=i, episode=c[0], station=c[1],
                        time_index=c[2], diameter_bin=c[3],
                        age_hr=float(truth[f"{c[0]}__{c[1]}__age_hr"][c[2], c[3]]),
                        frac=[float(x) for x in
                              truth[f"{c[0]}__{c[1]}__frac"][c[2], :, c[3]]])
                   for i, c in enumerate(picked)], f, indent=1)

    ep_index = json.load(open(os.path.join(ENV, "episode_index.json")))
    files = {}
    for name in sorted(os.listdir(ENV)):
        p = os.path.join(ENV, name)
        if os.path.isfile(p) and name != "data_manifest.json":
            files[name] = sha(p)
    manifest = dict(
        schema_version="growth-event-1.0",
        description=("synthetic regional transport and size-resolved "
                     "nanoparticle growth system; eight 10 h episodes on a "
                     "300 km by 240 km domain with one well-mixed "
                     "boundary-layer slab; four episodes are event days"),
        data_origin=("synthetic, from a seeded conservative finite-volume "
                     "simulator run at 2.5 km with a 30 s step and "
                     "source-tagged tracers; the public grid is 5 km; the "
                     "true parameter vector is one draw from the published "
                     "prior and the observation errors are one draw from the "
                     "published covariance"),
        spec="model_spec.md",
        grid=dict(nx=60, ny=48, dx_m=5000.0, dy_m=5000.0,
                  domain_m=[300000.0, 240000.0]),
        time=dict(episode_hours=10.0, forcing_records=41,
                  observation_times=41, interval_minutes=15,
                  interpolation="linear between consecutive records",
                  units="seconds since 2026-06-15T00:00:00Z"),
        size_bins=dict(count=12, d_min_nm=1.5, d_max_nm=15.0,
                       spacing="logarithmic", file="size_bins.csv",
                       operators="sizing_operators.nc"),
        backgrounds=dict(c_bg_ug_m3=0.06, n_bg_cm3=0.8),
        unknown_parameters=json.load(open(os.path.join(ENV, "priors.json")))["param_ids"],
        event_statistic=dict(
            grown_channels=list(range(M.GROWN_FIRST_BIN, M.N_BINS)),
            records=list(range(M.Q_FIRST_RECORD, E.N_OBS)),
            episodes=E.EVENT_EPISODES,
            definition=("domain mean over all cells of the number in the "
                        "grown channels, averaged over the listed records, "
                        "then averaged over the event episodes; cm-3")),
        observation=dict(
            detection_limit_cm3=O.DETECT_LIMIT,
            log_floor_counts_cm3=0.05,
            log_floor_vapour_ug_m3=0.001,
            error_covariance="error_covariance.nc",
            representation_floor_counts_log=O.REPR_FLOOR_N,
            representation_floor_vapour_log=O.REPR_FLOOR_V,
            note=("reported counts are post the station operator; the "
                  "lineage diagnostics are defined pre-operator")),
        split=dict(observed_stations=["S1", "S2", "S3", "S4", "S5"],
                   withheld_stations=WITHHELD_STATIONS,
                   observed_episodes=OBSERVED_EPISODES,
                   withheld_episodes=WITHHELD_EPISODES,
                   event_episodes=E.EVENT_EPISODES),
        episodes=ep_index,
        files=files)
    with open(os.path.join(ENV, "data_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("data_manifest.json written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
