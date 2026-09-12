"""Write withheld_index.csv, calibration_index.csv and dataset_manifest.json."""
import csv
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scenarios as SC
import simulator as SM
import instruments as IN

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENV = os.path.join(ROOT, "environment", "data")
OBSERVED = ["E01", "E02", "E03", "E04", "E05", "E06"]
WITHHELD_EP = ["E07", "E08"]
SEEN = ["S1", "S2", "S3", "S4", "S5", "S6"]
WITHHELD_ST = ["W1", "W2"]
ALL = [s["id"] for s in SC.STATIONS]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_rows(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "episode", "station", "time_index"])
        for i, (e, s, k) in enumerate(rows):
            w.writerow([i, e, s, k])
    return len(rows)


def main():
    wrows = [(e, s, k) for e in OBSERVED for s in WITHHELD_ST for k in range(SC.N_OBS)]
    wrows += [(e, s, k) for e in WITHHELD_EP for s in ALL for k in range(SC.N_OBS)]
    crows = [(e, s, k) for e in OBSERVED for s in SEEN for k in range(SC.N_OBS)]
    print("withheld rows", write_rows(os.path.join(ENV, "withheld_index.csv"), wrows))
    print("calibration rows", write_rows(os.path.join(ENV, "calibration_index.csv"), crows))

    files = {n: sha(os.path.join(ENV, n)) for n in sorted(os.listdir(ENV))
             if os.path.isfile(os.path.join(ENV, n)) and n != "dataset_manifest.json"}
    manifest = dict(
        schema_version="event-attribution-1.0",
        data_origin=("synthetic: a seeded conservative finite-volume simulator on the "
                     "published 4 km grid with a 30 s step, a limited unsplit flux-form "
                     "scheme and region-tagged number, on an analytic non-divergent "
                     "wind; the truth is one draw from prior.json and the errors one "
                     "draw from error_model.nc"),
        specification="specification.md",
        grid=dict(nx=SC.NX_PUB, ny=SC.NY_PUB, dx_m=SC.DX_PUB, dy_m=SC.DX_PUB,
                  domain_m=[SC.LX, SC.LY]),
        time=dict(episode_hours=SC.HOURS, records=SC.N_REC, interval_minutes=10,
                  interpolation="linear between records"),
        bins=dict(count=SM.N_BINS, d_min_nm=SM.D_MIN, d_max_nm=SM.D_MAX, file="diameter_bins.csv"),
        backgrounds=dict(vapour_ug_m3=SM.C_BG, number_per_bin_cm3=SM.N_BG),
        parameters=json.load(open(os.path.join(ENV, "prior.json")))["param_ids"],
        event_statistic=dict(grown_channels=list(range(SM.GROWN_FIRST_BIN, SM.N_BINS)),
                             records=list(range(SM.Q_FIRST_RECORD, SC.N_OBS)),
                             event_days=SC.EVENT_EPISODES),
        observation=dict(detection_limit_cm3=IN.DETECT_LIMIT,
                         log_floor_counts_cm3=IN.LOG_FLOOR_N,
                         log_floor_vapour_ug_m3=IN.LOG_FLOOR_V,
                         representation_floor_log=dict(vapour=IN.REPR_FLOOR_V,
                                                       counts=IN.REPR_FLOOR_N)),
        split=dict(observed_stations=SEEN, withheld_stations=WITHHELD_ST,
                   observed_episodes=OBSERVED, withheld_episodes=WITHHELD_EP),
        episodes=json.load(open(os.path.join(ENV, "episodes.json"))),
        files=files)
    with open(os.path.join(ENV, "dataset_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("manifest written")


if __name__ == "__main__":
    main()
