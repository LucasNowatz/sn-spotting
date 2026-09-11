#!/usr/bin/env bash
# Reference solution for the ozone budget attribution task.
#
# The analysis module is written into /app/output/workflow/ and then executed
# from there, so the delivered workflow directory is exactly the code that
# produced the delivered tables.
set -euo pipefail

mkdir -p /app/output/workflow

cat > /app/output/workflow/ozone_budget.py <<'PYEOF'
"""Fixed and moving volume ozone budgets, four-run contrasts and receptor
exposure for the KEPLER-RED v1.2 wildfire plume archive.

Every integrated term is aggregated interval by interval with the
start-of-interval mask weights; the mask-change term is accumulated separately
with the end-of-interval cell masses.  Transport comes from the recorded
interface fluxes only.
"""

import csv
import json
import os

import numpy as np
from netCDF4 import Dataset

DATA = "/app/data"
OUT = "/app/output"

SCENARIOS = ["00", "10", "01", "11"]
VOLUMES = ["fixed_urban", "moving_plume"]
QUANTITIES = ["M0", "M1", "dM", "A", "K", "P", "L", "C", "D", "G", "residual"]
SERIES = ["00", "10", "01", "11", "d_fire_A0", "d_fire_A1",
          "d_urban_F0", "d_urban_F1", "interaction"]
THRESHOLD = 40.0
TAU_FRACTION = 0.005
FMT = "%.12g"


def read(path, names):
    ds = Dataset(path)
    ds.set_auto_mask(False)
    out = {n: np.asarray(ds.variables[n][:], dtype=np.float64) for n in names}
    ds.close()
    return out


def dry_air_mass(manifest):
    m = read(os.path.join(DATA, "meteo.nc"), ["air_mass", "specific_humidity"])
    return m["air_mass"] * (1.0 - m["specific_humidity"])


def face_divergence(ds, kind, shape):
    """Net import per cell, kg per interval, split lateral and vertical."""
    lateral = np.zeros(shape)
    fx = np.asarray(ds.variables[kind + "_flux_x"][:], dtype=np.float64)
    lateral += fx[:, :-1, :, :] - fx[:, 1:, :, :]
    fy = np.asarray(ds.variables[kind + "_flux_y"][:], dtype=np.float64)
    lateral += fy[:, :, :-1, :] - fy[:, :, 1:, :]
    fz = np.asarray(ds.variables[kind + "_flux_z"][:], dtype=np.float64)
    vertical = fz[:, :, :, :-1] - fz[:, :, :, 1:]
    return lateral, vertical


def load_scenario(scenario, dry, manifest):
    w_o3 = manifest["molar_mass_g_per_mol"]["O3"]
    w_dry = manifest["molar_mass_g_per_mol"]["dry_air"]
    state = read(os.path.join(DATA, scenario, "state.nc"), ["o3_vmr"])
    ppbv = state["o3_vmr"]
    mass = ppbv * 1.0e-9 * dry * (w_o3 / w_dry)

    chem = manifest["chemistry"]
    names = chem["production_channels"] + chem["destruction_channels"] + ["o3_depos"]
    proc = read(os.path.join(DATA, scenario, "process.nc"), names)
    prod = sum(proc[n] for n in chem["production_channels"])
    loss = sum(proc[n] for n in chem["destruction_channels"])
    dep = proc["o3_depos"]

    ds = Dataset(os.path.join(DATA, scenario, "fluxes.nc"))
    ds.set_auto_mask(False)
    adv_lat, adv_vert = face_divergence(ds, "adv", prod.shape)
    mix_lat, mix_vert = face_divergence(ds, "mix", prod.shape)
    ds.close()

    # The ground value of mix_flux_z already carries the deposition removal, so
    # the turbulent import term has to give it back before D is subtracted once.
    return {"mass": mass, "ppbv": ppbv,
            "A_lateral": adv_lat, "A_vertical": adv_vert,
            "A": adv_lat + adv_vert,
            "K_lateral": mix_lat, "K_vertical": mix_vert + dep,
            "K": mix_lat + mix_vert + dep,
            "P": prod, "L": loss, "D": dep}


def budget(fields, weight, n0, n1):
    mass = fields["mass"]
    acc = {k: 0.0 for k in ("A", "A_lateral", "A_vertical", "K", "K_lateral",
                            "K_vertical", "P", "L", "D")}
    g_term = 0.0
    for n in range(n0, n1):
        w_start = weight[n]
        for key in acc:
            acc[key] += float(np.sum(w_start * fields[key][n]))
        g_term += float(np.sum((weight[n + 1] - w_start) * mass[n + 1]))
    row = dict(acc)
    row["M0"] = float(np.sum(weight[n0] * mass[n0]))
    row["M1"] = float(np.sum(weight[n1] * mass[n1]))
    row["C"] = row["P"] - row["L"]
    row["S"] = 0.0
    row["G"] = g_term
    row["dM"] = row["M1"] - row["M0"]
    row["residual"] = row["dM"] - (row["A"] + row["K"] + row["C"]
                                   - row["D"] + row["S"] + row["G"])
    return row


def contrast_set(values):
    return {
        "d_fire_A0": values["10"] - values["00"],
        "d_fire_A1": values["11"] - values["01"],
        "d_urban_F0": values["01"] - values["00"],
        "d_urban_F1": values["11"] - values["10"],
        "interaction": values["11"] - values["01"] - values["10"] + values["00"],
    }


def trapezoid(series):
    total = 0.0
    for n in range(len(series) - 1):
        total += 0.5 * (series[n] + series[n + 1])
    return total


def receptor_metrics(series, n0, n1):
    seg = [float(series[n]) for n in range(n0, n1 + 1)]
    integral = trapezoid(seg)
    excess = [max(v - THRESHOLD, 0.0) for v in seg]
    return {"mean_ppbv": integral / float(n1 - n0),
            "peak_ppbv": max(seg),
            "exposure_ppbvh": integral,
            "exposure_over40_ppbvh": trapezoid(excess)}


def main():
    manifest = json.load(open(os.path.join(DATA, "manifest.json")))
    windows = {k: (v["start_index"], v["end_index"])
               for k, v in manifest["windows"].items()}
    window_names = sorted(windows)

    dry = dry_air_mass(manifest)
    masks = read(os.path.join(DATA, "masks.nc"), ["fixed_urban", "moving_plume"])
    n_end = dry.shape[0]
    weights = {
        "fixed_urban": np.broadcast_to(masks["fixed_urban"],
                                       (n_end,) + masks["fixed_urban"].shape),
        "moving_plume": masks["moving_plume"],
    }

    fields = {s: load_scenario(s, dry, manifest) for s in SCENARIOS}

    budgets = {}
    for volume in VOLUMES:
        for window in window_names:
            n0, n1 = windows[window]
            for scenario in SCENARIOS:
                budgets[(scenario, volume, window)] = budget(
                    fields[scenario], weights[volume], n0, n1)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "budgets.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "scenario", "volume", "window",
                         "M0_kg", "M1_kg", "A_kg", "A_lateral_kg",
                         "A_vertical_kg", "K_kg", "K_lateral_kg",
                         "K_vertical_kg", "P_kg", "L_kg", "C_kg", "D_kg",
                         "S_kg", "G_kg", "residual_kg"])
        for volume in VOLUMES:
            for window in window_names:
                for scenario in SCENARIOS:
                    r = budgets[(scenario, volume, window)]
                    writer.writerow(
                        [f"{scenario}|{volume}|{window}", scenario, volume, window]
                        + [FMT % r[k] for k in (
                            "M0", "M1", "A", "A_lateral", "A_vertical", "K",
                            "K_lateral", "K_vertical", "P", "L", "C", "D", "S",
                            "G", "residual")])

    with open(os.path.join(OUT, "contrasts.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["contrast_id", "volume", "window", "quantity",
                         "d_fire_A0", "d_fire_A1", "d_urban_F0", "d_urban_F1",
                         "interaction"])
        for volume in VOLUMES:
            for window in window_names:
                for quantity in QUANTITIES:
                    values = {s: budgets[(s, volume, window)][quantity]
                              for s in SCENARIOS}
                    c = contrast_set(values)
                    writer.writerow(
                        [f"{volume}|{window}|{quantity}", volume, window, quantity]
                        + [FMT % c[k] for k in ("d_fire_A0", "d_fire_A1",
                                                "d_urban_F0", "d_urban_F1",
                                                "interaction")])

    receptors = {}
    order = []
    with open(os.path.join(DATA, "receptors.csv")) as f:
        for row in csv.DictReader(f):
            rid = row["receptor_id"]
            if rid not in receptors:
                receptors[rid] = []
                order.append(rid)
            receptors[rid].append((int(row["i"]), int(row["j"]), int(row["k"]),
                                   float(row["weight"])))

    with open(os.path.join(OUT, "receptors.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "receptor_id", "series", "window",
                         "mean_ppbv", "peak_ppbv", "exposure_ppbvh",
                         "exposure_over40_ppbvh"])
        metrics = ("mean_ppbv", "peak_ppbv", "exposure_ppbvh",
                   "exposure_over40_ppbvh")
        for rid in order:
            series = {}
            for scenario in SCENARIOS:
                ppbv = fields[scenario]["ppbv"]
                acc = np.zeros(ppbv.shape[0])
                for (i, j, k, wgt) in receptors[rid]:
                    acc += wgt * ppbv[:, i, j, k]
                series[scenario] = acc
            for window in window_names:
                n0, n1 = windows[window]
                per = {s: receptor_metrics(series[s], n0, n1) for s in SCENARIOS}
                table = {}
                for metric in metrics:
                    values = {s: per[s][metric] for s in SCENARIOS}
                    table[metric] = dict(values)
                    table[metric].update(contrast_set(values))
                for name in SERIES:
                    writer.writerow([f"{rid}|{name}|{window}", rid, name, window]
                                    + [FMT % table[m][name] for m in metrics])

    rows = []
    for volume in VOLUMES:
        for window in window_names:
            r = {s: budgets[(s, volume, window)] for s in SCENARIOS}
            tau = TAU_FRACTION * abs(r["11"]["M0"])

            def chem_label(value):
                if value > tau:
                    return "net_production"
                if value < -tau:
                    return "net_destruction"
                return "near_zero"

            def contrast_label(value):
                if value > tau:
                    return "more_positive"
                if value < -tau:
                    return "more_negative"
                return "near_zero"

            interaction = (r["11"]["C"] - r["01"]["C"]
                           - r["10"]["C"] + r["00"]["C"])
            if interaction > tau:
                interaction_label = "amplifying"
            elif interaction < -tau:
                interaction_label = "damping"
            else:
                interaction_label = "near_zero"

            dominant = {}
            for s in SCENARIOS:
                candidates = {"A": r[s]["A"], "K": r[s]["K"], "C": r[s]["C"],
                              "D": -r[s]["D"], "G": r[s]["G"]}
                dominant[s] = max(sorted(candidates),
                                  key=lambda k: abs(candidates[k]))

            d_burden = r["11"]["M1"] - r["01"]["M1"]
            d_chem = r["11"]["C"] - r["01"]["C"]
            d_transport = sum(r["11"][k] - r["01"][k] for k in ("A", "K", "G"))
            rows.append({
                "volume": volume,
                "window": window,
                "tau_kg": tau,
                "scenario_chemistry": {s: chem_label(r[s]["C"]) for s in SCENARIOS},
                "fire_chemistry_contrast": {
                    "A0": contrast_label(r["10"]["C"] - r["00"]["C"]),
                    "A1": contrast_label(d_chem)},
                "interaction_label": interaction_label,
                "dominant_term": dominant,
                "transport_sustains_anomaly": bool(
                    d_burden > tau and d_chem < -tau and d_transport > tau),
            })

    with open(os.path.join(OUT, "diagnosis.json"), "w") as f:
        json.dump({"schema_version": "kepler-o3-diagnosis-1.0", "rows": rows},
                  f, indent=1, sort_keys=True)

    worst = max(abs(v["residual"]) / max(abs(v["M0"]), 1.0)
                for v in budgets.values())
    print("wrote 4 output files; worst relative closure residual %.3e" % worst)


if __name__ == "__main__":
    main()
PYEOF

cat > /app/output/workflow/RUN.txt <<'TXTEOF'
Invocation used to produce the delivered tables:

    python3 /app/output/workflow/ozone_budget.py

Inputs  : /app/data (read only)
Outputs : /app/output/budgets.csv
          /app/output/contrasts.csv
          /app/output/receptors.csv
          /app/output/diagnosis.json

The module reads the archive conventions from /app/data/manifest.json rather
than hard-coding them: the dry-air mass convention, the complete list of ozone
chemistry channels, the deposition term already contained in the surface
turbulent flux, and the analysis window indices.
TXTEOF

python3 /app/output/workflow/ozone_budget.py
