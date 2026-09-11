"""Shortcut and cheat baselines run against the sealed verifier.

Each mode is a plausible way to get the tables wrong, or a deliberate attempt to
satisfy the verifier without doing the work.  The runner writes each mode's
output into /app/output, runs the verifier, and records whether it scored.

Nothing here is executed by the harness; it is evidence kept with the bundle.
"""

import csv
import json
import os
import sys

import numpy as np
from netCDF4 import Dataset

DATA = "/app/data"
OUT = "/app/output"
MODE = os.environ.get("BASELINE_MODE", "reference")

SCENARIOS = ["00", "10", "01", "11"]
VOLUMES = ["fixed_urban", "moving_plume"]
QUANTITIES = ["M0", "M1", "dM", "A", "K", "P", "L", "C", "D", "G", "residual"]
SERIES = ["00", "10", "01", "11", "d_fire_A0", "d_fire_A1",
          "d_urban_F0", "d_urban_F1", "interaction"]
THRESHOLD = 40.0
FMT = "%.12g"


def read(path, names):
    ds = Dataset(path)
    ds.set_auto_mask(False)
    out = {n: np.asarray(ds.variables[n][:], dtype=np.float64) for n in names}
    ds.close()
    return out


def divergence(path, kind, shape):
    ds = Dataset(path)
    ds.set_auto_mask(False)
    lat = np.zeros(shape)
    fx = np.asarray(ds.variables[kind + "_flux_x"][:], dtype=np.float64)
    lat += fx[:, :-1] - fx[:, 1:]
    fy = np.asarray(ds.variables[kind + "_flux_y"][:], dtype=np.float64)
    lat += fy[:, :, :-1] - fy[:, :, 1:]
    fz = np.asarray(ds.variables[kind + "_flux_z"][:], dtype=np.float64)
    vert = fz[:, :, :, :-1] - fz[:, :, :, 1:]
    ds.close()
    return lat, vert


def load(manifest):
    meteo = read(os.path.join(DATA, "meteo.nc"), ["air_mass", "specific_humidity"])
    if MODE == "moist_air_mass":
        air = meteo["air_mass"]                       # forgets the water vapour
    else:
        air = meteo["air_mass"] * (1.0 - meteo["specific_humidity"])
    w_o3 = manifest["molar_mass_g_per_mol"]["O3"]
    w_dry = manifest["molar_mass_g_per_mol"]["dry_air"]

    chem = manifest["chemistry"]
    losses = list(chem["destruction_channels"])
    if MODE == "drop_o1d_channel":
        losses.remove("o3_loss_o1d")
    if MODE == "duplicate_titration":
        losses.append("o3_loss_no")

    fields = {}
    for s in SCENARIOS:
        ppbv = read(os.path.join(DATA, s, "state.nc"), ["o3_vmr"])["o3_vmr"]
        mass = ppbv * 1.0e-9 * air * (w_o3 / w_dry)
        names = chem["production_channels"] + chem["destruction_channels"] + [
            "o3_depos", "o3_prod_ox_diag"]
        proc = read(os.path.join(DATA, s, "process.nc"), names)
        prod = proc["o3_prod_jno2"]
        if MODE == "radical_diagnostic_as_production":
            prod = proc["o3_prod_ox_diag"]
        loss = sum(proc[n] for n in losses)
        dep = proc["o3_depos"]
        flux = os.path.join(DATA, s, "fluxes.nc")
        adv_lat, adv_vert = divergence(flux, "adv", prod.shape)
        mix_lat, mix_vert = divergence(flux, "mix", prod.shape)
        if MODE == "deposition_double_counted":
            vert = mix_vert                            # forgets to add D back
        elif MODE == "deposition_omitted":
            vert = mix_vert + dep
            dep = np.zeros_like(dep)
        else:
            vert = mix_vert + dep
        fields[s] = {"mass": mass, "ppbv": ppbv,
                     "A_lateral": adv_lat, "A_vertical": adv_vert,
                     "A": adv_lat + adv_vert,
                     "K_lateral": mix_lat, "K_vertical": vert,
                     "K": mix_lat + vert, "P": prod, "L": loss, "D": dep}
    return fields


def budget(f, weight, n0, n1):
    mass = f["mass"]
    acc = {k: 0.0 for k in ("A", "A_lateral", "A_vertical", "K", "K_lateral",
                            "K_vertical", "P", "L", "D")}
    g_term = 0.0
    for n in range(n0, n1):
        w = weight[n + 1] if MODE == "end_of_step_weights" else weight[n]
        for key in acc:
            acc[key] += float(np.sum(w * f[key][n]))
        if MODE == "mask_term_start_state":
            g_term += float(np.sum((weight[n + 1] - weight[n]) * mass[n]))
        else:
            g_term += float(np.sum((weight[n + 1] - weight[n]) * mass[n + 1]))
    row = dict(acc)
    row["M0"] = float(np.sum(weight[n0] * mass[n0]))
    row["M1"] = float(np.sum(weight[n1] * mass[n1]))
    row["C"] = row["P"] - row["L"]
    row["S"] = 0.0
    row["G"] = 0.0 if MODE == "mask_term_omitted" else g_term
    row["dM"] = row["M1"] - row["M0"]
    row["residual"] = row["dM"] - (row["A"] + row["K"] + row["C"]
                                   - row["D"] + row["S"] + row["G"])
    if MODE == "transport_from_residual":
        row["A"] = row["dM"] - (row["K"] + row["C"] - row["D"]
                                + row["S"] + row["G"])
        row["A_lateral"] = row["A"]
        row["A_vertical"] = 0.0
        row["residual"] = 0.0
    if MODE == "all_zero":
        for k in row:
            row[k] = 0.0
    return row


def contrast_set(v):
    return {"d_fire_A0": v["10"] - v["00"], "d_fire_A1": v["11"] - v["01"],
            "d_urban_F0": v["01"] - v["00"], "d_urban_F1": v["11"] - v["10"],
            "interaction": v["11"] - v["01"] - v["10"] + v["00"]}


def trapezoid(seq):
    return sum(0.5 * (seq[n] + seq[n + 1]) for n in range(len(seq) - 1))


def main():
    manifest = json.load(open(os.path.join(DATA, "manifest.json")))
    windows = {k: (v["start_index"], v["end_index"])
               for k, v in manifest["windows"].items()}
    names = sorted(windows)
    masks = read(os.path.join(DATA, "masks.nc"), ["fixed_urban", "moving_plume"])
    fields = load(manifest)
    n_end = fields["00"]["mass"].shape[0]
    weights = {"fixed_urban": np.broadcast_to(masks["fixed_urban"],
                                              (n_end,) + masks["fixed_urban"].shape),
               "moving_plume": masks["moving_plume"]}

    order = dict(zip(SCENARIOS, SCENARIOS))
    if MODE == "scenarios_swapped":
        order["10"], order["01"] = "01", "10"

    budgets = {}
    for volume in VOLUMES:
        for window in names:
            n0, n1 = windows[window]
            for s in SCENARIOS:
                budgets[(s, volume, window)] = budget(
                    fields[order[s]], weights[volume], n0, n1)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "budgets.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "scenario", "volume", "window", "M0_kg", "M1_kg",
                    "A_kg", "A_lateral_kg", "A_vertical_kg", "K_kg",
                    "K_lateral_kg", "K_vertical_kg", "P_kg", "L_kg", "C_kg",
                    "D_kg", "S_kg", "G_kg", "residual_kg"])
        for volume in VOLUMES:
            for window in names:
                for s in SCENARIOS:
                    r = budgets[(s, volume, window)]
                    w.writerow([f"{s}|{volume}|{window}", s, volume, window]
                               + [FMT % r[k] for k in (
                                   "M0", "M1", "A", "A_lateral", "A_vertical",
                                   "K", "K_lateral", "K_vertical", "P", "L",
                                   "C", "D", "S", "G", "residual")])

    with open(os.path.join(OUT, "contrasts.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["contrast_id", "volume", "window", "quantity", "d_fire_A0",
                    "d_fire_A1", "d_urban_F0", "d_urban_F1", "interaction"])
        for volume in VOLUMES:
            for window in names:
                for q in QUANTITIES:
                    vals = {s: budgets[(s, volume, window)][q] for s in SCENARIOS}
                    c = contrast_set(vals)
                    if MODE == "interaction_always_positive":
                        c["interaction"] = abs(c["interaction"])
                    w.writerow([f"{volume}|{window}|{q}", volume, window, q]
                               + [FMT % c[k] for k in ("d_fire_A0", "d_fire_A1",
                                                       "d_urban_F0", "d_urban_F1",
                                                       "interaction")])

    receptors, rorder = {}, []
    with open(os.path.join(DATA, "receptors.csv")) as f:
        for row in csv.DictReader(f):
            rid = row["receptor_id"]
            if rid not in receptors:
                receptors[rid] = []
                rorder.append(rid)
            receptors[rid].append((int(row["i"]), int(row["j"]), int(row["k"]),
                                   float(row["weight"])))

    metrics = ("mean_ppbv", "peak_ppbv", "exposure_ppbvh", "exposure_over40_ppbvh")

    def series_metrics(chi, n0, n1):
        seg = [float(chi[n]) for n in range(n0, n1 + 1)]
        integral = trapezoid(seg)
        over = trapezoid([max(v - THRESHOLD, 0.0) for v in seg])
        if MODE == "threshold_after_integration":
            over = max(integral - THRESHOLD * (n1 - n0), 0.0)
        return {"mean_ppbv": integral / float(n1 - n0), "peak_ppbv": max(seg),
                "exposure_ppbvh": integral, "exposure_over40_ppbvh": over}

    with open(os.path.join(OUT, "receptors.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "receptor_id", "series", "window"] + list(metrics))
        for rid in rorder:
            chi = {}
            for s in SCENARIOS:
                ppbv = fields[order[s]]["ppbv"]
                acc = np.zeros(ppbv.shape[0])
                for (i, j, k, wgt) in receptors[rid]:
                    if MODE == "receptor_nearest_cell":
                        continue
                    acc += wgt * ppbv[:, i, j, k]
                if MODE == "receptor_nearest_cell":
                    i, j, k, _ = receptors[rid][0]
                    acc = ppbv[:, i, j, k]
                chi[s] = acc
            for window in names:
                n0, n1 = windows[window]
                per = {s: series_metrics(chi[s], n0, n1) for s in SCENARIOS}
                table = {}
                for m in metrics:
                    vals = {s: per[s][m] for s in SCENARIOS}
                    table[m] = dict(vals)
                    table[m].update(contrast_set(vals))
                if MODE == "peak_of_difference":
                    for name, (p, mn) in {"d_fire_A0": ("10", "00"),
                                          "d_fire_A1": ("11", "01"),
                                          "d_urban_F0": ("01", "00"),
                                          "d_urban_F1": ("11", "10")}.items():
                        diff = [chi[p][n] - chi[mn][n] for n in range(n0, n1 + 1)]
                        table["peak_ppbv"][name] = max(diff)
                for name in SERIES:
                    w.writerow([f"{rid}|{name}|{window}", rid, name, window]
                               + [FMT % table[m][name] for m in metrics])

    rows = []
    for volume in VOLUMES:
        for window in names:
            r = {s: budgets[(s, volume, window)] for s in SCENARIOS}
            tau = 0.005 * abs(r["11"]["M0"])

            def chem(v):
                return ("net_production" if v > tau else
                        "net_destruction" if v < -tau else "near_zero")

            def contrast(v):
                return ("more_positive" if v > tau else
                        "more_negative" if v < -tau else "near_zero")

            scen = {s: chem(r[s]["C"]) for s in SCENARIOS}
            if MODE == "enhancement_means_production":
                scen = {s: ("net_production" if r[s]["M1"] > r[s]["M0"]
                            else "net_destruction") for s in SCENARIOS}
            inter = r["11"]["C"] - r["01"]["C"] - r["10"]["C"] + r["00"]["C"]
            label = ("amplifying" if inter > tau else
                     "damping" if inter < -tau else "near_zero")
            if MODE == "interaction_always_positive":
                label = "amplifying"
            dom = {}
            for s in SCENARIOS:
                cand = {"A": r[s]["A"], "K": r[s]["K"], "C": r[s]["C"],
                        "D": -r[s]["D"], "G": r[s]["G"]}
                if MODE == "dominant_ignores_mask":
                    cand.pop("G")
                dom[s] = max(sorted(cand), key=lambda k: abs(cand[k]))
            d_burden = r["11"]["M1"] - r["01"]["M1"]
            d_chem = r["11"]["C"] - r["01"]["C"]
            d_trans = sum(r["11"][k] - r["01"][k] for k in ("A", "K", "G"))
            sustain = bool(d_burden > tau and d_chem < -tau and d_trans > tau)
            if MODE == "sustain_from_enhancement_only":
                sustain = bool(d_burden > tau)
            rows.append({"volume": volume, "window": window, "tau_kg": tau,
                         "scenario_chemistry": scen,
                         "fire_chemistry_contrast": {
                             "A0": contrast(r["10"]["C"] - r["00"]["C"]),
                             "A1": contrast(d_chem)},
                         "interaction_label": label, "dominant_term": dom,
                         "transport_sustains_anomaly": sustain})

    with open(os.path.join(OUT, "diagnosis.json"), "w") as f:
        json.dump({"schema_version": "o3-budget-diagnosis-1.0", "rows": rows},
                  f, indent=1, sort_keys=True)

    os.makedirs(os.path.join(OUT, "workflow"), exist_ok=True)
    with open(os.path.join(OUT, "workflow", "RUN.txt"), "w") as f:
        f.write("baseline mode: " + MODE + "\n" + "x" * 600 + "\n")
    print("baseline", MODE, "written")


if __name__ == "__main__":
    sys.exit(main())
