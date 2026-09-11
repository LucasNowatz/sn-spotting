"""Independent conservation evaluator for the wildfire plume ozone archive.

This is the second implementation.  It reads only the NetCDF files that were
written into environment/data/ and knows nothing about the forward model that
produced them: no shared module, no shared helper, no shared array.  It is
written with xarray and whole-array reductions, while the reference solution
under solution/ is written with netCDF4 and explicit interval loops, so an
agreement between the two is evidence about the archive rather than about one
piece of code.

Output: tests/truth.json, the sealed expectation the verifier grades against.
"""

import json
import os

import numpy as np
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "..", "environment", "data"))
TESTS = os.path.normpath(os.path.join(HERE, "..", "..", "tests"))

SCENARIOS = ["00", "10", "01", "11"]
VOLUMES = ["fixed_urban", "moving_plume"]
LOSS = ["o3_loss_no", "o3_loss_o1d", "o3_loss_ho2", "o3_loss_oh", "o3_loss_alkene"]
QUANTITIES = ["M0", "M1", "dM", "A", "K", "P", "L", "C", "D", "G", "residual"]
SERIES = ["00", "10", "01", "11", "d_fire_A0", "d_fire_A1",
          "d_urban_F0", "d_urban_F1", "interaction"]
THRESHOLD_PPBV = 40.0
TAU_FRACTION = 0.005


def load():
    man = json.load(open(os.path.join(DATA, "manifest.json")))
    grid = xr.open_dataset(os.path.join(DATA, "grid.nc"))
    meteo = xr.open_dataset(os.path.join(DATA, "meteo.nc"))
    masks = xr.open_dataset(os.path.join(DATA, "masks.nc"))
    return man, grid, meteo, masks


def cell_ozone_mass(o3_ppbv, meteo, man):
    """kg O3 per cell from the dry-air mole fraction."""
    w_o3 = man["molar_mass_g_per_mol"]["O3"]
    w_dry = man["molar_mass_g_per_mol"]["dry_air"]
    dry = meteo["air_mass"] * (1.0 - meteo["specific_humidity"])
    return o3_ppbv * 1.0e-9 * dry * (w_o3 / w_dry)


def divergence(ds, kind):
    """Net import per cell, split into the lateral and vertical interfaces."""
    fx = ds[f"{kind}_flux_x"]
    fy = ds[f"{kind}_flux_y"]
    fz = ds[f"{kind}_flux_z"]
    dx = fx.isel(x_face=slice(0, -1)).values - fx.isel(x_face=slice(1, None)).values
    dy = fy.isel(y_face=slice(0, -1)).values - fy.isel(y_face=slice(1, None)).values
    dz = fz.isel(z_face=slice(0, -1)).values - fz.isel(z_face=slice(1, None)).values
    return dx + dy, dz


def scenario_fields(s, meteo, man):
    st = xr.open_dataset(os.path.join(DATA, s, "state.nc"))
    pr = xr.open_dataset(os.path.join(DATA, s, "process.nc"))
    fx = xr.open_dataset(os.path.join(DATA, s, "fluxes.nc"))
    mass = cell_ozone_mass(st["o3_vmr"], meteo, man).values.astype(np.float64)
    adv_lat, adv_vert = (a.astype(np.float64) for a in divergence(fx, "adv"))
    mix_lat, mix_vert = (a.astype(np.float64) for a in divergence(fx, "mix"))
    dep = pr["o3_depos"].values.astype(np.float64)
    prod = pr["o3_prod_jno2"].values.astype(np.float64)
    loss = np.zeros_like(prod)
    for name in LOSS:
        loss += pr[name].values.astype(np.float64)
    return {"mass": mass, "A_lateral": adv_lat, "A_vertical": adv_vert,
            "A": adv_lat + adv_vert, "K_lateral": mix_lat,
            "K_vertical": mix_vert + dep, "K": mix_lat + mix_vert + dep,
            "P": prod, "L": loss, "D": dep,
            "o3_ppbv": st["o3_vmr"].values.astype(np.float64)}


def weights(volume, masks, n_end):
    if volume == "fixed_urban":
        w = masks["fixed_urban"].values.astype(np.float64)
        return np.broadcast_to(w, (n_end,) + w.shape).copy()
    return masks["moving_plume"].values.astype(np.float64)


def budget(fields, w, n0, n1):
    m = fields["mass"]
    b0 = float((w[n0] * m[n0]).sum())
    b1 = float((w[n1] * m[n1]).sum())
    ws = w[n0:n1]
    out = {"M0": b0, "M1": b1}
    for key in ("A", "A_lateral", "A_vertical", "K", "K_lateral",
                "K_vertical", "P", "L", "D"):
        out[key] = float((ws * fields[key][n0:n1]).sum())
    out["C"] = out["P"] - out["L"]
    out["S"] = 0.0
    out["G"] = float(((w[n0 + 1:n1 + 1] - w[n0:n1]) * m[n0 + 1:n1 + 1]).sum())
    out["dM"] = b1 - b0
    out["residual"] = out["dM"] - (out["A"] + out["K"] + out["C"]
                                   - out["D"] + out["S"] + out["G"])
    return out


def receptor_series(fields, cells):
    chi = np.zeros(fields["o3_ppbv"].shape[0])
    for (i, j, k, wgt) in cells:
        chi += wgt * fields["o3_ppbv"][:, i, j, k]
    return chi


def receptor_metrics(chi, n0, n1):
    seg = chi[n0:n1 + 1]
    hours = float(n1 - n0)
    integral = float(np.trapezoid(seg, dx=1.0))
    excess = np.clip(seg - THRESHOLD_PPBV, 0.0, None)
    return {"mean_ppbv": integral / hours,
            "peak_ppbv": float(seg.max()),
            "exposure_ppbvh": integral,
            "exposure_over40_ppbvh": float(np.trapezoid(excess, dx=1.0))}


def contrasts(vals):
    return {"d_fire_A0": vals["10"] - vals["00"],
            "d_fire_A1": vals["11"] - vals["01"],
            "d_urban_F0": vals["01"] - vals["00"],
            "d_urban_F1": vals["11"] - vals["10"],
            "interaction": vals["11"] - vals["01"] - vals["10"] + vals["00"]}


def label_chem(c, tau):
    if c > tau:
        return "net_production"
    if c < -tau:
        return "net_destruction"
    return "near_zero"


def label_contrast(d, tau):
    if d > tau:
        return "more_positive"
    if d < -tau:
        return "more_negative"
    return "near_zero"


def label_interaction(i, tau):
    if i > tau:
        return "amplifying"
    if i < -tau:
        return "damping"
    return "near_zero"


def dominant(row):
    cand = {"A": row["A"], "K": row["K"], "C": row["C"],
            "D": -row["D"], "G": row["G"]}
    return max(sorted(cand), key=lambda k: abs(cand[k]))


def main():
    man, grid, meteo, masks = load()
    windows = {k: (v["start_index"], v["end_index"])
               for k, v in man["windows"].items()}
    n_end = meteo.sizes["time"]
    fields = {s: scenario_fields(s, meteo, man) for s in SCENARIOS}

    receptors = {}
    for line in open(os.path.join(DATA, "receptors.csv")).read().splitlines()[1:]:
        rid, i, j, k, wgt = line.split(",")
        receptors.setdefault(rid, []).append((int(i), int(j), int(k), float(wgt)))

    budgets = {}
    for vol in VOLUMES:
        w = weights(vol, masks, n_end)
        for win, (n0, n1) in windows.items():
            for s in SCENARIOS:
                budgets[f"{s}|{vol}|{win}"] = budget(fields[s], w, n0, n1)

    contrast_rows = {}
    for vol in VOLUMES:
        for win in windows:
            for q in QUANTITIES:
                vals = {s: budgets[f"{s}|{vol}|{win}"][q] for s in SCENARIOS}
                contrast_rows[f"{vol}|{win}|{q}"] = contrasts(vals)

    rec_rows = {}
    chi = {rid: {s: receptor_series(fields[s], cells) for s in SCENARIOS}
           for rid, cells in receptors.items()}
    for rid in receptors:
        for win, (n0, n1) in windows.items():
            per = {s: receptor_metrics(chi[rid][s], n0, n1) for s in SCENARIOS}
            for metric in ("mean_ppbv", "peak_ppbv", "exposure_ppbvh",
                           "exposure_over40_ppbvh"):
                vals = {s: per[s][metric] for s in SCENARIOS}
                cs = contrasts(vals)
                for series in SERIES:
                    key = f"{rid}|{series}|{win}"
                    rec_rows.setdefault(key, {})
                    rec_rows[key][metric] = (vals[series] if series in vals
                                             else cs[series])

    diagnosis = []
    scales = {}
    for vol in VOLUMES:
        for win in windows:
            r = {s: budgets[f"{s}|{vol}|{win}"] for s in SCENARIOS}
            tau = TAU_FRACTION * abs(r["11"]["M0"])
            scales[f"{vol}|{win}"] = abs(r["11"]["M0"])
            ic = r["11"]["C"] - r["01"]["C"] - r["10"]["C"] + r["00"]["C"]
            d_m1 = r["11"]["M1"] - r["01"]["M1"]
            d_c = r["11"]["C"] - r["01"]["C"]
            d_trans = ((r["11"]["A"] - r["01"]["A"])
                       + (r["11"]["K"] - r["01"]["K"])
                       + (r["11"]["G"] - r["01"]["G"]))
            diagnosis.append({
                "volume": vol, "window": win, "tau_kg": tau,
                "scenario_chemistry": {s: label_chem(r[s]["C"], tau)
                                       for s in SCENARIOS},
                "fire_chemistry_contrast": {
                    "A0": label_contrast(r["10"]["C"] - r["00"]["C"], tau),
                    "A1": label_contrast(d_c, tau)},
                "interaction_label": label_interaction(ic, tau),
                "dominant_term": {s: dominant(r[s]) for s in SCENARIOS},
                "transport_sustains_anomaly": bool(
                    (d_m1 > tau) and (d_c < -tau) and (d_trans > tau)),
            })

    truth = {
        "schema_version": "o3-budget-truth-1.0",
        "windows": windows,
        "scales_kg": scales,
        "tolerance": {"rtol": 1.0e-4, "atol_scale_fraction": 2.0e-3,
                      "receptor_rtol": 1.0e-4, "receptor_atol_ppbv": 2.0e-3,
                      "receptor_atol_ppbvh": 2.0e-2},
        "budgets": budgets,
        "contrasts": contrast_rows,
        "receptors": rec_rows,
        "diagnosis": diagnosis,
    }
    os.makedirs(TESTS, exist_ok=True)
    with open(os.path.join(TESTS, "truth.json"), "w") as f:
        json.dump(truth, f, indent=1, sort_keys=True)
    print("budget rows", len(budgets), "contrast rows", len(contrast_rows),
          "receptor rows", len(rec_rows), "diagnosis rows", len(diagnosis))
    worst = max(abs(v["residual"]) / max(abs(v["M0"]), 1.0)
                for v in budgets.values())
    print("worst relative closure residual %.3e" % worst)


if __name__ == "__main__":
    main()
