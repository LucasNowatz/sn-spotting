"""Numerical validation of the wildfire plume ozone archive.  Writes validation.md.

Checks, in order:

1. the dry-air mass flux field is discretely non-divergent and has no flow
   through the ground or the lid;
2. the recorded process increments close each cell's ozone mass exactly;
3. the four runs share one meteorological realisation bit for bit;
4. analytic cases: closed-box chemistry, pure transport, deposition only, and
   and a prescribed mask-motion case with an analytic answer;
5. time-step refinement: the whole episode re-integrated at half the model step
   and half the chemistry sub-step, compared term by term;
6. determinism: a second generation run reproduces the archive byte for byte.

Authoring material.  Not mounted into any container.
"""

import hashlib
import json
import os
import subprocess
import sys

import numpy as np

import generate_dataset as G

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "..", "environment", "data"))
report = []


def say(line=""):
    print(line)
    report.append(line)


def check_flux_field():
    say("## 1. Mass flux field")
    say()
    worst_div = 0.0
    worst_ground = 0.0
    worst_lid = 0.0
    speeds = []
    for hour in range(0, 48, 3):
        fx, fy, fz = G.mass_fluxes(hour * 3600.0)
        div = ((fx[1:] - fx[:-1]) + (fy[:, 1:] - fy[:, :-1])
               + (fz[:, :, 1:] - fz[:, :, :-1]))
        scale = max(np.abs(fx).max(), np.abs(fy).max())
        worst_div = max(worst_div, np.abs(div).max() / scale)
        worst_ground = max(worst_ground, np.abs(fz[:, :, 0]).max())
        worst_lid = max(worst_lid, np.abs(fz[:, :, -1]).max())
        u = fx / (G.rho_dry[None, None, :] * G.DY * G.DZ[None, None, :])
        speeds.append(np.abs(u).max())
    say(f"* worst relative mass divergence: {worst_div:.3e}")
    say(f"* worst vertical mass flux at the ground: {worst_ground:.3e} kg/s")
    say(f"* worst vertical mass flux at the lid: {worst_lid:.3e} kg/s")
    say(f"* peak resolved horizontal wind speed: {max(speeds):.2f} m/s")
    say()
    return worst_div


def check_closure(runs):
    say("## 2. Per-cell closure of the recorded increments")
    say()
    say("| run | worst absolute residual (kg) | relative to the largest cell mass |")
    say("|-----|------------------------------|-----------------------------------|")
    worst = 0.0
    for name in ("00", "10", "01", "11"):
        absolute, relative = G.check_closure(runs[name])
        worst = max(worst, relative)
        say(f"| {name} | {absolute:.3e} | {relative:.3e} |")
    say()
    return worst


def check_shared_meteorology(runs):
    say("## 3. Shared meteorology")
    say()
    same = all(np.array_equal(runs[n]["air_mass"], runs["00"]["air_mass"])
               and np.array_equal(runs[n]["spec_humidity"], runs["00"]["spec_humidity"])
               for n in runs)
    say(f"* air mass and specific humidity identical in all four runs: {same}")
    say()
    return same


def analytic_cases():
    say("## 4. Analytic cases")
    say()

    # Closed box: chemistry alone must move the ozone mass by exactly P - L.
    n_air = G.rho_dry / (G.W_DRY * 1.0e-3) * G.N_AV * 1.0e-6
    n_air = n_air[None, None, :] * np.ones((G.NX, G.NY, 1))
    q = G.spec_humidity(3600.0)
    n_h2o = (q / (1.0 - q)) * (G.W_DRY / G.W_H2O) * n_air
    start = [np.full((G.NX, G.NY, G.NZ), v) * n_air for v in
             (45e-9, 3e-9, 4e-9, 150e-9, 8e-9, 2e-9, 1e-9, 1e-9, 1e-9)]
    before = start[0].copy()
    after, channels = G.chemistry([a.copy() for a in start], n_air, n_h2o,
                                  8.0 * 3600.0, G.DT)
    net = channels["prod"] - sum(channels[k] for k in
                                 ("l_no", "l_o1d", "l_ho2", "l_oh", "l_alk"))
    err = np.abs((after[0] - before) - net).max() / np.abs(before).max()
    say(f"* closed-box chemistry, |dO3 - (P - L)| / O3: {err:.3e}")

    # Pure transport: a uniform mixing ratio must stay uniform and conserved.
    uniform = np.ones((G.NX, G.NY, G.NZ)) * G.m_dry * 1.0e-6
    fx, fy, fz = G.mass_fluxes(5.0 * 3600.0)
    moved, _, _, _ = G.advect(uniform, fx, fy, fz,
                              np.full(G.NZ, 1.0e-6), G.DT)
    ratio = moved / G.m_dry
    say(f"* pure transport, departure from a uniform mixing ratio: "
        f"{np.abs(ratio - 1.0e-6).max() / 1.0e-6:.3e}")
    say(f"* pure transport, mass change against the net boundary flux: "
        f"{abs(moved.sum() - uniform.sum() - _net_boundary(fx, fy, 1.0e-6)):.3e} kg")

    # Deposition only: one well-mixed surface layer against the analytic decay.
    mass = np.zeros((G.NX, G.NY, G.NZ))
    mass[:, :, 0] = G.m_dry[:, :, 0] * 1.0e-6
    vd = G.deposition_velocity(12.0 * 3600.0)
    kz = np.zeros(G.NZ + 1)
    step = 60.0
    total = 3600.0
    state = mass.copy()
    for _ in range(int(total / step)):
        state, _flux, _dep = G.vertical_diffusion(state, kz, vd, step, True)
    expected = mass[:, :, 0] * np.exp(-vd * total / (G.DZ[0]))
    got = state[:, :, 0]
    say(f"* deposition only, worst relative departure from exp(-vd t / dz): "
        f"{np.abs(got / expected - 1.0).max():.3e} "
        f"(implicit first-order truncation at 60 s)")

    # Prescribed mask motion: the sampled burden jumps with no physics at all.
    cell_mass = np.array([[10.0, 30.0]])
    w0 = np.array([[1.0, 0.0]])
    w1 = np.array([[0.0, 1.0]])
    g_term = float(((w1 - w0) * cell_mass).sum())
    burden0 = float((w0 * cell_mass).sum())
    burden1 = float((w1 * cell_mass).sum())
    say(f"* prescribed mask motion, B0 = {burden0:.0f} kg, B1 = {burden1:.0f} kg, "
        f"G = {g_term:.0f} kg with every physical increment zero")
    say()
    return err


def _net_boundary(fx, fy, ratio):
    inflow = (fx[0] * ratio).sum() - (fx[-1] * ratio).sum()
    inflow += (fy[:, 0] * ratio).sum() - (fy[:, -1] * ratio).sum()
    return inflow * G.DT


def refinement(runs):
    say("## 5. Time-step refinement")
    say()
    say("The episode is re-integrated with the model step halved to 90 s and the "
        "chemistry sub-step halved to 2.5 s, holding the one-hour diagnostic "
        "interval fixed.  The table compares the fixed urban volume over window "
        "W2 in run 11, as a fraction of the initial burden of that row.")
    say()
    G.DT = 90.0
    G.STEPS_PER_DIAG = 40
    G.N_SUB = 36
    G.DT_SUB = G.DT / G.N_SUB
    fine = G.run_scenario(True, True)
    G.DT = 180.0
    G.STEPS_PER_DIAG = 20
    G.N_SUB = 36
    G.DT_SUB = G.DT / G.N_SUB

    n0, n1 = G.WINDOWS["W2"]
    fixed = np.zeros((G.NX, G.NY, G.NZ))
    fixed[G.FIXED_I, G.FIXED_J, G.FIXED_K] = 1.0
    say("| term | 180 s run (kg) | 90 s run (kg) | difference / M0 |")
    say("|------|----------------|---------------|-----------------|")
    base = _window_terms(runs["11"], fixed, n0, n1)
    ref = _window_terms(fine, fixed, n0, n1)
    worst = 0.0
    for key in ("M0", "M1", "A", "K", "P", "L", "C", "D"):
        rel = abs(ref[key] - base[key]) / abs(base["M0"])
        worst = max(worst, rel)
        say(f"| {key} | {base[key]:.1f} | {ref[key]:.1f} | {rel:.3e} |")
    say()
    say(f"Worst term-wise change under refinement: {worst:.3e} of the initial "
        "burden.  The archive itself is the graded object and its recorded "
        "increments close it exactly, so this number describes how converged the "
        "physics is, not the grading tolerance.")
    say()
    return worst


def _window_terms(run, w, n0, n1):
    mass = run["o3_vmr"] * G.m_dry[None] * (G.W_O3 / G.W_DRY)
    fl, pr = run["fluxes"], run["process"]
    adv = ((fl["adv_flux_x"][:, :-1] - fl["adv_flux_x"][:, 1:])
           + (fl["adv_flux_y"][:, :, :-1] - fl["adv_flux_y"][:, :, 1:])
           + (fl["adv_flux_z"][:, :, :, :-1] - fl["adv_flux_z"][:, :, :, 1:]))
    mix = ((fl["mix_flux_x"][:, :-1] - fl["mix_flux_x"][:, 1:])
           + (fl["mix_flux_y"][:, :, :-1] - fl["mix_flux_y"][:, :, 1:])
           + (fl["mix_flux_z"][:, :, :, :-1] - fl["mix_flux_z"][:, :, :, 1:]))
    dep = pr["o3_depos"]
    prod = pr["o3_prod_jno2"]
    loss = sum(pr[k] for k in ("o3_loss_no", "o3_loss_o1d", "o3_loss_ho2",
                               "o3_loss_oh", "o3_loss_alkene"))
    agg = lambda field: float((w[None] * field[n0:n1]).sum())
    out = {"M0": float((w * mass[n0]).sum()), "M1": float((w * mass[n1]).sum()),
           "A": agg(adv), "K": agg(mix) + agg(dep), "P": agg(prod),
           "L": agg(loss), "D": agg(dep)}
    out["C"] = out["P"] - out["L"]
    return out


def determinism():
    say("## 6. Determinism")
    say()
    manifest = json.load(open(os.path.join(DATA, "manifest.json")))
    before = dict(manifest["checksums_sha256"])
    subprocess.run([sys.executable, os.path.join(HERE, "generate_dataset.py")],
                   check=True, capture_output=True)
    after = json.load(open(os.path.join(DATA, "manifest.json")))["checksums_sha256"]
    same = before == after
    say(f"* regeneration reproduces every archive file byte for byte: {same}")
    say(f"* files compared: {len(before)}")
    say()
    return same


def main():
    say("# Validation record")
    say()
    say("Produced by `authoring/provenance/validate.py` against the archive in "
        "`environment/data`.  Every number below is measured, not asserted.")
    say()
    check_flux_field()
    runs = {}
    for name, (fire, urban) in {"00": (0, 0), "10": (1, 0),
                                "01": (0, 1), "11": (1, 1)}.items():
        runs[name] = G.run_scenario(bool(fire), bool(urban))
    check_closure(runs)
    check_shared_meteorology(runs)
    analytic_cases()
    refinement(runs)
    determinism()
    with open(os.path.join(HERE, "validation.md"), "w") as handle:
        handle.write("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
