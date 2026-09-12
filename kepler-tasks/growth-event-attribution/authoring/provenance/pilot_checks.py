"""Physics unit checks for the forward model (authoring record).

  1. number conservation with no source, loss, wind or growth
  2. zero-wind box: vapour approaches its analytic source/loss/dilution balance
  3. single column: mean diameter follows the analytic dD/dt = G(D) trajectory
  4. constant-wind Gaussian plume: spread matches sigma^2 = sigma0^2 + 2 Kh t
"""
import sys, os
import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import truth_model as M
import transport as TR

NX, NY, DX = 24, 20, 5000.0
P = dict(kh=6000.0, we=0.008, tauv=3.0, yorg=1.0, betaorg=1.2, heff=80.0,
         taup=15.0, sA=1.0, sB=1.0, sC=1.0, s_ev=1.0, q_ev=1.0)


def make_ep(u0=0.0, v0=0.0, T0=290.0, h0=900.0, q0=0.0, ca=0.0, s0=0.0,
            nrec=41, hours=10.0):
    """Uniform-wind episode: psi = -u0*Y + v0*X on the corners."""
    t = np.linspace(0.0, hours * 3600.0, nrec)
    one = np.ones((nrec, NY, NX))
    xf = np.arange(NX + 1) * DX
    yf = np.arange(NY + 1) * DX
    Xf, Yf = np.meshgrid(xf, yf)
    psi = np.tile((-u0 * Yf + v0 * Xf)[None], (nrec, 1, 1))
    maps = np.zeros((3, NY, NX))
    maps[0, NY // 2, NX // 4] = 1.0
    return M.Episode(NX, NY, DX, DX, t, psi, one * T0, one * h0,
                     one * q0, one * ca, maps,
                     np.array([[s0] * nrec, [0.0] * nrec, [0.0] * nrec]),
                     False)


def check_number_conservation():
    ep = make_ep(s0=0.0)
    p = dict(P); p["taup"] = 1e9; p["we"] = 0.0; p["betaorg"] = 0.0
    p["kh"] = 0.0
    nbg = np.zeros(M.N_BINS)
    ep2 = ep
    out = M.run_episode(ep2, p, 60.0, 10, 6, tags=False, cbg=0.0, nbg=nbg)
    n0 = out["n"][0]
    # seed a blob instead of relying on the source
    ny, nx = NY, NX
    n = np.zeros((M.N_BINS, ny, nx)); n[3, ny // 2, nx // 2] = 1000.0
    tot0 = n.sum()
    edges = M.bin_edges(); widths = np.diff(edges)[:, None, None]
    for _ in range(600):
        g = np.zeros((M.N_BINS - 1, ny, nx))
        n, _ = M.size_flux(n, n / widths, g, 1.0 / 60.0)
        n = TR.advect(n, np.zeros((ny, nx + 1)), np.zeros((ny + 1, nx)),
                      DX, DX, 60.0, 0.0)
    return abs(n.sum() - tot0) / tot0


def check_box_vapour():
    """Zero wind, uniform source: c -> yorg*q / (1/tauv + we/h) + balance."""
    q0, h0 = 0.5, 900.0
    ep = make_ep(q0=q0, h0=h0)
    p = dict(P)
    p["kh"] = 0.0
    out = M.run_episode(ep, p, 60.0, 40, 15, tags=False, cbg=0.06)
    c_end = out["c"][-1][NY // 2, NX // 2]
    kv = 1.0 / (p["tauv"] * 3600.0)
    ke = p["we"] / h0
    c_star = (p["yorg"] * q0 / 3600.0 + ke * 0.06) / (kv + ke)
    # analytic relaxation from c0 = cbg over 10 h
    tt = 10.0 * 3600.0
    c_ana = c_star + (0.06 - c_star) * np.exp(-(kv + ke) * tt)
    return abs(c_end - c_ana) / c_ana, c_end, c_ana


def check_column_growth():
    """Mean diameter of a narrow mode vs the analytic dD/dt = G(D)."""
    T0, c0, ca = 290.0, 1.0, 4.0
    edges = M.bin_edges(); widths = np.diff(edges)
    ctr = M.bin_centres()
    n = np.zeros(M.N_BINS); n[0] = 1000.0
    dt_h = 60.0 / 3600.0
    hours = 8.0
    steps = int(hours / dt_h)
    n3 = n[:, None, None].copy()
    for _ in range(steps):
        g = M.growth_rate(edges[1:-1, None, None], np.array([[[T0]]]),
                          np.array([[[c0]]]), np.array([[[ca]]]),
                          P["betaorg"], P["heff"])
        n3, _ = M.size_flux(n3, n3 / widths[:, None, None], g, dt_h)
    w = n3[:, 0, 0]
    d_num = np.exp((w * np.log(ctr)).sum() / w.sum())
    sol = solve_ivp(lambda t, D: M.growth_rate(D[0], T0, c0, ca, P["betaorg"],
                                               P["heff"]),
                    [0, hours], [ctr[0]], rtol=1e-9, atol=1e-11)
    d_ana = float(sol.y[0, -1])
    return abs(d_num - d_ana) / d_ana, d_num, d_ana


def check_plume_spread():
    sig0, U, hours = 12000.0, 6.0, 6.0
    nx, ny = 60, 48
    x = (np.arange(nx) + 0.5) * DX; y = (np.arange(ny) + 0.5) * DX
    X, Y = np.meshgrid(x, y)
    q = np.exp(-((X - 60e3) ** 2 + (Y - ny * DX / 2) ** 2) / (2 * sig0 ** 2))
    uf = np.full((ny, nx + 1), U); vf = np.zeros((ny + 1, nx))
    for _ in range(int(hours * 3600 / 60)):
        q = TR.advect(q, uf, vf, DX, DX, 60.0, 0.0)
        q = TR.diffuse(q, P["kh"], DX, DX, 60.0, 0.0)
    m = q.sum(); cy = (q * Y).sum() / m
    var_y = (q * (Y - cy) ** 2).sum() / m
    ana = sig0 ** 2 + 2 * P["kh"] * hours * 3600
    return abs(var_y - ana) / ana, var_y, ana


if __name__ == "__main__":
    e = check_number_conservation()
    print(f"1. number conservation, no source/loss/growth: rel error {e:.3e}")
    e, got, ana = check_box_vapour()
    print(f"2. zero-wind box vapour: {got:.6f} vs analytic {ana:.6f} "
          f"ug m-3, rel error {e:.3e}")
    e, got, ana = check_column_growth()
    print(f"3. column growth mean diameter: {got:.4f} vs analytic {ana:.4f} "
          f"nm, rel error {e:.3e}")
    e, got, ana = check_plume_spread()
    print(f"4. Gaussian plume cross-flow variance: {got:.4e} vs analytic "
          f"{ana:.4e} m2, rel error {e:.3e}")
