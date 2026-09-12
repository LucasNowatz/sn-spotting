"""Physics unit checks for the simulator (authoring record).

  1. number conservation with no source, loss, wind or growth
  2. zero-wind box: vapour approaches its analytic balance
  3. single column: mean diameter follows dD/dt = G(D)
  4. constant-wind Gaussian plume: variance grows as 2 Kh t
  5. uniform tracer through every episode on the public grid stays uniform
"""
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simulator as SM
import dynamics as DY
import driver as D
import scenarios as SC

NX, NY, DX = 24, 20, 4000.0
P = dict(kh=5000.0, we=0.008, tauv=2.5, yorg=1.0, betaorg=1.0, heff=70.0,
         taup=14.0, sA=1.0, sB=1.0, sC=1.0, s_ev=1.0, q_ev=1.0)


def make(u0=0.0, v0=0.0, T0=290.0, h0=900.0, q0=0.0, ca=0.0):
    t = np.linspace(0.0, 8 * 3600.0, 49)
    one = np.ones((49, NY, NX))
    xf = np.arange(NX + 1) * DX; yf = np.arange(NY + 1) * DX
    Xf, Yf = np.meshgrid(xf, yf)
    psi = np.tile((-u0 * Yf + v0 * Xf)[None], (49, 1, 1))
    maps = np.zeros((3, NY, NX))
    return SM.Scenario(NX, NY, DX, DX, t, psi, one * T0, one * h0, one * q0,
                       one * ca, maps, np.zeros((3, 49)), False)


def conservation():
    n = np.zeros((SM.N_BINS, NY, NX)); n[3, NY // 2, NX // 2] = 1000.0
    tot0 = n.sum(); widths = np.diff(SM.bin_edges())[:, None, None]
    for _ in range(600):
        n = SM.size_flux(n, widths, np.zeros((SM.N_BINS - 1, NY, NX)), 1 / 60)
        n = DY.advection_step(n, np.zeros((NY, NX + 1)), np.zeros((NY + 1, NX)), DX, DX, 60.0, 0.0)
    return abs(n.sum() - tot0) / tot0


def box_vapour():
    q0, h0 = 0.5, 900.0
    sc = make(q0=q0, h0=h0)
    p = dict(P); p["kh"] = 0.0
    res = SM.integrate(sc, p, 60.0, 48, 10)
    c_end = res["c"][-1][NY // 2, NX // 2]
    kv = 1.0 / (p["tauv"] * 3600.0); ke = p["we"] / h0
    c_star = (p["yorg"] * q0 / 3600.0 + ke * SM.C_BG) / (kv + ke)
    c_ana = c_star + (SM.C_BG - c_star) * np.exp(-(kv + ke) * 8 * 3600.0)
    return abs(c_end - c_ana) / c_ana


def column_growth():
    T0, c0, ca = 290.0, 1.0, 4.0
    edges = SM.bin_edges(); widths = np.diff(edges)[:, None, None]; ctr = SM.bin_centres()
    n = np.zeros((SM.N_BINS, 1, 1)); n[0] = 1000.0
    dt_h = 1 / 60
    for _ in range(int(8.0 / dt_h)):
        g = SM.growth_rate(edges[1:-1, None, None], np.array([[[T0]]]), np.array([[[c0]]]),
                           np.array([[[ca]]]), P["betaorg"], P["heff"])
        n = SM.size_flux(n, widths, g, dt_h)
    w = n[:, 0, 0]
    d_num = np.exp((w * np.log(ctr)).sum() / w.sum())
    sol = solve_ivp(lambda t, Dd: SM.growth_rate(Dd[0], T0, c0, ca, P["betaorg"], P["heff"]),
                    [0, 8.0], [ctr[0]], rtol=1e-9, atol=1e-11)
    return abs(d_num - sol.y[0, -1]) / sol.y[0, -1]


def plume():
    sig0, U, hours = 12000.0, 6.0, 6.0
    nx, ny = 70, 50
    x = (np.arange(nx) + 0.5) * DX; y = (np.arange(ny) + 0.5) * DX
    X, Y = np.meshgrid(x, y)
    q = np.exp(-((X - 60e3) ** 2 + (Y - ny * DX / 2) ** 2) / (2 * sig0 ** 2))
    uf = np.full((ny, nx + 1), U); vf = np.zeros((ny + 1, nx))
    for _ in range(int(hours * 60)):
        q = DY.transport_step(q, uf, vf, DX, DX, 60.0, P["kh"], 0.0)
    m = q.sum(); cy = (q * Y).sum() / m
    var_y = (q * (Y - cy) ** 2).sum() / m
    ana = sig0 ** 2 + 2 * P["kh"] * hours * 3600
    return abs(var_y - ana) / ana


def uniform():
    worst = 0.0
    for e in SC.EPISODES:
        sc = D.build(e, SC.NX_PUB, SC.NY_PUB)
        q = np.full((sc.ny, sc.nx), 1.0)
        for k in range(480):
            uf, vf, *_ = sc.at(k * 60.0)
            q = DY.advection_step(q, uf, vf, sc.dx, sc.dy, 60.0, 1.0)
        worst = max(worst, float(np.abs(q - 1.0).max()))
    return worst


if __name__ == "__main__":
    print(f"1. number conservation: {conservation():.3e}")
    print(f"2. zero-wind box vapour: {box_vapour():.3e}")
    print(f"3. column growth mean diameter: {column_growth():.3e}")
    print(f"4. Gaussian plume spread: {plume():.3e}")
    print(f"5. uniform tracer, all episodes: {uniform():.3e}")
