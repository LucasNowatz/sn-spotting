#!/usr/bin/env python3
"""Regional transport and nanoparticle-growth forward model (reference).

Written against model_spec.md rather than against the generator: this is an
independent implementation on the public 5 km grid with a 60 s step, using
the published corner streamfunction to build face-normal winds.

Units follow the specification: c in ug m-3, particle number in cm-3,
diameter in nm, growth in nm h-1, time in s.  Parameters are passed as a dict
of physical values (see PHYS below).
"""

import os

import numpy as np
from netCDF4 import Dataset

R_GAS = 8.314462618
T_REF = 298.15
G_ACID = 0.05
D_KELVIN = 1.0
SEC_H = 3600.0
N_BINS = 12
D_MIN, D_MAX = 1.5, 15.0
C_BG = 0.06
N_BG = np.full(N_BINS, 0.8)
GROWN_FIRST_BIN = 5
Q_FIRST_RECORD = 24

PHYS = ["kh", "we", "tauv", "yorg", "betaorg", "heff", "taup",
        "sA", "sB", "sC", "s_ev", "q_ev"]


def to_phys(theta):
    return {k: float(np.exp(v)) for k, v in zip(PHYS, np.asarray(theta))}


# --------------------------------------------------------------------------
# limited advection and diffusion
# --------------------------------------------------------------------------
def _limiter(r):
    return np.maximum(0.0, np.minimum(np.minimum(2.0 * r,
                                                 (1.0 + 2.0 * r) / 3.0), 2.0))


def _faces(q, axis, positive):
    tiny = 1e-300
    n = q.shape[axis]
    if axis == -1:
        a = np.concatenate((q[..., :1], q[..., :n - 2]), axis=-1)
        b = q[..., :n - 1]
        c = q[..., 1:]
        d = np.concatenate((q[..., 2:], q[..., n - 1:]), axis=-1)
    else:
        a = np.concatenate((q[..., :1, :], q[..., :n - 2, :]), axis=-2)
        b = q[..., :n - 1, :]
        c = q[..., 1:, :]
        d = np.concatenate((q[..., 2:, :], q[..., n - 1:, :]), axis=-2)
    fwd = c - b
    up = b + 0.5 * _limiter((b - a) / np.where(np.abs(fwd) < tiny, tiny, fwd)) * fwd
    back = b - c
    dn = c + 0.5 * _limiter((c - d) / np.where(np.abs(back) < tiny, tiny, back)) * back
    return np.where(positive, up, dn)


def transport(q, uf, vf, dx, dy, dt, kh, q_bg):
    """Advection plus diffusion in one call, on the trailing two axes."""
    bg = np.broadcast_to(np.asarray(q_bg, dtype=float), q.shape)

    fx = np.empty(q.shape[:-1] + (q.shape[-1] + 1,))
    fx[..., 1:-1] = uf[..., 1:-1] * _faces(q, -1, uf[..., 1:-1] >= 0.0)
    fx[..., 0] = uf[..., 0] * np.where(uf[..., 0] >= 0.0, bg[..., 0], q[..., 0])
    fx[..., -1] = uf[..., -1] * np.where(uf[..., -1] >= 0.0, q[..., -1],
                                         bg[..., -1])
    fy = np.empty(q.shape[:-2] + (q.shape[-2] + 1, q.shape[-1]))
    fy[..., 1:-1, :] = vf[..., 1:-1, :] * _faces(q, -2, vf[..., 1:-1, :] >= 0.0)
    fy[..., 0, :] = vf[..., 0, :] * np.where(vf[..., 0, :] >= 0.0, bg[..., 0, :],
                                             q[..., 0, :])
    fy[..., -1, :] = vf[..., -1, :] * np.where(vf[..., -1, :] >= 0.0,
                                               q[..., -1, :], bg[..., -1, :])
    out = (q - dt / dx * (fx[..., 1:] - fx[..., :-1])
             - dt / dy * (fy[..., 1:, :] - fy[..., :-1, :]))

    pad = np.empty(q.shape[:-2] + (q.shape[-2] + 2, q.shape[-1] + 2))
    pad[..., 1:-1, 1:-1] = out
    pad[..., 0, 1:-1] = bg[..., 0, :]
    pad[..., -1, 1:-1] = bg[..., -1, :]
    pad[..., 1:-1, 0] = bg[..., :, 0]
    pad[..., 1:-1, -1] = bg[..., :, -1]
    pad[..., 0, 0] = bg[..., 0, 0]; pad[..., 0, -1] = bg[..., 0, -1]
    pad[..., -1, 0] = bg[..., -1, 0]; pad[..., -1, -1] = bg[..., -1, -1]
    lap = ((pad[..., 1:-1, 2:] - 2.0 * out + pad[..., 1:-1, :-2]) / (dx * dx)
           + (pad[..., 2:, 1:-1] - 2.0 * out + pad[..., :-2, 1:-1]) / (dy * dy))
    return out + dt * kh * lap


# --------------------------------------------------------------------------
# size space
# --------------------------------------------------------------------------
def growth(D, T, c, c_acid, betaorg, heff):
    fT = np.exp(-(heff * 1.0e3) / R_GAS * (1.0 / T - 1.0 / T_REF))
    fK = np.exp(-D_KELVIN / D * (T_REF / T))
    return G_ACID * c_acid + betaorg * c * fT * fK


def size_advect(field, widths, g_edge, dt_h):
    """Conservative limited flux through the interior diameter edges."""
    tiny = 1e-300
    f = field / widths
    lo = np.concatenate((np.zeros_like(f[:1]), f[:-1]), axis=0)
    hi = np.concatenate((f[1:], np.zeros_like(f[-1:])), axis=0)
    fwd = hi - f
    face = f + 0.5 * _limiter((f - lo) / np.where(np.abs(fwd) < tiny, tiny,
                                                  fwd)) * fwd
    flux = g_edge * face[:-1]
    out = field.copy()
    out[:-1] -= dt_h * flux
    out[1:] += dt_h * flux
    return out


def edges():
    return D_MIN * (D_MAX / D_MIN) ** (np.arange(N_BINS + 1) / N_BINS)


def centres():
    e = edges()
    return np.sqrt(e[:-1] * e[1:])


# --------------------------------------------------------------------------
# episode input
# --------------------------------------------------------------------------
_CASE_CACHE = {}


def get_case(data_dir, episode):
    key = (data_dir, episode)
    if key not in _CASE_CACHE:
        _CASE_CACHE[key] = Case(os.path.join(data_dir, "episodes", episode))
    return _CASE_CACHE[key]


class Case:
    def __init__(self, ep_dir):
        with Dataset(ep_dir + "/met.nc") as d:
            self.time = np.array(d["time"][:], dtype=float)
            self.x = np.array(d["x"][:], dtype=float)
            self.y = np.array(d["y"][:], dtype=float)
            self.psi = np.array(d["psi"][:], dtype=float)
            self.T = np.array(d["T"][:], dtype=float)
            self.h = np.array(d["pbl_height"][:], dtype=float)
        with Dataset(ep_dir + "/forcing.nc") as d:
            self.q_org = np.array(d["q_org"][:], dtype=float)
            self.c_acid = np.array(d["c_h2so4"][:], dtype=float)
            self.maps = np.array(d["npf_region_map"][:], dtype=float)
            self.npf = np.array(d["npf_rate"][:], dtype=float)
            self.event = bool(int(d["event_day"][...]))
        self.dx = float(self.x[1] - self.x[0])
        self.dy = float(self.y[1] - self.y[0])
        self.ny, self.nx = self.T.shape[1:]

    def at(self, t):
        i = int(np.searchsorted(self.time, self.time[0] + t, side="right")) - 1
        i = min(max(i, 0), len(self.time) - 2)
        w = (self.time[0] + t - self.time[i]) / (self.time[i + 1] - self.time[i])
        w = min(max(w, 0.0), 1.0)
        f = lambda a: a[i] * (1.0 - w) + a[i + 1] * w
        psi = f(self.psi)
        uf = -(psi[1:, :] - psi[:-1, :]) / self.dy
        vf = (psi[:, 1:] - psi[:, :-1]) / self.dx
        npf = self.npf[:, i] * (1.0 - w) + self.npf[:, i + 1] * w
        return uf, vf, f(self.T), f(self.h), f(self.q_org), f(self.c_acid), npf


def _multipliers(case, p):
    s = np.array([p["sA"], p["sB"], p["sC"]])
    q = 1.0
    if case.event:
        s = s * p["s_ev"]
        q = p["q_ev"]
    return s, q


def run_vapour(case, p, dt=60.0, n_out=40):
    """Vapour only: independent of the particle parameters, and much cheaper."""
    steps_per_out = int(round((case.time[-1] - case.time[0]) / dt / n_out))
    _, qm = _multipliers(case, p)
    c = np.full((case.ny, case.nx), C_BG)
    snaps = [c.copy()]
    for k in range(n_out * steps_per_out):
        uf, vf, T, h, q, ca, npf = case.at(k * dt)
        c = c + dt / SEC_H * p["yorg"] * qm * q
        c = transport(c, uf, vf, case.dx, case.dy, dt, p["kh"], C_BG)
        c = c - dt * (c / (p["tauv"] * SEC_H) + p["we"] / h * (c - C_BG))
        np.maximum(c, 0.0, out=c)
        if (k + 1) % steps_per_out == 0:
            snaps.append(c.copy())
    return np.array(snaps)


def run_full(case, p, dt=60.0, n_out=40, tags=False, keep_fields=True):
    """Vapour and the twelve particle bins, optionally with source tags and
    the number-age moment."""
    steps_per_out = int(round((case.time[-1] - case.time[0]) / dt / n_out))
    e = edges()
    widths = np.diff(e)[:, None, None]
    nbg = N_BG[:, None, None]
    sm, qm = _multipliers(case, p)

    c = np.full((case.ny, case.nx), C_BG)
    n = np.tile(nbg, (1, case.ny, case.nx))
    tg = np.zeros((3, N_BINS, case.ny, case.nx)) if tags else None
    ag = np.zeros((N_BINS, case.ny, case.nx)) if tags else None

    out_c, out_n = [c.copy()], [n.copy()]
    out_tg, out_ag = ([tg.copy()], [ag.copy()]) if tags else (None, None)
    dt_h = dt / SEC_H

    for k in range(n_out * steps_per_out):
        uf, vf, T, h, q, ca, npf = case.at(k * dt)
        src = case.maps * (npf * sm)[:, None, None]
        s_tot = src.sum(axis=0)

        c = c + dt_h * p["yorg"] * qm * q
        if tags:
            ag = ag + dt_h * tg.sum(axis=0)
        inj = np.zeros_like(n)
        inj[0] = 0.7 * s_tot
        inj[1] = 0.3 * s_tot
        n = n + dt_h * inj
        if tags:
            tg[:, 0] += dt_h * 0.7 * src
            tg[:, 1] += dt_h * 0.3 * src

        c = transport(c, uf, vf, case.dx, case.dy, dt, p["kh"], C_BG)
        n = transport(n, uf, vf, case.dx, case.dy, dt, p["kh"], nbg)
        if tags:
            tg = transport(tg, uf, vf, case.dx, case.dy, dt, p["kh"], 0.0)
            ag = transport(ag, uf, vf, case.dx, case.dy, dt, p["kh"], 0.0)

        ent = p["we"] / h
        c = c - dt * (c / (p["tauv"] * SEC_H) + ent * (c - C_BG))
        decay = dt * (1.0 / (p["taup"] * SEC_H) + ent)
        n = n * (1.0 - decay) + dt * ent * nbg
        if tags:
            tg = tg * (1.0 - decay)
            ag = ag * (1.0 - decay)

        g = growth(e[1:-1, None, None], T[None], c[None], ca[None],
                   p["betaorg"], p["heff"])
        n = size_advect(n, widths, g, dt_h)
        if tags:
            ag = size_advect(ag, widths, g, dt_h)
            for r in range(3):
                tg[r] = size_advect(tg[r], widths, g, dt_h)

        np.maximum(c, 0.0, out=c)
        np.maximum(n, 0.0, out=n)
        if tags:
            np.maximum(tg, 0.0, out=tg)
            np.maximum(ag, 0.0, out=ag)

        if (k + 1) % steps_per_out == 0:
            out_c.append(c.copy())
            out_n.append(n.copy())
            if tags:
                out_tg.append(tg.copy())
                out_ag.append(ag.copy())

    res = dict(c=np.array(out_c), n=np.array(out_n))
    if tags:
        res["tg"] = np.array(out_tg)
        res["ag"] = np.array(out_ag)
    return res


def event_statistic(n_snaps):
    """Domain mean of the grown-channel number over the late records."""
    return float(n_snaps[Q_FIRST_RECORD:, GROWN_FIRST_BIN:].sum(axis=1).mean())
