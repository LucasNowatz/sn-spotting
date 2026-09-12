#!/usr/bin/env python3
"""Forward model of the reference solution, written from specification.md.

Runs on the public 4 km grid with a 60 s step.  Face winds are the discrete
curl of the published corner streamfunction; horizontal transport is an
unsplit limited flux-form step plus explicit diffusion; growth is a
conservative limited flux through diameter space.  Parameters are physical
values in a dict (see PHYS); `from_theta` converts a log vector.
"""

import os

import numpy as np
from netCDF4 import Dataset

R_GAS, T_REF, G_ACID, D_KELVIN = 8.314462618, 298.15, 0.05, 1.0
SEC_H = 3600.0
N_BINS, D_MIN, D_MAX = 12, 1.5, 15.0
C_BG, N_BG = 0.06, 0.8
GROWN_FIRST_BIN, Q_FIRST_RECORD = 5, 30
PHYS = ["kh", "we", "tauv", "yorg", "betaorg", "heff", "taup",
        "sA", "sB", "sC", "s_ev", "q_ev"]


def from_theta(theta):
    return {k: float(np.exp(v)) for k, v in zip(PHYS, np.asarray(theta))}


def edges():
    return D_MIN * (D_MAX / D_MIN) ** (np.arange(N_BINS + 1) / N_BINS)


def centres():
    e = edges()
    return np.sqrt(e[:-1] * e[1:])


# ---------------------------------------------------------------- numerics
def _lim(r):
    return np.maximum(0.0, np.minimum(np.minimum(2.0 * r, (1.0 + 2.0 * r) / 3.0), 2.0))


def _face(q, axis, pos):
    tiny = 1e-300
    n = q.shape[axis]
    sl = (lambda a, b: q[..., a:b]) if axis == -1 else (lambda a, b: q[..., a:b, :])
    first = sl(0, 1); last = sl(n - 1, n)
    a = np.concatenate((first, sl(0, n - 2)), axis=axis)
    b = sl(0, n - 1); c = sl(1, n)
    d = np.concatenate((sl(2, n), last), axis=axis)
    fwd = c - b
    up = b + 0.5 * _lim((b - a) / np.where(np.abs(fwd) < tiny, tiny, fwd)) * fwd
    back = b - c
    dn = c + 0.5 * _lim((c - d) / np.where(np.abs(back) < tiny, tiny, back)) * back
    return np.where(pos, up, dn)


def step_transport(q, uf, vf, dx, dy, dt, kh, bg_value):
    bg = np.broadcast_to(np.asarray(bg_value, float), q.shape)
    fx = np.empty(q.shape[:-1] + (q.shape[-1] + 1,))
    fx[..., 1:-1] = uf[..., 1:-1] * _face(q, -1, uf[..., 1:-1] >= 0.0)
    fx[..., 0] = uf[..., 0] * np.where(uf[..., 0] >= 0.0, bg[..., 0], q[..., 0])
    fx[..., -1] = uf[..., -1] * np.where(uf[..., -1] >= 0.0, q[..., -1], bg[..., -1])
    fy = np.empty(q.shape[:-2] + (q.shape[-2] + 1, q.shape[-1]))
    fy[..., 1:-1, :] = vf[..., 1:-1, :] * _face(q, -2, vf[..., 1:-1, :] >= 0.0)
    fy[..., 0, :] = vf[..., 0, :] * np.where(vf[..., 0, :] >= 0.0, bg[..., 0, :], q[..., 0, :])
    fy[..., -1, :] = vf[..., -1, :] * np.where(vf[..., -1, :] >= 0.0, q[..., -1, :], bg[..., -1, :])
    out = q - dt / dx * (fx[..., 1:] - fx[..., :-1]) - dt / dy * (fy[..., 1:, :] - fy[..., :-1, :])
    pad = np.empty(q.shape[:-2] + (q.shape[-2] + 2, q.shape[-1] + 2))
    pad[..., 1:-1, 1:-1] = out
    pad[..., 0, 1:-1] = bg[..., 0, :]; pad[..., -1, 1:-1] = bg[..., -1, :]
    pad[..., 1:-1, 0] = bg[..., :, 0]; pad[..., 1:-1, -1] = bg[..., :, -1]
    pad[..., 0, 0] = bg[..., 0, 0]; pad[..., 0, -1] = bg[..., 0, -1]
    pad[..., -1, 0] = bg[..., -1, 0]; pad[..., -1, -1] = bg[..., -1, -1]
    lap = ((pad[..., 1:-1, 2:] - 2.0 * out + pad[..., 1:-1, :-2]) / dx ** 2
           + (pad[..., 2:, 1:-1] - 2.0 * out + pad[..., :-2, 1:-1]) / dy ** 2)
    return out + dt * kh * lap


def growth(D, T, c, acid, betaorg, heff):
    fT = np.exp(-(heff * 1.0e3) / R_GAS * (1.0 / T - 1.0 / T_REF))
    fK = np.exp(-D_KELVIN / D * (T_REF / T))
    return G_ACID * acid + betaorg * c * fT * fK


def step_growth(field, widths, g_edge, dt_h):
    tiny = 1e-300
    f = field / widths
    lo = np.concatenate((np.zeros_like(f[:1]), f[:-1]), axis=0)
    hi = np.concatenate((f[1:], np.zeros_like(f[-1:])), axis=0)
    fwd = hi - f
    face = f + 0.5 * _lim((f - lo) / np.where(np.abs(fwd) < tiny, tiny, fwd)) * fwd
    flux = g_edge * face[:-1]
    out = field.copy()
    out[:-1] -= dt_h * flux
    out[1:] += dt_h * flux
    return out


# ------------------------------------------------------------------ inputs
_CACHE = {}


def episode(data_dir, eid):
    key = (data_dir, eid)
    if key not in _CACHE:
        _CACHE[key] = Episode(os.path.join(data_dir, "episodes", eid))
    return _CACHE[key]


class Episode:
    def __init__(self, d):
        with Dataset(os.path.join(d, "meteorology.nc")) as f:
            self.time = np.array(f["time"][:], float)
            self.x = np.array(f["x"][:], float); self.y = np.array(f["y"][:], float)
            self.psi = np.array(f["streamfunction"][:], float)
            self.T = np.array(f["temperature"][:], float)
            self.h = np.array(f["mixed_layer_depth"][:], float)
        with Dataset(os.path.join(d, "sources.nc")) as f:
            self.q = np.array(f["precursor_rate"][:], float)
            self.acid = np.array(f["sulfuric_acid"][:], float)
            self.masks = np.array(f["region_mask"][:], float)
            self.rate = np.array(f["nucleation_rate"][:], float)
            self.event = bool(int(f["event_day"][...]))
        self.dx = float(self.x[1] - self.x[0]); self.dy = float(self.y[1] - self.y[0])
        self.ny, self.nx = self.T.shape[1:]
        self.n_out = len(self.time) - 1
        self.duration = float(self.time[-1] - self.time[0])

    def at(self, t):
        i = int(np.searchsorted(self.time, self.time[0] + t, side="right")) - 1
        i = min(max(i, 0), len(self.time) - 2)
        w = (self.time[0] + t - self.time[i]) / (self.time[i + 1] - self.time[i])
        w = min(max(w, 0.0), 1.0)
        f = lambda a: a[i] * (1.0 - w) + a[i + 1] * w
        psi = f(self.psi)
        uf = -(psi[1:, :] - psi[:-1, :]) / self.dy
        vf = (psi[:, 1:] - psi[:, :-1]) / self.dx
        return uf, vf, f(self.T), f(self.h), f(self.q), f(self.acid), \
            self.rate[:, i] * (1.0 - w) + self.rate[:, i + 1] * w


def _factors(ep, p):
    s = np.array([p["sA"], p["sB"], p["sC"]])
    return (s * p["s_ev"], p["q_ev"]) if ep.event else (s, 1.0)


def run_vapour(ep, p, dt=60.0):
    """Vapour only; independent of the particle parameters."""
    per = int(round(ep.duration / dt / ep.n_out))
    _, qm = _factors(ep, p)
    c = np.full((ep.ny, ep.nx), C_BG)
    out = [c.copy()]
    for k in range(ep.n_out * per):
        uf, vf, T, h, q, acid, rate = ep.at(k * dt)
        c = c + dt / SEC_H * p["yorg"] * qm * q
        c = step_transport(c, uf, vf, ep.dx, ep.dy, dt, p["kh"], C_BG)
        c = c - dt * (c / (p["tauv"] * SEC_H) + p["we"] / h * (c - C_BG))
        np.maximum(c, 0.0, out=c)
        if (k + 1) % per == 0:
            out.append(c.copy())
    return np.array(out)


def run(ep, p, dt=60.0, tags=False):
    """Vapour and particles; with tags, also the number per source region."""
    per = int(round(ep.duration / dt / ep.n_out))
    e = edges(); widths = np.diff(e)[:, None, None]
    nbg = np.full((N_BINS, 1, 1), N_BG)
    sm, qm = _factors(ep, p)
    dt_h = dt / SEC_H
    c = np.full((ep.ny, ep.nx), C_BG)
    n = np.tile(nbg, (1, ep.ny, ep.nx))
    tg = np.zeros((3, N_BINS, ep.ny, ep.nx)) if tags else None
    oc, on, otg = [c.copy()], [n.copy()], ([tg.copy()] if tags else None)
    for k in range(ep.n_out * per):
        uf, vf, T, h, q, acid, rate = ep.at(k * dt)
        src = ep.masks * (rate * sm)[:, None, None]
        tot = src.sum(axis=0)
        c = c + dt_h * p["yorg"] * qm * q
        n[0] += dt_h * 0.7 * tot; n[1] += dt_h * 0.3 * tot
        if tags:
            tg[:, 0] += dt_h * 0.7 * src; tg[:, 1] += dt_h * 0.3 * src
        c = step_transport(c, uf, vf, ep.dx, ep.dy, dt, p["kh"], C_BG)
        n = step_transport(n, uf, vf, ep.dx, ep.dy, dt, p["kh"], nbg)
        if tags:
            tg = step_transport(tg, uf, vf, ep.dx, ep.dy, dt, p["kh"], 0.0)
        ent = p["we"] / h
        c = c - dt * (c / (p["tauv"] * SEC_H) + ent * (c - C_BG))
        decay = dt * (1.0 / (p["taup"] * SEC_H) + ent)
        n = n * (1.0 - decay) + dt * ent * nbg
        if tags:
            tg = tg * (1.0 - decay)
        g = growth(e[1:-1, None, None], T[None], c[None], acid[None], p["betaorg"], p["heff"])
        n = step_growth(n, widths, g, dt_h)
        if tags:
            for r in range(3):
                tg[r] = step_growth(tg[r], widths, g, dt_h)
        np.maximum(c, 0.0, out=c); np.maximum(n, 0.0, out=n)
        if tags:
            np.maximum(tg, 0.0, out=tg)
        if (k + 1) % per == 0:
            oc.append(c.copy()); on.append(n.copy())
            if tags:
                otg.append(tg.copy())
    res = dict(c=np.array(oc), n=np.array(on))
    if tags:
        res["tagged"] = np.array(otg)
    return res


def event_statistic(n_snaps):
    return float(n_snaps[Q_FIRST_RECORD:, GROWN_FIRST_BIN:].sum(axis=1).mean())


def domain_shares(res):
    tot = res["n"][Q_FIRST_RECORD:, GROWN_FIRST_BIN:].sum(axis=1).mean()
    tagged = res["tagged"][Q_FIRST_RECORD:, :, GROWN_FIRST_BIN:].sum(axis=2)
    return tagged.mean(axis=(0, 2, 3)) / tot
