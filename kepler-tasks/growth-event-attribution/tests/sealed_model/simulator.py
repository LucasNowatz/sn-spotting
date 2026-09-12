"""Truth-side simulator for the event-attribution benchmark.

One well-mixed boundary-layer slab carrying a condensable vapour c and
particle number n_j in twelve logarithmic diameter bins:

  dc/dt + div(u c)     = Kh lap(c) + Yorg*Qev*qorg - c/tauv - (we/h)(c - cbg)
  dn_j/dt + div(u n_j) = Kh lap(n_j) - n_j/taup - (we/h)(n_j - nbg_j)
                         + S_j - d/dD (G n)

G = gA*CA + betaorg * c * fT(T; Heff) * fK(D, T).  The nucleation source is
the published per-region rate times the unknown regional strength and, on
event days, the nucleation anomaly; on event days the precursor source is
also scaled by the vapour anomaly.

Alongside the total number, the simulator carries the number injected in
each source region as tagged tracers, so region-resolved attribution comes
from tagged transport rather than from anything a solution computes.
"""

import numpy as np

import dynamics as DY

R_GAS = 8.314462618
T_REF = 298.15
G_ACID = 0.05                # nm h-1 per 1e6 cm-3
D_KELVIN = 1.0               # nm
N_BINS = 12
D_MIN, D_MAX = 1.5, 15.0
GROWN_FIRST_BIN = 5          # channels 5..11, D > 3.92 nm
Q_FIRST_RECORD = 30          # records 30..48, t >= 5 h
C_BG = 0.06
N_BG = 0.8

PHYS = ["kh", "we", "tauv", "yorg", "betaorg", "heff", "taup",
        "sA", "sB", "sC", "s_ev", "q_ev"]


def bin_edges():
    return D_MIN * (D_MAX / D_MIN) ** (np.arange(N_BINS + 1) / N_BINS)


def bin_centres():
    e = bin_edges()
    return np.sqrt(e[:-1] * e[1:])


def growth_rate(D, T, c, c_acid, betaorg, heff):
    fT = np.exp(-(heff * 1.0e3) / R_GAS * (1.0 / T - 1.0 / T_REF))
    fK = np.exp(-D_KELVIN / D * (T_REF / T))
    return G_ACID * c_acid + betaorg * c * fT * fK


def size_flux(field, widths, g_edge, dt_h):
    """Conservative limited flux through the interior diameter edges."""
    tiny = 1e-300
    f = field / widths
    lo = np.concatenate((np.zeros_like(f[:1]), f[:-1]), axis=0)
    hi = np.concatenate((f[1:], np.zeros_like(f[-1:])), axis=0)
    dn = hi - f
    r = (f - lo) / np.where(np.abs(dn) < tiny, tiny, dn)
    face = f + 0.5 * DY.limiter(r) * dn
    flux = g_edge * face[:-1]
    out = field.copy()
    out[:-1] -= dt_h * flux
    out[1:] += dt_h * flux
    return out


class Scenario:
    """Meteorology and forcing of one episode on the solver grid."""

    def __init__(self, nx, ny, dx, dy, times, psi, T, h, qorg, c_acid,
                 s_maps, s_time, event, ident=""):
        self.nx, self.ny, self.dx, self.dy = nx, ny, dx, dy
        self.times = np.asarray(times)
        self.psi, self.T, self.h = psi, T, h
        self.qorg, self.c_acid = qorg, c_acid
        self.s_maps, self.s_time = s_maps, s_time
        self.event = bool(event)
        self.id = ident

    def at(self, t):
        i = int(np.searchsorted(self.times, t, side="right")) - 1
        i = min(max(i, 0), len(self.times) - 2)
        w = (t - self.times[i]) / (self.times[i + 1] - self.times[i])
        w = min(max(w, 0.0), 1.0)
        lerp = lambda a: a[i] * (1.0 - w) + a[i + 1] * w
        psi = lerp(self.psi)
        uf = -(psi[1:, :] - psi[:-1, :]) / self.dy
        vf = (psi[:, 1:] - psi[:, :-1]) / self.dx
        st = self.s_time[:, i] * (1.0 - w) + self.s_time[:, i + 1] * w
        return uf, vf, lerp(self.T), lerp(self.h), lerp(self.qorg), lerp(self.c_acid), st


def multipliers(sc, p):
    s = np.array([p["sA"], p["sB"], p["sC"]], dtype=float)
    q = 1.0
    if sc.event:
        s = s * p["s_ev"]
        q = p["q_ev"]
    return s, q


def integrate(sc, p, dt, n_out, out_every, tags=False):
    """Advance one episode; return snapshots of c, n and (optionally) the
    tagged number per source region."""
    ny, nx = sc.ny, sc.nx
    edges = bin_edges()
    widths = np.diff(edges)[:, None, None]
    nbg = np.full((N_BINS, 1, 1), N_BG)
    s_mult, q_mult = multipliers(sc, p)
    dt_h = dt / 3600.0

    c = np.full((ny, nx), C_BG)
    n = np.tile(nbg, (1, ny, nx))
    tg = np.zeros((3, N_BINS, ny, nx)) if tags else None
    out = dict(c=[], n=[], tagged=[])
    steps = n_out * out_every
    for k in range(steps + 1):
        if k % out_every == 0:
            out["c"].append(c.copy()); out["n"].append(n.copy())
            if tags:
                out["tagged"].append(tg.copy())
        if k == steps:
            break
        uf, vf, T, h, qorg, c_acid, st = sc.at(k * dt)
        src = sc.s_maps * (st * s_mult)[:, None, None]
        s_tot = src.sum(axis=0)

        c = c + dt_h * p["yorg"] * q_mult * qorg
        inj = np.zeros((N_BINS, ny, nx))
        inj[0] = 0.7 * s_tot; inj[1] = 0.3 * s_tot
        n = n + dt_h * inj
        if tags:
            tg[:, 0] += dt_h * 0.7 * src
            tg[:, 1] += dt_h * 0.3 * src

        c = DY.transport_step(c, uf, vf, sc.dx, sc.dy, dt, p["kh"], C_BG)
        n = DY.transport_step(n, uf, vf, sc.dx, sc.dy, dt, p["kh"], nbg)
        if tags:
            tg = DY.transport_step(tg, uf, vf, sc.dx, sc.dy, dt, p["kh"], 0.0)

        ent = p["we"] / h
        c = c - dt * (c / (p["tauv"] * 3600.0) + ent * (c - C_BG))
        loss = dt * (1.0 / (p["taup"] * 3600.0) + ent)
        n = n * (1.0 - loss) + dt * ent * nbg
        if tags:
            tg = tg * (1.0 - loss)

        g = growth_rate(edges[1:-1, None, None], T[None], c[None], c_acid[None],
                        p["betaorg"], p["heff"])
        n = size_flux(n, widths, g, dt_h)
        if tags:
            for r in range(3):
                tg[r] = size_flux(tg[r], widths, g, dt_h)
        np.maximum(c, 0.0, out=c); np.maximum(n, 0.0, out=n)
        if tags:
            np.maximum(tg, 0.0, out=tg)
    res = dict(c=np.array(out["c"]), n=np.array(out["n"]))
    if tags:
        res["tagged"] = np.array(out["tagged"])
    return res


def grown_number(n_snaps):
    """(time, y, x) number in the grown channels."""
    return n_snaps[:, GROWN_FIRST_BIN:].sum(axis=1)


def event_statistic(n_snaps):
    """Domain mean of the grown-channel number over the late records."""
    return float(grown_number(n_snaps)[Q_FIRST_RECORD:].mean())


def region_shares_domain(res):
    """Share of the late-record domain-mean grown number injected in each
    source region, from the tagged tracers.  The remainder is background."""
    tot = grown_number(res["n"])[Q_FIRST_RECORD:].mean()
    tagged = res["tagged"][Q_FIRST_RECORD:, :, GROWN_FIRST_BIN:].sum(axis=2)  # (t, 3, y, x)
    return tagged.mean(axis=(0, 2, 3)) / tot
