"""Forward model for the growth-event benchmark (generator side).

Solves, on one well-mixed boundary-layer slab:

  dc/dt + div(u c)     = Kh lap(c) + Yorg*Qev*qorg - c/tauv - (we/h)(c - cbg)
  dn_j/dt + div(u n_j) = Kh lap(n_j) - n_j/taup - (we/h)(n_j - nbg_j)
                         + S_j - d/dD (G n)

with G(D,T,c,CA) = gA*CA + betaorg * c * fT(T;Heff) * fK(D,T), and the
new-particle source S_j built from the published per-region rates scaled by
the unknown regional strengths s_A, s_B, s_C and, on event days only, by the
nucleation anomaly s_ev.  On event days the precursor source is scaled by the
vapour anomaly q_ev.

Alongside the total particle number it carries source-tagged number for the
three new-particle source regions, a background component, and a number-age
moment, so the lineage diagnostics come from tagged transport rather than from
anything the reference solution computes.

Units: c in ug m-3, n_j in cm-3, D in nm, G in nm h-1, t in s.
"""

import numpy as np

import transport as TR

R_GAS = 8.314462618          # J mol-1 K-1
T_REF = 298.15               # K
G_ACID = 0.05                # nm h-1 per 1e6 cm-3 of H2SO4
D_KELVIN = 1.0               # nm, Kelvin-style suppression scale
N_BINS = 12
D_MIN, D_MAX = 1.5, 15.0     # nm
GROWN_FIRST_BIN = 5          # bins 5..11 (D > 3.92 nm) are the "grown" channels
Q_FIRST_RECORD = 24          # records 24..40 (t >= 6 h) define the event statistic

PARAM_NAMES = ["kh", "we", "tauv", "yorg", "betaorg", "heff", "taup",
               "sA", "sB", "sC", "s_ev", "q_ev"]


def bin_edges():
    return D_MIN * (D_MAX / D_MIN) ** (np.arange(N_BINS + 1) / N_BINS)


def bin_centres():
    e = bin_edges()
    return np.sqrt(e[:-1] * e[1:])


def growth_rate(D, T, c, c_acid, betaorg, heff):
    """G in nm h-1.  D may be scalar or shaped like the edge array."""
    fT = np.exp(-(heff * 1.0e3) / R_GAS * (1.0 / T - 1.0 / T_REF))
    fK = np.exp(-D_KELVIN / D * (T_REF / T))
    return G_ACID * c_acid + betaorg * c * fT * fK


def size_flux(field, per_width, g_edge, dt_h):
    """Conservative limited flux through the interior size-space edges.

    `field` is (nbin, ny, nx) bin-integrated content, `per_width` the same
    divided by the bin width, `g_edge` the growth rate at each interior edge.
    Growth is strictly positive, so the upwind side is always the lower bin.
    """
    eps = 1e-300
    f = per_width
    lo = np.empty_like(f)
    lo[0] = 0.0
    lo[1:] = f[:-1]
    hi = np.empty_like(f)
    hi[:-1] = f[1:]
    hi[-1] = 0.0
    dq_up = f - lo
    dq_dn = hi - f
    r = dq_up / np.where(np.abs(dq_dn) < eps, eps, dq_dn)
    face = f + 0.5 * TR._koren(r) * dq_dn
    flux = g_edge * face[:-1]                  # interior edges only
    out = field.copy()
    out[:-1] -= dt_h * flux
    out[1:] += dt_h * flux
    return out, 0.0


class Episode:
    """Meteorology and forcing for one 10 h episode, on the solver grid.

    Winds are held as the corner streamfunction; the face-normal winds used by
    the transport operator are its discrete curl, so the flow the solver sees
    is non-divergent on whatever grid it is evaluated on.
    """

    def __init__(self, nx, ny, dx, dy, times, psi, T, h, qorg, c_acid,
                 s_maps, s_time, event):
        self.nx, self.ny, self.dx, self.dy = nx, ny, dx, dy
        self.times = np.asarray(times)
        self.psi = psi                         # (nrec, ny+1, nx+1)
        self.T, self.h = T, h
        self.qorg, self.c_acid = qorg, c_acid
        self.s_maps = s_maps                   # (3, ny, nx)
        self.s_time = s_time                   # (3, nrec)
        self.event = bool(event)

    def at(self, t):
        """Linear interpolation between forcing records."""
        i = int(np.searchsorted(self.times, t, side="right")) - 1
        i = min(max(i, 0), len(self.times) - 2)
        w = (t - self.times[i]) / (self.times[i + 1] - self.times[i])
        w = min(max(w, 0.0), 1.0)
        lerp = lambda a: a[i] * (1.0 - w) + a[i + 1] * w
        psi = lerp(self.psi)
        uf = -(psi[1:, :] - psi[:-1, :]) / self.dy
        vf = (psi[:, 1:] - psi[:, :-1]) / self.dx
        st = self.s_time[:, i] * (1.0 - w) + self.s_time[:, i + 1] * w
        return (uf, vf, lerp(self.T), lerp(self.h), lerp(self.qorg),
                lerp(self.c_acid), st)


def effective_multipliers(ep, p):
    """Regional source strengths and the precursor factor for this episode."""
    s = np.array([p["sA"], p["sB"], p["sC"]], dtype=float)
    q = 1.0
    if ep.event:
        s = s * p["s_ev"]
        q = p["q_ev"]
    return s, q


def run_episode(ep, p, dt, n_out, out_every, tags=True, cbg=0.06,
                nbg=None):
    """Integrate one episode.  Returns output snapshots.

    p holds the twelve parameters named in PARAM_NAMES, as physical values
    (multipliers for the source strengths and the two event anomalies).
    Output is every `out_every` sub-steps, `n_out` records in total.
    """
    ny, nx = ep.ny, ep.nx
    edges = bin_edges()
    widths = np.diff(edges)[:, None, None]
    if nbg is None:
        nbg = np.zeros(N_BINS)
    nbg_col = nbg[:, None, None]
    s_mult, q_mult = effective_multipliers(ep, p)

    c = np.full((ny, nx), cbg)
    n = np.tile(nbg_col, (1, ny, nx)).astype(np.float64)
    if tags:
        tg = np.zeros((3, N_BINS, ny, nx))
        ag = np.zeros((N_BINS, ny, nx))
    else:
        tg = ag = None

    out = dict(c=[], n=[], age=[], frac=[])
    dt_h = dt / 3600.0
    steps = n_out * out_every

    for k in range(steps + 1):
        if k % out_every == 0:
            out["c"].append(c.copy())
            out["n"].append(n.copy())
            if tags:
                out["age"].append(ag.copy())
                out["frac"].append(tg.copy())
        if k == steps:
            break
        t = k * dt
        uf, vf, T, h, qorg, c_acid, st = ep.at(t)
        src = ep.s_maps * (st * s_mult)[:, None, None]     # (3, ny, nx), cm-3 h-1

        # --- 1. sources and aging ---
        c = c + dt_h * p["yorg"] * q_mult * qorg
        s_tot = src.sum(axis=0)
        inj = np.zeros((N_BINS, ny, nx))
        inj[0] = 0.7 * s_tot
        inj[1] = 0.3 * s_tot
        if tags:
            ag = ag + dt_h * tg.sum(axis=0)
        n = n + dt_h * inj
        if tags:
            for r in range(3):
                tg[r, 0] += dt_h * 0.7 * src[r]
                tg[r, 1] += dt_h * 0.3 * src[r]

        # --- 2. horizontal advection and diffusion ---
        c = TR.advect(c, uf, vf, ep.dx, ep.dy, dt, cbg)
        c = TR.diffuse(c, p["kh"], ep.dx, ep.dy, dt, cbg)
        n = TR.advect(n, uf, vf, ep.dx, ep.dy, dt, nbg_col)
        n = TR.diffuse(n, p["kh"], ep.dx, ep.dy, dt, nbg_col)
        if tags:
            ag = TR.diffuse(TR.advect(ag, uf, vf, ep.dx, ep.dy, dt, 0.0),
                            p["kh"], ep.dx, ep.dy, dt, 0.0)
            tg = TR.diffuse(TR.advect(tg, uf, vf, ep.dx, ep.dy, dt, 0.0),
                            p["kh"], ep.dx, ep.dy, dt, 0.0)

        # --- 3. dilution and first-order loss ---
        ent = p["we"] / h
        c = c - dt * (c / (p["tauv"] * 3600.0) + ent * (c - cbg))
        loss = dt * (1.0 / (p["taup"] * 3600.0) + ent)
        n = n * (1.0 - loss) + dt * ent * nbg_col
        if tags:
            ag = ag * (1.0 - loss)
            tg = tg * (1.0 - loss)

        # --- 4. size-space growth ---
        g_edge = growth_rate(edges[1:-1, None, None], T[None], c[None],
                             c_acid[None], p["betaorg"], p["heff"])
        n, _ = size_flux(n, n / widths, g_edge, dt_h)
        if tags:
            ag, _ = size_flux(ag, ag / widths, g_edge, dt_h)
            for r in range(3):
                tg[r], _ = size_flux(tg[r], tg[r] / widths, g_edge, dt_h)

        np.maximum(c, 0.0, out=c)
        np.maximum(n, 0.0, out=n)
        if tags:
            np.maximum(ag, 0.0, out=ag)
            np.maximum(tg, 0.0, out=tg)

    for k in out:
        out[k] = np.array(out[k]) if out[k] else None
    return out


def grown_number_statistic(n_snaps):
    """The event statistic for one episode from (time, bin, y, x) snapshots.

    Domain mean of the number in the grown channels (bins GROWN_FIRST_BIN and
    above), averaged over the records from Q_FIRST_RECORD to the end.  A
    domain-integrated statistic is insensitive to the grid a solver uses.
    """
    g = n_snaps[Q_FIRST_RECORD:, GROWN_FIRST_BIN:, :, :].sum(axis=1)
    return float(g.mean())
