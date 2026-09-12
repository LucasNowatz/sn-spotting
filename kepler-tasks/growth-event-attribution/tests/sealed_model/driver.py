"""Build scenarios on a chosen grid, run them and sample the network."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scenarios as SC
import simulator as SM
import instruments as IN


def build(ep_id, nx, ny):
    x, y, dx, dy, X, Y = SC.grid(nx, ny)
    t = SC.record_times()
    sp = SC.EPISODES[ep_id]
    vf = SC.VAPOUR_FORCING[ep_id]
    psi = np.stack([SC.streamfunction(x, y, dx, dy, tt / 3600.0, sp) for tt in t])
    T = np.stack([SC.temperature(X, Y, tt / 3600.0, sp) for tt in t])
    h = np.stack([SC.pbl_height(X, Y, tt / 3600.0, sp) for tt in t])
    q = np.stack([SC.vapour_source(X, Y, tt / 3600.0, vf) for tt in t])
    ca = np.stack([SC.acid_field(X, Y, tt / 3600.0, vf) for tt in t])
    sc = SM.Scenario(nx, ny, dx, dy, t, psi, T, h, q, ca, SC.source_maps(X, Y),
                     SC.npf_time(ep_id), sp["event"], ep_id)
    sc.x, sc.y = x, y
    return sc


def weights(sc):
    return {s["id"]: IN.bilinear(sc.x, sc.y, s["x"], s["y"]) for s in SC.STATIONS}


def sample_network(res, sc, station_ids=None):
    """Per-station vapour, true and reported counts; tagged number if present."""
    w = weights(sc)
    ids = station_ids or [s["id"] for s in SC.STATIONS]
    out = {}
    for sid in ids:
        jj, ww = w[sid]
        v = IN.sample(res["c"], jj, ww)
        n = IN.sample(res["n"], jj, ww)
        d = dict(vapour=v, n_true=n, counts=n @ IN.operator(sid).T)
        if "tagged" in res:
            d["tagged"] = IN.sample(res["tagged"], jj, ww)     # (t, 3, bin)
        out[sid] = d
    return out


def run(ep_id, p, nx=SC.NX_PUB, ny=SC.NY_PUB, dt=60.0, tags=False):
    sc = build(ep_id, nx, ny)
    n_out = SC.N_OBS - 1
    out_every = int(round(SC.HOURS * 3600.0 / dt / n_out))
    return sc, SM.integrate(sc, p, dt, n_out, out_every, tags=tags)


def station_region_shares(d):
    """Share of the late-record grown number at a station injected in each
    region, from the sampled tagged number (pre-operator)."""
    g = d["n_true"][SM.Q_FIRST_RECORD:, SM.GROWN_FIRST_BIN:].sum(axis=1).mean()
    t = d["tagged"][SM.Q_FIRST_RECORD:, :, SM.GROWN_FIRST_BIN:].sum(axis=2).mean(axis=0)
    return t / g
