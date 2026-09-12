"""Build Episode objects on a chosen grid and sample the station network."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import episodes as E
import truth_model as M
import observe as O

C_BG = 0.06                                    # ug m-3 vapour background
N_BG = np.full(M.N_BINS, 0.8)                  # cm-3 per bin background


def build_episode(ep_id, nx, ny):
    x, y, dx, dy, X, Y = E.grid(nx, ny)
    t = E.rec_times()
    sp = E.EPISODES[ep_id]
    vf = E.VAPOUR_FORCING[ep_id]
    psi = np.stack([E.streamfunction(x, y, dx, dy, tt / 3600.0, sp) for tt in t])
    T = np.stack([E.temperature(X, Y, tt / 3600.0, sp) for tt in t])
    h = np.stack([E.pbl_height(X, Y, tt / 3600.0, sp) for tt in t])
    q = np.stack([E.vapour_source(X, Y, tt / 3600.0, vf) for tt in t])
    ca = np.stack([E.acid_field(X, Y, tt / 3600.0, vf) for tt in t])
    ep = M.Episode(nx, ny, dx, dy, t, psi, T, h, q, ca,
                   E.source_maps(X, Y), E.npf_time(ep_id), sp["event"])
    ep.x, ep.y = x, y
    ep.id = ep_id
    return ep


def station_weights(ep):
    return {s["id"]: O.bilinear_weights(ep.x, ep.y, s["x"], s["y"])
            for s in E.STATIONS}


def sample(out, ep, station_ids=None, tags=False):
    """Sample the model output at the stations.  Returns dict of arrays."""
    w = station_weights(ep)
    ids = station_ids or [s["id"] for s in E.STATIONS]
    nt = out["c"].shape[0]
    res = {}
    for sid in ids:
        jj_ii, ww = w[sid]
        v = np.array([O.sample_field(out["c"][k], jj_ii, ww) for k in range(nt)])
        n = np.array([O.sample_field(out["n"][k], jj_ii, ww) for k in range(nt)])
        d = dict(vapour=v, counts=O.apply_operator(n, sid), n_true=n)
        if tags:
            ag = np.array([O.sample_field(out["age"][k], jj_ii, ww)
                           for k in range(nt)])
            fr = np.array([[O.sample_field(out["frac"][k][r], jj_ii, ww)
                            for r in range(3)] for k in range(nt)])
            d["age_moment"] = ag
            d["tagged"] = fr
        res[sid] = d
    return res


def run(ep_id, p, nx=60, ny=48, dt=60.0, tags=False):
    ep = build_episode(ep_id, nx, ny)
    n_out = E.N_OBS - 1
    out_every = int(round(E.HOURS * 3600.0 / dt / n_out))
    out = M.run_episode(ep, p, dt, n_out, out_every, tags=tags, cbg=C_BG,
                        nbg=N_BG)
    return ep, out
