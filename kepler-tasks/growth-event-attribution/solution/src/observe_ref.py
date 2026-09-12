"""Station sampling, sizing operators and error blocks for the reference."""
import csv
import os

import numpy as np
from netCDF4 import Dataset

_STATIONS = {}
_OPS = {}
_COV_V = {}
_COV_N = {}
_DIR = None


def ensure_geometry(data_dir):
    if _STATIONS and _OPS and _DIR == data_dir:
        return
    load_geometry(data_dir)


def load_geometry(data_dir):
    global _DIR
    _DIR = data_dir
    _STATIONS.clear(); _OPS.clear(); _COV_V.clear(); _COV_N.clear()
    with open(os.path.join(data_dir, "stations.csv")) as f:
        for row in csv.DictReader(f):
            _STATIONS[row["station_id"]] = (float(row["x_m"]), float(row["y_m"]))
    with Dataset(os.path.join(data_dir, "sizing_operators.nc")) as d:
        ids = [str(s) for s in d["station_id"][:]]
        op = np.array(d["sizing_operator"][:], dtype=float)
        for i, s in enumerate(ids):
            _OPS[s] = op[i]
    with Dataset(os.path.join(data_dir, "error_covariance.nc")) as d:
        ids = [str(s) for s in d["station_id"][:]]
        cv = np.array(d["vapour_log_cov"][:], dtype=float)
        cn = np.array(d["counts_log_cov"][:], dtype=float)
        for i, s in enumerate(ids):
            _COV_V[s] = cv[i]
            _COV_N[s] = cn[i]


def operator(sid):
    return _OPS[sid]


def vapour_cov(sid):
    return _COV_V[sid]


def counts_cov(sid):
    return _COV_N[sid]


def station_ids():
    return list(_STATIONS)


def _weights(case, sid):
    sx, sy = _STATIONS[sid]
    fi = (sx - case.x[0]) / case.dx
    fj = (sy - case.y[0]) / case.dy
    i0 = int(np.clip(np.floor(fi), 0, case.nx - 2))
    j0 = int(np.clip(np.floor(fj), 0, case.ny - 2))
    a = float(np.clip(fi - i0, 0.0, 1.0))
    b = float(np.clip(fj - j0, 0.0, 1.0))
    w = np.array([[(1 - b) * (1 - a), (1 - b) * a],
                  [b * (1 - a), b * a]])
    return j0, i0, w


def sample_series(snaps, case, sid):
    """Bilinear sample of a (time, y, x) field at one station."""
    j0, i0, w = _weights(case, sid)
    return (snaps[:, j0:j0 + 2, i0:i0 + 2] * w[None]).sum(axis=(1, 2))


def sample_bins(snaps, case, sid):
    """Bilinear sample of a (time, bin, y, x) field at one station."""
    j0, i0, w = _weights(case, sid)
    return (snaps[:, :, j0:j0 + 2, i0:i0 + 2] * w[None, None]).sum(axis=(2, 3))


def reported_counts(snaps_n, case, sid):
    """Post-operator (time, channel) counts at a station."""
    return sample_bins(snaps_n, case, sid) @ operator(sid).T
