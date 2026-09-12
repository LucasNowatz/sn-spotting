"""Station sampling, instrument operators and error blocks (reference)."""
import csv
import os

import numpy as np
from netCDF4 import Dataset

_XY, _OP, _CV, _CN = {}, {}, {}, {}
_DIR = None


def load(data_dir):
    global _DIR
    if _DIR == data_dir and _XY:
        return
    _DIR = data_dir
    _XY.clear(); _OP.clear(); _CV.clear(); _CN.clear()
    with open(os.path.join(data_dir, "network.csv")) as f:
        for r in csv.DictReader(f):
            _XY[r["station_id"]] = (float(r["x_m"]), float(r["y_m"]))
    with Dataset(os.path.join(data_dir, "instruments.nc")) as d:
        ids = [str(s) for s in d["station_id"][:]]
        op = np.array(d["operator"][:], float)
        for i, s in enumerate(ids):
            _OP[s] = op[i]
    with Dataset(os.path.join(data_dir, "error_model.nc")) as d:
        ids = [str(s) for s in d["station_id"][:]]
        cv = np.array(d["vapour_cov"][:], float); cn = np.array(d["counts_cov"][:], float)
        for i, s in enumerate(ids):
            _CV[s] = cv[i]; _CN[s] = cn[i]


def stations():
    return list(_XY)


def operator(s):
    return _OP[s]


def vapour_cov(s):
    return _CV[s]


def counts_cov(s):
    return _CN[s]


def _w(ep, s):
    sx, sy = _XY[s]
    fi = (sx - ep.x[0]) / ep.dx; fj = (sy - ep.y[0]) / ep.dy
    i0 = int(np.clip(np.floor(fi), 0, ep.nx - 2)); j0 = int(np.clip(np.floor(fj), 0, ep.ny - 2))
    a = float(np.clip(fi - i0, 0.0, 1.0)); b = float(np.clip(fj - j0, 0.0, 1.0))
    return j0, i0, np.array([[(1 - b) * (1 - a), (1 - b) * a], [b * (1 - a), b * a]])


def at_station(field, ep, s):
    """Bilinear sample of the trailing two axes."""
    j0, i0, w = _w(ep, s)
    return (field[..., j0:j0 + 2, i0:i0 + 2] * w).sum(axis=(-2, -1))


def reported(res_n, ep, s):
    return at_station(res_n, ep, s) @ operator(s).T
