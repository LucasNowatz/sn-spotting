"""NetCDF and CSV writers for the published observing system."""
import csv
import hashlib
import os

import numpy as np
from netCDF4 import Dataset

import episodes as E
import observe as O
import truth_model as M

TIME_UNITS = "seconds since 2026-06-15T00:00:00Z"
Z = dict(zlib=True, complevel=6, shuffle=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_met(path, ep):
    with Dataset(path, "w", format="NETCDF4") as d:
        d.createDimension("time", len(ep.times))
        d.createDimension("y", ep.ny)
        d.createDimension("x", ep.nx)
        d.createDimension("y_corner", ep.ny + 1)
        d.createDimension("x_corner", ep.nx + 1)
        v = d.createVariable("time", "f8", ("time",))
        v[:] = ep.times; v.units = TIME_UNITS
        v.long_name = ("forcing record time; all forcing fields vary linearly "
                       "in time between consecutive records")
        v = d.createVariable("x", "f8", ("x",)); v[:] = ep.x; v.units = "m"
        v.long_name = "cell centre easting"
        v = d.createVariable("y", "f8", ("y",)); v[:] = ep.y; v.units = "m"
        v.long_name = "cell centre northing"
        v = d.createVariable("x_corner", "f8", ("x_corner",))
        v[:] = np.concatenate([[ep.x[0] - ep.dx / 2], ep.x + ep.dx / 2])
        v.units = "m"
        v = d.createVariable("y_corner", "f8", ("y_corner",))
        v[:] = np.concatenate([[ep.y[0] - ep.dy / 2], ep.y + ep.dy / 2])
        v.units = "m"
        v = d.createVariable("psi", "f4", ("time", "y_corner", "x_corner"), **Z)
        v[:] = ep.psi; v.units = "m2 s-1"
        v.long_name = ("horizontal streamfunction on the cell corners; the "
                       "flow is non-divergent, u = -dpsi/dy and v = dpsi/dx")
        uc = 0.5 * (-(ep.psi[:, 1:, :] - ep.psi[:, :-1, :]) / ep.dy)[:, :, 1:] \
            + 0.5 * (-(ep.psi[:, 1:, :] - ep.psi[:, :-1, :]) / ep.dy)[:, :, :-1]
        vc = 0.5 * ((ep.psi[:, :, 1:] - ep.psi[:, :, :-1]) / ep.dx)[:, 1:, :] \
            + 0.5 * ((ep.psi[:, :, 1:] - ep.psi[:, :, :-1]) / ep.dx)[:, :-1, :]
        v = d.createVariable("u", "f4", ("time", "y", "x"), **Z)
        v[:] = uc; v.units = "m s-1"
        v.long_name = ("eastward wind at cell centres, the average of the two "
                       "adjacent face values; diagnostic only, the "
                       "streamfunction is authoritative")
        v = d.createVariable("v", "f4", ("time", "y", "x"), **Z)
        v[:] = vc; v.units = "m s-1"
        v.long_name = "northward wind at cell centres, diagnostic only"
        v = d.createVariable("T", "f4", ("time", "y", "x"), **Z)
        v[:] = ep.T; v.units = "K"; v.long_name = "boundary-layer temperature"
        v = d.createVariable("pbl_height", "f4", ("time", "y", "x"), **Z)
        v[:] = ep.h; v.units = "m"
        v.long_name = "well-mixed boundary-layer depth"
        d.coordinate_system = "projected Cartesian metres, x east, y north"
        d.domain = "300 km by 240 km, one well-mixed boundary-layer slab"
        d.episode = ep.id
        d.event_day = int(ep.event)


def write_forcing(path, ep):
    with Dataset(path, "w", format="NETCDF4") as d:
        d.createDimension("time", len(ep.times))
        d.createDimension("y", ep.ny)
        d.createDimension("x", ep.nx)
        d.createDimension("region", 3)
        v = d.createVariable("time", "f8", ("time",))
        v[:] = ep.times; v.units = TIME_UNITS
        v = d.createVariable("q_org", "f4", ("time", "y", "x"), **Z)
        v[:] = ep.qorg; v.units = "ug m-3 h-1"
        v.long_name = ("published precursor-to-condensable-vapour source rate, "
                       "before the yield scale Y_org and, on event days, "
                       "before the vapour anomaly q_event")
        v = d.createVariable("c_h2so4", "f4", ("time", "y", "x"), **Z)
        v[:] = ep.c_acid; v.units = "1e6 cm-3"
        v.long_name = "known sulfuric acid concentration"
        v = d.createVariable("npf_region_map", "i1", ("region", "y", "x"), **Z)
        v[:] = ep.s_maps.astype(np.int8)
        v.long_name = ("new-particle source region masks A, B, C; the three "
                       "regions are disjoint")
        v = d.createVariable("npf_rate", "f4", ("region", "time"))
        v[:] = ep.s_time; v.units = "cm-3 h-1"
        v.long_name = ("published bottom-up new-particle production rate "
                       "inside each region mask, before the unknown regional "
                       "strength and, on event days, the nucleation anomaly; "
                       "70 per cent enters diameter bin 0 and 30 per cent bin 1")
        v = d.createVariable("event_day", "i1", ())
        v[...] = int(ep.event)
        v.long_name = ("1 if the nucleation anomaly s_event and the vapour "
                       "anomaly q_event apply throughout this episode, else 0")
        d.region_names = "A B C"
        d.episode = ep.id
        d.event_day = int(ep.event)


def write_observations(path, obs, station_ids, times, withheld):
    """Station observations.  Withheld entries carry a fill value and qc = -1."""
    nb = M.N_BINS
    with Dataset(path, "w", format="NETCDF4") as d:
        d.createDimension("time", len(times))
        d.createDimension("station", len(station_ids))
        d.createDimension("diameter_bin", nb)
        v = d.createVariable("time", "f8", ("time",))
        v[:] = times; v.units = TIME_UNITS
        v = d.createVariable("station_id", str, ("station",))
        for i, s in enumerate(station_ids):
            v[i] = s
        vv = d.createVariable("vapour", "f4", ("station", "time"),
                              fill_value=-999.0, **Z)
        nn = d.createVariable("counts", "f4",
                              ("station", "time", "diameter_bin"),
                              fill_value=-999.0, **Z)
        qq = d.createVariable("qc", "i1", ("station", "time", "diameter_bin"))
        qv = d.createVariable("qc_vapour", "i1", ("station", "time"))
        for i, s in enumerate(station_ids):
            if s in withheld:
                vv[i, :] = -999.0; nn[i, :, :] = -999.0
                qq[i, :, :] = -1; qv[i, :] = -1
                continue
            o = obs[s]
            vv[i, :] = o["vapour"]
            nn[i, :, :] = o["counts"]
            qq[i, :, :] = o["qc"]
            qv[i, :] = 1
        vv.units = "ug m-3"
        vv.long_name = "observed condensable organic vapour concentration"
        nn.units = "cm-3"
        nn.long_name = ("observed particle number per diameter channel, after "
                        "the station's sizing operator")
        qq.long_name = ("1 usable, 0 below the detection limit, -1 withheld")
        qv.long_name = "1 usable, -1 withheld"
        d.detection_limit_cm3 = O.DETECT_LIMIT
        d.withheld_stations = " ".join(sorted(withheld)) or "none"
        d.error_model = ("log-space Gaussian errors, correlated in time and "
                         "with a per-deployment bias; the covariance blocks "
                         "are in error_covariance.nc and depend only on the "
                         "station")


def write_operators(path, station_ids):
    nb = M.N_BINS
    with Dataset(path, "w", format="NETCDF4") as d:
        d.createDimension("station", len(station_ids))
        d.createDimension("channel", nb)
        d.createDimension("diameter_bin", nb)
        v = d.createVariable("station_id", str, ("station",))
        for i, s in enumerate(station_ids):
            v[i] = s
        op = d.createVariable("sizing_operator", "f8",
                              ("station", "channel", "diameter_bin"))
        eff = d.createVariable("counting_efficiency", "f8",
                               ("station", "channel"))
        br = d.createVariable("broadening", "f8",
                              ("station", "channel", "diameter_bin"))
        for i, s in enumerate(station_ids):
            ins = O.INSTRUMENTS[s]
            op[i] = O.sizing_operator(s)
            eff[i] = O.efficiency(ins["d50"], ins["k"])
            br[i] = O.broadening(ins["side"])
        op.long_name = ("reported_counts[channel] = sum_bin "
                        "sizing_operator[channel, bin] * n_true[bin]; equals "
                        "diag(counting_efficiency) @ broadening")
        eff.long_name = "size-dependent counting efficiency per channel"
        br.long_name = "row-stochastic broadening into neighbouring channels"
        d.note = ("each station has its own operator; the lineage "
                  "diagnostics are defined before the operator")


def write_error_covariance(path, station_ids):
    nb, nt = M.N_BINS, O.N_T
    with Dataset(path, "w", format="NETCDF4") as d:
        d.createDimension("station", len(station_ids))
        d.createDimension("time", nt)
        d.createDimension("time2", nt)
        d.createDimension("tc", nt * nb)
        d.createDimension("tc2", nt * nb)
        v = d.createVariable("station_id", str, ("station",))
        for i, s in enumerate(station_ids):
            v[i] = s
        cv = d.createVariable("vapour_log_cov", "f8",
                              ("station", "time", "time2"), **Z)
        cn = d.createVariable("counts_log_cov", "f8",
                              ("station", "tc", "tc2"), **Z)
        for i, s in enumerate(station_ids):
            cv[i] = O.vapour_block(s)
            cn[i] = O.counts_block(s)
        cv.long_name = ("covariance of the natural-log vapour error over the "
                        "41 record times of one station-episode deployment")
        cn.long_name = ("covariance of the natural-log count error over the "
                        "41 times x 12 channels of one deployment, flattened "
                        "as index = time_index * 12 + channel")
        d.structure = ("vapour: sw^2 I + ss^2 AR1(rho) + sb^2 11^T over time, "
                       "times the station's vapour error scale squared. "
                       "counts: (sw^2 I + ss^2 AR1(rho)) over time, Kronecker "
                       "the identity over channels, plus sb^2 11^T over every "
                       "time and channel, times the station's count error "
                       "scale squared. Errors are independent between "
                       "deployments and between vapour and counts.")
        d.vapour_sw = O.ERR_VAPOUR["sw"]; d.vapour_ss = O.ERR_VAPOUR["ss"]
        d.vapour_rho = O.ERR_VAPOUR["rho"]; d.vapour_sb = O.ERR_VAPOUR["sb"]
        d.counts_sw = O.ERR_COUNTS["sw"]; d.counts_ss = O.ERR_COUNTS["ss"]
        d.counts_rho = O.ERR_COUNTS["rho"]; d.counts_sb = O.ERR_COUNTS["sb"]
        es = d.createVariable("vapour_error_scale", "f8", ("station",))
        es[:] = [O.INSTRUMENTS[s]["ev"] for s in station_ids]
        es = d.createVariable("counts_error_scale", "f8", ("station",))
        es[:] = [O.INSTRUMENTS[s]["en"] for s in station_ids]


def write_stations(path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["station_id", "x_m", "y_m", "observed_in_E01_E06"])
        for s in E.STATIONS:
            w.writerow([s["id"], f"{s['x']:.1f}", f"{s['y']:.1f}",
                        "no" if s["id"].startswith("W") else "yes"])


def write_size_bins(path):
    e = M.bin_edges()
    c = M.bin_centres()
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["diameter_bin", "edge_lower_nm", "edge_upper_nm",
                    "centre_nm"])
        for j in range(M.N_BINS):
            w.writerow([j, f"{e[j]:.6f}", f"{e[j+1]:.6f}", f"{c[j]:.6f}"])
