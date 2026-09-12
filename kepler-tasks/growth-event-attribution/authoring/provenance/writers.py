"""NetCDF and CSV writers for the published dataset."""
import csv
import hashlib

import numpy as np
from netCDF4 import Dataset

import scenarios as SC
import simulator as SM
import instruments as IN

TIME_UNITS = "seconds since 2026-07-03T00:00:00Z"
Z = dict(zlib=True, complevel=6, shuffle=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_meteorology(path, sc):
    with Dataset(path, "w", format="NETCDF4") as d:
        d.createDimension("time", len(sc.times))
        d.createDimension("y", sc.ny); d.createDimension("x", sc.nx)
        d.createDimension("y_corner", sc.ny + 1); d.createDimension("x_corner", sc.nx + 1)
        v = d.createVariable("time", "f8", ("time",)); v[:] = sc.times; v.units = TIME_UNITS
        v.long_name = "record time; every field is linear in time between records"
        v = d.createVariable("x", "f8", ("x",)); v[:] = sc.x; v.units = "m"
        v = d.createVariable("y", "f8", ("y",)); v[:] = sc.y; v.units = "m"
        v = d.createVariable("x_corner", "f8", ("x_corner",))
        v[:] = np.concatenate([[sc.x[0] - sc.dx / 2], sc.x + sc.dx / 2]); v.units = "m"
        v = d.createVariable("y_corner", "f8", ("y_corner",))
        v[:] = np.concatenate([[sc.y[0] - sc.dy / 2], sc.y + sc.dy / 2]); v.units = "m"
        v = d.createVariable("streamfunction", "f4", ("time", "y_corner", "x_corner"), **Z)
        v[:] = sc.psi; v.units = "m2 s-1"
        v.long_name = "corner streamfunction; u = -d/dy, v = +d/dx; authoritative"
        uf = -(sc.psi[:, 1:, :] - sc.psi[:, :-1, :]) / sc.dy
        vf = (sc.psi[:, :, 1:] - sc.psi[:, :, :-1]) / sc.dx
        v = d.createVariable("u_centre", "f4", ("time", "y", "x"), **Z)
        v[:] = 0.5 * (uf[:, :, 1:] + uf[:, :, :-1]); v.units = "m s-1"
        v.long_name = "eastward wind at cell centres, diagnostic only"
        v = d.createVariable("v_centre", "f4", ("time", "y", "x"), **Z)
        v[:] = 0.5 * (vf[:, 1:, :] + vf[:, :-1, :]); v.units = "m s-1"
        v.long_name = "northward wind at cell centres, diagnostic only"
        v = d.createVariable("temperature", "f4", ("time", "y", "x"), **Z)
        v[:] = sc.T; v.units = "K"
        v = d.createVariable("mixed_layer_depth", "f4", ("time", "y", "x"), **Z)
        v[:] = sc.h; v.units = "m"
        d.episode = sc.id; d.event_day = int(sc.event)
        d.domain = "280 km by 200 km, projected metres, one well-mixed slab"


def write_sources(path, sc):
    with Dataset(path, "w", format="NETCDF4") as d:
        d.createDimension("time", len(sc.times))
        d.createDimension("y", sc.ny); d.createDimension("x", sc.nx)
        d.createDimension("region", 3)
        v = d.createVariable("time", "f8", ("time",)); v[:] = sc.times; v.units = TIME_UNITS
        v = d.createVariable("precursor_rate", "f4", ("time", "y", "x"), **Z)
        v[:] = sc.qorg; v.units = "ug m-3 h-1"
        v.long_name = ("published precursor-to-vapour source before the yield "
                       "scale and, on event days, before the vapour anomaly")
        v = d.createVariable("sulfuric_acid", "f4", ("time", "y", "x"), **Z)
        v[:] = sc.c_acid; v.units = "1e6 cm-3"
        v = d.createVariable("region_mask", "i1", ("region", "y", "x"), **Z)
        v[:] = sc.s_maps.astype(np.int8)
        v.long_name = "nucleation source regions A, B, C (disjoint)"
        v = d.createVariable("nucleation_rate", "f4", ("region", "time"))
        v[:] = sc.s_time; v.units = "cm-3 h-1"
        v.long_name = ("published bottom-up nucleation rate inside each region "
                       "before the regional strength and the event anomaly; "
                       "70% enters bin 0, 30% bin 1")
        v = d.createVariable("event_day", "i1", ()); v[...] = int(sc.event)
        d.episode = sc.id; d.event_day = int(sc.event); d.region_names = "A B C"


def write_records(path, obs, station_ids, times, withheld):
    nb = SM.N_BINS
    with Dataset(path, "w", format="NETCDF4") as d:
        d.createDimension("time", len(times)); d.createDimension("station", len(station_ids))
        d.createDimension("channel", nb)
        v = d.createVariable("time", "f8", ("time",)); v[:] = times; v.units = TIME_UNITS
        v = d.createVariable("station_id", str, ("station",))
        for i, s in enumerate(station_ids):
            v[i] = s
        vv = d.createVariable("vapour", "f4", ("station", "time"), fill_value=-999.0, **Z)
        nn = d.createVariable("counts", "f4", ("station", "time", "channel"), fill_value=-999.0, **Z)
        qq = d.createVariable("counts_flag", "i1", ("station", "time", "channel"))
        qv = d.createVariable("vapour_flag", "i1", ("station", "time"))
        for i, s in enumerate(station_ids):
            if s in withheld:
                vv[i, :] = -999.0; nn[i, :, :] = -999.0; qq[i, :, :] = -1; qv[i, :] = -1
                continue
            o = obs[s]
            vv[i, :] = o["vapour"]; nn[i, :, :] = o["counts"]; qq[i, :, :] = o["qc"]; qv[i, :] = 1
        vv.units = "ug m-3"; nn.units = "cm-3"
        nn.long_name = "reported counts per channel after the station operator"
        qq.long_name = "1 usable, 0 below the detection limit, -1 withheld"
        qv.long_name = "1 usable, -1 withheld"
        d.detection_limit_cm3 = IN.DETECT_LIMIT
        d.withheld_stations = " ".join(sorted(withheld)) or "none"


def write_instruments(path, station_ids):
    nb = SM.N_BINS
    with Dataset(path, "w", format="NETCDF4") as d:
        d.createDimension("station", len(station_ids))
        d.createDimension("channel", nb); d.createDimension("bin", nb)
        v = d.createVariable("station_id", str, ("station",))
        for i, s in enumerate(station_ids):
            v[i] = s
        op = d.createVariable("operator", "f8", ("station", "channel", "bin"))
        ef = d.createVariable("efficiency", "f8", ("station", "channel"))
        br = d.createVariable("broadening", "f8", ("station", "channel", "bin"))
        for i, s in enumerate(station_ids):
            ins = IN.INSTRUMENTS[s]
            op[i] = IN.operator(s); ef[i] = IN.efficiency(ins["d50"], ins["k"])
            br[i] = IN.broadening(ins["side"])
        op.long_name = "reported[channel] = sum_bin operator[channel, bin] * n_true[bin]"


def write_error_model(path, station_ids):
    nb, nt = SM.N_BINS, IN.N_T
    with Dataset(path, "w", format="NETCDF4") as d:
        d.createDimension("station", len(station_ids))
        d.createDimension("time", nt); d.createDimension("time2", nt)
        d.createDimension("tc", nt * nb); d.createDimension("tc2", nt * nb)
        v = d.createVariable("station_id", str, ("station",))
        for i, s in enumerate(station_ids):
            v[i] = s
        cv = d.createVariable("vapour_cov", "f8", ("station", "time", "time2"), **Z)
        cn = d.createVariable("counts_cov", "f8", ("station", "tc", "tc2"), **Z)
        for i, s in enumerate(station_ids):
            cv[i] = IN.vapour_block(s); cn[i] = IN.counts_block(s)
        cv.long_name = "covariance of the log vapour error over one deployment (49 times)"
        cn.long_name = ("covariance of the log count error over one deployment, "
                        "flattened as time_index * 12 + channel")
        d.structure = ("white + AR(1) in time + a bias shared by the whole "
                       "deployment (for counts across every channel and time); "
                       "independent between deployments and between vapour and counts")
        for k, val in IN.ERR_VAPOUR.items():
            setattr(d, "vapour_" + k, val)
        for k, val in IN.ERR_COUNTS.items():
            setattr(d, "counts_" + k, val)


def write_network(path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["station_id", "x_m", "y_m", "role"])
        for s in SC.STATIONS:
            w.writerow([s["id"], f"{s['x']:.1f}", f"{s['y']:.1f}",
                        "withheld" if s["id"].startswith("W") else "observed in E01-E06"])


def write_bins(path):
    e = SM.bin_edges(); c = SM.bin_centres()
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bin", "lower_nm", "upper_nm", "centre_nm"])
        for j in range(SM.N_BINS):
            w.writerow([j, f"{e[j]:.6f}", f"{e[j + 1]:.6f}", f"{c[j]:.6f}"])
