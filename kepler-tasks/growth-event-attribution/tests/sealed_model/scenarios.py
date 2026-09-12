"""Meteorology, sources and the observing network (generator side).

Eight 8-hour episodes on a 280 km by 200 km domain, forcing every 10 min.
Winds are an analytic streamfunction (mean flow, a sheared jet, a travelling
eddy), so the flow is non-divergent to rounding on any grid.  Four episodes
are event days on which the nucleation and vapour anomalies act.
"""

import numpy as np

LX, LY = 280.0e3, 200.0e3
DX_PUB, NX_PUB, NY_PUB = 4000.0, 70, 50
HOURS = 8.0
N_REC = 49                      # forcing records, every 10 min
N_OBS = 49

SOURCE_REGIONS = [
    dict(name="A", x=48.0e3, y=58.0e3, r=24.0e3),
    dict(name="B", x=60.0e3, y=150.0e3, r=21.0e3),
    dict(name="C", x=150.0e3, y=34.0e3, r=20.0e3),
]

STATIONS = [
    dict(id="S1", x=116.0e3, y=92.0e3),
    dict(id="S2", x=178.0e3, y=132.0e3),
    dict(id="S3", x=138.0e3, y=166.0e3),
    dict(id="S4", x=196.0e3, y=62.0e3),
    dict(id="S5", x=92.0e3, y=124.0e3),
    dict(id="S6", x=236.0e3, y=150.0e3),
    dict(id="W1", x=228.0e3, y=98.0e3),
    dict(id="W2", x=158.0e3, y=108.0e3),
]

# spd m s-1, dirdeg the direction the wind blows FROM, rot the rotation over
# the episode in degrees, jet the amplitude of a sheared jet, eddy the eddy
# amplitude, T0/dTdy/dTdt the thermal field, h0/dhdt/hgx the boundary layer
EPISODES = {
    "E01": dict(status="observed", event=False, spd=6.0, dirdeg=232.0, rot=10.0,
                jet=0.6, ekx=1.0, eky=1.0, T0=292.0, dTdy=-2.6, dTdt=1.0,
                h0=1100.0, dhdt=200.0, hgx=150.0,
                note="warm, deep boundary layer, near-steady south-westerly"),
    "E02": dict(status="observed", event=True, spd=6.8, dirdeg=262.0, rot=0.0,
                jet=0.0, ekx=1.0, eky=1.0, T0=281.0, dTdy=-2.2, dTdt=0.6,
                h0=520.0, dhdt=80.0, hgx=60.0,
                note="cool, shallow boundary layer, westerly; event day"),
    "E03": dict(status="observed", event=True, spd=3.4, dirdeg=205.0, rot=25.0,
                jet=0.0, ekx=2.0, eky=1.0, T0=294.5, dTdy=-1.8, dTdt=1.1,
                h0=1350.0, dhdt=260.0, hgx=200.0,
                note="warm, slack flow with a strong eddy; event day"),
    "E04": dict(status="observed", event=False, spd=5.4, dirdeg=245.0, rot=70.0,
                jet=0.0, ekx=1.0, eky=1.0, T0=286.5, dTdy=-2.4, dTdt=0.5,
                h0=760.0, dhdt=140.0, hgx=100.0,
                note="mild, wind backs through 70 degrees"),
    "E05": dict(status="observed", event=False, spd=5.8, dirdeg=270.0, rot=0.0,
                jet=3.0, ekx=1.0, eky=2.0, T0=280.0, dTdy=-3.2, dTdt=0.4,
                h0=600.0, dhdt=100.0, hgx=80.0,
                note="cool, sheared westerly jet across the middle of the domain"),
    "E06": dict(status="observed", event=True, spd=9.6, dirdeg=250.0, rot=-15.0,
                jet=1.2, ekx=1.0, eky=1.0, T0=289.0, dTdy=-2.0, dTdt=0.8,
                h0=900.0, dhdt=160.0, hgx=120.0,
                note="mild, fast transit with a weak jet; event day"),
    "E07": dict(status="withheld", event=False, spd=4.8, dirdeg=228.0, rot=55.0,
                jet=1.8, ekx=1.0, eky=2.0, T0=282.5, dTdy=-2.8, dTdt=0.9,
                h0=560.0, dhdt=280.0, hgx=100.0,
                note="cool, growing boundary layer, turning sheared flow"),
    "E08": dict(status="withheld", event=True, spd=7.2, dirdeg=256.0, rot=20.0,
                jet=2.4, ekx=2.0, eky=1.0, T0=293.5, dTdy=-2.3, dTdt=-1.0,
                h0=1300.0, dhdt=-340.0, hgx=140.0,
                note="warm, collapsing boundary layer, moderate shear; event day"),
}

EVENT_EPISODES = [e for e, d in EPISODES.items() if d["event"]]
EDDY_AMPLITUDE = 0.9            # m s-1, travelling non-divergent eddy

# published bottom-up nucleation rate at the region centre, cm-3 h-1, as
# (peak hour, width h, amplitude) per region A, B, C
NPF_TIME = {
    "E01": [(2.0, 2.2, 5400.0), (3.6, 2.0, 3400.0), (2.8, 2.6, 4300.0)],
    "E02": [(1.6, 2.0, 4600.0), (3.0, 2.4, 5000.0), (4.2, 2.2, 3200.0)],
    "E03": [(2.4, 2.8, 4900.0), (1.8, 1.8, 4000.0), (3.8, 2.6, 4700.0)],
    "E04": [(2.6, 3.0, 5800.0), (4.4, 2.2, 3000.0), (2.2, 2.0, 4100.0)],
    "E05": [(1.8, 2.2, 4200.0), (3.8, 2.8, 5200.0), (3.0, 2.4, 4600.0)],
    "E06": [(1.4, 1.8, 5000.0), (2.6, 2.2, 4500.0), (3.6, 2.4, 3800.0)],
    "E07": [(2.4, 2.6, 4800.0), (4.0, 2.0, 4200.0), (2.0, 2.2, 5000.0)],
    "E08": [(2.2, 2.0, 5300.0), (3.2, 2.6, 3700.0), (4.6, 2.4, 4400.0)],
}

# precursor source amplitude and diurnal shape, acid amplitude and peak hour
VAPOUR_FORCING = {
    "E01": dict(q0=0.64, qkx=1.0, qpk=4.2, qw=2.8, ca0=4.4, capk=4.0),
    "E02": dict(q0=0.46, qkx=1.0, qpk=3.8, qw=2.5, ca0=3.0, capk=3.7),
    "E03": dict(q0=0.72, qkx=2.0, qpk=4.0, qw=3.2, ca0=5.0, capk=4.2),
    "E04": dict(q0=0.56, qkx=2.0, qpk=4.5, qw=3.0, ca0=3.8, capk=4.4),
    "E05": dict(q0=0.50, qkx=2.0, qpk=3.9, qw=2.6, ca0=3.3, capk=3.9),
    "E06": dict(q0=0.60, qkx=1.0, qpk=4.3, qw=2.8, ca0=4.0, capk=4.0),
    "E07": dict(q0=0.53, qkx=2.0, qpk=4.0, qw=2.9, ca0=3.5, capk=4.1),
    "E08": dict(q0=0.68, qkx=1.0, qpk=4.2, qw=2.7, ca0=4.8, capk=3.9),
}

PRECURSOR_PATCHES = [
    dict(x=40.0e3, y=96.0e3, sx=20.0e3, sy=24.0e3, w=1.00),
    dict(x=88.0e3, y=30.0e3, sx=18.0e3, sy=16.0e3, w=0.80),
    dict(x=76.0e3, y=172.0e3, sx=22.0e3, sy=18.0e3, w=0.90),
    dict(x=164.0e3, y=128.0e3, sx=16.0e3, sy=20.0e3, w=0.65),
    dict(x=120.0e3, y=60.0e3, sx=14.0e3, sy=14.0e3, w=0.55),
]


def record_times():
    return np.linspace(0.0, HOURS * 3600.0, N_REC)


def _stream(x_f, y_f, t_h, sp):
    ang = np.deg2rad(sp["dirdeg"] + sp["rot"] * t_h / HOURS)
    u_mean = -sp["spd"] * np.sin(ang)
    v_mean = -sp["spd"] * np.cos(ang)
    X, Y = np.meshgrid(x_f, y_f)
    psi = -u_mean * Y + v_mean * X
    # a jet centred on the domain mid-line: u adds a Gaussian bump in y
    yc = LY / 2.0
    w = 0.22 * LY
    psi = psi - sp["jet"] * w * np.sqrt(np.pi) / 2.0 * _erf((Y - yc) / w)
    kx = 2.0 * np.pi * sp["ekx"] / LX
    ky = 2.0 * np.pi * sp["eky"] / LY
    ph = 2.0 * np.pi * 0.10 * t_h
    psi = psi + (EDDY_AMPLITUDE / ky) * np.sin(kx * X - ph) * np.cos(ky * Y)
    return psi


def _erf(z):
    # Abramowitz-Stegun 7.1.26, adequate for a smooth streamfunction
    s = np.sign(z); a = np.abs(z)
    t = 1.0 / (1.0 + 0.3275911 * a)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * np.exp(-a * a)
    return s * y


def corner_axes(x_c, y_c, dx, dy):
    return (np.concatenate([[x_c[0] - dx / 2], x_c + dx / 2]),
            np.concatenate([[y_c[0] - dy / 2], y_c + dy / 2]))


def streamfunction(x_c, y_c, dx, dy, t_h, sp):
    x_f, y_f = corner_axes(x_c, y_c, dx, dy)
    return _stream(x_f, y_f, t_h, sp)


def temperature(X, Y, t_h, sp):
    return sp["T0"] + sp["dTdy"] * (Y / LY - 0.5) * 2.0 + sp["dTdt"] * t_h / HOURS


def pbl_height(X, Y, t_h, sp):
    shape = 0.5 * (1.0 - np.cos(np.pi * min(t_h / HOURS, 1.0)))
    h = sp["h0"] + sp["dhdt"] * shape + sp["hgx"] * (X / LX - 0.5)
    return np.maximum(h, 200.0)


def vapour_source(X, Y, t_h, vf):
    spatial = np.zeros_like(X)
    for i, pa in enumerate(PRECURSOR_PATCHES):
        amp = pa["w"] * (1.0 + 0.12 * np.cos(2.0 * np.pi * vf["qkx"] * i / 5.0))
        spatial += amp * np.exp(-0.5 * (((X - pa["x"]) / pa["sx"]) ** 2
                                        + ((Y - pa["y"]) / pa["sy"]) ** 2))
    diurnal = np.exp(-0.5 * ((t_h - vf["qpk"]) / vf["qw"]) ** 2)
    return vf["q0"] * 3.4 * spatial * diurnal


def acid_field(X, Y, t_h, vf):
    spatial = 0.75 + 0.45 * np.cos(2.0 * np.pi * X / LX + 0.3) * np.sin(np.pi * Y / LY + 0.5)
    diurnal = np.exp(-0.5 * ((t_h - vf["capk"]) / 2.8) ** 2)
    return vf["ca0"] * spatial * diurnal


def source_maps(X, Y):
    out = np.zeros((3,) + X.shape)
    for i, r in enumerate(SOURCE_REGIONS):
        out[i] = ((X - r["x"]) ** 2 + (Y - r["y"]) ** 2 <= r["r"] ** 2).astype(float)
    return out


def npf_time(ep_id):
    t = record_times() / 3600.0
    return np.array([amp * np.exp(-0.5 * ((t - pk) / w) ** 2)
                     for (pk, w, amp) in NPF_TIME[ep_id]])


def grid(nx, ny):
    dx, dy = LX / nx, LY / ny
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dy
    X, Y = np.meshgrid(x, y)
    return x, y, dx, dy, X, Y
