"""Meteorology, forcing and observing-system definition (generator side).

Winds come from an analytic streamfunction, so the supplied horizontal wind
field is non-divergent to machine precision.  That is disclosed in the model
specification: it removes spurious mass sources from the transport equation
without removing any of the intended difficulty.

Every episode is 10 h with forcing records every 15 min.  The eight episodes
differ in flow topology and thermal/PBL regime; the two withheld ones
recombine ranges that appear in the observed six rather than leaving them.
Four of the eight are flagged as event days, on which the unknown nucleation
and vapour anomalies apply.
"""

import numpy as np

LX, LY = 300.0e3, 240.0e3
DX_PUB, NX_PUB, NY_PUB = 5000.0, 60, 48
HOURS = 10.0
N_REC = 41                      # forcing records, every 15 min
N_OBS = 41                      # observation times, every 15 min

# --- source regions: three disjoint discs of new-particle formation ---
SOURCE_REGIONS = [
    dict(name="A", x=52.0e3, y=68.0e3, r=27.0e3),
    dict(name="B", x=72.0e3, y=182.0e3, r=23.0e3),
    dict(name="C", x=163.0e3, y=42.0e3, r=22.0e3),
]

# --- stations: S1-S5 observed, W1-W2 spatially withheld ---
STATIONS = [
    dict(id="S1", x=128.0e3, y=108.0e3),
    dict(id="S2", x=192.0e3, y=152.0e3),
    dict(id="S3", x=148.0e3, y=192.0e3),
    dict(id="S4", x=208.0e3, y=72.0e3),
    dict(id="S5", x=98.0e3, y=142.0e3),
    dict(id="W1", x=248.0e3, y=118.0e3),
    dict(id="W2", x=172.0e3, y=128.0e3),
]

EPISODES = {
    # id:   mean wind speed/dir, rotation, shear, eddy, thermal and PBL regime
    "E01": dict(status="observed", event=False, spd=5.6, dirdeg=225.0, rot=0.0,
                shear=0.0, eddy=0.6, ekx=1.0, eky=1.0, T0=294.0, dTdy=-3.0,
                dTdt=1.2, h0=1250.0, dhdt=180.0, hgx=180.0,
                note="warm, deep boundary layer, steady south-westerly"),
    "E02": dict(status="observed", event=True, spd=6.4, dirdeg=270.0, rot=0.0,
                shear=0.0, eddy=0.4, ekx=1.0, eky=1.0, T0=278.0, dTdy=-2.0,
                dTdt=0.8, h0=430.0, dhdt=60.0, hgx=60.0,
                note="cool, shallow boundary layer, steady westerly; event day"),
    "E03": dict(status="observed", event=False, spd=5.0, dirdeg=250.0, rot=62.0,
                shear=0.0, eddy=0.7, ekx=1.0, eky=1.0, T0=286.0, dTdy=-2.5,
                dTdt=0.6, h0=800.0, dhdt=120.0, hgx=120.0,
                note="mild, wind direction rotates 62 degrees"),
    "E04": dict(status="observed", event=True, spd=2.6, dirdeg=210.0, rot=15.0,
                shear=0.0, eddy=1.5, ekx=2.0, eky=1.0, T0=295.0, dTdy=-2.0,
                dTdt=1.0, h0=1500.0, dhdt=220.0, hgx=220.0,
                note="warm, weak flow with strong eddy mixing; event day"),
    "E05": dict(status="observed", event=False, spd=5.2, dirdeg=265.0, rot=0.0,
                shear=3.4, eddy=0.5, ekx=1.0, eky=2.0, T0=279.0, dTdy=-3.5,
                dTdt=0.5, h0=560.0, dhdt=90.0, hgx=90.0,
                note="cool, strongly sheared northern and southern sectors"),
    "E06": dict(status="observed", event=True, spd=9.2, dirdeg=248.0, rot=0.0,
                shear=0.0, eddy=0.3, ekx=1.0, eky=1.0, T0=288.0, dTdy=-2.0,
                dTdt=0.7, h0=950.0, dhdt=140.0, hgx=140.0,
                note="mild, fast transit across the domain; event day"),
    "E07": dict(status="withheld", event=False, spd=4.6, dirdeg=238.0, rot=48.0,
                shear=1.8, eddy=1.0, ekx=1.0, eky=2.0, T0=280.5, dTdy=-3.0,
                dTdt=0.9, h0=520.0, dhdt=250.0, hgx=110.0,
                note="cool, growing boundary layer, curved sheared flow"),
    "E08": dict(status="withheld", event=True, spd=6.8, dirdeg=258.0, rot=22.0,
                shear=2.6, eddy=0.7, ekx=2.0, eky=1.0, T0=293.0, dTdy=-2.5,
                dTdt=-0.9, h0=1400.0, dhdt=-320.0, hgx=150.0,
                note="warm, collapsing boundary layer, moderate shear; event day"),
}

EVENT_EPISODES = [e for e, d in EPISODES.items() if d["event"]]

# NPF temporal modulation, cm-3 h-1 at the region centre, one row per region:
# (peak hour, width h, amplitude).  These are the published bottom-up rates,
# before the unknown regional strengths and the event anomaly.
NPF_TIME = {
    "E01": [(2.2, 2.6, 5200.0), (4.0, 2.2, 3600.0), (3.1, 3.0, 4100.0)],
    "E02": [(1.8, 2.2, 4300.0), (3.4, 2.6, 4800.0), (5.0, 2.4, 3300.0)],
    "E03": [(2.6, 3.0, 4700.0), (2.0, 2.0, 4200.0), (4.6, 2.8, 4500.0)],
    "E04": [(3.0, 3.4, 5600.0), (5.2, 2.4, 3100.0), (2.4, 2.2, 3900.0)],
    "E05": [(2.0, 2.4, 4100.0), (4.4, 3.0, 5000.0), (3.6, 2.6, 4400.0)],
    "E06": [(1.6, 2.0, 4900.0), (3.0, 2.4, 4300.0), (4.2, 2.6, 3700.0)],
    "E07": [(2.8, 2.8, 4600.0), (4.8, 2.2, 4000.0), (2.2, 2.4, 4800.0)],
    "E08": [(2.4, 2.2, 5100.0), (3.8, 2.8, 3500.0), (5.4, 2.6, 4200.0)],
}

# precursor-to-condensable-vapour source and H2SO4 forcing, per episode
VAPOUR_FORCING = {
    "E01": dict(q0=0.62, qkx=1.0, qky=1.0, qpk=5.2, qw=3.4, ca0=4.6, capk=5.0),
    "E02": dict(q0=0.44, qkx=1.0, qky=2.0, qpk=4.6, qw=3.0, ca0=3.1, capk=4.6),
    "E03": dict(q0=0.55, qkx=2.0, qky=1.0, qpk=5.6, qw=3.6, ca0=3.9, capk=5.4),
    "E04": dict(q0=0.70, qkx=1.0, qky=1.0, qpk=5.0, qw=4.0, ca0=5.2, capk=5.2),
    "E05": dict(q0=0.48, qkx=2.0, qky=2.0, qpk=4.8, qw=3.2, ca0=3.4, capk=4.8),
    "E06": dict(q0=0.58, qkx=1.0, qky=1.0, qpk=5.4, qw=3.4, ca0=4.1, capk=5.0),
    "E07": dict(q0=0.51, qkx=2.0, qky=1.0, qpk=5.0, qw=3.5, ca0=3.6, capk=5.1),
    "E08": dict(q0=0.66, qkx=1.0, qky=2.0, qpk=5.2, qw=3.3, ca0=4.9, capk=4.9),
}


def rec_times():
    return np.linspace(0.0, HOURS * 3600.0, N_REC)


def _stream(x_f, y_f, t_h, sp):
    """Streamfunction on the corner grid, m2 s-1."""
    ang = np.deg2rad(sp["dirdeg"] + sp["rot"] * t_h / HOURS)
    # meteorological direction: the wind blows FROM dirdeg
    u_mean = -sp["spd"] * np.sin(ang)
    v_mean = -sp["spd"] * np.cos(ang)
    X, Y = np.meshgrid(x_f, y_f)
    psi = -u_mean * Y + v_mean * X
    yc = LY / 2.0
    psi = psi - sp["shear"] * ((Y - yc) ** 2) / (2.0 * (LY / 2.0))
    kx = 2.0 * np.pi * sp["ekx"] / LX
    ky = 2.0 * np.pi * sp["eky"] / LY
    ph = 2.0 * np.pi * 0.12 * t_h
    psi = psi + (sp["eddy"] / ky) * np.sin(kx * X + ph) * np.cos(ky * Y)
    return psi


def corner_axes(x_c, y_c, dx, dy):
    return (np.concatenate([[x_c[0] - dx / 2], x_c + dx / 2]),
            np.concatenate([[y_c[0] - dy / 2], y_c + dy / 2]))


def streamfunction(x_c, y_c, dx, dy, t_h, sp):
    """Streamfunction on the cell corners, shape (ny+1, nx+1), m2 s-1."""
    x_f, y_f = corner_axes(x_c, y_c, dx, dy)
    return _stream(x_f, y_f, t_h, sp)


def face_winds(psi, dx, dy):
    uf = -(psi[1:, :] - psi[:-1, :]) / dy          # (ny, nx+1)
    vf = (psi[:, 1:] - psi[:, :-1]) / dx           # (ny+1, nx)
    return uf, vf


def temperature(X, Y, t_h, sp):
    return (sp["T0"] + sp["dTdy"] * (Y / LY - 0.5) * 2.0
            + sp["dTdt"] * t_h / HOURS)


def pbl_height(X, Y, t_h, sp):
    shape = 0.5 * (1.0 - np.cos(np.pi * min(t_h / HOURS, 1.0)))
    h = sp["h0"] + sp["dhdt"] * shape + sp["hgx"] * (X / LX - 0.5)
    return np.maximum(h, 200.0)


# Precursor patches: the condensable-vapour source is localised rather than
# spread over the domain, so the concentration a station sees depends on how
# long the air has travelled since leaving a patch; that separates the yield
# scale from the vapour lifetime.
PRECURSOR_PATCHES = [
    dict(x=46.0e3, y=112.0e3, sx=22.0e3, sy=26.0e3, w=1.00),
    dict(x=96.0e3, y=36.0e3, sx=20.0e3, sy=18.0e3, w=0.85),
    dict(x=82.0e3, y=204.0e3, sx=24.0e3, sy=20.0e3, w=0.90),
    dict(x=176.0e3, y=152.0e3, sx=18.0e3, sy=22.0e3, w=0.70),
]


def vapour_source(X, Y, t_h, vf):
    """Known precursor-to-condensable-vapour source, ug m-3 h-1."""
    spatial = np.zeros_like(X)
    for i, pa in enumerate(PRECURSOR_PATCHES):
        amp = pa["w"] * (1.0 + 0.15 * np.cos(2.0 * np.pi * vf["qkx"] * i / 4.0))
        spatial += amp * np.exp(-0.5 * (((X - pa["x"]) / pa["sx"]) ** 2
                                        + ((Y - pa["y"]) / pa["sy"]) ** 2))
    diurnal = np.exp(-0.5 * ((t_h - vf["qpk"]) / vf["qw"]) ** 2)
    return vf["q0"] * 3.2 * spatial * diurnal


def acid_field(X, Y, t_h, vf):
    """Known gridded H2SO4 concentration, 1e6 cm-3."""
    spatial = 0.8 + 0.4 * np.cos(2.0 * np.pi * X / LX) * np.sin(
        np.pi * Y / LY + 0.4)
    diurnal = np.exp(-0.5 * ((t_h - vf["capk"]) / 3.2) ** 2)
    return vf["ca0"] * spatial * diurnal


def source_maps(X, Y):
    """Three disjoint region maps, 1 inside the disc and 0 outside."""
    out = np.zeros((3, ) + X.shape)
    for i, r in enumerate(SOURCE_REGIONS):
        out[i] = ((X - r["x"]) ** 2 + (Y - r["y"]) ** 2
                  <= r["r"] ** 2).astype(float)
    return out


def npf_time(ep_id):
    t = rec_times() / 3600.0
    rows = []
    for (pk, w, amp) in NPF_TIME[ep_id]:
        rows.append(amp * np.exp(-0.5 * ((t - pk) / w) ** 2))
    return np.array(rows)


def grid(nx, ny):
    dx, dy = LX / nx, LY / ny
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dy
    X, Y = np.meshgrid(x, y)
    return x, y, dx, dy, X, Y
