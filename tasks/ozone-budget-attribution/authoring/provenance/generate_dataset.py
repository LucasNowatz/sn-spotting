"""Forward chemistry-transport model used to build the wildfire plume ozone archive.

This script is authoring material.  It never enters the agent container and it
never enters the verifier container.  It writes the agent-visible NetCDF archive
under environment/data/ and nothing else; every regional aggregate that the
verifier grades against is produced separately by build_truth.py, which reads
only the written files.

Model summary
-------------
* Regular Cartesian grid, 24 x 18 x 8 cells, dx = dy = 9 km, rigid lid at 3.3 km.
* Dry-air mass flux field is the discrete curl of an edge-based vector
  potential, so the discrete mass divergence is exactly zero and the dry-air
  mass of every cell is constant in time.  Moist air mass varies because the
  specific humidity field varies.
* Tracer transport: flux-form first-order upwind on the dry mass fluxes,
  explicit horizontal diffusion, implicit vertical diffusion with a dry
  deposition flux boundary condition at the ground.
* Gas-phase chemistry: RNV-LITE v1.2, a 12-species reduced NOx-VOC mechanism
  integrated with 5 s sub-steps and radicals at photostationary steady state.
* Every process operator writes its own integrated increment, so the sum of the
  recorded increments reproduces the cell ozone mass change exactly.

Scenarios
---------
00  fire off, urban off      10  fire on, urban off
01  fire off, urban on       11  fire on, urban on

Only the explicitly named fire and urban emission blocks are switched.  Boundary
composition, background and biogenic emissions, meteorology, photolysis and
deposition parameters are byte-identical across the four runs.
"""

import hashlib
import json
import os

import numpy as np
from netCDF4 import Dataset

# --------------------------------------------------------------------------
# Grid and run configuration
# --------------------------------------------------------------------------
NX, NY, NZ = 24, 18, 8
DX = DY = 9000.0                      # m
ZI = np.array([0.0, 120.0, 300.0, 560.0, 900.0, 1350.0, 1900.0, 2550.0, 3300.0])
DZ = np.diff(ZI)                      # m, layer thickness
ZM = 0.5 * (ZI[:-1] + ZI[1:])         # m, layer midpoint
AREA_H = DX * DY                      # m2, horizontal cell face area

DT = 180.0                            # s, model time step
STEPS_PER_DIAG = 20                   # 20 x 180 s = 1 h diagnostic interval
N_DIAG = 48                           # 48 diagnostic intervals = 48 h
DT_DIAG = DT * STEPS_PER_DIAG
N_SUB = 36                            # chemistry sub-steps per model step (5 s)
DT_SUB = DT / N_SUB
RADICAL_REFRESH = 6                   # recompute radical steady state every 30 s

W_O3 = 47.9982                        # g/mol
W_DRY = 28.9644                       # g/mol
W_H2O = 18.0153                       # g/mol
N_AV = 6.02214076e23

RHO0 = 1.200                          # kg/m3, dry-air density at the surface
SCALE_H = 8000.0                      # m

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "..", "environment", "data")
OUT = os.path.normpath(OUT)

SPECIES = ["O3", "NO", "NO2", "CO", "VOC", "HCHO", "HNO3", "H2O2", "ROOH"]

# --------------------------------------------------------------------------
# Static meteorology
# --------------------------------------------------------------------------
rho_dry = RHO0 * np.exp(-ZM / SCALE_H)                      # (NZ,)
m_dry = (rho_dry * DZ * AREA_H)[None, None, :] * np.ones((NX, NY, 1))   # kg
temp = (298.0 - 6.5e-3 * ZM)[None, None, :] * np.ones((NX, NY, 1))      # K

xc = (np.arange(NX) + 0.5) * DX
yc = (np.arange(NY) + 0.5) * DY
xe = np.arange(NX + 1) * DX
ye = np.arange(NY + 1) * DY

# Land / water mask drives the deposition velocity and the biogenic source.
XX, YY = np.meshgrid(xc, yc, indexing="ij")
water = (YY < 2.5 * DY) | ((XX > 16.5 * DX) & (YY > 13.5 * DY))
land = ~water

# Urban block (fixed diagnostic volume sits on top of it) and fire block.
URBAN_I = slice(12, 19)
URBAN_J = slice(7, 14)
URBAN_K = slice(0, 2)
FIRE_I = slice(2, 5)
FIRE_J = slice(7, 11)
FIRE_K = slice(3, 7)
FIRE_HOURS = (0.0, 10.0)


def hour_of_run(t_seconds):
    return t_seconds / 3600.0


def local_hour(t_seconds):
    """Run starts at 06:00 local time."""
    return (6.0 + hour_of_run(t_seconds)) % 24.0


def cos_sza(t_seconds):
    """Idealised solar elevation: daylight from 05:00 to 19:00 local."""
    lh = local_hour(t_seconds)
    arg = np.pi * (lh - 5.0) / 14.0
    return float(max(0.0, np.sin(arg)))


def pbl_top(t_seconds):
    """Diurnal boundary-layer depth in metres."""
    return 180.0 + 1750.0 * cos_sza(t_seconds) ** 1.4


def spec_humidity(t_seconds):
    """Specific humidity, kg water per kg moist air.  Varies in space and time."""
    lh = local_hour(t_seconds)
    base = 0.0165 * np.exp(-ZM / 2200.0) + 0.0035
    horiz = 1.0 + 0.22 * np.sin(2.0 * np.pi * XX / (NX * DX)) \
        + 0.16 * np.cos(np.pi * YY / (NY * DY))
    diurnal = 1.0 + 0.10 * np.sin(2.0 * np.pi * (lh - 9.0) / 24.0)
    q = base[None, None, :] * horiz[:, :, None] * diurnal
    q[:, :, 0] += 0.0016 * water[:, :, None][:, :, 0]
    return np.clip(q, 0.0035, 0.0215)


# --------------------------------------------------------------------------
# Dry-air mass flux field: discrete curl of an edge vector potential
# --------------------------------------------------------------------------
def mass_fluxes(t_seconds):
    """Return (Fx, Fy, Fz) dry-air mass fluxes in kg/s on the staggered faces.

    Fx has shape (NX+1, NY, NZ) and is positive towards +x, and so on.  The
    fields are the discrete curl of an edge-based vector potential, so the
    discrete divergence vanishes to round-off and the dry-air mass of every
    cell is time invariant.
    """
    th = hour_of_run(t_seconds)
    # Weak, stagnant summer flow that veers slowly: a classic ozone episode.
    u_prof = (1.15 + 0.32 * ZM / 1000.0) * (1.0 + 0.12 * np.sin(2.0 * np.pi * th / 19.0))
    v_prof = (0.42 + 0.10 * ZM / 1000.0) * np.sin(2.0 * np.pi * (th - 4.0) / 27.0)

    alpha = rho_dry * u_prof * DZ                      # kg/s per metre of y
    beta = rho_dry * v_prof * DZ                       # kg/s per metre of x
    Az = (alpha[None, None, :] * ye[None, :, None]
          - beta[None, None, :] * xe[:, None, None]) * np.ones((1, 1, 1))
    Az = Az * np.ones((NX + 1, NY + 1, NZ))

    kx = 2.0 * np.pi / (NX * DX)
    ky = 2.0 * np.pi / (NY * DY)
    eddy = 1.5e6 * (1.0 + 0.30 * np.sin(2.0 * np.pi * th / 23.0))
    Az = Az + eddy * np.sin(kx * xe[:, None, None] + 0.35 * np.sin(2 * np.pi * th / 31.0)) \
        * np.cos(ky * ye[None, :, None]) * np.exp(-ZM / 2600.0)[None, None, :]

    # Ax and Ay vanish on the ground and lid edge levels, so w = 0 there.
    shape_z = np.sin(np.pi * ZI / ZI[-1])
    Ax = 1.2e6 * shape_z[None, None, :] \
        * np.sin(kx * xc[:, None, None] + 0.26 * th) \
        * np.cos(ky * ye[None, :, None])
    Ay = 1.0e6 * shape_z[None, None, :] \
        * np.cos(ky * yc[None, :, None]) \
        * np.sin(kx * xe[:, None, None] + 0.18 * th)

    Fx = (Az[:, 1:, :] - Az[:, :-1, :]) - (Ay[:, :, 1:] - Ay[:, :, :-1])
    Fy = (Ax[:, :, 1:] - Ax[:, :, :-1]) - (Az[1:, :, :] - Az[:-1, :, :])
    Fz = (Ay[1:, :, :] - Ay[:-1, :, :]) - (Ax[:, 1:, :] - Ax[:, :-1, :])
    return Fx, Fy, Fz


def eddy_diffusivity(t_seconds):
    """Vertical diffusivity on the NZ+1 interfaces, m2/s."""
    h = pbl_top(t_seconds)
    kz = np.zeros(NZ + 1)
    inner = ZI[1:-1]
    frac = np.clip(inner / h, 0.0, 1.0)
    kmax = 12.0 + 95.0 * cos_sza(t_seconds) ** 1.2
    kz[1:-1] = kmax * frac * (1.0 - frac) ** 2 * 6.75 + 0.6
    kz[1:-1] = np.where(inner > 1.35 * h, 0.35, kz[1:-1])
    return kz


def deposition_velocity(t_seconds):
    """Ozone dry deposition velocity at the surface, m/s."""
    day = cos_sza(t_seconds)
    vd = np.where(land, 0.0010 + 0.0034 * day, 0.00035 + 0.00015 * day)
    return vd


# --------------------------------------------------------------------------
# Emissions (kg of species per second per cell)
# --------------------------------------------------------------------------
def emission_fields(t_seconds, fire_on, urban_on):
    """Background, fire and urban emissions for the transported species."""
    em = {s: np.zeros((NX, NY, NZ)) for s in SPECIES}
    lh = local_hour(t_seconds)
    day = cos_sza(t_seconds)

    # ---- background and biogenic, present in every scenario ----------------
    # Biogenic VOC over land, temperature and light dependent.
    bio = np.zeros((NX, NY, NZ))
    bio[:, :, 0] = land * 5.2e-4 * (0.15 + 0.85 * day)
    em["VOC"] += bio
    # Soil NO, always on.
    soil = np.zeros((NX, NY, NZ))
    soil[:, :, 0] = land * 2.1e-5 * (0.5 + 0.5 * day)
    em["NO"] += soil
    # Diffuse regional CO source.
    em["CO"][:, :, 0] += land * 1.4e-3

    # ---- selected urban precursor block -----------------------------------
    if urban_on:
        rush = 1.0 + 0.85 * np.exp(-0.5 * ((lh - 8.0) / 1.6) ** 2) \
            + 0.70 * np.exp(-0.5 * ((lh - 18.0) / 1.9) ** 2)
        daynight = 0.42 + 0.58 * np.exp(-0.5 * ((lh - 13.0) / 6.0) ** 2)
        scale = rush * daynight
        nox_total = 1.38e-2 * scale          # kg NO_x as NO per second per cell
        blk = (URBAN_I, URBAN_J, URBAN_K)
        em["NO"][blk] += 0.90 * nox_total
        em["NO2"][blk] += 0.10 * nox_total * (46.0055 / 30.0061)
        em["VOC"][blk] += 2.7e-2 * scale
        em["CO"][blk] += 1.15e-1 * scale
        em["HCHO"][blk] += 1.9e-3 * scale

    # ---- selected wildfire precursor block --------------------------------
    if fire_on and FIRE_HOURS[0] <= hour_of_run(t_seconds) < FIRE_HOURS[1]:
        ramp = np.sin(np.pi * (hour_of_run(t_seconds) - FIRE_HOURS[0])
                      / (FIRE_HOURS[1] - FIRE_HOURS[0])) ** 0.6
        blk = (FIRE_I, FIRE_J, FIRE_K)
        em["NO"][blk] += 2.75e-2 * ramp
        em["NO2"][blk] += 0.41e-2 * ramp * (46.0055 / 30.0061)
        em["VOC"][blk] += 1.15e-1 * ramp
        em["CO"][blk] += 1.05e0 * ramp
        em["HCHO"][blk] += 1.3e-2 * ramp
    return em


def fire_tracer_emission(t_seconds):
    e = np.zeros((NX, NY, NZ))
    if FIRE_HOURS[0] <= hour_of_run(t_seconds) < FIRE_HOURS[1]:
        ramp = np.sin(np.pi * (hour_of_run(t_seconds) - FIRE_HOURS[0])
                      / (FIRE_HOURS[1] - FIRE_HOURS[0])) ** 0.6
        e[FIRE_I, FIRE_J, FIRE_K] = 1.0 * ramp
    return e


# --------------------------------------------------------------------------
# Boundary composition (dry-air mole fraction, mol/mol)
# --------------------------------------------------------------------------
def boundary_vmr(t_seconds):
    prof = 33.0 + 22.0 * (ZM / ZI[-1])
    o3 = prof * (1.0 + 0.05 * np.sin(2.0 * np.pi * hour_of_run(t_seconds) / 24.0))
    bg = {
        "O3": o3 * 1e-9,
        "NO": np.full(NZ, 0.03e-9),
        "NO2": np.full(NZ, 0.12e-9),
        "CO": np.full(NZ, 105.0e-9),
        "VOC": np.full(NZ, 1.4e-9),
        "HCHO": np.full(NZ, 0.35e-9),
        "HNO3": np.full(NZ, 0.20e-9),
        "H2O2": np.full(NZ, 0.9e-9),
        "ROOH": np.full(NZ, 0.35e-9),
    }
    return bg


# --------------------------------------------------------------------------
# Transport operators
# --------------------------------------------------------------------------
def advect(mass, Fx, Fy, Fz, bg_mmr, dt):
    """Flux-form upwind advection.  Returns new mass and the face fluxes (kg)."""
    r = mass / m_dry                      # kg tracer per kg dry air
    r_bg = bg_mmr                         # kg tracer per kg dry air, inflow

    # x faces -------------------------------------------------------------
    rx = np.empty((NX + 1, NY, NZ))
    rx[1:-1] = np.where(Fx[1:-1] > 0.0, r[:-1], r[1:])
    rx[0] = np.where(Fx[0] > 0.0, r_bg[None, :], r[0])
    rx[-1] = np.where(Fx[-1] > 0.0, r[-1], r_bg[None, :])
    fx = Fx * rx * dt

    ry = np.empty((NX, NY + 1, NZ))
    ry[:, 1:-1] = np.where(Fy[:, 1:-1] > 0.0, r[:, :-1], r[:, 1:])
    ry[:, 0] = np.where(Fy[:, 0] > 0.0, r_bg[None, :], r[:, 0])
    ry[:, -1] = np.where(Fy[:, -1] > 0.0, r[:, -1], r_bg[None, :])
    fy = Fy * ry * dt

    rz = np.zeros((NX, NY, NZ + 1))
    rz[:, :, 1:-1] = np.where(Fz[:, :, 1:-1] > 0.0, r[:, :, :-1], r[:, :, 1:])
    fz = Fz * rz * dt

    div = (fx[:-1] - fx[1:]) + (fy[:, :-1] - fy[:, 1:]) + (fz[:, :, :-1] - fz[:, :, 1:])
    return mass + div, fx, fy, fz


def horizontal_diffusion(mass, kh, dt):
    r = mass / m_dry
    cx = (kh * rho_dry * DY * DZ / DX)[None, None, :]
    gx = np.zeros((NX + 1, NY, NZ))
    gx[1:-1] = -cx * (r[1:] - r[:-1]) * dt
    cy = (kh * rho_dry * DX * DZ / DY)[None, None, :]
    gy = np.zeros((NX, NY + 1, NZ))
    gy[:, 1:-1] = -cy * (r[:, 1:] - r[:, :-1]) * dt
    div = (gx[:-1] - gx[1:]) + (gy[:, :-1] - gy[:, 1:])
    return mass + div, gx, gy


def vertical_diffusion(mass, kz, vd, dt, deposits):
    """Implicit vertical diffusion with a deposition flux at the ground.

    Returns the new mass, the upward turbulent flux on the NZ+1 interfaces
    (kg, the ground value carrying the deposition removal) and the deposited
    mass per column-bottom cell.
    """
    r = (mass / m_dry).reshape(-1, NZ).copy()
    mk = m_dry.reshape(-1, NZ)
    c = np.zeros((r.shape[0], NZ + 1))
    rho_i = np.interp(ZI[1:-1], ZM, rho_dry)
    dz_i = np.diff(ZM)
    c[:, 1:-1] = (kz[1:-1] * rho_i * AREA_H / dz_i)[None, :]
    d0 = (vd.reshape(-1) * rho_dry[0] * AREA_H) if deposits else np.zeros(r.shape[0])

    a = np.zeros_like(r)
    b = np.zeros_like(r)
    cc = np.zeros_like(r)
    for k in range(NZ):
        lower = c[:, k] if k > 0 else np.zeros(r.shape[0])
        upper = c[:, k + 1]
        a[:, k] = -dt * lower
        cc[:, k] = -dt * upper
        b[:, k] = mk[:, k] + dt * (lower + upper)
    b[:, 0] += dt * d0
    rhs = mk * r

    # Thomas algorithm, vectorised over columns.
    cp = np.zeros_like(r)
    dp = np.zeros_like(r)
    cp[:, 0] = cc[:, 0] / b[:, 0]
    dp[:, 0] = rhs[:, 0] / b[:, 0]
    for k in range(1, NZ):
        den = b[:, k] - a[:, k] * cp[:, k - 1]
        cp[:, k] = cc[:, k] / den
        dp[:, k] = (rhs[:, k] - a[:, k] * dp[:, k - 1]) / den
    rn = np.zeros_like(r)
    rn[:, -1] = dp[:, -1]
    for k in range(NZ - 2, -1, -1):
        rn[:, k] = dp[:, k] - cp[:, k] * rn[:, k + 1]

    flux = np.zeros((r.shape[0], NZ + 1))
    flux[:, 1:-1] = -c[:, 1:-1] * (rn[:, 1:] - rn[:, :-1]) * dt
    flux[:, 0] = -d0 * rn[:, 0] * dt
    dep = -flux[:, 0]
    new_mass = (rn * mk).reshape(NX, NY, NZ)
    return (new_mass,
            flux.reshape(NX, NY, NZ + 1),
            dep.reshape(NX, NY))


# --------------------------------------------------------------------------
# RNV-LITE v1.2 gas-phase chemistry
# --------------------------------------------------------------------------
F_ALKENE = 0.15            # alkene fraction of the lumped VOC
J1_MAX = 9.2e-3            # s-1, NO2 photolysis
J_O1D_MAX = 3.4e-5         # s-1, O3 -> O(1D)
J_HCHO_MAX = 3.6e-5        # s-1, radical-yielding HCHO photolysis


def rate_constants():
    t = temp
    return {
        "k2": 1.40e-12 * np.exp(-1310.0 / t),
        "k3": 1.05e-11 * np.ones_like(t),
        "k4": 2.40e-13 * np.ones_like(t),
        "k5": 8.00e-12 * np.ones_like(t),
        "k6": 3.50e-12 * np.exp(250.0 / t),
        "k7": 1.10e-11 * np.ones_like(t),
        "k8": 2.50e-12 * np.ones_like(t),
        "k9": 1.00e-11 * np.ones_like(t),
        "k11": 2.00e-15 * np.ones_like(t),
        "k12": 7.00e-14 * np.ones_like(t),
        "k13": 1.00e-17 * np.ones_like(t),
    }


KC = rate_constants()


def chemistry(conc, n_air, n_h2o, t_seconds, dt):
    """Integrate the mechanism for one model step.

    conc holds number densities in molec/cm3 for the nine transported species.
    Returns the updated concentrations and the integrated ozone channels in
    molec/cm3 (production and each destruction channel, all non-negative).
    """
    cs = cos_sza(t_seconds)
    j1 = J1_MAX * cs
    j_o1d = J_O1D_MAX * cs ** 2.1
    j_hcho = J_HCHO_MAX * cs ** 1.4

    # Fraction of O(1D) that reacts with water rather than being quenched.
    f_h2o = (2.2e-10 * n_h2o) / (2.2e-10 * n_h2o + 2.9e-11 * n_air)
    j_o3_eff = j_o1d * f_h2o

    o3, no, no2, co, voc, hcho, hno3, h2o2, rooh = conc
    acc = {k: np.zeros_like(o3) for k in
           ("prod", "l_no", "l_o1d", "l_ho2", "l_oh", "l_alk", "pox")}

    oh = np.full_like(o3, 1.0e5)
    ho2 = np.full_like(o3, 1.0e7)
    ro2 = np.full_like(o3, 1.0e7)

    for s in range(N_SUB):
        if s % RADICAL_REFRESH == 0:
            for _ in range(8):
                s_oh = 2.0 * j_o3_eff * o3 + KC["k6"] * ho2 * no + KC["k11"] * ho2 * o3
                l_oh = (KC["k3"] * voc + KC["k4"] * co + KC["k7"] * no2
                        + KC["k12"] * o3 + 1.0e-3)
                oh = s_oh / l_oh
                s_ro2 = KC["k3"] * voc * oh
                l_ro2 = KC["k5"] * no + KC["k9"] * ho2 + 1.0e-4
                ro2 = s_ro2 / l_ro2
                s_ho2 = (KC["k4"] * co * oh + KC["k5"] * ro2 * no
                         + KC["k12"] * oh * o3 + 2.0 * j_hcho * hcho)
                l_ho2 = (KC["k6"] * no + 2.0 * KC["k8"] * ho2
                         + KC["k9"] * ro2 + KC["k11"] * o3 + 1.0e-4)
                ho2 = s_ho2 / l_ho2

        p_o3 = j1 * no2
        l_no = KC["k2"] * no * o3
        l_o1d = j_o3_eff * o3
        l_ho2r = KC["k11"] * ho2 * o3
        l_ohr = KC["k12"] * oh * o3
        l_alk = KC["k13"] * F_ALKENE * voc * o3
        r5 = KC["k5"] * ro2 * no
        r6 = KC["k6"] * ho2 * no
        r7 = KC["k7"] * oh * no2
        r3 = KC["k3"] * voc * oh
        r4 = KC["k4"] * co * oh
        r8 = KC["k8"] * ho2 * ho2
        r9 = KC["k9"] * ro2 * ho2

        o3 = o3 + (p_o3 - l_no - l_o1d - l_ho2r - l_ohr - l_alk) * DT_SUB
        no = no + (j1 * no2 - l_no - r5 - r6) * DT_SUB
        no2 = no2 + (l_no + r5 + r6 - j1 * no2 - r7) * DT_SUB
        co = co - r4 * DT_SUB
        voc = voc - r3 * DT_SUB
        hcho = hcho + (0.55 * r3 - j_hcho * hcho) * DT_SUB
        hno3 = hno3 + r7 * DT_SUB
        h2o2 = h2o2 + r8 * DT_SUB
        rooh = rooh + r9 * DT_SUB
        np.clip(no, 1.0e2, None, out=no)
        np.clip(no2, 1.0e2, None, out=no2)
        np.clip(voc, 1.0e3, None, out=voc)
        np.clip(hcho, 1.0e3, None, out=hcho)

        acc["prod"] += p_o3 * DT_SUB
        acc["l_no"] += l_no * DT_SUB
        acc["l_o1d"] += l_o1d * DT_SUB
        acc["l_ho2"] += l_ho2r * DT_SUB
        acc["l_oh"] += l_ohr * DT_SUB
        acc["l_alk"] += l_alk * DT_SUB
        acc["pox"] += (r5 + r6) * DT_SUB

    return [o3, no, no2, co, voc, hcho, hno3, h2o2, rooh], acc


# --------------------------------------------------------------------------
# Spin-up and the scenario integration
# --------------------------------------------------------------------------
def initial_state():
    """Common spun-up initial condition shared by all four scenarios."""
    bg = boundary_vmr(0.0)
    vmr = {}
    for s in SPECIES:
        vmr[s] = np.ones((NX, NY, NZ)) * bg[s][None, None, :]
    # A modest pre-existing regional plume of aged urban air near the city.
    bump = np.exp(-(((XX - 13.5 * DX) / (4.0 * DX)) ** 2
                    + ((YY - 10.0 * DY) / (3.5 * DY)) ** 2))
    vmr["O3"] += (6.5e-9 * bump)[:, :, None] * np.exp(-ZM / 2400.0)[None, None, :]
    vmr["NO2"] += (0.55e-9 * bump)[:, :, None] * np.exp(-ZM / 900.0)[None, None, :]
    vmr["VOC"] += (2.2e-9 * bump)[:, :, None] * np.exp(-ZM / 900.0)[None, None, :]
    vmr["CO"] += (24.0e-9 * bump)[:, :, None] * np.exp(-ZM / 1200.0)[None, None, :]
    return vmr


MOLAR = {"O3": 47.9982, "NO": 30.0061, "NO2": 46.0055, "CO": 28.0101,
         "VOC": 58.12, "HCHO": 30.026, "HNO3": 63.0128, "H2O2": 34.0147,
         "ROOH": 48.042}


def run_scenario(fire_on, urban_on, want_tracer=False):
    """Integrate one scenario and return the hourly diagnostic archive."""
    vmr0 = initial_state()
    mass = {s: vmr0[s] * m_dry * (MOLAR[s] / W_DRY) for s in SPECIES}
    tracer = np.zeros((NX, NY, NZ))

    o3_vmr_out = np.zeros((N_DIAG + 1, NX, NY, NZ), dtype=np.float64)
    air_out = np.zeros((N_DIAG + 1, NX, NY, NZ), dtype=np.float64)
    q_out = np.zeros((N_DIAG + 1, NX, NY, NZ), dtype=np.float64)
    trc_out = np.zeros((N_DIAG + 1, NX, NY, NZ), dtype=np.float64)

    proc_names = ["o3_prod_jno2", "o3_loss_no", "o3_loss_o1d", "o3_loss_ho2",
                  "o3_loss_oh", "o3_loss_alkene", "o3_depos", "o3_prod_ox_diag"]
    proc = {n: np.zeros((N_DIAG, NX, NY, NZ), dtype=np.float64) for n in proc_names}
    fl = {
        "adv_flux_x": np.zeros((N_DIAG, NX + 1, NY, NZ)),
        "adv_flux_y": np.zeros((N_DIAG, NX, NY + 1, NZ)),
        "adv_flux_z": np.zeros((N_DIAG, NX, NY, NZ + 1)),
        "mix_flux_x": np.zeros((N_DIAG, NX + 1, NY, NZ)),
        "mix_flux_y": np.zeros((N_DIAG, NX, NY + 1, NZ)),
        "mix_flux_z": np.zeros((N_DIAG, NX, NY, NZ + 1)),
    }

    def snapshot(idx, t):
        q = spec_humidity(t)
        o3_vmr_out[idx] = mass["O3"] / m_dry * (W_DRY / W_O3)
        air_out[idx] = m_dry / (1.0 - q)
        q_out[idx] = q
        trc_out[idx] = tracer

    snapshot(0, 0.0)
    n_air_const = rho_dry / (W_DRY * 1.0e-3) * N_AV * 1.0e-6   # molec/cm3 per layer

    for d in range(N_DIAG):
        acc_proc = {n: np.zeros((NX, NY, NZ)) for n in proc_names}
        acc_fl = {k: np.zeros(v.shape[1:]) for k, v in fl.items()}
        for st in range(STEPS_PER_DIAG):
            t = (d * STEPS_PER_DIAG + st) * DT
            Fx, Fy, Fz = mass_fluxes(t)
            kz = eddy_diffusivity(t)
            vd = deposition_velocity(t)
            bg = boundary_vmr(t)
            em = emission_fields(t, fire_on, urban_on)

            for s in SPECIES:
                mass[s] = mass[s] + em[s] * DT
            tracer = tracer + fire_tracer_emission(t) * DT

            for s in SPECIES:
                bgm = bg[s] * (MOLAR[s] / W_DRY)
                new, fx, fy, fz = advect(mass[s], Fx, Fy, Fz, bgm, DT)
                mass[s] = new
                if s == "O3":
                    acc_fl["adv_flux_x"] += fx
                    acc_fl["adv_flux_y"] += fy
                    acc_fl["adv_flux_z"] += fz
            tracer, _, _, _ = advect(tracer, Fx, Fy, Fz, np.zeros(NZ), DT)

            for s in SPECIES:
                new, gx, gy = horizontal_diffusion(mass[s], 210.0, DT)
                mass[s] = new
                if s == "O3":
                    acc_fl["mix_flux_x"] += gx
                    acc_fl["mix_flux_y"] += gy
            tracer, _, _ = horizontal_diffusion(tracer, 210.0, DT)

            for s in SPECIES:
                dep_on = (s == "O3")
                new, vfz, dep = vertical_diffusion(mass[s], kz, vd, DT, dep_on)
                mass[s] = new
                if s == "O3":
                    acc_fl["mix_flux_z"] += vfz
                    acc_proc["o3_depos"][:, :, 0] += dep
            tracer, _, _ = vertical_diffusion(tracer, kz, vd, DT, False)

            q = spec_humidity(t)
            n_air = n_air_const[None, None, :] * np.ones((NX, NY, 1))
            n_h2o = (q / (1.0 - q)) * (W_DRY / W_H2O) * n_air
            conc = [mass[s] / m_dry * (W_DRY / MOLAR[s]) * n_air for s in SPECIES]
            conc, chem = chemistry(conc, n_air, n_h2o, t, DT)
            for i, s in enumerate(SPECIES):
                mass[s] = conc[i] / n_air * m_dry * (MOLAR[s] / W_DRY)
            conv = m_dry / n_air * (W_O3 / W_DRY)
            acc_proc["o3_prod_jno2"] += chem["prod"] * conv
            acc_proc["o3_loss_no"] += chem["l_no"] * conv
            acc_proc["o3_loss_o1d"] += chem["l_o1d"] * conv
            acc_proc["o3_loss_ho2"] += chem["l_ho2"] * conv
            acc_proc["o3_loss_oh"] += chem["l_oh"] * conv
            acc_proc["o3_loss_alkene"] += chem["l_alk"] * conv
            acc_proc["o3_prod_ox_diag"] += chem["pox"] * conv

        for n in proc_names:
            proc[n][d] = acc_proc[n]
        for k in fl:
            fl[k][d] = acc_fl[k]
        snapshot(d + 1, (d + 1) * STEPS_PER_DIAG * DT)

    out = {"o3_vmr": o3_vmr_out, "air_mass": air_out, "spec_humidity": q_out,
           "process": proc, "fluxes": fl}
    if want_tracer:
        out["tracer"] = trc_out
    return out


# --------------------------------------------------------------------------
# Diagnostic volumes
# --------------------------------------------------------------------------
FIXED_I = slice(12, 19)      # 7 cells, x = 108 - 171 km
FIXED_J = slice(7, 14)       # 7 cells, y = 63 - 126 km
FIXED_K = slice(0, 4)        # 4 layers, z = 0 - 900 m, includes the ground

MASK_LO = 6.0e-9             # fire-tracer mixing ratio at which the mask opens
MASK_HI = 5.0e-8             # mixing ratio at which the mask weight saturates

WINDOWS = {"W1": (6, 14), "W2": (16, 24), "W3": (26, 34), "W4": (34, 44)}

RECEPTORS = [
    ("R1_urban_core", [(15, 10, 0, 1.0)]),
    ("R2_downwind_suburb", [(19, 9, 0, 0.36), (20, 9, 0, 0.24),
                            (19, 10, 0, 0.24), (20, 10, 0, 0.16)]),
    ("R3_rural_upwind", [(5, 4, 0, 0.5), (5, 5, 0, 0.5)]),
]


def build_masks(tracer):
    """Common moving plume weights and the fixed urban volume."""
    rt = tracer / m_dry[None, :, :, :]
    w = np.clip((rt - MASK_LO) / (MASK_HI - MASK_LO), 0.0, 1.0)
    w = np.round(w, 4)
    fixed = np.zeros((NX, NY, NZ))
    fixed[FIXED_I, FIXED_J, FIXED_K] = 1.0
    return fixed, w


def check_closure(run):
    """Per-cell closure of the recorded increments, over every hour."""
    o3 = run["o3_vmr"]
    q = run["spec_humidity"]
    mdry = m_dry[None, :, :, :] * np.ones((o3.shape[0], 1, 1, 1))
    mass = o3 * mdry * (W_O3 / W_DRY)
    fl, pr = run["fluxes"], run["process"]
    adv = ((fl["adv_flux_x"][:, :-1] - fl["adv_flux_x"][:, 1:])
           + (fl["adv_flux_y"][:, :, :-1] - fl["adv_flux_y"][:, :, 1:])
           + (fl["adv_flux_z"][:, :, :, :-1] - fl["adv_flux_z"][:, :, :, 1:]))
    mix = ((fl["mix_flux_x"][:, :-1] - fl["mix_flux_x"][:, 1:])
           + (fl["mix_flux_y"][:, :, :-1] - fl["mix_flux_y"][:, :, 1:])
           + (fl["mix_flux_z"][:, :, :, :-1] - fl["mix_flux_z"][:, :, :, 1:]))
    chem = pr["o3_prod_jno2"] - sum(pr[k] for k in
                                    ("o3_loss_no", "o3_loss_o1d", "o3_loss_ho2",
                                     "o3_loss_oh", "o3_loss_alkene"))
    resid = (mass[1:] - mass[:-1]) - (adv + mix + chem)
    scale = np.abs(mass[:-1]).max()
    return np.abs(resid).max(), np.abs(resid).max() / scale


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Archive writer
# --------------------------------------------------------------------------
def _var(ds, name, dims, data, dtype="f4", **attrs):
    v = ds.createVariable(name, dtype, dims, zlib=True, complevel=5, shuffle=True)
    v[...] = data
    for k, val in attrs.items():
        setattr(v, k, val)


def write_grid(path):
    with Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("x", NX)
        ds.createDimension("y", NY)
        ds.createDimension("z", NZ)
        ds.createDimension("x_face", NX + 1)
        ds.createDimension("y_face", NY + 1)
        ds.createDimension("z_face", NZ + 1)
        _var(ds, "x_centre", ("x",), xc, "f8", units="m",
             long_name="cell centre, eastward distance from the west edge")
        _var(ds, "y_centre", ("y",), yc, "f8", units="m",
             long_name="cell centre, northward distance from the south edge")
        _var(ds, "z_centre", ("z",), ZM, "f8", units="m",
             long_name="layer mid height above ground")
        _var(ds, "x_face_position", ("x_face",), xe, "f8", units="m",
             long_name="x face position; face i lies between cell i-1 and cell i")
        _var(ds, "y_face_position", ("y_face",), ye, "f8", units="m",
             long_name="y face position; face j lies between cell j-1 and cell j")
        _var(ds, "z_interface", ("z_face",), ZI, "f8", units="m",
             long_name="layer interface height; interface k lies below layer k")
        _var(ds, "layer_thickness", ("z",), DZ, "f8", units="m")
        _var(ds, "cell_volume", ("z",), DZ * AREA_H, "f8", units="m3",
             long_name="cell volume, identical for every column")
        _var(ds, "face_area_x", ("z",), DY * DZ, "f8", units="m2")
        _var(ds, "face_area_y", ("z",), DX * DZ, "f8", units="m2")
        _var(ds, "face_area_z", ("z_face",), np.full(NZ + 1, AREA_H), "f8",
             units="m2")
        _var(ds, "temperature", ("z",), 298.0 - 6.5e-3 * ZM, "f8", units="K",
             long_name="layer temperature, constant in time")
        _var(ds, "land_fraction", ("x", "y"), land.astype(np.float32), "f4",
             units="1", long_name="1 over land, 0 over water")
        ds.title = "Wildfire plume ozone budget archive: grid and topology"
        ds.dx_metres = DX
        ds.dy_metres = DY
        ds.face_orientation = (
            "adv_flux_x and mix_flux_x are positive towards +x and are stored on "
            "x_face; cell i receives flux_x[i] through its west face and loses "
            "flux_x[i+1] through its east face. The same pattern holds in y and z, "
            "with z_face index 0 the ground and index NZ the model lid.")
        ds.lid = "rigid; the resolved vertical mass flux is zero at z_face 0 and NZ"


def write_meteo(path, air_mass, spec_hum, times):
    with Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", len(times))
        ds.createDimension("x", NX)
        ds.createDimension("y", NY)
        ds.createDimension("z", NZ)
        _var(ds, "time", ("time",), times, "f8", units="s",
             long_name="seconds since run start")
        _var(ds, "air_mass", ("time", "x", "y", "z"), air_mass, "f4", units="kg",
             long_name="total (moist) air mass of the cell",
             note="includes the mass of water vapour")
        _var(ds, "specific_humidity", ("time", "x", "y", "z"), spec_hum, "f4",
             units="kg kg-1",
             long_name="mass of water vapour per kg of moist air")
        ds.title = "Wildfire plume ozone budget archive: meteorology"
        ds.identical_across_scenarios = (
            "yes; the four scenarios share one meteorological realisation")
        ds.dry_air_note = (
            "dry air mass = air_mass * (1 - specific_humidity); it is invariant in "
            "time because the resolved dry-air mass flux field is non-divergent")


def write_masks(path, fixed, moving, times):
    with Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", len(times))
        ds.createDimension("x", NX)
        ds.createDimension("y", NY)
        ds.createDimension("z", NZ)
        _var(ds, "time", ("time",), times, "f8", units="s",
             long_name="seconds since run start")
        _var(ds, "fixed_urban", ("x", "y", "z"), fixed, "f4", units="1",
             long_name="fixed diagnostic volume over the urban area")
        _var(ds, "moving_plume", ("time", "x", "y", "z"), moving, "f4", units="1",
             long_name="common moving plume weight w between 0 and 1")
        ds.title = "Wildfire plume ozone budget archive: diagnostic volumes"
        ds.moving_mask_origin = (
            "derived from a passive fire source tracer transported by the same "
            "wind field with no chemistry and no deposition, then rescaled to "
            "[0,1]; the weights are therefore identical in all four scenarios")


def write_state(path, o3_ppbv, times):
    with Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", len(times))
        ds.createDimension("x", NX)
        ds.createDimension("y", NY)
        ds.createDimension("z", NZ)
        _var(ds, "time", ("time",), times, "f8", units="s",
             long_name="seconds since run start")
        _var(ds, "o3_vmr", ("time", "x", "y", "z"), o3_ppbv, "f4", units="ppbv",
             long_name="ozone dry-air mole fraction",
             note="instantaneous value at the interval endpoint; multiply by 1e-9 "
                  "for mol per mol of dry air")
        ds.title = "Wildfire plume ozone budget archive: ozone state"


def write_process(path, proc, t0, t1):
    with Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("dtime", N_DIAG)
        ds.createDimension("x", NX)
        ds.createDimension("y", NY)
        ds.createDimension("z", NZ)
        _var(ds, "interval_start", ("dtime",), t0, "f8", units="s",
             long_name="interval start, seconds since run start")
        _var(ds, "interval_end", ("dtime",), t1, "f8", units="s",
             long_name="interval end, seconds since run start")
        meta = {
            "o3_prod_jno2": "gross ozone production, NO2 + hv -> NO + O3",
            "o3_loss_no": "gross ozone loss, NO + O3 -> NO2 + O2",
            "o3_loss_o1d": "gross ozone loss, O3 + hv -> O(1D), counted only for "
                           "the O(1D) fraction that reacts with water vapour",
            "o3_loss_ho2": "gross ozone loss, HO2 + O3 -> OH + 2 O2",
            "o3_loss_oh": "gross ozone loss, OH + O3 -> HO2 + O2",
            "o3_loss_alkene": "gross ozone loss, alkene + O3 -> products",
            "o3_depos": "ozone removed by dry deposition at the ground",
            "o3_prod_ox_diag": "radical-driven NO to NO2 conversion, "
                               "k(RO2+NO)[RO2][NO] + k(HO2+NO)[HO2][NO]; reported "
                               "for reference only and NOT a term of the ozone "
                               "mass budget",
        }
        for name, desc in meta.items():
            _var(ds, name, ("dtime", "x", "y", "z"), proc[name], "f4",
                 units="kg", long_name=desc,
                 note="integrated over the interval; non-negative by construction")
        ds.title = "Wildfire plume ozone budget archive: integrated process increments"
        ds.mechanism = "RNV-LITE v1.2"
        ds.completeness = (
            "the six gross channels o3_prod_jno2 and o3_loss_* are every reaction "
            "in RNV-LITE v1.2 that creates or destroys O3; o3_prod_ox_diag is a "
            "radical diagnostic and is not one of them")
        ds.deposition_note = (
            "o3_depos is the same removal that appears inside mix_flux_z at "
            "z_face index 0 in fluxes.nc, reported here as a separate "
            "non-negative quantity")


def write_fluxes(path, fl, t0, t1):
    with Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("dtime", N_DIAG)
        ds.createDimension("x", NX)
        ds.createDimension("y", NY)
        ds.createDimension("z", NZ)
        ds.createDimension("x_face", NX + 1)
        ds.createDimension("y_face", NY + 1)
        ds.createDimension("z_face", NZ + 1)
        _var(ds, "interval_start", ("dtime",), t0, "f8", units="s",
             long_name="interval start, seconds since run start")
        _var(ds, "interval_end", ("dtime",), t1, "f8", units="s",
             long_name="interval end, seconds since run start")
        dims = {"x": ("dtime", "x_face", "y", "z"),
                "y": ("dtime", "x", "y_face", "z"),
                "z": ("dtime", "x", "y", "z_face")}
        for kind, pretty in (("adv", "resolved advective"), ("mix", "turbulent and diffusive")):
            for ax in "xyz":
                _var(ds, f"{kind}_flux_{ax}", dims[ax], fl[f"{kind}_flux_{ax}"],
                     "f4", units="kg",
                     long_name=f"{pretty} ozone flux through the {ax} faces, "
                               f"integrated over the interval, positive towards +{ax}")
        ds.title = "Wildfire plume ozone budget archive: integrated interface fluxes"
        ds.independence = (
            "these fluxes are recorded by the transport operators themselves and "
            "are not derived from the stored ozone state")
        ds.surface_note = (
            "mix_flux_z at z_face index 0 is the total turbulent ozone flux at the "
            "ground and therefore carries the dry deposition removal; "
            "adv_flux_z is zero at z_face index 0 and at z_face index NZ")


def write_receptors(path):
    lines = ["receptor_id,i,j,k,weight"]
    for rid, cells in RECEPTORS:
        for (i, j, k, wgt) in cells:
            lines.append(f"{rid},{i},{j},{k},{wgt:.2f}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    scen = {"00": (False, False), "10": (True, False),
            "01": (False, True), "11": (True, True)}
    runs = {}
    for name, (f, u) in scen.items():
        print("integrating scenario", name, flush=True)
        runs[name] = run_scenario(f, u, want_tracer=(name == "11"))
        a, r = check_closure(runs[name])
        print(f"   per-cell closure: max {a:.3e} kg, relative {r:.3e}")
        assert r < 1e-10, "recorded increments do not close the cell ozone mass"

    # Meteorology must be bit-identical between the runs.
    for name in scen:
        assert np.array_equal(runs[name]["air_mass"], runs["00"]["air_mass"])
        assert np.array_equal(runs[name]["spec_humidity"], runs["00"]["spec_humidity"])

    times = np.arange(N_DIAG + 1) * DT_DIAG
    t0 = times[:-1]
    t1 = times[1:]
    fixed, moving = build_masks(runs["11"]["tracer"])

    write_grid(os.path.join(OUT, "grid.nc"))
    write_meteo(os.path.join(OUT, "meteo.nc"), runs["00"]["air_mass"],
                runs["00"]["spec_humidity"], times)
    write_masks(os.path.join(OUT, "masks.nc"), fixed, moving, times)
    write_receptors(os.path.join(OUT, "receptors.csv"))
    for name in scen:
        d = os.path.join(OUT, name)
        os.makedirs(d, exist_ok=True)
        write_state(os.path.join(d, "state.nc"), runs[name]["o3_vmr"] * 1e9, times)
        write_process(os.path.join(d, "process.nc"), runs[name]["process"], t0, t1)
        write_fluxes(os.path.join(d, "fluxes.nc"), runs[name]["fluxes"], t0, t1)

    write_manifest(runs)
    print("archive written to", OUT)


def write_manifest(runs):
    files = ["grid.nc", "meteo.nc", "masks.nc", "receptors.csv"]
    for s in ("00", "10", "01", "11"):
        files += [f"{s}/state.nc", f"{s}/process.nc", f"{s}/fluxes.nc"]
    checks = {f: sha256(os.path.join(OUT, f)) for f in files}
    man = {
        "archive_version": "o3-budget-archive-1.0",
        "mechanism": "RNV-LITE v1.2",
        "data_status": "synthetic; produced by a purpose-built regional "
                       "chemistry-transport model, not by a public archive",
        "grid": {"nx": NX, "ny": NY, "nz": NZ, "dx_m": DX, "dy_m": DY,
                 "z_interfaces_m": list(ZI),
                 "array_order": "variables are stored as (time, x, y, z) or "
                                "(dtime, x, y, z); face variables replace the "
                                "staggered axis by its x_face / y_face / z_face "
                                "counterpart"},
        "time": {"diagnostic_interval_s": DT_DIAG,
                 "n_intervals": N_DIAG,
                 "n_endpoints": N_DIAG + 1,
                 "model_time_step_s": DT,
                 "start_local_time": "06:00",
                 "note": "state.nc holds instantaneous endpoint values; "
                         "process.nc and fluxes.nc hold quantities already "
                         "integrated over the interval between two endpoints, so "
                         "no quadrature of a rate is required"},
        "units": {"o3_vmr": "ppbv, dry-air mole fraction (multiply by 1e-9)",
                  "air_mass": "kg of moist air per cell",
                  "specific_humidity": "kg water vapour per kg moist air",
                  "process_increments": "kg O3 per interval, non-negative",
                  "fluxes": "kg O3 per interval, signed, positive towards +x/+y/+z"},
        "molar_mass_g_per_mol": {"O3": W_O3, "dry_air": W_DRY, "H2O": W_H2O},
        "mass_convention": (
            "cell ozone mass = o3_vmr * 1e-9 * dry_air_mass * M(O3)/M(dry air), "
            "with dry_air_mass = air_mass * (1 - specific_humidity)"),
        "chemistry": {
            "production_channels": ["o3_prod_jno2"],
            "destruction_channels": ["o3_loss_no", "o3_loss_o1d", "o3_loss_ho2",
                                     "o3_loss_oh", "o3_loss_alkene"],
            "not_a_budget_term": ["o3_prod_ox_diag"],
            "note": "the listed channels are the complete set of O3 sources and "
                    "sinks in RNV-LITE v1.2"},
        "deposition": {
            "variable": "o3_depos",
            "double_counting_warning":
                "the same removal is already contained in mix_flux_z at z_face "
                "index 0; a turbulent import term that excludes deposition must "
                "add o3_depos back to the summed mixing fluxes"},
        "scenarios": {
            "00": "selected fire emissions off, selected urban emissions off",
            "10": "selected fire emissions on, selected urban emissions off",
            "01": "selected fire emissions off, selected urban emissions on",
            "11": "selected fire emissions on, selected urban emissions on",
            "switched_species": ["NO", "NO2", "CO", "VOC", "HCHO"],
            "unchanged": ["meteorology", "initial state", "boundary composition",
                          "biogenic and soil emissions", "background CO source",
                          "photolysis forcing", "deposition parameters"],
            "direct_ozone_emission": 0.0},
        "diagnostic_volumes": {
            "fixed_urban": "masks.nc variable fixed_urban, constant in time",
            "moving_plume": "masks.nc variable moving_plume, weights in [0,1] on "
                            "the same time axis as state.nc"},
        "windows": {k: {"start_index": v[0], "end_index": v[1],
                        "start_s": v[0] * DT_DIAG, "end_s": v[1] * DT_DIAG}
                    for k, v in WINDOWS.items()},
        "additional_source_S": "no ozone source other than chemistry is applied; "
                               "S is zero in every scenario, volume and window",
        "checksums_sha256": checks,
    }
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(man, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
