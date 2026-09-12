# Model specification

Everything an expert needs to reproduce the system that produced the
observations. The numerical method, the optimiser and the uncertainty method
are yours to choose; the physics, the operators, the error model and the
definitions below are not.

## 1. Domain, coordinates and time

Projected Cartesian metres, `x` eastward and `y` northward. The domain is
300 km by 240 km. The public grid is 60 by 48 cells of 5 km, with cell centres
in `x` and `y` in `met.nc`; you may work on any internal grid.

One well-mixed boundary-layer slab of spatially and temporally varying depth
`h`. There is no resolved vertical structure.

Each episode is 10 h. Meteorology and forcing are given at 41 records, every
15 min, and **every forcing field varies linearly in time between consecutive
records**. Observations are reported on the same 41 times. Times are seconds
since 2026-06-15T00:00:00Z.

## 2. Wind field

The flow is horizontally non-divergent and is defined by the corner
streamfunction `psi` in `met.nc`, of shape `(time, y_corner, x_corner)`:

    u = -d(psi)/dy        v = +d(psi)/dx

Taking face-normal winds as the discrete curl of `psi`,

    u_face[j, i] = -(psi[j+1, i] - psi[j, i]) / dy
    v_face[j, i] = +(psi[j, i+1] - psi[j, i]) / dx

gives a field whose discrete divergence is zero on any grid, so a
flux-form transport operator built on it conserves mass exactly. `psi` is
authoritative. The cell-centred `u` and `v` in the same file are the average
of the two adjacent face values and are published as diagnostics only;
building transport from them introduces a spurious source of several per cent.

## 3. Boundaries

All four lateral boundaries are open. Inflow carries the fixed background
values in section 8; outflow carries the adjacent interior value, that is a
zero-gradient advective outflow. There is no flux of tracer through the top of
the slab other than the entrainment term in section 5.

## 4. Condensable organic vapour

`c` is a condensable organic vapour concentration in ug m-3:

    dc/dt + div(u c) = Kh * lap(c) + Yorg * Qev * q_org - c / tauv
                       - (we / h) * (c - c_bg)

`q_org` in `forcing.nc` is the published precursor-to-condensable-vapour
source rate in ug m-3 h-1, before the yield scale. `Qev` is the vapour
anomaly of section 7: it equals `q_event` on event days and 1 otherwise.
`Yorg`, `Kh`, `tauv` and `we` are unknown and shared across every episode.

## 5. Size-resolved particle number

Particle number is carried in 12 logarithmic diameter bins from 1.5 to 15 nm,
with exact edges in `size_bins.csv`. For each bin `j`, `n_j` in cm-3:

    dn_j/dt + div(u n_j) = Kh * lap(n_j) - n_j / taup
                           - (we / h) * (n_j - n_bg_j)
                           + S_j - d/dD (G n)|_j

The last term is the flux through diameter space caused by condensational
growth, written as a conservation law: in the absence of source and loss,
**your discretisation of it must conserve total particle number**, apart from
the flux out of the upper edge at 15 nm, where particles leave the represented
range. No flux enters through the lower edge at 1.5 nm.

`S_j` is the new-particle source. `forcing.nc` gives three disjoint region
masks A, B, C and a published bottom-up production rate `npf_rate` per region
in cm-3 h-1. The rate that actually applies inside region `r` is

    npf_rate[r] * s_r * Sev

where `s_A`, `s_B`, `s_C` are unknown regional strengths and `Sev` is the
nucleation anomaly of section 7: it equals `s_event` on event days and 1
otherwise. Inside a region mask, 70 per cent of the applied rate enters bin 0
and 30 per cent enters bin 1. `taup` is unknown and shared.

Coagulation is not represented.

## 6. Growth law

    G(D, T, c, C_H2SO4) = g_acid * C_H2SO4
                          + betaorg * c * f_T(T) * f_K(D, T)

    f_T(T) = exp[ -(Heff * 1000) / R * (1/T - 1/T_ref) ]
    f_K(D, T) = exp[ -(D_kelvin / D) * (T_ref / T) ]

with `G` in nm h-1 and the public constants

| constant | value |
| --- | --- |
| `g_acid` | 0.05 nm h-1 per 1e6 cm-3 |
| `T_ref` | 298.15 K |
| `R` | 8.314462618 J mol-1 K-1 |
| `D_kelvin` | 1.0 nm |

`C_H2SO4` is supplied as a known gridded field in `forcing.nc` in units of
1e6 cm-3, so sulfur chemistry is not part of the inverse problem. `betaorg`,
in nm h-1 per ug m-3, and `Heff`, in kJ mol-1, are unknown and shared.

## 7. Event days and the twelve unknown parameters

Four of the eight episodes are **event days**, flagged by `event_day = 1` in
`forcing.nc`, in `met.nc`, in `episode_index.json` and in
`data_manifest.json`: E02, E04, E06 and E08. On an event day, and for its
whole duration, the nucleation anomaly `s_event` multiplies every regional
new-particle rate and the vapour anomaly `q_event` multiplies the precursor
source. On the other four days both multipliers are exactly 1.

The unknowns are twelve natural-log parameters, one value each, shared across
every episode, in the fixed order of `priors.json`:

| # | id | physical quantity | unit |
| --- | --- | --- | --- |
| 0 | `log_Kh` | horizontal eddy diffusivity | m2 s-1 |
| 1 | `log_we` | entrainment velocity | m s-1 |
| 2 | `log_tauv` | condensable-vapour lifetime | h |
| 3 | `log_Yorg` | organic vapour yield scale | 1 |
| 4 | `log_betaorg` | organic growth coefficient | nm h-1 per ug m-3 |
| 5 | `log_Heff` | effective temperature-sensitivity enthalpy | kJ mol-1 |
| 6 | `log_taup` | particle-loss timescale | h |
| 7 | `log_sA` | strength of source region A | 1 |
| 8 | `log_sB` | strength of source region B | 1 |
| 9 | `log_sC` | strength of source region C | 1 |
| 10 | `log_s_event` | nucleation anomaly on event days | 1 |
| 11 | `log_q_event` | vapour anomaly on event days | 1 |

`priors.json` gives the Gaussian prior in these log coordinates: mean `mu`,
standard deviations `sigma`, the full correlation and covariance matrices, and
hard bounds. **The covariance is not diagonal**: the three regional strengths
share a nucleation-parameterisation calibration term, the yield scale and the
vapour lifetime are anticorrelated, and the growth coefficient and its
temperature sensitivity were fitted together.

The truth is a single draw from this prior. Draws were rejected until every
parameter lay within 1.5 prior standard deviations of its mean, the
diffusivity was below 12000 m2 s-1, and both event anomalies were positive
and non-negligible, so that event days really are days of enhanced nucleation
and enhanced condensable vapour. The seed and the accepted draw index are not
disclosed.

Note that the vapour equation contains no particle parameter. Vapour therefore
constrains `Kh`, `we`, `tauv`, `Yorg` and `q_event` without reference to the
particle field at all; the particle field depends on all twelve.

## 8. Backgrounds

Episode-independent, and used both as the lateral inflow value and as the
entrainment reservoir:

| quantity | value |
| --- | --- |
| `c_bg` | 0.06 ug m-3 |
| `n_bg_j` | 0.8 cm-3 in every bin |

## 9. Numerical accuracy

Your transport operator's own numerical diffusion adds to `Kh` and will bias
it if it is large. First-order upwind advection on a 5 km grid at 6 m s-1
carries an effective diffusivity of order `u*dx/2`, which is thousands of
m2 s-1. Two published checks let you measure your own scheme:

- **Uniform tracer.** Set the tracer to a constant equal to the background and
  advect it through any episode. With face winds from `psi` and a
  divergence-free flux form it must stay constant. A dimension-split flux
  update does not have that property.
- **Gaussian plume.** In a constant wind with `Kh` constant and no sources or
  losses, an initially Gaussian blob of variance `sigma0^2` satisfies
  `sigma^2(t) = sigma0^2 + 2*Kh*t`. Fitting that line and subtracting `2*Kh*t`
  gives your scheme's numerical diffusivity directly.

### Time step

The reference discretisation of this system is converged at a 60 s step and
is not converged at 120 s: halving the step from 60 s to 30 s changes the
withheld-station predictions by less than the measurement uncertainty, while
doubling it to 120 s degrades them by a large factor on every graded quantity.
A Courant number alone does not tell you this, because the growth term moves
particles through diameter bins that are narrow at the small end. Whatever
scheme you choose, verify convergence against your own step before you trust
an inversion built on it.

## 10. Observation operator

Each station reading is produced from the model field by, in order:

1. **Bilinear interpolation** in `x` and `y` to the station position in
   `stations.csv`, for the vapour field and for each diameter bin separately.
2. **Station sizing operator.** Every station has its own 12 by 12 matrix
   `sizing_operator[station, channel, bin]` in `sizing_operators.nc`, and
   reported counts are `sizing_operator @ n_true`. The matrix is the product
   of a size-dependent counting efficiency (low in the smallest channels,
   different at every station) and a row-stochastic broadening into
   neighbouring channels with a station-specific width; both factors are also
   published separately. Using one station's operator for another station is
   a modelling error.
3. **Error.** Errors are Gaussian in natural-log space, with covariance
   blocks published in `error_covariance.nc` that depend only on the station.
   For vapour, one 41 by 41 block covers the 41 record times of a deployment,
   one station in one episode. For counts, one 492 by 492 block covers the
   41 times by 12 channels of a deployment, flattened as
   `time_index * 12 + channel`. Each block is white noise plus a first-order
   autoregressive term in time plus a bias term shared by every entry of the
   deployment, which for counts means every channel and every time. Errors
   are independent between deployments and between vapour and counts. The
   realisation is not disclosed; the full covariance is.
4. **Detection limit and flags.** A channel whose true post-operator count is
   below 5 cm-3 is reported as 0 with `qc = 0`; usable channels have `qc = 1`;
   withheld entries carry the fill value -999 with `qc = -1`. The covariance
   block of a deployment applies to its usable channels, that is to the
   corresponding rows and columns.

No temporal averaging is applied: each reported value is an instantaneous
sample at the record time. There is no other smoothing or filtering.

**Log floors.** Wherever a model value enters a natural logarithm, for the
likelihood, the chi-square or a predictive spread, counts are first clipped
from below at 0.05 cm-3 and vapour at 0.001 ug m-3. These two floors are
part of the definition of the calibration chi-square.

## 11. Air-mass history diagnostics

Two lineage quantities are defined per station, time and diameter bin. Both
refer only to particles injected by the new-particle source fields; entrained
and inflowing background particles take no part in either.

Let `n^r_j` be the number in bin `j` that was injected inside source region
`r`, transported by the same operator as `n_j` with zero background and zero
entrainment gain, and let `N_j = sum_r n^r_j`.

- **Source fractions.** `source_fraction_r = n^r_j / N_j`, so the three
  fractions sum to one.
- **Mean particle age.** Carry the age moment `A_j`, which obeys the same
  transport, loss and size-space growth as `N_j` with the extra source term
  `+N_j` per unit time and zero injection at birth. Then
  `mean_particle_age_hr = A_j / N_j`, in hours since injection.

Both are sampled at the station with the same bilinear interpolation as the
fields, and **before** the station sizing operator.

## 12. The event statistic and its decomposition

The grown-particle statistic of an episode is the domain mean, over every
cell of the domain, of the total number in diameter bins 5 to 11 (all
particles larger than 3.92 nm), averaged over observation records 24 to 40
inclusive (the last four hours). It is in cm-3 and is a property of the model
state, not of any station. The **event statistic** `Q` is the mean of that
quantity over the four event days E02, E04, E06 and E08.

Group the event controls into a nucleation group `E` (`log_s_event`) and a
condensation group `C` (`log_q_event`). Switching a group off means setting
its log-parameter to zero; every other parameter keeps its inferred value and
every run is a full re-integration of the system. With

    Q00 = Q(E off, C off)   Q10 = Q(E on,  C off)
    Q01 = Q(E off, C on )   Q11 = Q(E on,  C on )

the order-averaged decomposition is

    A_E = 0.5 * [ (Q10 - Q00) + (Q11 - Q01) ]
    A_C = 0.5 * [ (Q01 - Q00) + (Q11 - Q10) ]

which satisfies `A_E + A_C = Q11 - Q00` exactly. The statistic is not
additive in the two groups, because the number that survives to the grown
channels depends on how fast the particles grow, so a linearisation around
the optimum is not the same quantity.

## 13. What is observed and what is withheld

| | |
| --- | --- |
| observed | stations S1 to S5 in episodes E01 to E06 |
| withheld, spatially | stations W1 and W2, in every episode |
| withheld, meteorologically | every station in E07 and E08 |

Meteorology and forcing are supplied for all eight episodes. E07 and E08
recombine wind speed, temperature and boundary-layer ranges that appear among
E01 to E06; they contain no new physics and no forcing outside the range you
can see. E08 is an event day whose anomalies are the same two numbers that
act on E02, E04 and E06.

## 14. Verification

Scoring compares your withheld vapour and size-distribution predictions and
your lineage diagnostics against truth from an independent seeded simulator
run at finer resolution than the public grid, using tolerances that combine
the published error model with a representation-error floor of about 5 per
cent in log space. Your predictive spreads are scored against the withheld
noisy observations themselves. Your reported parameters are re-run through a
trusted forward model, which must reproduce the predictions you submit, your
fitted values at the visible observations, and the four counterfactual
statistics you report; the chi-square you report is recomputed from your own
fitted values with the full published covariance. Your attribution and its
uncertainty are compared with the generator's own counterfactual runs.
