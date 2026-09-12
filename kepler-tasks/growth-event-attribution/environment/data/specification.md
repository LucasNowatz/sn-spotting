# System specification

This document fixes the physics, the operators, the error model and every
definition the verifier relies on. Numerics, optimiser and uncertainty method
are yours.

## A. Grid and time

Cartesian metres, `x` east and `y` north, a 280 km by 200 km domain. The
published grid has 70 by 50 cells of 4 km with centres in `x` and `y` of
`meteorology.nc`; any internal grid is allowed. The atmosphere is one
well-mixed slab of depth `h(x, y, t)` (`mixed_layer_depth`); nothing is
resolved vertically.

Each episode lasts 8 h with 49 records every 10 min. Every field in
`meteorology.nc` and `sources.nc` is **linear in time between records**, and
station records sit on the same 49 times.

## B. Winds

The flow is defined by the corner `streamfunction` psi (`time, y_corner,
x_corner`), with `u = -dpsi/dy` and `v = +dpsi/dx`. Face-normal winds taken
as the discrete curl,

    u_face[j, i] = -(psi[j+1, i] - psi[j, i]) / dy
    v_face[j, i] = +(psi[j, i+1] - psi[j, i]) / dx

are exactly non-divergent on any grid, so a flux-form scheme built on them
conserves mass. `u_centre` and `v_centre` are face averages published for
inspection only; transport built on them carries a spurious source of a few
per cent.

Lateral boundaries are open: inflow faces carry the backgrounds of section
G, outflow faces the adjacent interior value. The slab exchanges tracer with
the free troposphere only through the entrainment term.

## C. Condensable vapour

`c` in ug m-3:

    dc/dt + div(u c) = Kh lap(c) + Yorg * Qev * P - c/tauv - (we/h) (c - c_bg)

`P` is `precursor_rate` in `sources.nc` (ug m-3 h-1). `Qev` is the vapour
anomaly of section F: `q_event` on event days, 1 otherwise. `Kh`, `we`,
`tauv` and `Yorg` are unknown and common to every episode.

## D. Particle number

Twelve logarithmic bins from 1.5 to 15 nm (`diameter_bins.csv`). For bin j,
`n_j` in cm-3:

    dn_j/dt + div(u n_j) = Kh lap(n_j) - n_j/taup - (we/h) (n_j - n_bg)
                           + S_j - d/dD (G n)|_j

The last term is condensational growth as a conservation law through
diameter space. With sources and losses off, **your discretisation must
conserve total number** except for the flux leaving through the 15 nm edge;
nothing enters at 1.5 nm.

`S_j` comes from `sources.nc`: three disjoint `region_mask`s A, B, C and a
published `nucleation_rate` per region (cm-3 h-1). Inside region r the rate
that applies is

    nucleation_rate[r] * s_r * Sev

with `s_A`, `s_B`, `s_C` unknown regional strengths and `Sev` the nucleation
anomaly of section F (`s_event` on event days, 1 otherwise); 70 per cent
enters bin 0 and 30 per cent bin 1. `taup` is unknown and common. There is no
coagulation.

## E. Growth rate

    G = g_acid * C_acid + betaorg * c * f_T(T) * f_K(D, T)      [nm h-1]
    f_T = exp(-(Heff * 1000)/R * (1/T - 1/T_ref))
    f_K = exp(-(D_kelvin / D) * (T_ref / T))

| constant | value |
| --- | --- |
| `g_acid` | 0.05 nm h-1 per 1e6 cm-3 |
| `T_ref` | 298.15 K |
| `R` | 8.314462618 J mol-1 K-1 |
| `D_kelvin` | 1.0 nm |

`C_acid` is `sulfuric_acid` in `sources.nc` (1e6 cm-3), a known field.
`betaorg` (nm h-1 per ug m-3) and `Heff` (kJ mol-1) are unknown and common.

## F. Event days and the unknowns

E02, E03, E06 and E08 are event days (`event_day = 1` in both NetCDF files
and in `episodes.json`). Throughout an event day `s_event` multiplies every
regional nucleation rate and `q_event` multiplies the precursor rate. On the
other days both are exactly 1.

The unknowns are twelve natural logs, one value each for all episodes, in the
order of `prior.json`:

| # | id | quantity | unit |
| --- | --- | --- | --- |
| 0 | `log_Kh` | eddy diffusivity | m2 s-1 |
| 1 | `log_we` | entrainment velocity | m s-1 |
| 2 | `log_tauv` | vapour lifetime | h |
| 3 | `log_Yorg` | vapour yield scale | 1 |
| 4 | `log_betaorg` | organic growth coefficient | nm h-1 per ug m-3 |
| 5 | `log_Heff` | temperature sensitivity of growth | kJ mol-1 |
| 6 | `log_taup` | particle loss timescale | h |
| 7 | `log_sA` | strength of region A | 1 |
| 8 | `log_sB` | strength of region B | 1 |
| 9 | `log_sC` | strength of region C | 1 |
| 10 | `log_s_event` | nucleation anomaly, event days | 1 |
| 11 | `log_q_event` | vapour anomaly, event days | 1 |

`prior.json` gives the Gaussian prior: mean, standard deviations, the full
correlation and covariance (**not diagonal**: the three regional strengths
share a calibration term, lifetime and yield are anticorrelated, growth
coefficient and its temperature sensitivity are correlated) and hard bounds.
The truth is a single draw from this prior, accepted only if every parameter
lay within 1.8 prior standard deviations of its mean, the diffusivity was
below 11000 m2 s-1, the nucleation anomaly was at least 0.4 and the vapour
anomaly at least 0.3 in log, so that event days are unmistakably events.
Seed and draw index are not disclosed.

The vapour equation contains no particle parameter: vapour alone constrains
`Kh`, `we`, `tauv`, `Yorg` and `q_event`.

## G. Backgrounds

`c_bg = 0.06 ug m-3` and `n_bg = 0.8 cm-3 in every bin`, used as the inflow
value and as the entrainment reservoir, in every episode.

## H. Numerical fidelity

Numerical diffusion adds to `Kh`. First-order upwind at 4 km and 6 m s-1 is
worth thousands of m2 s-1 and would swamp the parameter. Two checks let you
measure your own scheme: a uniform field advected with face winds from psi
must stay uniform (dimension-split updates fail this at open boundaries), and
a Gaussian blob in a constant wind with constant `Kh` and no sources or
losses must obey `sigma^2(t) = sigma0^2 + 2 Kh t`, the excess being your
scheme's diffusivity.

On the published grid the growth term, not the Courant number, sets the
step: the smallest bins are about 0.3 nm wide. A converged limited flux-form
scheme is converged at 60 s (30 s changes nothing beyond measurement
uncertainty) and badly under-resolved at 120 s. Whatever you use, check
convergence against your own step before trusting an inversion.

## I. Station operator and error model

A station reading is produced from the model field by, in order:

1. bilinear interpolation to the station position in `network.csv`, for the
   vapour and for each bin;
2. the station's own 12 by 12 `operator` from `instruments.nc`, so that
   `reported[channel] = sum_bin operator[channel, bin] * n_true[bin]`; it is
   the product of a size-dependent counting `efficiency` and a row-stochastic
   `broadening`, both also published, and differs between stations;
3. Gaussian errors in natural-log space with the covariance blocks of
   `error_model.nc`, which depend only on the station: `vapour_cov` is 49 by
   49 over the record times of one deployment (one station in one episode);
   `counts_cov` is 588 by 588 over the 49 times by 12 channels of a
   deployment, flattened as `time_index * 12 + channel`. Each block is white
   noise plus a first-order autoregressive term in time plus a bias shared by
   the whole deployment, for counts by every channel and time. Deployments
   are independent of one another, as are vapour and counts. The realisation
   is not disclosed;
4. a detection limit: a channel whose true reported count is below 4 cm-3 is
   written as 0 with `counts_flag = 0`; usable channels carry 1; withheld
   entries carry -999 and flag -1. A deployment's covariance applies to its
   usable rows and columns.

Values are instantaneous at the record time; nothing is averaged.

**Log floors.** Wherever a model value enters a logarithm (likelihood,
chi-square, predictive spread) counts are first clipped from below at
0.05 cm-3 and vapour at 0.001 ug m-3. The floors are part of the definition
of the calibration chi-square.

## J. The event statistic, its decomposition and its regional shares

The **grown number** of an episode is the domain mean, over all cells, of
the number in bins 5 to 11 (particles above 3.92 nm), averaged over records
30 to 48 (the last three hours). The **event statistic** `Q` is the mean of
the grown number over the four event days, in cm-3.

Two control groups: `E` = {`log_s_event`}, `C` = {`log_q_event`}. Switching
a group off sets its log-parameter to zero; every other parameter keeps its
inferred value and every run is a complete re-integration. With

    Q00 = Q(E off, C off)   Q10 = Q(E on,  C off)
    Q01 = Q(E off, C on )   Q11 = Q(E on,  C on )

the order-averaged decomposition is

    A_E = 0.5 [ (Q10 - Q00) + (Q11 - Q01) ]
    A_C = 0.5 [ (Q01 - Q00) + (Q11 - Q10) ]

with `A_E + A_C = Q11 - Q00` exactly. The statistic is not additive in the
two groups: the number that survives into the grown bins depends on how fast
it grows, so a linearisation around the optimum is a different quantity.

**Region shares.** Let `n^r_j` be the number in bin j that was injected in
region r, transported by the same operator as `n_j` with zero background
and no entrainment gain. For an event day, the domain share of region r is
the domain mean of `sum_{j>=5} n^r_j` over records 30 to 48 divided by the
grown number of that day; the shares of A, B and C sum to at most one, the
remainder being background particles. The station share at a withheld
station is the same ratio formed from the bilinear samples of the true fields
at that station (before the operator), averaged over the same records.

## K. Observed and withheld

| | |
| --- | --- |
| observed | S1 to S6 in E01 to E06 |
| withheld stations | W1 and W2, every episode |
| withheld episodes | every station in E07 and E08 |

Meteorology and sources are published for all eight episodes. E07 and E08
recombine flow, temperature and mixed-layer ranges that occur in E01 to E06;
E08 is an event day governed by the same two anomalies as E02, E03 and E06.

## L. How submissions are scored

Withheld vapour and size distributions, a number-weighted diameter derived
from them, and the region shares are compared with a seeded simulation of
the same equations by a separate implementation with a shorter step on the
published grid, under tolerances combining the error model above with a
discretisation floor of about 5 per cent (vapour) and 6 per cent (counts) in
log space, measured so that a correct scheme with a different flux limiter
passes. Predictive spreads are scored against the withheld noisy records.
The submitted estimate is re-run through a sealed model that must reproduce
the submitted predictions, fitted values and counterfactual statistics; the
reported chi-square is recomputed from the submitted fitted values with the
full published covariance.
