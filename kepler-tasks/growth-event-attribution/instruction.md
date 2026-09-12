# Attributing a nanoparticle growth event under regional transport

The synthetic regional observing system in /app/data covers eight 10-hour episodes of new-particle formation and growth in a 300 km by 240 km boundary layer: gridded meteorology, prescribed precursor and nucleation forcing, and sparse station records of condensable organic vapour and nanoparticle size distributions. Four of the eight episodes are event days. On those days the grown-particle population is enhanced, and the enhancement can come from two places: more nucleation, or more condensable vapour that grows the freshly nucleated particles to larger sizes before they are lost. Your job is to separate those two causes, with an honest statement of how well the data can separate them, and to say how much of that uncertainty comes from the poorly known strength of the three nucleation source regions.

The station records are Eulerian, so different air parcels move past each instrument over time. A change in modal diameter at a fixed station is not condensational growth alone; it can be transport, mixing, or one plume replacing another. The two causes of the event enhancement interact through survival: how many particles reach the grown size range depends on how fast they grow, so the decomposition is a property of the full coupled system and not of one-at-a-time perturbations around a fitted optimum.

Twelve unknown log-parameters control transport, dilution, vapour lifetime and yield, condensational growth, particle loss, the strength of each nucleation source region and the two event-day anomalies. The same values apply to every episode. A Gaussian prior on all twelve is published and is not diagonal. Observation errors are correlated in time and carry a per-deployment bias, and their full covariance is published. Each station has its own sizing operator. The data are synthetic, generated from the equations in the specification with one draw of the parameters from the prior and one draw of the errors from the published covariance.

## Inputs

`/app/data/model_spec.md` is the scientific contract: coordinates, the wind convention, the governing equations, the event-day multipliers, the growth law and its public constants, the backgrounds, the observation operators and error model, the exact definition of the lineage diagnostics, and the definition of the event statistic and its counterfactual decomposition. `/app/data/data_manifest.json` lists every file, its checksum and the observed/withheld split.

Also under `/app/data`: `priors.json`, `stations.csv`, `size_bins.csv`, `sizing_operators.nc`, `error_covariance.nc`, `prediction_index.csv`, `calibration_index.csv`, `history_queries.csv`, `episode_index.json`, and `episodes/E01` through `episodes/E08`, each holding `met.nc`, `forcing.nc` and `observations.nc`.

## What you are asked for

Infer the twelve parameters from the visible observations and the published prior, with a posterior you would defend. Then use the inferred system to answer three questions.

**How large is the event enhancement, and what caused it?** The event statistic `Q` is defined in section 12 of the specification: the domain-mean number of particles larger than 3.92 nm over the last four hours, averaged over the four event days. Group the controls into a nucleation group `E` (`log_s_event`) and a condensation group `C` (`log_q_event`). Switching a group off means setting its log-parameter to zero while every other parameter keeps its inferred value; every counterfactual is a full re-integration. With

```
Q00 = Q(E off, C off)   Q10 = Q(E on,  C off)
Q01 = Q(E off, C on )   Q11 = Q(E on,  C on )
```

report the order-averaged decomposition

```
A_E = 0.5 * [ (Q10 - Q00) + (Q11 - Q01) ]
A_C = 0.5 * [ (Q01 - Q00) + (Q11 - Q10) ]
```

which satisfies `A_E + A_C = Q11 - Q00` exactly, together with the posterior standard deviation of each term.

**How much do the source strengths cost you?** Repeat the uncertainty analysis with `log_sA`, `log_sB` and `log_sC` held fixed at their prior means and everything else free, and report the posterior standard deviations of `A_E` and `A_C` from that restricted analysis alongside those from the full joint analysis. Report the posterior correlation between `log_s_event` and `log_sA` as well.

**What do the observations say where you cannot see them?** Predict vapour and the post-operator size distribution at every row of `prediction_index.csv` (the two withheld stations in the six observed episodes, and every station in the two withheld episodes), each with a predictive standard deviation in natural-log space for the noisy observation, not just a point value. Reconstruct the air-mass history behind every row of `history_queries.csv`.

Say what the data cannot determine. Some directions in this parameter space stay close to the prior, and a tight interval on one of them is a worse answer than a wide one.

## Deliverables

Write exactly these four files. Paths are absolute; the directory `/app/output` exists.

`/app/output/posterior.json`

```json
{
  "param_ids": ["log_Kh", "..."],
  "estimate": [12 numbers], "posterior_sd": [12 numbers],
  "ci_lower": [12 numbers], "ci_upper": [12 numbers],
  "posterior_prior_sd_ratio": [12 numbers],
  "dof_signal": 0.0,
  "chi2_calibration": 0.0,
  "chi2_by_stream": {"vapour": 0.0, "counts": 0.0}
}
```

`param_ids` must equal the order in `priors.json`. `estimate` is your posterior estimate in log coordinates, `ci_lower` and `ci_upper` the central 95 per cent credible interval, `posterior_prior_sd_ratio` each posterior standard deviation divided by the prior one, and `dof_signal` is `trace(I - Sigma_posterior B^-1)` with `B` the prior covariance. `chi2_calibration` is `r^T R^-1 r` over every usable visible observation at your estimate, in natural-log space with the log floors of the specification, with the full published block covariance `R` restricted to the usable channels; `chi2_by_stream` splits it into the vapour and the count blocks.

`/app/output/predictions.npz` with six arrays. Four are indexed by `row_id` of `prediction_index.csv`, in that order: `vapour` of shape `(n_rows,)` in ug m-3, `vapour_log_sd` of shape `(n_rows,)`, `pnsd` of shape `(n_rows, 12)` in cm-3 after the station's sizing operator, and `pnsd_log_sd` of shape `(n_rows, 12)`. The two `_log_sd` arrays are the predictive standard deviation of the natural log of the noisy observation, including observation error. Two are indexed by `row_id` of `calibration_index.csv`, the visible station-times: `fit_vapour` of shape `(n_cal,)` and `fit_pnsd` of shape `(n_cal, 12)`, your model's values at your estimate, in the same units and after the station operator; these are the values your `chi2_calibration` is computed from.

`/app/output/airmass_history.csv` with one row per row of `history_queries.csv` and columns `query_id, mean_particle_age_hr, source_fraction_A, source_fraction_B, source_fraction_C`. The three fractions must sum to one.

`/app/output/attribution.json`

```json
{
  "units": "cm-3",
  "Q00": 0.0, "Q10": 0.0, "Q01": 0.0, "Q11": 0.0,
  "A_E": 0.0, "A_C": 0.0,
  "A_E_sd": 0.0, "A_C_sd": 0.0,
  "A_E_sd_sources_fixed": 0.0, "A_C_sd_sources_fixed": 0.0,
  "corr_s_event_sA": 0.0
}
```

Every numeric field must be finite. Further keys are ignored.

## How the result is judged

A sealed verifier applies these checks, and every one must pass:

1. **Schema and bounds.** Every file present with the right shapes and ordering, all values finite, spreads positive, fractions normalised, the estimate inside the published bounds.
2. **Withheld predictions.** Vapour, size distributions, a number-weighted diameter derived from them, mean particle age and source fractions are compared with an independent simulation of the same system at finer resolution, under tolerances that combine the published error model with a representation-error floor. Both withheld episodes must pass on their own.
3. **Predictive calibration.** Your 90 per cent predictive intervals must cover the withheld noisy observations at a rate between 80 and 97 per cent, for vapour and for counts separately, and the mean Gaussian log score must beat a threshold that penalises intervals that are too wide as well as too narrow.
4. **Posterior honesty.** The posterior-to-prior standard-deviation ratio is checked for three parameters against bands set from what these data support, and `dof_signal` must lie in a band. Point values are not graded against the truth beyond the published bounds, because two correct solvers with different numerical diffusion land at different points along the same valley while predicting the observations equally well.
5. **Attribution.** The four statistics must match the verifier's own recomputation from your estimate, `A_E + A_C` must equal `Q11 - Q00`, `A_E` and `A_C` must each lie within a tolerance of the generator's counterfactual truth, and their reported standard deviations must lie in a band. A decomposition built from a linearisation, or one that never lets the anomalies move, does not survive this.
6. **Sensitivity.** The restricted standard deviations must be smaller than the joint ones, the implied variance share `1 - (sd_fixed / sd_full)^2` for `A_E` must lie in a band, and the reported correlation must lie in a band.
7. **Forward consistency.** Your estimate is re-run through a trusted forward model, which must reproduce the predictions and fitted values you submitted and the counterfactual statistics you reported, and your reported chi-square must equal the one recomputed from your own fitted values with the full published covariance. A fit performed with a diagonal covariance that honestly reports the chi-square it computed does not survive this.

The numerical method, the internal grid, the optimiser, the uncertainty method and the language are yours to choose, so long as they implement the system in `model_spec.md`. Partial credit does not exist.

You have 28800 seconds to complete this task. Do not cheat by using online solutions or hints specific to this task.
