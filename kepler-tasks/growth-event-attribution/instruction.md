# Attributing a nanoparticle growth event under regional transport

The synthetic regional observing system in /app/data covers eight 10-hour episodes of new-particle formation and growth in a 300 km by 240 km boundary layer: gridded meteorology, prescribed precursor and nucleation forcing, and sparse station records of condensable organic vapour and nanoparticle size distributions. Four episodes are event days on which the grown-particle population is enhanced. The enhancement can come from more nucleation or from more condensable vapour that grows fresh particles to larger sizes before they are lost. Separate those two causes, state honestly how well the data separate them, and say how much of that uncertainty comes from the poorly known strength of the three nucleation source regions.

Station records are Eulerian: different air parcels pass each instrument over time, so a change in modal diameter at a station is not growth alone. The two causes interact through survival, since how many particles reach the grown size range depends on how fast they grow, so the decomposition is a property of the full coupled system and not of one-at-a-time perturbations around a fitted optimum.

Twelve unknown log-parameters, shared across all episodes, control transport, dilution, vapour lifetime and yield, growth, particle loss, the strength of each source region and the two event-day anomalies. A non-diagonal Gaussian prior on all twelve is published. Observation errors are correlated in time with a per-deployment bias, and their full covariance is published. Each station has its own sizing operator. The data are synthetic: one draw of the parameters from the prior and one draw of the errors from the published covariance, run through the equations in the specification.

## Inputs

`/app/data/model_spec.md` is the scientific contract: coordinates, wind convention, governing equations, event-day multipliers, growth law and constants, backgrounds, observation operators and error model, the lineage diagnostics, and the event statistic with its counterfactual decomposition. `/app/data/data_manifest.json` lists every file, its checksum and the observed/withheld split.

Also under `/app/data`: `priors.json`, `stations.csv`, `size_bins.csv`, `sizing_operators.nc`, `error_covariance.nc`, `prediction_index.csv`, `calibration_index.csv`, `history_queries.csv`, `episode_index.json`, and `episodes/E01` to `episodes/E08`, each with `met.nc`, `forcing.nc` and `observations.nc`.

## What you are asked for

Infer the twelve parameters from the visible observations and the prior, with a posterior you would defend, then answer three questions.

**How large is the event enhancement, and what caused it?** The event statistic `Q` (specification section 12) is the domain-mean number of particles larger than 3.92 nm over the last four hours, averaged over the four event days. Group the controls into a nucleation group `E` (`log_s_event`) and a condensation group `C` (`log_q_event`). Switching a group off sets its log-parameter to zero with every other parameter at its inferred value; every counterfactual is a full re-integration. With

```
Q00 = Q(E off, C off)   Q10 = Q(E on,  C off)
Q01 = Q(E off, C on )   Q11 = Q(E on,  C on )
```

report

```
A_E = 0.5 * [ (Q10 - Q00) + (Q11 - Q01) ]
A_C = 0.5 * [ (Q01 - Q00) + (Q11 - Q10) ]
```

which satisfies `A_E + A_C = Q11 - Q00` exactly, with the posterior standard deviation of each term.

**How much do the source strengths cost you?** Repeat the uncertainty analysis with `log_sA`, `log_sB` and `log_sC` fixed at their prior means and everything else free, and report the standard deviations of `A_E` and `A_C` from that restricted analysis beside the joint ones. Report the posterior correlation between `log_s_event` and `log_sA` as well.

**What do the observations say where you cannot see them?** Predict vapour and the post-operator size distribution at every row of `prediction_index.csv`, each with a predictive standard deviation in natural-log space for the noisy observation. Reconstruct the air-mass history behind every row of `history_queries.csv`.

Say what the data cannot determine: a tight interval on a prior-dominated direction is a worse answer than a wide one.

## Deliverables

Write exactly these four files; `/app/output` exists.

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

`param_ids` must equal the order in `priors.json`. `estimate` is in log coordinates, `ci_lower`/`ci_upper` the central 95 per cent credible interval, `posterior_prior_sd_ratio` each posterior sd divided by the prior sd, and `dof_signal` is `trace(I - Sigma_posterior B^-1)` with `B` the prior covariance. `chi2_calibration` is `r^T R^-1 r` over every usable visible observation at your estimate, in log space with the specification's log floors, using the full published block covariance restricted to the usable channels; `chi2_by_stream` splits it into vapour and count blocks.

`/app/output/predictions.npz` with six arrays. Indexed by `row_id` of `prediction_index.csv`: `vapour` `(n_rows,)` in ug m-3, `vapour_log_sd` `(n_rows,)`, `pnsd` `(n_rows, 12)` in cm-3 after the station operator, `pnsd_log_sd` `(n_rows, 12)`; the `_log_sd` arrays are predictive standard deviations of the log of the noisy observation, observation error included. Indexed by `row_id` of `calibration_index.csv`: `fit_vapour` `(n_cal,)` and `fit_pnsd` `(n_cal, 12)`, your model's values at your estimate at the visible station-times, same units, after the operator; these are what `chi2_calibration` is computed from.

`/app/output/airmass_history.csv` with one row per row of `history_queries.csv` and columns `query_id, mean_particle_age_hr, source_fraction_A, source_fraction_B, source_fraction_C`. The fractions must sum to one.

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

A sealed verifier applies these checks; all must pass.

1. **Schema and bounds.** Files present with the right shapes and order, values finite, spreads positive, fractions normalised, the estimate inside the published bounds.
2. **Withheld predictions.** Vapour, size distributions, a number-weighted diameter derived from them, mean particle age and source fractions are compared with an independent finer-resolution simulation, under tolerances combining the published error model with a representation-error floor. Both withheld episodes must pass on their own.
3. **Predictive calibration.** Your 90 per cent intervals must cover the withheld noisy observations at 80 to 97 per cent, for vapour and counts separately, and the mean Gaussian log score must beat a threshold that penalises intervals too wide as well as too narrow.
4. **Posterior honesty.** The posterior-to-prior sd ratio is checked for three parameters against bands, and `dof_signal` must lie in a band. Point values are graded only through the bounds, because correct solvers with different numerical diffusion land at different points of the same valley.
5. **Attribution.** The four statistics must match the verifier's recomputation from your estimate, `A_E + A_C` must equal `Q11 - Q00`, `A_E` and `A_C` must lie within a tolerance of the generator's counterfactual truth, and their standard deviations within a band. A linearised decomposition, or one that never moves the anomalies, does not survive this.
6. **Sensitivity.** The restricted standard deviations must be smaller than the joint ones, the variance share `1 - (sd_fixed / sd_full)^2` for `A_E` must lie in a band, and so must the reported correlation.
7. **Forward consistency.** Your estimate is re-run through a trusted forward model that must reproduce your predictions, fitted values and counterfactual statistics, and your reported chi-square must equal the one recomputed from your fitted values with the full covariance. A diagonal-covariance fit that reports its own chi-square fails here.

The numerical method, internal grid, optimiser, uncertainty method and language are yours, so long as they implement `model_spec.md`. Partial credit does not exist.

You have 28800 seconds to complete this task. Do not cheat by using online solutions or hints specific to this task.
