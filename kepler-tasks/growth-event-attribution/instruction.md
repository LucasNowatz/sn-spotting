# Nucleation or condensation: attributing a growth event

/app/data holds a synthetic regional observing system: eight 8-hour episodes of new-particle formation and growth in a 280 km by 200 km mixed layer, with gridded meteorology, published precursor and nucleation forcing, and station records of condensable vapour and nanoparticle size distributions from six stations on six days. Two further stations and two further days are withheld. On four event days the population of particles that have grown past 3.9 nm is larger than usual. Two mechanisms can produce that: stronger nucleation, or more condensable vapour that grows fresh particles into the larger bins before they are lost. Your job is to say how much of the enhancement each mechanism is responsible for, how confidently the data support that split, how much of the uncertainty is caused by the poorly known strength of the three nucleation regions, and which regions the grown particles came from.

The records are Eulerian. Air masses replace one another at a fixed station, so a shift in the modal diameter there is not growth alone. The two mechanisms also interact: survival into the grown bins depends on growth rate, so the decomposition belongs to the coupled system and cannot be read off one-at-a-time perturbations around a fitted optimum.

Twelve unknown log-parameters, common to every episode, govern mixing, dilution, vapour lifetime and yield, growth, particle loss, the strength of each source region, and the two event-day anomalies. Their Gaussian prior is published and is not diagonal. Station errors are correlated in time and carry a per-deployment bias; their full covariance is published, and every station has its own sizing operator. The truth is one draw of the parameters from the prior and one draw of the errors from that covariance.

## Inputs

`/app/data/specification.md` fixes everything the verifier relies on: coordinates, winds, equations, event-day multipliers, growth law, backgrounds, operators, error model, log floors, the event statistic, its decomposition, and the region shares. `/app/data/dataset_manifest.json` lists the files with checksums and the observed/withheld split.

Also in `/app/data`: `prior.json`, `network.csv`, `diameter_bins.csv`, `instruments.nc`, `error_model.nc`, `withheld_index.csv`, `calibration_index.csv`, `episodes.json`, and `episodes/E01` to `E08` each with `meteorology.nc`, `sources.nc` and `station_records.nc`.

## Questions to answer

Infer the twelve parameters from the visible records and the prior, with a posterior you would stand behind, then:

**Decompose the event.** With `Q` the event statistic of specification section J, the nucleation group `E` = {`log_s_event`} and the condensation group `C` = {`log_q_event`}, switch each group off by zeroing its log-parameter while everything else keeps its inferred value, re-integrate the system in full for each of the four combinations, and report

```
Q00 = Q(E off, C off)   Q10 = Q(E on,  C off)
Q01 = Q(E off, C on )   Q11 = Q(E on,  C on )
A_E = 0.5 [ (Q10 - Q00) + (Q11 - Q01) ]
A_C = 0.5 [ (Q01 - Q00) + (Q11 - Q10) ]
```

with `A_E + A_C = Q11 - Q00` exactly, and the posterior standard deviation of `A_E` and `A_C`.

**Price the source strengths.** Redo the uncertainty analysis with `log_sA`, `log_sB`, `log_sC` held at their prior means and all else free, and report the standard deviations of `A_E` and `A_C` from that restricted analysis next to the joint ones, plus the posterior correlation between `log_s_event` and `log_sA`.

**Predict what was withheld.** For every row of `withheld_index.csv` give vapour and the reported size distribution with a predictive standard deviation, in natural-log space, for the noisy record. For every event day give the region shares of the grown number, domain-wide and at W1 and W2, as defined in section J.

Where the data leave a direction close to the prior, say so with a wide interval rather than a narrow one.

## Deliverables

Exactly these four files; `/app/output` exists.

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

`param_ids` in the order of `prior.json`; `estimate` in log coordinates; `ci_lower`/`ci_upper` the central 95 per cent credible interval; `posterior_prior_sd_ratio` each posterior sd over the prior sd; `dof_signal = trace(I - Sigma_post B^-1)` with `B` the prior covariance; `chi2_calibration = r^T R^-1 r` over every usable visible value at your estimate, in log space with the specification's floors, with the full published covariance restricted to usable channels; `chi2_by_stream` its vapour and count parts.

`/app/output/predictions.npz` with six arrays. By `row_id` of `withheld_index.csv`: `vapour` `(n,)` ug m-3, `vapour_log_sd` `(n,)`, `pnsd` `(n, 12)` cm-3 after the station operator, `pnsd_log_sd` `(n, 12)`; the `_log_sd` arrays are the predictive sd of the log of the noisy record, observation error included. By `row_id` of `calibration_index.csv`: `fit_vapour` `(m,)` and `fit_pnsd` `(m, 12)`, your model at your estimate at the visible station-times, same units, after the operator; `chi2_calibration` is computed from these.

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

`/app/output/region_shares.json`

```json
{
  "domain":   {"E02": {"A": 0.0, "B": 0.0, "C": 0.0}, "E03": {...}, "E06": {...}, "E08": {...}},
  "stations": {"E02": {"W1": {"A": 0.0, "B": 0.0, "C": 0.0}, "W2": {...}}, "E03": {...}, "E06": {...}, "E08": {...}}
}
```

Shares are non-negative and sum to at most one per vector. All numbers must be finite; extra keys are ignored.

## Grading

A sealed verifier applies twelve checks and all must pass:

1. Schema and bounds: files, shapes, order, finiteness, positive spreads, the estimate inside the published bounds.
2. Withheld vapour, size distributions and a number-weighted diameter derived from them, against a finer-resolution seeded simulation, within tolerances built from the error model and a representation floor; both withheld episodes must pass on their own.
3. Region shares, domain-wide and at the withheld stations, against tagged-tracer truth.
4. Predictive calibration: 90 per cent intervals must cover the withheld noisy records at the stated rate, vapour and counts separately, and a Gaussian log score must beat a threshold that punishes hedging as well as overconfidence.
5. Posterior: width ratios for three parameters and `dof_signal` within bands set from what these data support. Point values are graded only through the bounds, since correct solvers with different numerical diffusion land at different points of one valley.
6. Attribution: the four statistics must agree with a sealed re-run from your estimate, `A_E + A_C` must equal `Q11 - Q00`, `A_E` and `A_C` must lie within a tolerance of the generator's own counterfactuals, and their spreads within a band. A linearised split, or one that never moves the anomalies, fails.
7. Sensitivity: restricted spreads smaller than joint ones, the variance share `1 - (sd_fixed/sd_full)^2` of `A_E` within a band, and the correlation within a band.
8. Sealed re-run: predictions and fitted values must match a re-run from your estimate, and your reported chi-square must equal the value recomputed from your fitted values with the full covariance. A diagonal-covariance fit that reports its own chi-square fails.

Numerics, grid, optimiser, uncertainty method and language are free, provided they implement `specification.md`. There is no partial credit.

You have 28800 seconds to complete this task. Do not cheat by using online solutions or hints specific to this task.
