# Limit calibration

Every limit clears two bars: comfortably above what a correct solution
achieves (the measured floor and the oracle's own score) and comfortably
below the cheapest shortcut that the gate exists to catch. `set_limits.py`
applies that mechanically and refuses to write `limits.json` if any accuracy
limit cannot satisfy both; its inputs are in `limit_inputs.json`.

## The achievable floor

`calibrate.py`: the reference solver on the public 4 km grid at the true
parameters against the generator's own run. No solution can do better;
the difference is discretisation error, not a modelling mistake.

| metric | 30 s | 60 s (reference) | 120 s |
| --- | --- | --- | --- |
| withheld vapour, log NRMSE | 2.08e-07 | 0.05994 | 0.1821 |
| withheld size distributions, weighted log RMSE | 3.074e-06 | 0.2932 | 0.9266 |
| number-weighted diameter MAE, nm | 1.121e-07 | 0.005513 | 0.02074 |
| domain region shares, L1 | 4.817e-08 | 0.002838 | 0.008598 |
| station region shares, L1 | 1.816e-08 | 0.002356 | 0.006943 |
| chi-square at the truth | 7929 | 8619 | 1.669e+04 |

30 s matches 60 s on every quantity; 120 s is under-resolved, as the
specification discloses.

Two other correct solvers were measured at the truth with the generator's own
code, so that no limit fails a solver for differing from the generator: the
monotonized-central limiter on the published grid, and Koren's limiter on a
2 km grid with a 30 s step. Every limit uses the worst of the three floors.

| metric | reference scheme | other limiter | 2 km grid |
| --- | --- | --- | --- |
| vapour | 0.05994 | 0.05645 | 0.03118 |
| size distributions | 0.2932 | 0.2801 | 0.4911 |
| diameter, nm | 0.005513 | 0.006784 | 0.01315 |
| domain shares | 0.002838 | 0.00805 | 0.02752 |
| station shares | 0.002356 | 0.003721 | 0.01777 |
| chi-square at the truth | 8619 | 8679 | 1.025e+04 |
| worst counterfactual difference, cm-3 | 0.47 | 0.86 | 13.89 |

Per episode at 60 s (vapour, size distributions):

- E01: 0.078, 0.218
- E02: 0.073, 0.208
- E03: 0.054, 0.161
- E04: 0.058, 0.152
- E05: 0.077, 0.113
- E06: 0.089, 0.462
- E07: 0.049, 0.164
- E08: 0.049, 0.365

Counterfactual statistics, reference solver at the truth against the generator's runs:

| statistic | reference at truth | generator | difference |
| --- | --- | --- | --- |
| Q00 | 388.23 | 388.51 | -0.28 |
| Q10 | 636.73 | 637.20 | -0.47 |
| Q01 | 555.48 | 555.61 | -0.13 |
| Q11 | 912.60 | 912.82 | -0.21 |
| A_E | 302.82 | 302.95 | -0.13 |
| A_C | 221.56 | 221.35 | +0.21 |

## The oracle

Vapour 0.110, size distributions 0.412, diameter 0.0093 nm,
domain shares 0.012, station shares 0.005; coverage
0.952 (vapour) and 0.930 (counts) with log scores -0.80 and -0.59;
dof 11.88; chi-square 8312.5. A_E 256.6 (truth 302.9),
sd 16.87, restricted 13.90, variance share 0.32; A_C 213.9
(truth 221.4), sd 3.29; corr(log_s_event, log_sA) -0.571.

Parameter z-scores against the truth: +14.1, +0.7, -1.8, +0.9, -0.0, -3.8, -0.2, +1.3, -0.1, +2.4, -2.5, -4.1.
Discretisation differences are absorbed into the parameters; see
DESIGN_NOTES.md for why point values are graded only through the bounds.

## Rules

Accuracy limits: the larger of a multiple of the floor and of the oracle
(vapour 4x/2.5x, size distributions 1.8x/1.8x, diameter 3x/2.5x, shares
3x/2.5x), below 0.40 to 0.70 of the nearest shortcut. Per-episode limits
1.6x the global ones, both withheld days alone. Re-run envelopes 2x the
global limits. Coverage lower bound min(0.80, oracle - 0.05), upper 0.97;
log-score limits oracle + 0.25; width-ratio bands one third to three times
the oracle's ratio for the tightest and two loosest parameters; dof band
oracle - 1.5 to 12; attribution tolerance the largest of 2x the oracle's
error, 4x the floor difference and 25% of the truth; spread band 0.5x to 2x
the oracle's; variance-share band oracle - 0.20 to + 0.15; correlation band
oracle +- 0.22; chi-square report tolerance 1%, ceiling the larger of 1.5x
the oracle and 1.3x the worst floor; statistic tolerance the largest of 4x
the oracle's re-run discrepancy, 1.5x the worst floor difference and 1% of
Q11.

## Limits

| limit | value |
| --- | --- |
| `attr_sd_band` | {"A_E": [8.436, 33.7442], "A_C": [1.6455, 6.582]} |
| `attr_tol` | {"A_E": 92.6645, "A_C": 55.3384} |
| `chi2_max` | 1.332e+04 |
| `chi2_report_tol` | 83.12 |
| `chi2_stream_report_tol` | 83.12 |
| `consistency_pnsd_max` | 1.768 |
| `consistency_vapour_max` | 0.5503 |
| `corr_band` | [-0.7905, -0.3505] |
| `corr_key` | corr_s_event_sA |
| `count_valid_min` | 4 |
| `coverage_counts` | [0.8, 0.97] |
| `coverage_vapour` | [0.8, 0.97] |
| `diam_min_total` | 40 |
| `diameter_max_nm` | 0.03945 |
| `dof_signal` | [10.3797, 12.0] |
| `domain_share_l1_max` | 0.08257 |
| `episode_pnsd_max` | 1.414 |
| `episode_vapour_max` | 0.4402 |
| `hard_episodes` | ["E07", "E08"] |
| `log_score_counts_max` | -0.3423 |
| `log_score_vapour_max` | -0.5525 |
| `median_sd_max` | 0.5863 |
| `pnsd_max` | 0.884 |
| `sd_ratio_bands` | {"log_q_event": [0.0016, 0.0147], "log_we": [0.0853, 0.7674], "log_taup": [0.0639, 0.5751]} |
| `sensitivity_key` | A_E |
| `share_stations` | ["W1", "W2"] |
| `share_sum_tol` | 0.01 |
| `source_variance_share` | [0.121, 0.471] |
| `station_share_l1_max` | 0.05331 |
| `statistic_tol` | 20.84 |
| `vapour_max` | 0.2751 |
