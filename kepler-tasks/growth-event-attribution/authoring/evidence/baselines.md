# Baselines and adversarial attempts

Measured on the frozen dataset before any frontier trial; limits were set
from these numbers. Truth: A_E = 302.9, A_C = 221.4 cm-3.

| attempt | vapour | pnsd | diam | shares D | shares S | cov V | cov N | ls V | ls N | dof | A_E err | A_C err | sd A_E | share | corr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reference | 0.110 | 0.412 | 0.0093 | 0.012 | 0.005 | 0.952 | 0.930 | -0.80 | -0.59 | 11.88 | -46.3 | -7.4 | 16.87 | 0.32 | -0.57 |
| climatology | 8.186 | 11.400 | 0.6882 | 0.616 | 1.037 | 0.469 | 0.381 | 6.68 | 6.85 | 0.00 | -202.9 | -131.4 | 10.00 | 0.51 | -0.50 |
| copied_attribution | 8.186 | 11.400 | 0.6882 | 0.616 | 1.037 | 0.469 | 0.381 | 6.68 | 6.85 | 0.00 | +0.0 | +0.0 | 10.00 | 0.51 | -0.50 |
| diagonal_chi2 | 0.110 | 0.412 | 0.0093 | 0.012 | 0.005 | 0.952 | 0.930 | -0.80 | -0.59 | 11.88 | -46.3 | -7.4 | 16.87 | 0.32 | -0.57 |
| false_sensitivity | 0.110 | 0.412 | 0.0093 | 0.012 | 0.005 | 0.952 | 0.930 | -0.80 | -0.59 | 11.88 | -46.3 | -7.4 | 16.87 | -0.44 | -0.57 |
| hedged | 0.110 | 0.412 | 0.0093 | 0.012 | 0.005 | 1.000 | 1.000 | 0.95 | 1.08 | 11.88 | -46.3 | -7.4 | 134.98 | 0.32 | -0.57 |
| linearised | 0.110 | 0.412 | 0.0093 | 0.012 | 0.005 | 0.952 | 0.930 | -0.80 | -0.59 | 11.88 | -41.2 | -12.5 | 16.87 | 0.32 | -0.57 |
| nearest_station | 6.155 | 15.253 | 0.7090 | 0.616 | 1.037 | 0.407 | 0.231 | 3.23 | 42.21 | 0.00 | -202.9 | -131.4 | 10.00 | 0.51 | -0.50 |
| no_anomaly | 0.110 | 0.412 | 0.0093 | 0.012 | 0.005 | 0.952 | 0.930 | -0.80 | -0.59 | 11.88 | -302.9 | -221.4 | 16.87 | 0.32 | -0.57 |
| shuffled | 12.558 | 21.086 | 1.4246 | 0.012 | 0.005 | 0.105 | 0.039 | 76.88 | 432.16 | 11.88 | -46.3 | -7.4 | 16.87 | 0.32 | -0.57 |
| uniform_shares | 0.110 | 0.412 | 0.0093 | 0.616 | 1.037 | 0.952 | 0.930 | -0.80 | -0.59 | 11.88 | -46.3 | -7.4 | 16.87 | 0.32 | -0.57 |
| wrong_operator | 0.110 | 1.827 | 0.0740 | 0.012 | 0.005 | 0.952 | 0.673 | -0.80 | 1.42 | 11.88 | -46.3 | -7.4 | 16.87 | 0.32 | -0.57 |
| zero_width | 0.110 | 0.412 | 0.0093 | 0.012 | 0.005 | 0.014 | 0.009 | 5750.86 | 8821.78 | 12.00 | -46.3 | -7.4 | 0.05 | 0.36 | -0.57 |

Where a re-run decides the outcome: the diagonal-covariance attempt reports
9115.2 against 8312.5 recomputed from its own fitted values
(tolerance 83.1); the linearised attribution's Q10 and Q01 differ from the
sealed re-run by +50.6 and +40.4 cm-3 (tolerance 20.8); the borrowed-operator attempt's
withheld size distributions differ by 1.73 sigma from the re-run and it
is caught by the size-distribution, diameter and calibration gates.

## Why each fails

- **climatology, nearest_station**: every accuracy gate by one to two orders
  of magnitude, the calibration gates, the shares, and the sealed re-run,
  since their fitted values are the observations and their estimate the prior.
- **zero_width**: coverage collapses, log scores explode, width ratios and
  attribution spreads fall below their bands.
- **hedged**: coverage 1.0 above the band, log scores above their limits,
  width ratios and attribution spreads above their bands.
- **copied_attribution**: the statistics copied at the truth with prior
  parameters; the re-run from the submitted estimate gives the prior-only
  values, and every accuracy gate fails.
- **linearised**: the split built from one-at-a-time responses without the
  interaction; its implied Q10 and Q01 differ from the re-run beyond tolerance.
- **false_sensitivity**: a restricted spread larger than the joint one.
- **shuffled**: correct values in permuted rows; accuracy, calibration and
  re-run gates fail.
- **wrong_operator**: S1's operator applied at the withheld stations; the
  size-distribution, diameter and count-calibration gates fail.
- **no_anomaly**: anomalies frozen at zero; the attribution is 0 against the
  truth and the event-day predictions no longer match a re-run of the estimate.
- **diagonal_chi2**: the honest chi-square of a diagonal fit differs from
  the full-covariance value by hundreds against a one-per-cent tolerance.
- **uniform_shares**: constant region shares; the shares gate fails.
