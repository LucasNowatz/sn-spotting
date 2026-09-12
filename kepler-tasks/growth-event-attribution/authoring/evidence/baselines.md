# Baseline and adversarial measurements on the frozen dataset

All numbers were measured on the frozen dataset before any frontier trial,
and the thresholds in `tests/thresholds.json` were set from them by
`authoring/provenance/set_thresholds.py`. Truth for this instance:
A_E = 125.6 cm-3, A_C = 68.6 cm-3.

| analysis | vapour | pnsd | diam nm | age h | src L1 | cov V | cov N | ls V | ls N | dof | A_E err | A_C err | sd A_E | share | corr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reference | 0.119 | 0.759 | 0.0166 | 0.133 | 0.108 | 0.936 | 0.827 | -0.77 | -0.26 | 11.86 | +15.4 | -6.2 | 9.07 | 0.39 | -0.53 |
| climatology | 11.099 | 11.371 | 0.7357 | 1.831 | 0.692 | 0.429 | 0.343 | 12.78 | 6.62 | 0.00 | -5.6 | +1.4 | 10.00 | 0.51 | -0.60 |
| diagonal_chi2 | 0.119 | 0.759 | 0.0166 | 0.133 | 0.108 | 0.936 | 0.827 | -0.77 | -0.26 | 11.86 | +15.4 | -6.2 | 9.07 | 0.39 | -0.53 |
| false_sensitivity | 0.119 | 0.759 | 0.0166 | 0.133 | 0.108 | 0.936 | 0.827 | -0.77 | -0.26 | 11.86 | +15.4 | -6.2 | 9.07 | -0.44 | -0.53 |
| guessed_attribution | 11.099 | 11.371 | 0.7357 | 1.831 | 0.692 | 0.429 | 0.343 | 12.78 | 6.62 | 0.00 | +0.0 | +0.0 | 10.00 | 0.51 | -0.60 |
| hedged | 0.119 | 0.759 | 0.0166 | 0.133 | 0.108 | 1.000 | 1.000 | 0.92 | 1.04 | 11.86 | +15.4 | -6.2 | 72.58 | 0.39 | -0.53 |
| linearised_attribution | 0.119 | 0.759 | 0.0166 | 0.133 | 0.108 | 0.936 | 0.827 | -0.77 | -0.26 | 11.86 | +21.4 | -12.2 | 9.07 | 0.39 | -0.53 |
| nearest_station | 8.590 | 13.859 | 0.5522 | 1.831 | 0.620 | 0.280 | 0.293 | 6.37 | 36.78 | 0.00 | -5.6 | +1.4 | 10.00 | 0.51 | -0.60 |
| no_anomaly | 0.119 | 0.759 | 0.0166 | 0.133 | 0.108 | 0.936 | 0.827 | -0.77 | -0.26 | 11.86 | -125.6 | -68.6 | 9.07 | 0.39 | -0.53 |
| shuffled_predictions | 15.052 | 18.081 | 1.3608 | 0.133 | 0.108 | 0.089 | 0.047 | 113.02 | 376.74 | 11.86 | +15.4 | -6.2 | 9.07 | 0.39 | -0.53 |
| wrong_operator | 0.119 | 1.612 | 0.0652 | 0.133 | 0.108 | 0.936 | 0.622 | -0.77 | 1.24 | 11.86 | +15.4 | -6.2 | 9.07 | 0.39 | -0.53 |
| zero_width | 0.119 | 0.759 | 0.0166 | 0.133 | 0.108 | 0.012 | 0.007 | 6194.33 | 14946.00 | 12.00 | +15.4 | -6.2 | 0.05 | 0.36 | -0.53 |

Replay quantities where they decide the outcome: the reference agrees with
the trusted forward to 1e-6 sigma on every prediction and 1e-6 cm-3 on every
counterfactual statistic; the diagonal-covariance analysis reports a
chi-square of 8045.9 against 7144.4 recomputed from its own fitted values
(tolerance 71.4); the linearised attribution's Q10 and Q01 differ from the
trusted recomputation by +19.7 and +7.5 cm-3 (tolerance 8.8); the wrong-operator
attempt's withheld size distributions differ by 1.40 sigma, inside the consistency
envelope, and it is caught instead by the size-distribution, diameter and
calibration gates.

## Why each one fails

**Climatology and nearest-station copying.** Fail every accuracy gate by one to
two orders of magnitude, the calibration gates, and the consistency gates,
since their fitted values are the observations themselves and their estimate
is the prior mean. Solving the inversion is necessary.

**Zero-width intervals.** The reference's numbers with every uncertainty
collapsed. Coverage falls to about one per cent, the log scores explode, the
width ratios drop below their bands and the attribution spreads fall below
theirs. Overconfidence is caught on four independent gates.

**Hedged intervals.** Every uncertainty inflated eightfold. Coverage rises to
100 per cent, the log scores rise above their limits, the width ratios exceed
their bands and the attribution spreads exceed theirs. Refusing to commit is
caught on the same four gates.

**Guessed attribution.** Prior-only parameters and climatology predictions,
with the four statistics and the decomposition copied at their exact truth
values. The verifier recomputes the four statistics from the submitted
estimate and gets the prior-only values instead, and every accuracy gate
fails as well.

**Linearised attribution.** The reference's estimate and predictions, but the
decomposition built from the two one-at-a-time responses without the
interaction term. Its A_E and A_C errors (21 and 12 cm-3) sit inside the
accuracy tolerance, which is set by the reference's own representation-error
bias; it is rejected because the Q10 and Q01 it implies differ from the
trusted recomputation by 20 and 8 cm-3 against a tolerance of 9. The
counterfactuals have to be run.

**False sensitivity.** A correct analysis with the restricted standard
deviation inflated above the joint one. Fixing parameters cannot widen a
posterior, and the gate says so.

**Shuffled predictions.** Correct values in a permuted row order. Every
accuracy and calibration gate fails, and so does the consistency gate.

**Wrong sizing operator.** The reference's fields with the withheld stations
sampled through station S1's operator instead of their own. The
size-distribution log RMSE doubles to 1.61 against a limit of 1.37, the
diameter error rises to 0.065 nm against 0.043, count coverage falls to
0.62, and the count log score rises to 1.24 against a limit of 0.0.

**No anomaly.** The reference's estimate with both event anomalies set to
zero and the four statistics all equal. A_E and A_C are zero against 126 and
69, and the predictions no longer match a trusted run of the submitted
estimate on the event days.

**Diagonal covariance.** The reference's analysis with the chi-square
computed from the diagonal of the published covariance, as an inversion that
ignored the correlations would honestly report. It differs from the value
recomputed from the same fitted values with the full covariance by about
900 against a tolerance of 71.
