# Design decisions and measurements

Where this task departs from the two designs it grew out of, a regional
transport-and-growth inversion graded on withheld predictions and lineage,
and a global trace-gas attribution graded on a counterfactual decomposition
with an honest posterior, and the measurement that settled each choice.

## One system, not two stapled together

The attribution question is asked of the transport-and-growth system itself,
not of a second model. Four of the eight episodes are event days on which a
nucleation anomaly and a vapour anomaly act; the graded statistic is the
domain-mean grown-particle number over the last four hours of those days,
and the decomposition is the order-averaged four-run counterfactual used in
top-down attribution. Measured at the prior mean, doubling the nucleation
factor scales the statistic linearly while a 1.6-fold vapour factor raises it
by 47 per cent through survival, and the two multiply: the interaction term
at the truth is 25 cm-3 of a 194 cm-3 enhancement. That is what makes a
linearised decomposition a different number from the counterfactual one.

## The grown-channel threshold

With channels 7 to 11 (above 5.7 nm) the statistic was carried almost
entirely by one warm, stagnant event day; the cool event days sat at the
background. Channels 5 to 11 (above 3.9 nm, the survival threshold used in
nucleation studies) spread the signal over the four event days while keeping
it a property of growth rather than of injection.

## Correlated errors with a per-deployment bias

The first design carried independent lognormal noise, under which 5376
values pin every parameter to a fraction of a per cent. The published error
model now has white, AR(1) and per-deployment bias terms in log space, and
for counts the bias is shared by every channel and every time of one
instrument in one episode. That is realistic (counting-efficiency
calibration drifts per deployment) and it changes the inference: a diagonal
fit gives a different chi-square, which the verifier exposes by recomputing
the chi-square from the submitted fitted values with the full covariance.

## Why the truth is not graded through credible intervals

The reference recovers the withheld observations to 0.12 and 0.76 sigma,
but its parameter estimates sit 2 to 27 posterior standard deviations from
the truth. That is not a bug in the reference: the posterior is data-limited
to a fraction of a per cent while the 5 km solver differs from the 2.5 km
truth by a representation error that is not in the published covariance, so
the fit absorbs it by moving the parameters along the valley, most visibly
the diffusivity (plus 9 per cent, the scheme's numerical diffusion) and the
source strengths. Any correct 5 km solver does the same to a different
degree. The task therefore grades the posterior on what a solver-independent
quantity can support: width ratios against the prior, degrees of freedom,
predictive coverage and log score against the withheld noisy observations
(where the representation floor is disclosed), and bands on the propagated
attribution spread. The truth-in-interval count is kept as a diagnostic
only.

## Fitted values as a deliverable

The first version recomputed the chi-square with the verifier's own 5 km
solver and compared it to the reported one. At the true parameters that
solver's chi-square is 7965 while a solver on the 2.5 km grid would report
about 5400 on the same data, so a one-per-cent tolerance would have failed a
correct fine-grid solution. The agent now submits its fitted values at the
1230 visible station-times; the chi-square check is arithmetic on those
values with the full covariance (tolerance one per cent), and the fitted
values themselves are held to the trusted forward run under the same loose
envelope as the withheld predictions.

## The sensitivity question

Clamping the three regional source strengths at their prior means reduces
the posterior standard deviation of the nucleation term from 9.07 to
7.09 cm-3, a 39 per cent variance share, and the posterior correlation
between the nucleation anomaly and the strength of region A is -0.53. Those
two numbers are structural: they come from event days constraining the
product of base strength and anomaly while non-event days constrain the base
alone. The correlation between the two anomalies is +0.08 and is not graded.

## Cost

On the 5 km grid with a 60 s step: 0.3 s per episode for vapour alone,
3.6 s with the twelve particle bins, 14.6 s with source tags and the age
moment. A full twelve-parameter Jacobian over the six observed episodes is
78 forward runs, about 75 s on four cores, and the joint inversion converges
in five iterations from the vapour-only start. The reference, with three
starts, the restricted analysis, the counterfactual gradients and the tagged
pass, takes about twenty minutes on four cores; the agent's budget of eight
hours is for building and checking the model, not for running it.
