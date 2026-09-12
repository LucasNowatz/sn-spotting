# Design notes

## One system, one question

The attribution is asked of the transport-and-growth system itself. Four of
eight days are event days on which a nucleation anomaly and a vapour anomaly
act; the graded statistic is the domain-mean grown number (above 3.9 nm)
over the last three hours of those days; the split is the order-averaged
four-run counterfactual used in top-down attribution. At the truth the
interaction term is 18 cm-3 of a 139 cm-3 enhancement, which is why a
linearised split is a different number from the counterfactual one, and why
the sealed re-run recomputes all four statistics from the submitted estimate.

## Region shares instead of per-sample lineage

Tagged transport is still required, but the graded quantity is the share of
the grown number injected in each source region, domain-wide and at the
withheld stations, for each event day. That is the question an attribution
study asks (where did the grown particles come from) and it is defined on the
same fields as the statistic. The remainder of each share vector is
background inflow and entrainment, so the shares sum to at most one.

## Correlated errors with a per-deployment bias

Independent noise on thousands of values would pin every parameter to a
fraction of a per cent. The published error model has white, AR(1) and
deployment-bias terms in log space; for counts the bias is shared by every
channel and time of one instrument on one day. A diagonal fit reports a
chi-square that differs from the full-covariance value by hundreds, which
the recomputation from the submitted fitted values exposes.

## Why the truth is not run at finer resolution

The first build ran the truth at 2 km against a 4 km public grid. The
reference then recovered the withheld records to a fraction of a sigma but
its estimate sat many posterior widths from the truth: the posterior is
data-limited to below a per cent while the representation error of the
coarser grid is not in the published covariance, so the fit absorbed it into
the parameters, most visibly the diffusivity (+16 per cent), one regional
strength (-16 per cent) and the nucleation anomaly (-13 per cent). On this
instance that turned a nucleation term of 53 cm-3 into 16, and no tolerance
could then separate a correct solution from a frozen-anomaly one. A second
build ran the truth with a different flux limiter on the published grid; the
limiter difference alone was worth half a sigma at the stations on that
instance and still biased the nucleation anomaly by a tenth. The truth is
now the generator's own implementation with a 30 s step on the published
grid, the event is screened to be strong (nucleation anomaly at least 0.4
in log, so the attribution signal is several times any scheme bias), and
every limit folds in a measured alternative-scheme floor: the generator's
solver with the monotonized-central limiter at the truth, so that a correct
solver with a different limiter is never failed for the difference.

## Why credible intervals are still not graded against the truth

Even so, the data-limited posterior is far narrower than the spread between
correct solvers with different numerical diffusion, so point values are
graded only through the bounds and the posterior on solver-independent
quantities: width ratios against the prior, degrees of freedom, predictive
coverage and log score against the withheld noisy records, and bands on the
propagated attribution spread.

## Fitted values as a deliverable

Recomputing the chi-square with the verifier's own solver would fail a
correct fine-grid solution, whose chi-square at the same estimate is
thousands lower. The agent submits its fitted values at the visible
station-times; the chi-square check is arithmetic on those values with the
full covariance at one per cent, and the fitted values themselves are held to
the sealed re-run under the same loose envelope as the withheld predictions.

## Cost

On the 4 km grid with a 60 s step: about 0.4 s per episode for vapour alone,
6 s with the twelve bins, 20 s with tags. A twelve-parameter Jacobian over
the six observed days is 78 forward runs. The reference, with three starts,
the restricted analysis, the counterfactual gradients and the tagged pass,
takes about half an hour on four cores; the agent's eight hours are for
building and checking the model.
