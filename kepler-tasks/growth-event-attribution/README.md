# growth-event-attribution

Decompose a nanoparticle growth-event enhancement into its nucleation and
condensation causes, with calibrated uncertainty, from sparse Eulerian
station records of a synthetic regional transport-and-growth system whose
observation errors are correlated and whose source strengths are poorly
known.

The agent is given eight 10-hour episodes of gridded meteorology, prescribed
precursor and nucleation forcing, and station records of condensable organic
vapour and nanoparticle size distributions from five stations in six
episodes; two stations and two episodes are withheld. Four episodes are event
days on which an unknown nucleation anomaly and an unknown vapour anomaly
act. It must build the coupled transport and size-resolved growth model from
the published equations, invert twelve log-parameters under a published
non-diagonal prior and a published correlated error covariance, decompose
the event statistic with four full counterfactual re-integrations, propagate
the posterior onto that decomposition and onto withheld predictions, repeat
the analysis with the regional source strengths clamped, and reconstruct the
air-mass history behind selected observations.

The scientific motivation is the limitation stated in the supplement of
Zhang, Kang, Poschl and Berkemeier (ACP 26, 2026): analysis of nanoparticle
growth in air masses is biased by the difference between Lagrangian and
Eulerian perspectives, and coupling growth microphysics to regional transport
is identified there as future work. The paper does not perform this
inversion. The attribution framing, an order-averaged counterfactual
decomposition of a burden-like statistic between an emission control and a
chemistry control under a poorly known nuisance sink, follows the practice of
top-down trace-gas attribution studies. Everything about the domain, the
forcing, the hidden parameters, the station network, the error model and the
grading is original to this task.

## Layout

```
instruction.md          the brief the agent reads
task.toml               metadata, resources, timeouts, declared artifacts
environment/
  Dockerfile            agent image, pinned, no truth
  data/                 model_spec.md, manifest, priors, operators, error
                        covariance, eight episodes, indices
solution/
  solve.sh              oracle entry point
  src/                  reference forward model, whitened inversion, driver
tests/
  Dockerfile            verifier image, offline
  test.sh               writes /logs/verifier/reward.txt and a CTRF report
  test_science.py       the thirteen gates
  verifier/             gate logic and scored metrics
  trusted_forward/      sealed forward model for the consistency gates
  hidden_truth/         sealed station truth, withheld noisy observations,
                        lineage truth, true parameters, attribution truth
  inputs/               public files the verifier needs
  thresholds.json       calibrated pass thresholds
authoring/
  provenance/           generator, physics checks, identifiability,
                        calibration, threshold setting
  evidence/             shortcut and adversarial baselines, validation records
```

## Difficulty

The data are synthetic and the manifest says so: a seeded conservative
finite-volume simulator at 2.5 km with a 30 s step and source-tagged
tracers, on an analytic non-divergent wind field, with the truth drawn once
from the published prior and the errors drawn once from the published
covariance. No episode reproduces a published dataset and the answer is not
online.

Two hard problems have to be solved together, and each one silently
corrupts the other if done carelessly.

The forward model is the first. Seven physical parameters and three regional
source strengths are shared across eight episodes. Recovering them means
building a regional transport operator whose numerical diffusion is measured
and kept well below the diffusivity being inferred (first-order upwind on
this grid carries thousands of m2 s-1), built from the published
streamfunction so that its discrete divergence vanishes, coupled to a
conservative flux through diameter space whose converged time step is set by
the narrow small-diameter bins and not by the Courant number, and carrying
source-tagged number and an age moment for the lineage diagnostics. Both
withheld meteorological episodes must pass on their own, so handling the
growing boundary layer of E07 but not the collapsing one of E08 does not
score.

The inference is the second. The observation errors are correlated in time
with an AR(1) term and carry a per-deployment bias shared by every channel
and every time of one instrument in one episode, so the 5376 usable values
carry far fewer independent constraints than their count suggests; a
diagonal-covariance fit gives a confident wrong posterior. The prior is not
diagonal. Every station has its own sizing operator, with a size-dependent
counting efficiency and a station-specific broadening, so using one
station's operator for the withheld stations is a modelling error the
forward-consistency gate rejects. Channels below the detection limit are
censored and the covariance must be restricted to what is usable. The
objective has curved valleys between the vapour lifetime, the yield scale,
the growth coefficient, the particle-loss timescale and the source
strengths, and between the base source strengths and the nucleation anomaly.

The graded questions are derived quantities. The event enhancement is not
additive in its two causes: the number that survives to the grown size range
depends on growth rate, so the interaction term is about a fifth of the
enhancement and a decomposition built from one-at-a-time perturbations or
from a gradient lands outside the tolerance. The four counterfactuals are
recomputed by the verifier from the submitted estimate. Their uncertainty has
to be propagated through that nonlinear map; the analysis has to be repeated
with the source strengths clamped to expose how much of the attribution
uncertainty they carry; and the predictive spreads at the withheld rows are
scored against the withheld noisy observations with a coverage band and a
log score, so collapsed and inflated intervals both fail.

Failure modes recorded in `authoring/evidence`: climatology and nearest-
station copying, the prior alone, collapsed uncertainties, inflated
uncertainties, an attribution copied at its true values with prior
parameters, a linearised attribution, a wrong sizing operator, shuffled
predictions, a false sensitivity claim, an analysis that never lets the
anomalies move, and a diagonal-covariance fit reporting its own chi-square.

An expert needs several days: reading the specification, building and
checking the forward model against the two published numerical self-tests
and against step refinement, assembling the whitened objective, running and
debugging a multi-start Gauss-Newton inversion, propagating the posterior,
running the restricted analysis and the tagged pass, and checking every
number before believing it.

## Reference solution

`solution/solve.sh` copies `solution/src` to `/app/work` and runs it on
`/app/data`. The vapour equation contains no particle parameter, so a
vapour-only fit of the diffusivity, entrainment velocity, vapour lifetime,
yield scale and vapour anomaly gives a cheap, well-conditioned start. The
joint inversion whitens every deployment block with the Cholesky factor of
its published covariance restricted to the usable channels and the prior
term with the Cholesky factor of the prior covariance, and runs
Levenberg-Marquardt with forward-difference Jacobians from that start and
two random prior draws. The Gauss-Newton Hessian gives the posterior
covariance; the counterfactuals are four full re-integrations over the event
days and their uncertainty comes from the delta method with a finite-
difference gradient of the counterfactual map under both the joint and the
restricted posterior; predictive spreads are the Jacobian-propagated latent
variance plus the published marginal error variance and a representation
floor; a final tagged pass yields the lineage diagnostics. It runs in about
an hour on four cores.

## Verification

Ground truth is independent of the reference: a seeded simulator at twice
the public resolution with a shorter step, carrying source-tagged number and
a number-age moment, whose own counterfactual runs define the attribution
truth. Thirteen gates must all pass: schema, bounds, withheld vapour,
withheld size distributions, a number-weighted diameter derived from them,
mean particle age, source attribution, a per-episode floor on both withheld
episodes, predictive calibration against the withheld noisy observations,
posterior honesty (width ratios and degrees of freedom), the attribution against the generator truth with a
band on its uncertainty, the source-strength sensitivity, and a trusted
forward run of the submitted estimate that must reproduce the submitted
predictions, the reported chi-square under the full covariance and the four
submitted counterfactual statistics.

Thresholds come from measurement, recorded in
`authoring/provenance/CALIBRATION.md`: the achievable floor (a correct 5 km
solver at the true parameters against the 2.5 km truth), the oracle's own
scores and the nearest shortcut baseline, with a script that refuses to
write a limit that cannot separate the two. Point values of the parameters
are graded only through the published bounds, because two correct solvers with different numerical diffusion sit at
different points along the same valley while predicting the observations
equally well.

The reward is exactly 0 or 1, written on every path including crashes.

## Before submitting

1. Rewrite `instruction.md` in your own words. The file here is a complete
   and accurate brief, but the project requires the instruction to be
   written by the submitting expert and screens it for machine-generated
   text. Keep the deliverables, the absolute paths, the output schemas and
   the exact closing sentence; rewrite the prose.
2. Check the author metadata and the three explanations in `task.toml` are
   yours and in your voice.
3. Run the harness gates where Docker is available:
   `harbor run -p . -a oracle -e docker` (must score 1),
   `harbor run -p . -a nop -e docker` (must score 0),
   `harbor check . -m anthropic/claude-opus-4-8`.
   The verifier logic has been run against the reference artifacts and every
   baseline outside Docker; the record is in `authoring/evidence`.
