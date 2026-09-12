# growth-event-attribution

Was a nanoparticle growth event caused by more nucleation or by more
condensable vapour? The agent answers that for a synthetic regional
observing system, with calibrated uncertainty, under correlated station
errors, a non-diagonal prior and poorly known regional source strengths, and
says which source regions the grown particles came from.

The system is a 280 km by 200 km mixed layer observed over eight 8-hour
episodes by six stations, with two stations and two days withheld. Four days
are event days on which an unknown nucleation anomaly and an unknown vapour
anomaly act. Twelve log-parameters are inferred; the event statistic (the
domain-mean number of particles above 3.9 nm over the last three hours of the
event days) is decomposed between the two anomalies by four complete
counterfactual re-integrations; the decomposition's spread is propagated from
the posterior, then recomputed with the regional strengths clamped; region
shares of the grown number come from tagged transport; and withheld records
are predicted with calibrated spreads.

## Layout

```
instruction.md                 the brief
task.toml                      metadata, resources, artifacts
environment/
  Dockerfile                   agent image
  data/                        specification.md, manifest, prior, network,
                               instruments, error model, indices, episodes
solution/
  solve.sh, src/               reference forward model, inversion, pipeline
tests/
  Dockerfile, test.sh          verifier image and entry point
  test_verify.py               the twelve gates
  gates/                       gate logic and metrics
  sealed_model/                the generator's solver and the sealed re-run
  sealed/                      truth, withheld noisy records, shares
  public/                      public files the verifier needs
  limits.json                  measured limits
authoring/
  provenance/                  generator, checks, calibration, limit setting
  evidence/                    baselines, verifier runs, validation record
```

## Difficulty

The data are synthetic and the manifest says so. Two hard problems have to be
solved at once, and each conceals the other's errors.

The forward model: a regional transport operator whose numerical diffusion
must be measured and kept well below the diffusivity being inferred,
built from the published streamfunction so that its discrete divergence
vanishes, coupled to a conservative flux through diameter space whose
converged step is set by the narrowest bins rather than the Courant number,
and carrying region-tagged number. Both withheld days must pass on their own,
so handling the growing mixed layer of E07 but not the collapsing one of E08
does not score.

The inference: the errors are AR(1) in time with a per-deployment bias shared
by every channel and time of one instrument in one episode, so the visible
values carry far fewer independent constraints than their count suggests and
a diagonal fit gives a confident wrong posterior; the prior is not diagonal;
every station has its own sizing operator with a size-dependent efficiency;
channels below the detection limit are censored; and the objective has
curved valleys between lifetime, yield, growth coefficient, loss timescale
and the regional strengths, and between the base strengths and the
nucleation anomaly.

The graded quantities are derived: the enhancement is not additive in its two
causes, so a linearised split lands outside the sealed re-run's tolerance;
its spread must be propagated through the nonlinear counterfactual map; the
analysis must be repeated with the strengths clamped; shares must come from
tagged transport; and predictive spreads are scored against withheld noisy
records with a coverage band and a log score that reject collapsed and
inflated intervals alike.

Recorded failure modes (`authoring/evidence/baselines.md`): climatology and
nearest-station copying, collapsed and inflated spreads, an attribution
copied at its truth with prior parameters, a linearised attribution, a false
sensitivity claim, shuffled rows, a borrowed sizing operator, frozen
anomalies, a diagonal-covariance chi-square, and constant region shares.

An expert needs several days across sessions: build and verify the forward
model, assemble the whitened objective, run and debug the multi-start
inversion, propagate the posterior, run the restricted analysis and the
tagged pass, and audit every number.

## Reference solution

`solution/solve.sh` copies `solution/src` to `/app/work` and runs the
pipeline on `/app/data`. A vapour-only fit of the five parameters the vapour
depends on seeds a joint Levenberg-Marquardt over all twelve, with every
deployment block whitened by the Cholesky factor of its published covariance
restricted to usable channels and the prior term by the Cholesky factor of
the prior, from that start and two random prior draws. The Gauss-Newton
Hessian gives the posterior; the counterfactuals are four full
re-integrations over the event days with the delta method for their spread
under both the joint and the restricted posterior; predictive spreads are the
propagated latent variance plus the published marginal error and a
representation floor; a tagged pass gives the region shares. About half an
hour on four cores.

## Verification

Ground truth is a seeded simulator separate from the reference: its own
implementation of the same equations with a shorter step on the published
grid, carrying region-tagged number; its own counterfactual runs define the
attribution truth. Every limit also allows for a correct scheme with a
different flux limiter, whose distance from the truth is measured. Twelve gates:
schema, bounds, withheld vapour, withheld size distributions, a
number-weighted diameter derived from them, region shares, a per-episode
floor on both withheld days, predictive calibration against the withheld
noisy records, posterior width ratios and degrees of freedom, the
attribution against the generator truth with a band on its spread, the
source-strength sensitivity, and a sealed re-run of the estimate that must
reproduce the submitted predictions, fitted values and counterfactual
statistics, with the reported chi-square recomputed from the submitted fitted
values under the full covariance.

Every limit comes from measurement (`authoring/provenance/CALIBRATION.md`):
the achievable floor of the reference solver at the true parameters against
the generator's discretisation, the oracle's own scores and the nearest
shortcut, with a
script that refuses to write a limit that cannot separate them. Parameter
point values are graded only through the published bounds, because correct
solvers with different numerical diffusion sit at different points of the
same valley.

The reward is exactly 0 or 1, written on every path including crashes.

## Before submitting

1. Rewrite `instruction.md` in your own words; the pipeline screens it for
   machine-generated text. Keep the paths, schemas and the exact closing
   sentence, and stay under 10000 characters.
2. Check the author metadata and the three explanations in `task.toml`.
3. Run the harness gates where Docker is available:
   `harbor run -p . -a oracle -e docker`, `harbor run -p . -a nop -e docker`,
   `harbor check . -m anthropic/claude-opus-4-8`.
