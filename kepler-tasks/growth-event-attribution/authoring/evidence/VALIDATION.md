# Validation record

Nothing in this directory is mounted into either container, and none of it is
executed by the verifier.

## A. Physics, before anything else

`authoring/provenance/pilot_checks.py`, on the model with the regional
strengths and event anomalies in place, all passing:

| check | result |
| --- | --- |
| number conservation with no source, loss or growth | exact |
| zero-wind box vapour against the analytic balance | 0.58 per cent |
| single-column mean diameter against the analytic dD/dt = G(D) | 0.73 per cent |
| Gaussian plume spread against sigma^2 = sigma0^2 + 2 Kh t | 3.9e-7 |

The last two are the ones published in `model_spec.md` so an agent can measure
its own scheme. Step refinement is in `authoring/provenance/CALIBRATION.md`:
30 s matches 60 s on every graded quantity and 120 s degrades every one
five- to seven-fold, as the specification discloses.

## B. Identifiability

`authoring/provenance/identifiability_report.txt`. Whitened with the full
published covariance, the visible data give a condition number of 204 and
11.9 degrees of freedom for signal out of 12. The weakest direction is the
split between entrainment dilution and first-order particle loss. The yield
scale and the growth coefficient are correlated at -0.92, the three regional
strengths at +0.74 with one another and at -0.53 to -0.59 with the
nucleation anomaly. These are curved valleys rather than null directions,
and the reference reaches the same optimum from three starts.

## C. Thresholds

`authoring/provenance/CALIBRATION.md` and `threshold_inputs.json`. Every
limit is set by `set_thresholds.py` from the measured floor, the oracle's own
score and the nearest baseline, and the script refuses to write a threshold
file if any accuracy limit fails to separate the two.

## D. Recovery and the posterior

The reference recovers the withheld observations to 0.12 (vapour) and 0.76
(size distributions) in sigma units, the diameter to 0.017 nm, the age to
0.13 h and the source fractions to 0.11 in L1. Its parameter estimates sit
several data-limited posterior widths from the truth because the 5 km grid's
representation error is absorbed into them, which is why point values and
credible intervals are graded only through the bounds; `DESIGN_NOTES.md`
records the measurement.

## E. Baselines and adversarial attempts

`baselines.md`. Eleven attempts, from climatology to a diagonal-covariance
fit that honestly reports its own chi-square, all score zero, each on the
gate built for it.

## F. Verifier runs

From `runs.txt`, the sealed verifier run outside Docker against the
calibrated thresholds with `finish_validation.sh`; per-gate logs are in
`verifier_runs/`:

| run | result |
| --- | --- |
| climatology | 10 failed, 3 passed, 1 warning in 43.13s |
| diagonal_chi2 | 1 failed, 12 passed, 1 warning in 42.12s |
| false_sensitivity | 1 failed, 12 passed, 1 warning in 42.20s |
| guessed_attribution | 10 failed, 3 passed, 1 warning in 42.52s |
| hedged | 3 failed, 10 passed, 1 warning in 41.42s |
| linearised_attribution | 1 failed, 12 passed, 1 warning in 42.11s |
| nearest_station | 10 failed, 3 passed, 1 warning in 42.56s |
| no_anomaly | 2 failed, 11 passed, 1 warning in 42.98s |
| nop | 13 failed in 0.23s |
| oracle_1 | 13 passed, 1 warning in 43.89s |
| oracle_2 | 13 passed, 1 warning in 44.96s |
| oracle_3 | 13 passed, 1 warning in 43.42s |
| shuffled_predictions | 6 failed, 7 passed, 1 warning in 42.46s |
| wrong_operator | 3 failed, 10 passed, 1 warning in 41.13s |
| zero_width | 3 failed, 10 passed, 1 warning in 40.71s |

The oracle passes every gate on three consecutive runs of the same output;
the verifier is deterministic and takes about 45 s, against a declared
timeout of 3600 s. An empty submission fails every gate in a fraction of a
second.

## G. Harness gates

Docker was not available on the authoring machine, so the container builds
and the three `harbor` gates have not been executed here. They must be run
from the task directory before submission:

```
harbor run -p . -a oracle -e docker    # must score exactly 1
harbor run -p . -a nop    -e docker    # must score exactly 0
harbor check . -m anthropic/claude-opus-4-8
```

Both Dockerfiles pin every pip install with `==`, pin no apt package, run
`apt-get update` before installing and remove the apt lists afterwards.
The pins are the same resolved sets used by the two designs this task grew
out of. The verifier image bakes everything under `tests/` and installs
nothing at run time; the reference writes only the four declared artifacts
under `/app/output`, which the environment image creates.

## H. Still the author's to do

`instruction.md` must be rewritten in the submitting author's own words, and
the author metadata and the three explanations in `task.toml` must be checked
and owned. See the README.
