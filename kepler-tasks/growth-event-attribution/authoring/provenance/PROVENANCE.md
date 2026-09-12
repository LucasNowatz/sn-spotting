# Provenance

Nothing in this directory is mounted into either container.

## What produced the data

```
python pilot_checks.py            # the four physics unit checks
python generate_truth.py 4        # truth draw, eight fine-grid episodes,
                                  # counterfactuals, noise, public files
python build_indices.py           # prediction index, lineage queries, manifest
python calibrate.py 4             # the achievable error floor at 60 s
python calibrate.py 4 30          # step-refinement check
python calibrate.py 4 120         # step-coarsening check
python identifiability.py 4       # whitened sensitivity, spectrum, correlations
python set_thresholds.py <oracle> <baselines>   # every verifier limit
```

| file | role |
| --- | --- |
| `transport.py` | unsplit limited-flux advection and explicit diffusion |
| `truth_model.py` | vapour, twelve particle bins, conservative size-space growth, regional strengths, event anomalies, source tags, age moment, event statistic |
| `episodes.py` | eight episodes with event flags, precursor patches, source regions, station network |
| `prior.py` | the published non-diagonal Gaussian prior and bounds |
| `observe.py` | sampling, per-station sizing operators, correlated log-space errors, detection limit |
| `writers.py` | NetCDF and CSV output |
| `generate_truth.py` | the seeded generator |
| `build_indices.py` | prediction index, lineage queries, manifest |
| `calibrate.py` | achievable floor, which sets every threshold |
| `identifiability.py` | whitened sensitivity matrix, spectrum, posterior correlations |
| `score_submission.py` | raw gate scores, used to separate the oracle from the baselines |
| `set_thresholds.py` | mechanical threshold setting with a refusal rule |

## Seeds and the truth

Master seed 20260912. The true parameter vector is the first draw from the
published prior, in a seeded sequence, that satisfies the disclosed screening
rules: every parameter within 1.5 prior standard deviations of its mean, the
diffusivity below 12000 m2 s-1, and both event anomalies positive with a
log magnitude of at least 0.2. The accepted draw index is recorded in
`tests/hidden_truth/parameters.json`. The observation-error realisation
uses the master seed offset by 4711. Both live only in `tests/hidden_truth`,
never in `environment/`.

Everything else in the generator is deterministic: the wind field, the
temperature and boundary-layer fields, the precursor patches, the
new-particle source modulation, the station operators and the error blocks
are analytic.

## Independence of the truth

| quantity | source |
| --- | --- |
| station vapour and counts | 2.5 km, 30 s forward run, sampled through the published station operators |
| withheld noisy observations | the same, with one draw of the published correlated error |
| source fractions | three tagged number tracers, transported by the same operator with zero background |
| mean particle age | a number-age moment carried alongside the tagged number |
| event statistic and counterfactuals | four 2.5 km runs per event day with the anomaly log-parameters zeroed in every combination |

The reference solution runs on the public 5 km grid with a 60 s step and its
own code. It never produces a graded value.

## Physics checks

From `pilot_checks.py`, on the modified model:

| check | result |
| --- | --- |
| number conservation with no source, loss or growth | exact |
| zero-wind box vapour against the analytic balance | 0.58 per cent |
| single-column mean diameter against the analytic dD/dt = G(D) | 0.73 per cent |
| Gaussian plume spread against sigma^2 = sigma0^2 + 2 Kh t | 3.9e-7 |

## Source and originality

Motivation is the limitation stated in the supplement of Zhang, Kang, Poschl
and Berkemeier, *Buffering of atmospheric nanoparticle growth by
temperature-dependent shifts in molecular composition, volatility and
diffusivity*, Atmospheric Chemistry and Physics 26, 12037-12047, 2026, which
notes that Lagrangian and Eulerian differences can bias the analysis of
nanoparticle growth and identifies coupling to regional transport as future
work. The attribution framing follows the practice of top-down trace-gas
attribution studies, where an anomaly is decomposed between an emission
control and a chemistry control by counterfactual re-integration under a
poorly known nuisance sink. Neither source performs this inversion. The
domain, forcing, hidden parameters, station network, error model,
diagnostics and grading are original to this task.
