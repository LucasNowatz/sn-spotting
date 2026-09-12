# Provenance

Nothing here is mounted into either container.

## Pipeline

```
python pilot_checks.py                 # five physics unit checks
python build_dataset.py 4              # truth draw, tagged runs at 30 s,
                                       # counterfactuals, errors, public files
python build_indices.py                # withheld and calibration indices, manifest
python calibrate.py 4 60 koren         # achievable floor at 60 s
python calibrate.py 4 60 mc            # alternative-limiter floor
python calibrate.py 4 30               # step refinement
python calibrate.py 4 120              # step coarsening
python identifiability.py 4            # whitened sensitivity at the truth
python set_limits.py <oracle> <bases>  # every verifier limit
```

| file | role |
| --- | --- |
| `dynamics.py` | unsplit limited flux-form advection and explicit diffusion |
| `simulator.py` | vapour, twelve bins, conservative growth flux, regional strengths, event anomalies, tagged number, the event statistic and region shares |
| `scenarios.py` | eight episodes with event flags, jet and eddy streamfunction, sources, network |
| `instruments.py` | per-station operators, correlated log-space error blocks, detection limit |
| `prior.py` | the published non-diagonal prior and bounds |
| `driver.py`, `writers.py` | scenario construction, sampling, NetCDF and CSV output |
| `build_dataset.py` | the seeded generator |
| `build_indices.py` | indices and manifest |
| `calibrate.py` | the achievable floor |
| `identifiability.py` | whitened sensitivity, spectrum, posterior correlations |
| `score_submission.py`, `set_limits.py` | raw gate scores and mechanical limit setting |

## Seeds and truth

Master seed 20260703. The truth is the first draw of a seeded sequence from
the published prior that satisfies the disclosed screen: every parameter
within 1.8 prior standard deviations of its mean, diffusivity below
11000 m2 s-1, nucleation anomaly at least 0.4 and vapour anomaly at least
0.3 in log. The
accepted index and the error seed (master plus 9109) are recorded in
`tests/sealed/truth.json` only.

Everything else is deterministic and analytic: the streamfunction (mean flow,
a Gaussian jet across the mid-line, a travelling eddy), the thermal and
mixed-layer fields, the five precursor patches, the nucleation-rate
modulation, the operators and the error blocks.

## Independence of the truth

| quantity | source |
| --- | --- |
| station vapour and counts | 4 km, 30 s run through the published operators |
| withheld noisy records | the same plus one draw of the published correlated error |
| region shares, domain and station | tagged number per region in the same runs |
| event statistic and counterfactuals | four generator runs per event day with the anomaly logs zeroed in every combination |

The reference is a separate implementation on the same grid with a 60 s
step and never produces a graded value. Earlier builds ran the truth at 2 km
and with a different limiter; DESIGN_NOTES.md records the measurements that
led to the present design.

## Physics checks

`pilot_checks.py`: number conservation exact; zero-wind box vapour 0.68 per
cent from the analytic balance; single-column mean diameter 0.76 per cent
from the analytic trajectory; Gaussian-plume spread 8.9e-6 from
`sigma0^2 + 2 Kh t`; a uniform tracer through all eight episodes on the
published grid drifts by 3e-15.

## Motivation and originality

The scientific setting follows the limitation noted in the supplement of
Zhang, Kang, Poschl and Berkemeier, ACP 26, 12037-12047 (2026): Eulerian
station analysis of nanoparticle growth is biased by air-mass replacement,
and coupling growth microphysics to regional transport is left as future
work. The attribution framing, an order-averaged counterfactual split of a
burden-like statistic between an emission control and a chemistry control
under a poorly known nuisance term, is the practice of top-down trace-gas
attribution. Neither source performs this inversion. The domain, episodes,
network, forcing, error model, statistic, shares and grading are original.
