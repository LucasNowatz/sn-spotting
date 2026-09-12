# Validation record

Nothing here is mounted into a container or executed by the verifier.

## A. Physics

`pilot_checks.py`: number conservation exact; zero-wind box vapour 6.8e-3;
column growth 7.6e-3; Gaussian plume 8.9e-6; uniform tracer through all
eight episodes 3e-15. Step refinement in `CALIBRATION.md`: 30 s matches 60 s,
120 s degrades every quantity several-fold, as disclosed.

## B. Identifiability

```
Whitened identifiability at the truth
========================================
usable observations: 8292
singular values: 2.7e+03 708 534 438 199 156 100 72.6 63.7 25.6 14.4 7.03
condition number: 384.1
degrees of freedom for signal: 11.87 of 12

parameter        post sd  prior sd   ratio
log_Kh            0.0023     0.350   0.007
log_we            0.1224     0.450   0.272
log_tauv          0.0062     0.400   0.016
log_Yorg          0.0107     0.300   0.036
log_betaorg       0.0096     0.400   0.024
log_Heff          0.0031     0.300   0.010
log_taup          0.0869     0.450   0.193
log_sA            0.0258     0.350   0.074
log_sB            0.0234     0.350   0.067
log_sC            0.0225     0.350   0.064
log_s_event       0.0271     0.350   0.078
log_q_event       0.0015     0.300   0.005

posterior correlations beyond 0.5
  log_we         log_tauv       +0.595
  log_we         log_Heff       -0.820
  log_we         log_taup       +0.569
  log_tauv       log_Heff       -0.515
  log_Yorg       log_betaorg    -0.917
  log_sA         log_sB         +0.892
  log_sA         log_sC         +0.825
  log_sA         log_s_event    -0.570
  log_sB         log_sC         +0.766
  log_sB         log_s_event    -0.599
  log_sC         log_s_event    -0.649
```

## C. Limits

`CALIBRATION.md` and `limit_inputs.json`; `set_limits.py` refuses to write a
limit that cannot separate the oracle from the nearest shortcut.

## D. Baselines

`baselines.md`: twelve attempts, all scoring zero on the gate built for them.

## E. Verifier runs

`runs.txt`, from `finish_validation.sh` outside Docker; per-gate logs in
`verifier_runs/`:

| run | result |
| --- | --- |
| climatology | 10 failed, 2 passed, 1 warning in 71.79s (0:01:11) |
| copied_attribution | 10 failed, 2 passed, 1 warning in 66.69s (0:01:06) |
| diagonal_chi2 | 1 failed, 11 passed, 1 warning in 66.04s (0:01:06) |
| false_sensitivity | 1 failed, 11 passed, 1 warning in 67.78s (0:01:07) |
| hedged | 3 failed, 9 passed, 1 warning in 70.68s (0:01:10) |
| linearised | 1 failed, 11 passed, 1 warning in 70.78s (0:01:10) |
| nearest_station | 10 failed, 2 passed, 1 warning in 66.59s (0:01:06) |
| no_anomaly | 2 failed, 10 passed, 1 warning in 65.80s (0:01:05) |
| nop | 12 failed in 0.17s |
| oracle_1 | 12 passed, 1 warning in 72.50s (0:01:12) |
| oracle_2 | 12 passed, 1 warning in 71.03s (0:01:11) |
| oracle_3 | 12 passed, 1 warning in 74.84s (0:01:14) |
| shuffled | 6 failed, 6 passed, 1 warning in 66.22s (0:01:06) |
| uniform_shares | 1 failed, 11 passed, 1 warning in 64.54s (0:01:04) |
| wrong_operator | 4 failed, 8 passed, 1 warning in 63.95s (0:01:03) |
| zero_width | 3 failed, 9 passed, 1 warning in 68.42s (0:01:08) |

The oracle passes every gate on three runs of the same output; the verifier
is deterministic and takes about a minute against a 3600 s timeout. An empty
submission fails every gate immediately.

## F. Harness gates

Docker was unavailable on the authoring machine, so the image builds and the
harbor gates (`harbor run -p . -a oracle -e docker`, `... -a nop ...`,
`harbor check .`) must be run before submission. Both Dockerfiles pin every
pip install, pin no apt package, and clean the apt lists; the verifier
installs nothing at run time.

## G. The author's part

`instruction.md` must be rewritten by the submitting author, and the metadata
and explanations in `task.toml` owned; see the README.
