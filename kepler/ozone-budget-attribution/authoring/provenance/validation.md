# Validation record

Produced by `authoring/provenance/validate.py` against the archive in `environment/data`.  Every number below is measured, not asserted.

## 1. Mass flux field

* worst relative mass divergence: 2.816e-16
* worst vertical mass flux at the ground: 0.000e+00 kg/s
* worst vertical mass flux at the lid: 1.562e-10 kg/s
* peak resolved horizontal wind speed: 2.45 m/s

## 2. Per-cell closure of the recorded increments

| run | worst absolute residual (kg) | relative to the largest cell mass |
|-----|------------------------------|-----------------------------------|
| 00 | 2.998e-11 | 6.511e-15 |
| 10 | 3.465e-11 | 5.018e-15 |
| 01 | 3.100e-11 | 6.734e-15 |
| 11 | 4.123e-11 | 5.971e-15 |

## 3. Shared meteorology

* air mass and specific humidity identical in all four runs: True

## 4. Analytic cases

* closed-box chemistry, |dO3 - (P - L)| / O3: 7.942e-16
* pure transport, departure from a uniform mixing ratio: 0.000e+00
* pure transport, mass change against the net boundary flux: 0.000e+00 kg
* deposition only, worst relative departure from exp(-vd t / dz): 2.313e-05 (implicit first-order truncation at 60 s)
* prescribed mask motion, B0 = 10 kg, B1 = 30 kg, G = 20 kg with every physical increment zero

## 5. Time-step refinement

The episode is re-integrated with the model step halved to 90 s and the chemistry sub-step halved to 2.5 s, holding the one-hour diagnostic interval fixed.  The table compares the fixed urban volume over window W2 in run 11, as a fraction of the initial burden of that row.

| term | 180 s run (kg) | 90 s run (kg) | difference / M0 |
|------|----------------|---------------|-----------------|
| M0 | 247985.5 | 248224.0 | 9.615e-04 |
| M1 | 246007.2 | 246154.8 | 5.949e-04 |
| A | 31029.5 | 30874.3 | 6.261e-04 |
| K | 596.2 | 598.0 | 7.146e-06 |
| P | 121359.2 | 124774.9 | 1.377e-02 |
| L | 148968.0 | 152319.1 | 1.351e-02 |
| C | -27608.8 | -27544.1 | 2.610e-04 |
| D | 5995.2 | 5997.3 | 8.653e-06 |

Worst term-wise change under refinement: 1.377e-02 of the initial burden.  The archive itself is the graded object and its recorded increments close it exactly, so this number describes how converged the physics is, not the grading tolerance.

## 6. Determinism

* regeneration reproduces every archive file byte for byte: True
* files compared: 16

