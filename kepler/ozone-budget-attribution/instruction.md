# Where does the extra ozone over the city come from?

A boreal wildfire plume drifts across a mid-size urban region during a stagnant
summer episode. Ozone inside the plume is clearly higher than outside it. That
observation on its own settles nothing: ozone made hours earlier and hundreds of
kilometres upwind can be imported into a region that is, right now, destroying
ozone faster than it makes it. Your job is to settle it properly, by closing an
ozone mass budget over the episode and by separating what the fire did from what
the city did.

A regional chemistry-transport archive sits in the image at `/app/data`. It
holds four runs of the same model over the same 48 hours, differing only in which
of two named precursor emission blocks were switched on:

| run | selected fire emissions | selected urban emissions |
|-----|-------------------------|--------------------------|
| `00` | off | off |
| `10` | on  | off |
| `01` | off | on  |
| `11` | on  | on  |

Meteorology, the initial state, boundary composition, biogenic and soil sources,
photolysis and the deposition parameters are the same realisation in all four.
Switching an emission block off removes only the named emissions of that block.
It does not remove the city, the background, or pre-existing ozone. No ozone is
emitted directly in any run.

Read `/app/data/manifest.json` before anything else. It carries the units, the
array ordering, the face orientation, the molar masses, the mass convention, the
complete list of ozone chemistry channels, the analysis windows and the file
checksums. The per-file NetCDF attributes repeat the same information. Where the
manifest describes a convention, that convention is what the archive uses, and
several of them will bite you if you assume the usual default instead.

## The budget you have to close

Work in kilograms of ozone. For a diagnostic volume and a window running from
endpoint index `n0` to endpoint index `n1`:

```
M1 - M0 = A + K + C - D + S + G + R
C = P - L
```

`M0` and `M1` are the sampled ozone burdens at the two endpoints. `A` is net
resolved advective import, `K` net turbulent and diffusive import, `P` and `L`
the gross chemical production and destruction, `D` the deposition removal, `S`
any additional explicit source, `G` the contribution of a changing diagnostic
volume, and `R` the closure residual. Imports carry a positive sign; `D` and `L`
are non-negative removals.

`A` and `K` must come from the recorded interface fluxes and `P`, `L` and `D`
from the recorded process increments. Do not obtain any of them by subtracting
the others from the burden change: a transport term back-calculated from storage
makes closure an identity and tells you nothing.

Two diagnostic volumes are graded, both supplied in `/app/data/masks.nc` and both
identical across the four runs. `fixed_urban` is a fixed box over the city.
`moving_plume` is a set of weights `w` between 0 and 1, on the same time axis as
the state files, derived from a passive fire source tracer that is blind to the
chemistry switches. Do not build a volume of your own from any ozone field.

For the moving volume, use the exact discrete identity rather than an estimated
boundary velocity. With `B^n = sum_i w_i^n M_i^n`,

```
B^(n+1) - B^n = sum_i w_i^n [ M_i^(n+1) - M_i^n ] + G^n
G^n = sum_i [ w_i^(n+1) - w_i^n ] M_i^(n+1)
```

so every process increment over the interval `[n, n+1]` is aggregated with the
start-of-interval weights, and the mask-change term is accumulated separately
with the end-of-interval cell masses. `G` is zero for the fixed volume. `G` is
not chemistry: it is ozone that entered or left the sample because the sample
itself moved.

## The four-run contrasts

For any quantity `X` computed identically in each run:

```
d_fire_A0  = X10 - X00        fire effect without the selected urban emissions
d_fire_A1  = X11 - X01        fire effect with them
d_urban_F0 = X01 - X00        urban effect without the fire
d_urban_F1 = X11 - X10        urban effect with it
interaction = X11 - X01 - X10 + X00
```

Apply these to the burdens, to every budget term and to the receptor metrics.
The windows do not start when the emissions were switched, so the initial
anomalies are generally not zero and the final anomaly is not the property of the
window alone.

## Receptors

`/app/data/receptors.csv` gives, for each receptor, the cells and weights that
define it. Its ozone series is the weighted sum of `o3_vmr` over those cells at
the endpoint times, in ppbv. Over a window from `n0` to `n1` inclusive, with a
one-hour sample spacing and the composite trapezoidal rule:

* `exposure_ppbvh` is the integral of the series, in ppbv h;
* `exposure_over40_ppbvh` is the integral of `max(chi - 40, 0)`, clipping each
  sample first and integrating afterwards;
* `mean_ppbv` is `exposure_ppbvh` divided by the window length in hours;
* `peak_ppbv` is the largest sample in the closed window.

For a contrast series, every metric is the arithmetic difference of the metric
values of the runs involved. The peak of a contrast is therefore a difference of
peaks, which can be negative, and is not the peak of a difference series.

## What to deliver

Write all of the following, under the exact paths given.

**`/app/output/budgets.csv`** — header exactly

```
case_id,scenario,volume,window,M0_kg,M1_kg,A_kg,A_lateral_kg,A_vertical_kg,K_kg,K_lateral_kg,K_vertical_kg,P_kg,L_kg,C_kg,D_kg,S_kg,G_kg,residual_kg
```

one row per run, volume and window, 32 rows, with
`case_id = "<scenario>|<volume>|<window>"`, scenario in `00,10,01,11`, volume in
`fixed_urban,moving_plume`, window in `W1,W2,W3,W4` as defined in the manifest.
All values in kg O3. `residual_kg` is `R` from the equation above.

Report both transport terms split by interface orientation as well as in total.
`A_lateral_kg` is the part of `A` carried by the x and y faces and
`A_vertical_kg` the part carried by the z faces; `K_lateral_kg` and
`K_vertical_kg` split `K` the same way, with the deposition bookkeeping belonging
to the vertical part. Each split is graded on its own and has to add up to the
total in the same row.

**`/app/output/contrasts.csv`** — header exactly

```
contrast_id,volume,window,quantity,d_fire_A0,d_fire_A1,d_urban_F0,d_urban_F1,interaction
```

with `contrast_id = "<volume>|<window>|<quantity>"` and `quantity` running over
`M0,M1,dM,A,K,P,L,C,D,G,residual`, where `dM = M1 - M0`. 88 rows, in kg O3.

**`/app/output/receptors.csv`** — header exactly

```
row_id,receptor_id,series,window,mean_ppbv,peak_ppbv,exposure_ppbvh,exposure_over40_ppbvh
```

with `row_id = "<receptor_id>|<series>|<window>"` and `series` running over
`00,10,01,11,d_fire_A0,d_fire_A1,d_urban_F0,d_urban_F1,interaction`. 108 rows.

**`/app/output/diagnosis.json`** — one object:

```json
{
  "schema_version": "kepler-o3-diagnosis-1.0",
  "rows": [
    {
      "volume": "fixed_urban",
      "window": "W1",
      "tau_kg": 0.0,
      "scenario_chemistry": {"00": "...", "10": "...", "01": "...", "11": "..."},
      "fire_chemistry_contrast": {"A0": "...", "A1": "..."},
      "interaction_label": "...",
      "dominant_term": {"00": "...", "10": "...", "01": "...", "11": "..."},
      "transport_sustains_anomaly": false
    }
  ]
}
```

with one row for each of the eight volume and window pairs. Every label is fixed
by numbers you have already computed, using the near-zero half-width
`tau = 0.005 * |M0|` of run `11` in that volume and window:

* `scenario_chemistry` is `net_production` when `C > tau`, `net_destruction` when
  `C < -tau`, otherwise `near_zero`.
* `fire_chemistry_contrast` reports the fire contrast in `C`, that is
  `C10 - C00` under key `A0` and `C11 - C01` under key `A1`, as `more_positive`
  when the contrast exceeds `tau`, `more_negative` when it is below `-tau`, and
  `near_zero` in between. A run can be destroying ozone while the fire still
  makes its chemistry more positive; report the two statements separately.
* `interaction_label` applies the same thresholds to the interaction in `C`, and
  is `amplifying`, `damping` or `near_zero`.
* `dominant_term` names whichever of `A`, `K`, `C`, `-D` and `G` has the largest
  absolute value in that run, reported as `"A"`, `"K"`, `"C"`, `"D"` or `"G"`.
* `transport_sustains_anomaly` is true only when all three of these hold for the
  fire effect with the urban emissions on: the burden anomaly `M1_11 - M1_01`
  exceeds `tau`, the chemistry contrast `C11 - C01` is below `-tau`, and the
  summed transport and mask contrast `(A + K + G)11 - (A + K + G)01` exceeds
  `tau`. Otherwise it is false.

**`/app/output/workflow/`** — the analysis code that produced the four files,
runnable as it stands, together with `/app/output/workflow/RUN.txt` giving the
exact invocation and naming the inputs and outputs.

Grading is on the numbers. Each scalar is checked as
`|reported - reference| <= atol + rtol * |reference|` with `rtol = 1e-4` and
`atol = 2e-3 * S`, where `S` is the reference initial ozone burden of run `11`
for that volume and window. The receptor metrics use `rtol = 1e-4` with
`atol = 2e-3 ppbv` for concentrations and `2e-2 ppbv h` for exposures. Every
term is compared on its own, so a budget that closes because two wrong terms
cancel will not pass.

You have 14400 seconds to complete this task. Do not cheat by using online solutions or hints specific to this task.
