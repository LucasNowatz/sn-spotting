# Ozone budget attribution for a wildfire plume over a city

A wildfire plume crosses an urban region during a stagnant summer episode. The
agent is given a four-run regional chemistry-transport archive and has to close
fixed-volume and moving-mask ozone mass budgets, separate the fire effect from
the urban effect and their interaction, compute receptor exposure, and say in
numbers whether the ozone enhancement over the city is locally produced or
imported.

## Difficulty

The data are synthetic and are declared as such in the manifest and the
metadata. They come from a purpose-built regional chemistry-transport model
written for this task: a 24 x 18 x 8 grid over 48 hours, a non-divergent dry-air
mass flux field built as the discrete curl of an edge vector potential,
flux-form upwind advection, explicit horizontal and implicit vertical diffusion
with a dry deposition flux boundary condition at the ground, and a twelve-species
reduced NOx-VOC mechanism integrated with five-second sub-steps and radicals at
photostationary steady state. Every operator records its own integrated
increment, so the recorded terms reproduce each cell's ozone mass change to
around fourteen significant figures.

The difficulty is dependent reasoning, not volume. A long chain of conventions
has to be right simultaneously, and one wrong link moves every downstream number
well past tolerance:

* ozone is a dry-air mole fraction while the archive supplies moist air mass and
  specific humidity;
* the surface turbulent flux already carries the deposition removal that is also
  reported separately;
* production is one channel, destruction is five, one of them the fast NO
  titration that is an order of magnitude larger than the net signal, and the
  archive also carries a radical-based ozone-production diagnostic that is not a
  budget term;
* the moving volume is graded through an exact discrete identity in which the
  process increments take start-of-interval weights and the mask-change term
  takes end-of-interval cell masses;
* transport has to be assembled from staggered face fluxes with a stated
  orientation and reported split by interface orientation.

The physics is set up so that interpretation cannot be guessed. Four windows
span four regimes: daytime net production with a negligible interaction, a night
window where the plume arrives over the city and the enhancement grows while
local chemistry is a net sink, a daytime window where urban NOx makes the fire's
chemical contribution strongly more positive, and a window where the plume leaves
the domain and the mask-change term dominates. Exactly one of the eight volume
and window pairs meets the numeric test for transport sustaining the anomaly.

Expert time estimate: about twelve focused hours, across several sessions.

## Reference solution

`solution/solve.sh` writes `workflow/ozone_budget.py` into `/app/output` and runs
it from there, so the delivered workflow directory is the code that produced the
delivered tables. It reads the conventions out of `manifest.json` rather than
hard-coding them, converts ozone to mass on dry air, differences the staggered
face fluxes into lateral and vertical net import, adds the deposition array back
into the vertical turbulent term, sums the six gross chemistry channels, and
aggregates interval by interval with start-of-interval weights while
accumulating the mask-change term on the end-of-interval masses. It then
differences matched runs for the counterfactual contrasts, integrates the
receptor series, and derives the labels. Runtime is under two seconds on the
task environment.

## Verification

Ground truth never touches the reference solution. The forward model writes
per-cell diagnostics into the archive; `authoring/provenance/build_truth.py`, an
independent evaluator written with xarray whole-array reductions and sharing no
code with the reference, aggregates the written NetCDF files into the regional
expectations sealed in `tests/truth.json`. The reference is a third,
structurally different implementation. The two agree to within 1e-5 of the
grading tolerance on all 1,352 scored numbers and on all 104 label fields.

`tests/test.sh` runs pytest offline in the verifier image and always writes a
binary reward to `/logs/verifier/reward.txt`, including on a crash or a missing
artifact. The suite grades every term on its own, so a total that closes because
two wrong terms cancel is rejected. Closure is recomputed from the agent's own
reported terms against the sealed endpoint burdens rather than trusting the
residual column. The mask gate requires the change term to vanish on the fixed
volume, to match on the moving one, and not to have been folded into transport.
Contrasts and receptor contrasts are checked against the sealed values and for
internal consistency with the agent's own tables. Labels are compared to the
sealed labels and separately required to follow from the reported numbers.

Twenty shortcut and cheat baselines were run through the sealed verifier and all
scored zero; `authoring/evidence/baseline_results.txt` records the run. The
reference scores 1 and an untouched environment scores 0.

## Repository layout of the authoring material

```
authoring/provenance/generate_dataset.py   forward model, writes environment/data
authoring/provenance/build_truth.py        independent evaluator, writes tests/truth.json
authoring/provenance/validation.md         invariants checked and their measured values
authoring/provenance/checksums.txt         sha256 of every archive file
authoring/evidence/baselines.py            shortcut and cheat implementations
authoring/evidence/run_baselines.sh        runs them through the sealed verifier
authoring/evidence/baseline_results.txt    recorded outcome of that run
```

Nothing under `authoring/` is mounted into any container.
