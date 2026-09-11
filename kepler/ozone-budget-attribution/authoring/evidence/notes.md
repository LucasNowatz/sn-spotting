# Authoring notes

## The failure mode this task targets

The task is built around one specific thing a solver has to get right and will
not get right by reading carefully once: **a chain of archive conventions that
each look like the obvious default and are not.**

Most budget tasks fail agents on arithmetic or on volume. This one fails them on
bookkeeping that is individually small and jointly fatal, because every scored
number depends on all of it:

| convention | the plausible wrong choice | size of the error |
|------------|----------------------------|-------------------|
| ozone is a dry-air mole fraction, air mass is moist | multiply by `air_mass` | 0.4 to 2.2 per cent of every burden and every term |
| the surface turbulent flux already carries deposition | sum the mixing fluxes, then subtract `o3_depos` | 1 to 4 per cent of the burden, and it makes closure fail in a way that looks like a transport error |
| one production channel, five destruction channels | drop the smallest, or use the radical diagnostic | from 0.4 per cent to an order of magnitude |
| process increments take start-of-interval weights, the mask term takes end-of-interval masses | one weight set for both, or `M^n` in the mask term | tens of per cent on the moving volume |
| transport comes from staggered face fluxes with a stated orientation | infer it from the burden change | passes a closure check and nothing else |

The last one is why `A` and `K` are also graded split by interface orientation:
a residual-derived transport total cannot be decomposed, so the shortcut that
would otherwise survive static-output grading does not.

Beyond the conventions sits the interpretation the problem is named after. The
enhancement over the city is real and positive while local chemistry is a net
sink, and only one of the eight volume-and-window pairs meets the numeric test
for that. A solver who reasons from the concentration anomaly alone gets that row
wrong and several label fields with it.

## What this task is not

It is not a reconstruction of numbers a missing tool would have produced, and it
is not a search through a large filesystem. The archive is 27 MB in thirteen
files and every convention it uses is written down in `manifest.json` and
repeated in the NetCDF attributes. Nothing is withheld that an expert opening a
model archive would have. The difficulty is that acting on all of it correctly,
at once, across 32 budget rows, 88 contrasts, 108 receptor rows and 8 label rows,
is a multi-session job.

It is also not an unresolved historical question. The specification this bundle
implements is explicit that the BORTAS model-observation discrepancy should not
become a benchmark question with an invented unique answer. What is graded here
is a defined property of four simulations that exist only in this bundle.

## Cheat attempts

`baselines.py` implements twenty shortcut and cheat routes and
`run_baselines.sh` puts each through the sealed verifier.
`baseline_results.txt` records the outcome: every one scores 0, the reference
scores 1, and an untouched environment scores 0. The set covers the laziest
attempts (empty output, all zeros), every convention error in the table above,
both directions of the deposition mistake, swapped runs, three receptor
mistakes, and four ways of producing a label table without the physics.

The one route worth naming separately is `transport_from_residual`. It scored 1
against an earlier draft of the verifier, because when every other term is
already exact a back-calculated transport total is also exact. That is what led
to grading the lateral and vertical split, and it now scores 0.

## Data status

Synthetic, and declared as such in `manifest.json`, in `task.toml` and in the
README. The generator, the seeds implicit in its deterministic construction, the
intervention definitions and the convergence evidence are under
`authoring/provenance/`. No random number generator is used anywhere in the
forward model, so the archive is reproducible byte for byte; `validation.md`
records that check.
