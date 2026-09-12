"""Write CALIBRATION.md, baselines.md and VALIDATION.md from the measured
numbers (floor*.json, limit_inputs.json, limits.json, runs.txt,
identifiability_report.txt)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROV = os.path.join(HERE, "..", "provenance")
ROOT = os.path.join(HERE, "..", "..")

fl = json.load(open(os.path.join(PROV, "floor.json")))
f30 = json.load(open(os.path.join(PROV, "floor_dt30.json")))
f120 = json.load(open(os.path.join(PROV, "floor_dt120.json")))
li = json.load(open(os.path.join(PROV, "limit_inputs.json")))
lim = json.load(open(os.path.join(ROOT, "tests", "limits.json")))
o, B = li["oracle"], li["baselines"]
ta, qp = fl["q_truth"], fl["q_public"]
runs = sorted(open(os.path.join(HERE, "runs.txt")).read().strip().split("\n"))


def fmt(v):
    if isinstance(v, dict):
        return json.dumps({k: ([round(x, 4) for x in vv] if isinstance(vv, list) else round(vv, 4))
                           for k, vv in v.items()})
    if isinstance(v, list):
        return json.dumps([round(x, 4) if isinstance(x, (int, float)) else x for x in v])
    if isinstance(v, str):
        return v
    return f"{v:.4g}"


# ------------------------------------------------------------ CALIBRATION.md
L = ["# Limit calibration", "",
     "Every limit clears two bars: comfortably above what a correct solution",
     "achieves (the measured floor and the oracle's own score) and comfortably",
     "below the cheapest shortcut that the gate exists to catch. `set_limits.py`",
     "applies that mechanically and refuses to write `limits.json` if any accuracy",
     "limit cannot satisfy both; its inputs are in `limit_inputs.json`.", "",
     "## The achievable floor", "",
     "`calibrate.py`: the reference solver on the public 4 km grid at the true",
     "parameters against the generator's own run. No solution can do better;",
     "the difference is discretisation error, not a modelling mistake.", "",
     "| metric | 30 s | 60 s (reference) | 120 s |", "| --- | --- | --- | --- |"]
for k, name in [("vapour", "withheld vapour, log NRMSE"), ("pnsd", "withheld size distributions, weighted log RMSE"),
                ("diameter", "number-weighted diameter MAE, nm"), ("domain_share_l1", "domain region shares, L1"),
                ("station_share_l1", "station region shares, L1"), ("chi2_truth", "chi-square at the truth")]:
    L.append(f"| {name} | {f30[k]:.4g} | {fl[k]:.4g} | {f120[k]:.4g} |")
fmc = json.load(open(os.path.join(PROV, "floor_mc.json"))); ffi = json.load(open(os.path.join(PROV, "floor_fine.json")))
L += ["", "30 s matches 60 s on every quantity; 120 s is under-resolved, as the",
      "specification discloses.", "",
      "Two other correct solvers were measured at the truth with the generator's own",
      "code, so that no limit fails a solver for differing from the generator: the",
      "monotonized-central limiter on the published grid, and Koren's limiter on a",
      "2 km grid with a 30 s step. Every limit uses the worst of the three floors.", "",
      "| metric | reference scheme | other limiter | 2 km grid |", "| --- | --- | --- | --- |"]
for k, name in [("vapour", "vapour"), ("pnsd", "size distributions"), ("diameter", "diameter, nm"),
                ("domain_share_l1", "domain shares"), ("station_share_l1", "station shares"), ("chi2_truth", "chi-square at the truth")]:
    L.append(f"| {name} | {fl[k]:.4g} | {fmc[k]:.4g} | {ffi[k]:.4g} |")
L += [f"| worst counterfactual difference, cm-3 | "
      f"{max(abs(fl['q_public'][k] - fl['q_truth'][k]) for k in ('Q00','Q10','Q01','Q11')):.2f} | "
      f"{max(abs(fmc['q_public'][k] - fmc['q_truth'][k]) for k in ('Q00','Q10','Q01','Q11')):.2f} | "
      f"{max(abs(ffi['q_public'][k] - ffi['q_truth'][k]) for k in ('Q00','Q10','Q01','Q11')):.2f} |",
      "", "Per episode at 60 s (vapour, size distributions):", ""]
for e in sorted(fl["per_episode"]):
    L.append(f"- {e}: {fl['per_episode'][e][0]:.3f}, {fl['per_episode'][e][1]:.3f}")
L += ["", "Counterfactual statistics, reference solver at the truth against the generator's runs:", "",
      "| statistic | reference at truth | generator | difference |", "| --- | --- | --- | --- |"]
for k in ("Q00", "Q10", "Q01", "Q11", "A_E", "A_C"):
    L.append(f"| {k} | {qp[k]:.2f} | {ta[k]:.2f} | {qp[k] - ta[k]:+.2f} |")
c = o["calibration"]; a = o["attr"]
L += ["", "## The oracle", "",
      f"Vapour {o['vapour']:.3f}, size distributions {o['pnsd']:.3f}, diameter {o['diameter']:.4f} nm,",
      f"domain shares {o['shares_domain']:.3f}, station shares {o['shares_station']:.3f}; coverage",
      f"{c['cov_v']:.3f} (vapour) and {c['cov_n']:.3f} (counts) with log scores {c['ls_v']:.2f} and {c['ls_n']:.2f};",
      f"dof {o['dof']:.2f}; chi-square {o['chi2_reported']:.1f}. A_E {a['A_E']:.1f} (truth {ta['A_E']:.1f}),",
      f"sd {a['A_E_sd']:.2f}, restricted {a['A_E_sd_fixed']:.2f}, variance share {a['share_E']:.2f}; A_C {a['A_C']:.1f}",
      f"(truth {ta['A_C']:.1f}), sd {a['A_C_sd']:.2f}; corr(log_s_event, log_sA) {a['corr']:.3f}.", "",
      "Parameter z-scores against the truth: " + ", ".join(f"{z:+.1f}" for z in o["z_scores"]) + ".",
      "Discretisation differences are absorbed into the parameters; see",
      "DESIGN_NOTES.md for why point values are graded only through the bounds.", "",
      "## Rules", "",
      "Accuracy limits: the larger of a multiple of the floor and of the oracle",
      "(vapour 4x/2.5x, size distributions 1.8x/1.8x, diameter 3x/2.5x, shares",
      "3x/2.5x), below 0.40 to 0.70 of the nearest shortcut. Per-episode limits",
      "1.6x the global ones, both withheld days alone. Re-run envelopes 2x the",
      "global limits. Coverage lower bound min(0.80, oracle - 0.05), upper 0.97;",
      "log-score limits oracle + 0.25; width-ratio bands one third to three times",
      "the oracle's ratio for the tightest and two loosest parameters; dof band",
      "oracle - 1.5 to 12; attribution tolerance the largest of 2x the oracle's",
      "error, 4x the floor difference and 25% of the truth; spread band 0.5x to 2x",
      "the oracle's; variance-share band oracle - 0.20 to + 0.15; correlation band",
      "oracle +- 0.22; chi-square report tolerance 1%, ceiling the larger of 1.5x",
      "the oracle and 1.3x the worst floor; statistic tolerance the largest of 4x",
      "the oracle's re-run discrepancy, 1.5x the worst floor difference and 1% of",
      "Q11.", "", "## Limits", "", "| limit | value |", "| --- | --- |"]
for k in sorted(k for k in lim if not k.startswith("_") and k not in ("sigma_vapour_log", "sigma_counts_log")):
    L.append(f"| `{k}` | {fmt(lim[k])} |")
open(os.path.join(PROV, "CALIBRATION.md"), "w").write("\n".join(L) + "\n")

# ------------------------------------------------------------ baselines.md
M = ["# Baselines and adversarial attempts", "",
     "Measured on the frozen dataset before any frontier trial; limits were set",
     f"from these numbers. Truth: A_E = {ta['A_E']:.1f}, A_C = {ta['A_C']:.1f} cm-3.", "",
     "| attempt | vapour | pnsd | diam | shares D | shares S | cov V | cov N | ls V | ls N | dof | A_E err | A_C err | sd A_E | share | corr |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]


def row(name, b):
    c = b["calibration"]; a = b["attr"]
    return (f"| {name} | {b['vapour']:.3f} | {b['pnsd']:.3f} | {b['diameter']:.4f} | {b['shares_domain']:.3f} | "
            f"{b['shares_station']:.3f} | {c['cov_v']:.3f} | {c['cov_n']:.3f} | {c['ls_v']:.2f} | {c['ls_n']:.2f} | "
            f"{b['dof']:.2f} | {a['A_E_err']:+.1f} | {a['A_C_err']:+.1f} | {a['A_E_sd']:.2f} | {a['share_E']:.2f} | {a['corr']:.2f} |")


M.append(row("reference", o))
for n in sorted(B):
    M.append(row(n, B[n]))
M += ["", "Where a re-run decides the outcome: the diagonal-covariance attempt reports",
      f"{B['diagonal_chi2']['chi2_reported']:.1f} against {o['chi2_reported']:.1f} recomputed from its own fitted values",
      f"(tolerance {lim['chi2_report_tol']:.1f}); the linearised attribution's Q10 and Q01 differ from the",
      f"sealed re-run by {B['linearised']['replay']['dq']['Q10']:+.1f} and {B['linearised']['replay']['dq']['Q01']:+.1f} cm-3 "
      f"(tolerance {lim['statistic_tol']:.1f}); the borrowed-operator attempt's",
      f"withheld size distributions differ by {B['wrong_operator']['replay']['dl']:.2f} sigma from the re-run and it",
      "is caught by the size-distribution, diameter and calibration gates.", "",
      "## Why each fails", "",
      "- **climatology, nearest_station**: every accuracy gate by one to two orders",
      "  of magnitude, the calibration gates, the shares, and the sealed re-run,",
      "  since their fitted values are the observations and their estimate the prior.",
      "- **zero_width**: coverage collapses, log scores explode, width ratios and",
      "  attribution spreads fall below their bands.",
      "- **hedged**: coverage 1.0 above the band, log scores above their limits,",
      "  width ratios and attribution spreads above their bands.",
      "- **copied_attribution**: the statistics copied at the truth with prior",
      "  parameters; the re-run from the submitted estimate gives the prior-only",
      "  values, and every accuracy gate fails.",
      "- **linearised**: the split built from one-at-a-time responses without the",
      "  interaction; its implied Q10 and Q01 differ from the re-run beyond tolerance.",
      "- **false_sensitivity**: a restricted spread larger than the joint one.",
      "- **shuffled**: correct values in permuted rows; accuracy, calibration and",
      "  re-run gates fail.",
      "- **wrong_operator**: S1's operator applied at the withheld stations; the",
      "  size-distribution, diameter and count-calibration gates fail.",
      "- **no_anomaly**: anomalies frozen at zero; the attribution is 0 against the",
      "  truth and the event-day predictions no longer match a re-run of the estimate.",
      "- **diagonal_chi2**: the honest chi-square of a diagonal fit differs from",
      "  the full-covariance value by hundreds against a one-per-cent tolerance.",
      "- **uniform_shares**: constant region shares; the shares gate fails."]
open(os.path.join(HERE, "baselines.md"), "w").write("\n".join(M) + "\n")

# ------------------------------------------------------------ VALIDATION.md
ident = open(os.path.join(PROV, "identifiability_report.txt")).read().strip()
V = ["# Validation record", "",
     "Nothing here is mounted into a container or executed by the verifier.", "",
     "## A. Physics", "",
     "`pilot_checks.py`: number conservation exact; zero-wind box vapour 6.8e-3;",
     "column growth 7.6e-3; Gaussian plume 8.9e-6; uniform tracer through all",
     "eight episodes 3e-15. Step refinement in `CALIBRATION.md`: 30 s matches 60 s,",
     "120 s degrades every quantity several-fold, as disclosed.", "",
     "## B. Identifiability", "", "```", ident, "```", "",
     "## C. Limits", "",
     "`CALIBRATION.md` and `limit_inputs.json`; `set_limits.py` refuses to write a",
     "limit that cannot separate the oracle from the nearest shortcut.", "",
     "## D. Baselines", "",
     "`baselines.md`: twelve attempts, all scoring zero on the gate built for them.", "",
     "## E. Verifier runs", "",
     "`runs.txt`, from `finish_validation.sh` outside Docker; per-gate logs in",
     "`verifier_runs/`:", "", "| run | result |", "| --- | --- |"]
for r in runs:
    n, res = r.split(": ", 1)
    V.append(f"| {n} | {res} |")
V += ["", "The oracle passes every gate on three runs of the same output; the verifier",
      "is deterministic and takes about a minute against a 3600 s timeout. An empty",
      "submission fails every gate immediately.", "",
      "## F. Harness gates", "",
      "Docker was unavailable on the authoring machine, so the image builds and the",
      "harbor gates (`harbor run -p . -a oracle -e docker`, `... -a nop ...`,",
      "`harbor check .`) must be run before submission. Both Dockerfiles pin every",
      "pip install, pin no apt package, and clean the apt lists; the verifier",
      "installs nothing at run time.", "",
      "## G. The author's part", "",
      "`instruction.md` must be rewritten by the submitting author, and the metadata",
      "and explanations in `task.toml` owned; see the README."]
open(os.path.join(HERE, "VALIDATION.md"), "w").write("\n".join(V) + "\n")
print("records written")
