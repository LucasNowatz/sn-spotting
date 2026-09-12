"""Sealed verifier.  Twelve gates; all must pass for a reward of 1.

  schema        artifacts present, shapes and order right, values finite
  bounds        the estimate lies inside the published bounds
  vapour        withheld vapour, uncertainty-normalised log RMSE
  pnsd          withheld size distributions, weighted log RMSE
  growth        number-weighted diameter derived from the predictions
  shares        region-resolved attribution, domain-wide and at the withheld
                stations, against tagged-tracer truth
  per_episode   both withheld episodes pass on their own
  calibration   predictive spreads against the withheld noisy observations
  posterior     width ratios and degrees of freedom are what the data support
  attribution   the counterfactual decomposition against the generator truth
  sensitivity   the restricted analysis shrinks the spread by the measured share
  replay        a sealed re-run of the estimate reproduces the predictions,
                fitted values and counterfactual statistics, and the reported
                chi-square equals the one recomputed from the fitted values
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gates import checks as G                               # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SUBMISSION = os.environ.get("EA_SUBMISSION", "/app/output")
PUB = os.path.join(HERE, "public")
SEALED = os.path.join(HERE, "sealed")


@pytest.fixture(scope="session")
def graded():
    lim = json.load(open(os.path.join(HERE, "limits.json")))
    prior = json.load(open(os.path.join(PUB, "prior.json")))
    windex = G.read_rows(os.path.join(PUB, "withheld_index.csv"))
    cindex = G.read_rows(os.path.join(PUB, "calibration_index.csv"))
    truth = np.load(os.path.join(SEALED, "network_truth.npz"))
    tj = json.load(open(os.path.join(SEALED, "truth.json")))
    events = list(tj["domain_region_shares"])
    centres = np.loadtxt(os.path.join(PUB, "diameter_bins.csv"), delimiter=",", skiprows=1, usecols=3)
    out = dict(lim=lim, windex=windex, cindex=cindex, truth_json=tj)
    errs, sub = G.read_submission(SUBMISSION, windex, cindex, prior["param_ids"], events, lim)
    out["schema"] = errs
    keys = ("bounds", "vapour", "pnsd", "growth", "shares", "per_episode", "calibration",
            "posterior", "attribution", "sensitivity")
    if sub is None:
        for k in keys:
            out[k] = ["not evaluated: the schema gate failed"]
        return out
    out["sub"] = sub
    T = G.truth_arrays(truth, windex)
    out["bounds"] = G.bounds(sub, prior)
    out["vapour"], _ = G.vapour(sub, T, lim)
    out["pnsd"], _ = G.pnsd(sub, T, lim)
    out["growth"], _ = G.growth(sub, T, centres, lim)
    out["shares"], _ = G.region_shares(sub, truth, tj, events, lim)
    out["per_episode"], _ = G.per_episode(sub, T, windex, lim)
    out["calibration"], _ = G.calibration(sub, T, lim)
    out["posterior"] = G.posterior(sub, prior, lim)
    out["attribution"] = G.attribution(sub, tj["attribution"], lim)
    out["sensitivity"] = G.sensitivity(sub, lim)
    return out


@pytest.fixture(scope="session")
def replayed(graded):
    if "sub" not in graded:
        return ["not evaluated: the schema gate failed"]
    sys.path.insert(0, os.path.join(HERE, "sealed_model"))
    import replay                                            # noqa: PLC0415
    errs, _ = replay.check(graded["sub"], graded["windex"], graded["cindex"], PUB, graded["lim"])
    return errs


def _fail(name, errs):
    if errs:
        pytest.fail(f"{name}: " + "; ".join(errs[:8]), pytrace=False)


def test_schema(graded): _fail("schema", graded["schema"])
def test_bounds(graded): _fail("bounds", graded["bounds"])
def test_vapour(graded): _fail("withheld vapour", graded["vapour"])
def test_pnsd(graded): _fail("withheld size distributions", graded["pnsd"])
def test_growth(graded): _fail("growth summary", graded["growth"])
def test_shares(graded): _fail("region shares", graded["shares"])
def test_per_episode(graded): _fail("per-episode floor", graded["per_episode"])
def test_calibration(graded): _fail("predictive calibration", graded["calibration"])
def test_posterior(graded): _fail("posterior", graded["posterior"])
def test_attribution(graded): _fail("attribution", graded["attribution"])
def test_sensitivity(graded): _fail("sensitivity", graded["sensitivity"])
def test_replay(replayed): _fail("sealed replay", replayed)
