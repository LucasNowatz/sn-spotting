"""Sealed verifier for the growth-event benchmark.

Gates, all of which must pass for a reward of 1:

  schema           artifacts present, shapes and ordering right, values
                   finite, spreads positive, source fractions normalised
  bounds           the estimate lies inside the published bounds
  vapour           withheld vapour, uncertainty-normalised log RMSE
  pnsd             withheld size distributions, weighted log RMSE
  growth           number-weighted diameter derived from the predictions
  age              mean particle age at the lineage queries
  sources          source-region attribution at the lineage queries
  per_episode      both withheld meteorological episodes pass on their own
  calibration      predictive intervals cover the withheld noisy observations
                   at the stated rate, with a log score that penalises
                   intervals that are too wide as well as too narrow
  posterior        posterior widths and degrees of freedom are what the
                   data support
  attribution      the counterfactual decomposition of the event statistic
                   matches the truth with a supportable uncertainty
  sensitivity      fixing the regional source strengths shrinks that
                   uncertainty by the measured share
  consistency      a trusted forward run of the submitted estimate reproduces
                   the submitted predictions and fitted values and the
                   submitted counterfactual statistics, and the reported
                   chi-square equals the one recomputed from the submitted
                   fitted values with the full covariance

Truth comes from a seeded generator at 2.5 km with source-tagged tracers and
an age moment; the reference solution never produces a graded value.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verifier import grade as G                             # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SUBMISSION = os.environ.get("GEA_SUBMISSION", "/app/output")
DATA = os.path.join(HERE, "inputs")
HID = os.path.join(HERE, "hidden_truth")
THR = os.path.join(HERE, "thresholds.json")


@pytest.fixture(scope="session")
def graded():
    thr = G.load_thresholds(THR)
    prior = json.load(open(os.path.join(DATA, "priors.json")))
    index = G.read_index(os.path.join(DATA, "prediction_index.csv"))
    cal_index = G.read_index(os.path.join(DATA, "calibration_index.csv"))
    queries = json.load(open(os.path.join(HID, "query_truth.json")))
    truth = np.load(os.path.join(HID, "station_truth.npz"))
    params = json.load(open(os.path.join(HID, "parameters.json")))
    centres = np.loadtxt(os.path.join(DATA, "size_bins.csv"), delimiter=",",
                         skiprows=1, usecols=3)

    out = dict(thr=thr, index=index, cal_index=cal_index, queries=queries,
               params=params)
    errs, sub = G.gate_schema(SUBMISSION, index, queries, prior["param_ids"],
                              thr, cal_index)
    out["schema"] = errs
    keys = ("bounds", "vapour", "pnsd", "growth", "age", "sources",
            "per_episode", "calibration", "posterior", "attribution",
            "sensitivity")
    if sub is None:
        for k in keys:
            out[k] = ["not evaluated: the schema gate failed"]
        return out
    out["sub"] = sub
    T = G.truth_arrays(truth, index)
    out["bounds"] = G.gate_bounds(sub, prior)
    out["vapour"], out["vapour_score"] = G.gate_vapour(sub, T, thr)
    out["pnsd"], out["pnsd_score"] = G.gate_pnsd(sub, T, thr)
    out["growth"], out["growth_score"] = G.gate_growth(sub, T, centres, thr)
    hist, hs = G.gate_history(sub, queries, thr)
    out["age"] = [e for e in hist if "age" in e]
    out["sources"] = [e for e in hist if "source" in e]
    out["per_episode"], out["episode_detail"] = G.gate_per_episode(
        sub, T, index, thr)
    out["calibration"], out["calibration_score"] = G.gate_calibration(
        sub, T, thr)
    out["posterior"], out["n_in_ci"] = G.gate_posterior(
        sub, np.array(params["theta_true"]), prior, thr)
    out["attribution"] = G.gate_attribution(sub, params["attribution"], thr)
    out["sensitivity"] = G.gate_sensitivity(sub, thr)
    return out


@pytest.fixture(scope="session")
def consistency(graded):
    if "sub" not in graded:
        return ["not evaluated: the schema gate failed"]
    sys.path.insert(0, os.path.join(HERE, "trusted_forward"))
    import replay                                            # noqa: PLC0415
    errs, _ = replay.check(graded["sub"], graded["index"], graded["cal_index"],
                           DATA, graded["thr"])
    return errs


def _fail(name, errs):
    if errs:
        pytest.fail(f"{name}: " + "; ".join(errs[:8]), pytrace=False)


def test_schema(graded):
    _fail("schema", graded["schema"])


def test_bounds(graded):
    _fail("parameter bounds", graded["bounds"])


def test_vapour(graded):
    _fail("withheld vapour", graded["vapour"])


def test_pnsd(graded):
    _fail("withheld size distributions", graded["pnsd"])


def test_growth(graded):
    _fail("growth summary", graded["growth"])


def test_age(graded):
    _fail("air-mass age", graded["age"])


def test_sources(graded):
    _fail("source attribution", graded["sources"])


def test_per_episode(graded):
    _fail("per-episode floor", graded["per_episode"])


def test_calibration(graded):
    _fail("predictive calibration", graded["calibration"])


def test_posterior(graded):
    _fail("posterior honesty", graded["posterior"])


def test_attribution(graded):
    _fail("event attribution", graded["attribution"])


def test_sensitivity(graded):
    _fail("source-strength sensitivity", graded["sensitivity"])


def test_consistency(consistency):
    _fail("forward consistency", consistency)
