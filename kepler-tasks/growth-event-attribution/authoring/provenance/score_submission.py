"""Report every gate's raw score for a submission, without pass/fail.

Used to set thresholds from measurement: run it on the oracle and on each
baseline, then choose limits that separate them.  With --replay the trusted
forward model is also run (about two minutes per submission) to report the
consistency, chi-square and counterfactual discrepancies.

    python score_submission.py [--replay] <submission_dir> [...]
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tests"))
from verifier import grade as G                              # noqa: E402
from verifier import metrics as MT                           # noqa: E402

ENV = os.path.join(ROOT, "environment", "data")
TESTS = os.path.join(ROOT, "tests")
HID = os.path.join(TESTS, "hidden_truth")
INPUTS = os.path.join(TESTS, "inputs")


def context():
    thr = json.load(open(os.path.join(TESTS, "thresholds.json")))
    prior = json.load(open(os.path.join(ENV, "priors.json")))
    truth = np.load(os.path.join(HID, "station_truth.npz"))
    index = list(csv.DictReader(open(os.path.join(ENV, "prediction_index.csv"))))
    cal_index = list(csv.DictReader(open(os.path.join(ENV, "calibration_index.csv"))))
    queries = json.load(open(os.path.join(HID, "query_truth.json")))
    params = json.load(open(os.path.join(HID, "parameters.json")))
    centres = np.loadtxt(os.path.join(ENV, "size_bins.csv"), delimiter=",",
                         skiprows=1, usecols=3)
    return dict(thr=thr, prior=prior, truth=truth, index=index,
                cal_index=cal_index, queries=queries, params=params,
                centres=centres)


def score(sub_dir, ctx, replay=False):
    thr, prior, index = ctx["thr"], ctx["prior"], ctx["index"]
    errs, sub = G.gate_schema(sub_dir, index, ctx["queries"],
                              prior["param_ids"], thr, ctx["cal_index"])
    if sub is None:
        return dict(schema=errs)
    T = G.truth_arrays(ctx["truth"], index)
    out = dict(schema=errs, bounds=G.gate_bounds(sub, prior))
    out["vapour"] = G.gate_vapour(sub, T, thr)[1]
    out["pnsd"] = G.gate_pnsd(sub, T, thr)[1]
    out["diameter"] = G.gate_growth(sub, T, ctx["centres"], thr)[1]
    out["age"], out["source"] = G.gate_history(sub, ctx["queries"], thr)[1]
    _, detail = G.gate_per_episode(sub, T, index, thr)
    out["per_episode"] = detail
    out["calibration"] = G.gate_calibration(sub, T, thr)[1]
    P = sub["post"]
    truth_theta = np.array(ctx["params"]["theta_true"])
    out["n_in_ci"] = int(np.sum((truth_theta >= P["ci_lower"]) &
                                (truth_theta <= P["ci_upper"])))
    out["z_scores"] = ((P["estimate"] - truth_theta) / P["posterior_sd"]).round(2).tolist()
    out["sd_ratio"] = P["posterior_prior_sd_ratio"].round(3).tolist()
    out["dof"] = P["dof_signal"]
    out["chi2_reported"] = P["chi2_calibration"]
    ta = ctx["params"]["attribution"]
    A = sub["attr"]
    out["attr"] = dict(A_E=A["A_E"], A_C=A["A_C"], A_E_err=A["A_E"] - ta["A_E"],
                       A_C_err=A["A_C"] - ta["A_C"], A_E_sd=A["A_E_sd"],
                       A_C_sd=A["A_C_sd"],
                       A_E_sd_fixed=A["A_E_sd_sources_fixed"],
                       A_C_sd_fixed=A["A_C_sd_sources_fixed"],
                       corr=A[thr["corr_key"]],
                       share_E=1 - (A["A_E_sd_sources_fixed"] / A["A_E_sd"]) ** 2,
                       share_C=1 - (A["A_C_sd_sources_fixed"] / A["A_C_sd"]) ** 2)
    if replay:
        sys.path.insert(0, os.path.join(TESTS, "trusted_forward"))
        import replay as RP                                  # noqa: PLC0415
        _, info = RP.check(sub, index, ctx["cal_index"], INPUTS, thr)
        out["replay"] = dict(dv=info["dv"], dl=info["dl"], fdv=info["fdv"],
                             fdl=info["fdl"], chi2_fit=info["chi2_fit"],
                             chi2_fit_v=info["chi2_fit_v"],
                             chi2_trusted=info["chi2_trusted"],
                             dq={k: A[k] - info["q"][k] for k in info["q"]})
    return out


def main():
    args = sys.argv[1:]
    replay = "--replay" in args
    dirs = [a for a in args if a != "--replay"]
    ctx = context()
    for d in dirs:
        s = score(d, ctx, replay)
        print(f"\n=== {os.path.basename(d)}")
        print(json.dumps(s, indent=1, default=float))


if __name__ == "__main__":
    main()
