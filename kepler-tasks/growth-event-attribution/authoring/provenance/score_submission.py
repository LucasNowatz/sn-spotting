"""Raw gate scores for a submission directory, without pass/fail.

    python score_submission.py [--replay] <dir> [...]
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tests"))
from gates import checks as G                                # noqa: E402

ENV = os.path.join(ROOT, "environment", "data")
TESTS = os.path.join(ROOT, "tests")
SEALED = os.path.join(TESTS, "sealed")
PUB = os.path.join(TESTS, "public")


def context():
    lim = json.load(open(os.path.join(TESTS, "limits.json")))
    prior = json.load(open(os.path.join(ENV, "prior.json")))
    tj = json.load(open(os.path.join(SEALED, "truth.json")))
    return dict(lim=lim, prior=prior, truth=np.load(os.path.join(SEALED, "network_truth.npz")),
                windex=list(csv.DictReader(open(os.path.join(ENV, "withheld_index.csv")))),
                cindex=list(csv.DictReader(open(os.path.join(ENV, "calibration_index.csv")))),
                tj=tj, events=list(tj["domain_region_shares"]),
                centres=np.loadtxt(os.path.join(ENV, "diameter_bins.csv"), delimiter=",", skiprows=1, usecols=3))


def score(d, ctx, replay=False):
    lim = ctx["lim"]
    errs, sub = G.read_submission(d, ctx["windex"], ctx["cindex"], ctx["prior"]["param_ids"], ctx["events"], lim)
    if sub is None:
        return dict(schema=errs)
    T = G.truth_arrays(ctx["truth"], ctx["windex"])
    out = dict(schema=errs, bounds=G.bounds(sub, ctx["prior"]))
    out["vapour"] = G.vapour(sub, T, lim)[1]
    out["pnsd"] = G.pnsd(sub, T, lim)[1]
    out["diameter"] = G.growth(sub, T, ctx["centres"], lim)[1]
    out["shares_domain"], out["shares_station"] = G.region_shares(sub, ctx["truth"], ctx["tj"], ctx["events"], lim)[1]
    out["per_episode"] = G.per_episode(sub, T, ctx["windex"], lim)[1]
    out["calibration"] = G.calibration(sub, T, lim)[1]
    P = sub["post"]; tt = np.array(ctx["tj"]["theta_true"])
    out["z_scores"] = ((P["estimate"] - tt) / P["posterior_sd"]).round(2).tolist()
    out["sd_ratio"] = P["posterior_prior_sd_ratio"].round(4).tolist()
    out["dof"] = P["dof_signal"]; out["chi2_reported"] = P["chi2_calibration"]
    ta = ctx["tj"]["attribution"]; A = sub["attr"]
    out["attr"] = dict(A_E=A["A_E"], A_C=A["A_C"], A_E_err=A["A_E"] - ta["A_E"], A_C_err=A["A_C"] - ta["A_C"],
                       A_E_sd=A["A_E_sd"], A_C_sd=A["A_C_sd"], A_E_sd_fixed=A["A_E_sd_sources_fixed"],
                       A_C_sd_fixed=A["A_C_sd_sources_fixed"], corr=A[lim["corr_key"]],
                       share_E=1 - (A["A_E_sd_sources_fixed"] / A["A_E_sd"]) ** 2,
                       share_C=1 - (A["A_C_sd_sources_fixed"] / A["A_C_sd"]) ** 2)
    if replay:
        sys.path.insert(0, os.path.join(TESTS, "sealed_model"))
        import replay as RP                                  # noqa: PLC0415
        _, info = RP.check(sub, ctx["windex"], ctx["cindex"], PUB, lim)
        out["replay"] = dict(dv=info["dv"], dl=info["dl"], fdv=info["fdv"], fdl=info["fdl"],
                             chi2_fit=info["chi2_fit"], chi2_fit_v=info["chi2_fit_v"],
                             dq={k: A[k] - info["q"][k] for k in info["q"]})
    return out


def main():
    args = sys.argv[1:]
    replay = "--replay" in args
    ctx = context()
    for d in [a for a in args if a != "--replay"]:
        print(f"\n=== {os.path.basename(d)}")
        print(json.dumps(score(d, ctx, replay), indent=1, default=float))


if __name__ == "__main__":
    main()
