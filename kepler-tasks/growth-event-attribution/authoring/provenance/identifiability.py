"""Whitened sensitivity analysis at the truth (authoring record).

    python identifiability.py [n_jobs]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "solution", "src"))
import bayes as BY                                          # noqa: E402
import network as NW                                        # noqa: E402

ENV = os.path.join(ROOT, "environment", "data")
OBSERVED = ["E01", "E02", "E03", "E04", "E05", "E06"]


def main():
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    theta = np.array(json.load(open(os.path.join(ROOT, "tests", "sealed", "truth.json")))["theta_true"])
    NW.load(ENV)
    rec = BY.Records(ENV, OBSERVED)
    inv = BY.Inversion(ENV, rec, list(range(12)), n_jobs)
    J, r = inv.jacobian(theta)
    Jd = J[:rec.n_obs]
    sv = np.linalg.svd(Jd, compute_uv=False)
    C = np.linalg.inv(J.T @ J); sd = np.sqrt(np.diag(C))
    corr = C / np.outer(sd, sd)
    prior_sd = np.sqrt(np.diag(inv.B))
    dof = float(np.trace(np.eye(12) - C @ np.linalg.inv(inv.B)))
    ids = inv.ids
    L = ["Whitened identifiability at the truth", "=" * 40,
         f"usable observations: {rec.n_obs}",
         "singular values: " + " ".join(f"{x:.3g}" for x in sv),
         f"condition number: {sv[0] / sv[-1]:.1f}",
         f"degrees of freedom for signal: {dof:.2f} of 12", "",
         f"{'parameter':14s} {'post sd':>9s} {'prior sd':>9s} {'ratio':>7s}"]
    for i, pid in enumerate(ids):
        L.append(f"{pid:14s} {sd[i]:9.4f} {prior_sd[i]:9.3f} {sd[i] / prior_sd[i]:7.3f}")
    L += ["", "posterior correlations beyond 0.5"]
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(corr[i, j]) > 0.5:
                L.append(f"  {ids[i]:14s} {ids[j]:14s} {corr[i, j]:+.3f}")
    text = "\n".join(L)
    print(text)
    open(os.path.join(HERE, "identifiability_report.txt"), "w").write(text + "\n")


if __name__ == "__main__":
    main()
