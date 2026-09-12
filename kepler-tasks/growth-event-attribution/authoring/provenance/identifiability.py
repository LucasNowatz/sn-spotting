"""Identifiability analysis for the twelve log-parameters (authoring record).

Builds the whitened finite-difference sensitivity matrix of the visible
observations with respect to each log-parameter, using the full published
error covariance, then reports its singular values, the prior-whitened
degrees of freedom for signal, and the posterior correlations.  A near-null
singular direction means the episode set must be redesigned, not that the
prior should be tightened.

    python identifiability.py [n_jobs]
"""
import json
import os
import sys

import numpy as np
from scipy.linalg import cholesky, solve_triangular

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "solution", "src"))
import invert as IV                                         # noqa: E402
import observe_ref as OB                                    # noqa: E402

ENV = os.path.join(ROOT, "environment", "data")
HID = os.path.join(ROOT, "tests", "hidden_truth")
OBSERVED = ["E01", "E02", "E03", "E04", "E05", "E06"]


def main():
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    theta = np.array(json.load(open(os.path.join(HID, "parameters.json")))["theta_true"])
    OB.load_geometry(ENV)
    obs = IV.Observations(ENV, OBSERVED)
    inv = IV.Inversion(ENV, obs, list(range(12)), n_jobs)
    J, r = inv.jacobian(theta)
    n_data = obs.n_obs
    Jd = J[:n_data]                       # data part, whitened by R
    u, sv, vt = np.linalg.svd(Jd, full_matrices=False)
    C_data = np.linalg.inv(Jd.T @ Jd)
    H = J.T @ J
    C_post = np.linalg.inv(H)
    sd_post = np.sqrt(np.diag(C_post))
    sd_data = np.sqrt(np.diag(C_data))
    prior_sd = np.sqrt(np.diag(inv.B))
    corr = C_post / np.outer(sd_post, sd_post)
    dof = float(np.trace(np.eye(12) - C_post @ np.linalg.inv(inv.B)))
    ids = inv.ids

    lines = []
    lines.append("Identifiability of the twelve log-parameters")
    lines.append("=" * 46)
    lines.append(f"visible observations: {n_data} (vapour and usable count "
                 f"channels), whitened with the full published covariance")
    lines.append("")
    lines.append("per-parameter rms whitened response to +0.02 in the log")
    for i, pid in enumerate(ids):
        lines.append(f"  {pid:14s} {np.sqrt(np.mean(Jd[:, i] ** 2)) * 0.02:8.4f}")
    lines.append("")
    lines.append("singular values of the whitened data Jacobian: "
                 + "  ".join(f"{x:.3g}" for x in sv))
    lines.append(f"condition number: {sv[0] / sv[-1]:.1f}")
    lines.append("weakest direction:")
    for pid, w in sorted(zip(ids, vt[-1]), key=lambda t: -abs(t[1]))[:5]:
        lines.append(f"  {pid:14s} {w:+.3f}")
    lines.append("")
    lines.append(f"degrees of freedom for signal: {dof:.2f} of 12")
    lines.append("")
    lines.append("linearised posterior sd at the truth")
    lines.append(f"  {'parameter':14s} {'data only':>10s} {'with prior':>11s} "
                 f"{'prior':>7s} {'post/prior':>11s}")
    for i, pid in enumerate(ids):
        lines.append(f"  {pid:14s} {sd_data[i]:10.4f} {sd_post[i]:11.4f} "
                     f"{prior_sd[i]:7.3f} {sd_post[i] / prior_sd[i]:11.3f}")
    lines.append("")
    lines.append("posterior correlations above 0.6 in magnitude")
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(corr[i, j]) > 0.6:
                lines.append(f"  {ids[i]:14s} {ids[j]:14s} {corr[i, j]:+.3f}")
    lines.append("")
    lines.append("correlations involving the event anomalies")
    for j in (10, 11):
        for i in range(12):
            if i != j and abs(corr[i, j]) > 0.25:
                lines.append(f"  {ids[j]:14s} {ids[i]:14s} {corr[i, j]:+.3f}")
    text = "\n".join(lines)
    print(text)
    open(os.path.join(HERE, "identifiability_report.txt"), "w").write(text + "\n")


if __name__ == "__main__":
    main()
