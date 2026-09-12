"""Shortcut and adversarial submissions that must all score zero.

    python baselines.py <data_dir> <tests_dir> <oracle_dir> <out_root> [names...]
"""
import csv
import json
import os
import shutil
import sys

import numpy as np
from netCDF4 import Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
NB = 12


def rows(p):
    return list(csv.DictReader(open(p)))


def observed(data_dir, episodes):
    out = {}
    for e in episodes:
        with Dataset(os.path.join(data_dir, "episodes", e, "station_records.nc")) as d:
            sid = [str(s) for s in d["station_id"][:]]
            v = np.array(d["vapour"][:], float); n = np.array(d["counts"][:], float)
            qv = np.array(d["vapour_flag"][:], int)
        for i, s in enumerate(sid):
            if qv[i, 0] > 0:
                out[(e, s)] = (v[i], n[i])
    return out


def xy(data_dir):
    return {r["station_id"]: (float(r["x_m"]), float(r["y_m"])) for r in rows(os.path.join(data_dir, "network.csv"))}


def load_oracle(d):
    return (json.load(open(os.path.join(d, "posterior.json"))),
            dict(np.load(os.path.join(d, "predictions.npz"))),
            json.load(open(os.path.join(d, "attribution.json"))),
            json.load(open(os.path.join(d, "region_shares.json"))))


def write(out, post, pr, att, shr):
    shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
    json.dump(post, open(os.path.join(out, "posterior.json"), "w"), indent=1)
    np.savez(os.path.join(out, "predictions.npz"), **pr)
    json.dump(att, open(os.path.join(out, "attribution.json"), "w"), indent=1)
    json.dump(shr, open(os.path.join(out, "region_shares.json"), "w"), indent=1)


def prior_post(data_dir):
    p = json.load(open(os.path.join(data_dir, "prior.json")))
    mu, sd = np.array(p["mu"]), np.array(p["sigma"])
    return dict(param_ids=p["param_ids"], estimate=mu.tolist(), posterior_sd=sd.tolist(),
                ci_lower=(mu - 1.96 * sd).tolist(), ci_upper=(mu + 1.96 * sd).tolist(),
                posterior_prior_sd_ratio=[1.0] * 12, dof_signal=0.0, chi2_calibration=1.0e5,
                chi2_by_stream=dict(vapour=2.0e4, counts=8.0e4))


def guessed_att():
    return dict(units="cm-3", Q00=250.0, Q10=330.0, Q01=320.0, Q11=440.0, A_E=100.0, A_C=90.0,
                A_E_sd=10.0, A_C_sd=8.0, A_E_sd_sources_fixed=7.0, A_C_sd_sources_fixed=6.0,
                corr_s_event_sA=-0.5)


def guessed_shares(events):
    return dict(domain={e: dict(A=0.4, B=0.3, C=0.2) for e in events},
                stations={e: {s: dict(A=0.4, B=0.3, C=0.2) for s in ("W1", "W2")} for e in events})


def observed_fit(ctx):
    cal = rows(os.path.join(ctx["data"], "calibration_index.csv"))
    fv = np.zeros(len(cal)); fn = np.zeros((len(cal), NB))
    for r in cal:
        i = int(r["row_id"]); v, n = ctx["obs"][(r["episode"], r["station"])]; k = int(r["time_index"])
        fv[i] = max(v[k], 1e-3); fn[i] = np.maximum(n[k], 0.0)
    return fv, fn


def _shortcut(ctx, vap, pns, frac_guess):
    fv, fn = observed_fit(ctx)
    pr = dict(row_id=np.arange(len(vap)), vapour=np.maximum(vap, 1e-3),
              vapour_log_sd=np.full(len(vap), 0.3), pnsd=pns,
              pnsd_log_sd=np.full((len(vap), NB), 0.5), fit_vapour=fv, fit_pnsd=fn)
    return prior_post(ctx["data"]), pr, guessed_att(), guessed_shares(ctx["events"])


def climatology(ctx):
    idx = ctx["windex"]; nt = 49
    vs = np.zeros(nt); ns = np.zeros((nt, NB)); c = 0
    for (v, n) in ctx["obs"].values():
        vs += v; ns += n; c += 1
    vap = np.array([vs[int(r["time_index"])] / c for r in idx])
    pns = np.array([ns[int(r["time_index"])] / c for r in idx])
    return _shortcut(ctx, vap, pns, None)


def nearest_station(ctx):
    idx = ctx["windex"]; pos = xy(ctx["data"])
    seen = sorted({s for (_, s) in ctx["obs"]})
    vap = np.zeros(len(idx)); pns = np.zeros((len(idx), NB))
    for r in idx:
        i = int(r["row_id"]); k = int(r["time_index"]); tx, ty = pos[r["station"]]
        best = min((s for s in seen if (r["episode"], s) in ctx["obs"]),
                   key=lambda s: (pos[s][0] - tx) ** 2 + (pos[s][1] - ty) ** 2, default=None)
        key = (r["episode"], best) if best else sorted(ctx["obs"])[0]
        v, n = ctx["obs"][key]; vap[i] = v[k]; pns[i] = n[k]
    return _shortcut(ctx, vap, pns, None)


def zero_width(ctx):
    post, pr, att, shr = load_oracle(ctx["oracle"])
    post, pr, att = dict(post), dict(pr), dict(att)
    sd = np.full(12, 1e-4); est = np.array(post["estimate"])
    sig = np.array(json.load(open(os.path.join(ctx["data"], "prior.json")))["sigma"])
    post.update(posterior_sd=sd.tolist(), ci_lower=(est - 1.96 * sd).tolist(),
                ci_upper=(est + 1.96 * sd).tolist(), posterior_prior_sd_ratio=(sd / sig).tolist(), dof_signal=12.0)
    pr["vapour_log_sd"] = np.full_like(pr["vapour_log_sd"], 1e-3)
    pr["pnsd_log_sd"] = np.full_like(pr["pnsd_log_sd"], 1e-3)
    att.update(A_E_sd=0.05, A_C_sd=0.05, A_E_sd_sources_fixed=0.04, A_C_sd_sources_fixed=0.04)
    return post, pr, att, shr


def hedged(ctx):
    post, pr, att, shr = load_oracle(ctx["oracle"])
    post, pr, att = dict(post), dict(pr), dict(att)
    sig = np.array(json.load(open(os.path.join(ctx["data"], "prior.json")))["sigma"])
    sd = np.minimum(np.array(post["posterior_sd"]) * 8.0, sig); est = np.array(post["estimate"])
    post.update(posterior_sd=sd.tolist(), ci_lower=(est - 1.96 * sd).tolist(),
                ci_upper=(est + 1.96 * sd).tolist(), posterior_prior_sd_ratio=(sd / sig).tolist())
    pr["vapour_log_sd"] = pr["vapour_log_sd"] * 8.0; pr["pnsd_log_sd"] = pr["pnsd_log_sd"] * 8.0
    for k in ("A_E_sd", "A_C_sd", "A_E_sd_sources_fixed", "A_C_sd_sources_fixed"):
        att[k] *= 8.0
    return post, pr, att, shr


def copied_attribution(ctx):
    post, pr, _, shr = climatology(ctx)
    t = json.load(open(os.path.join(ctx["tests"], "sealed", "truth.json")))["attribution"]
    att = guessed_att(); att.update({k: t[k] for k in ("Q00", "Q10", "Q01", "Q11", "A_E", "A_C")})
    return post, pr, att, shr


def linearised(ctx):
    post, pr, att, shr = load_oracle(ctx["oracle"])
    att = dict(att); q00, q11 = att["Q00"], att["Q11"]; tot = q11 - q00
    d_e, d_c = att["Q10"] - q00, att["Q01"] - q00
    att["A_E"] = tot * d_e / (d_e + d_c); att["A_C"] = tot * d_c / (d_e + d_c)
    att["Q10"] = q00 + att["A_E"]; att["Q01"] = q00 + att["A_C"]
    return post, pr, att, shr


def false_sensitivity(ctx):
    post, pr, att, shr = load_oracle(ctx["oracle"])
    att = dict(att); att["A_E_sd_sources_fixed"] = att["A_E_sd"] * 1.2; att["A_C_sd_sources_fixed"] = att["A_C_sd"] * 1.2
    return post, pr, att, shr


def shuffled(ctx):
    post, pr, att, shr = load_oracle(ctx["oracle"])
    pr = dict(pr); perm = np.random.default_rng(3).permutation(len(pr["vapour"]))
    for k in ("vapour", "vapour_log_sd", "pnsd", "pnsd_log_sd"):
        pr[k] = pr[k][perm]
    return post, pr, att, shr


def wrong_operator(ctx):
    post, pr, att, shr = load_oracle(ctx["oracle"])
    pr = dict(pr)
    with Dataset(os.path.join(ctx["data"], "instruments.nc")) as d:
        ids = [str(s) for s in d["station_id"][:]]; ops = np.array(d["operator"][:])
    M = {s: ops[i] for i, s in enumerate(ids)}; pinv = {s: np.linalg.pinv(M[s]) for s in ids}
    pns = pr["pnsd"].copy()
    for r in ctx["windex"]:
        i = int(r["row_id"]); s = r["station"]
        pns[i] = M["S1"] @ np.maximum(pinv[s] @ pns[i], 0.0)
    pr["pnsd"] = pns
    return post, pr, att, shr


def no_anomaly(ctx):
    post, pr, att, shr = load_oracle(ctx["oracle"])
    post, att = dict(post), dict(att)
    est = np.array(post["estimate"]); est[10] = 0.0; est[11] = 0.0
    sd = np.array(post["posterior_sd"])
    post.update(estimate=est.tolist(), ci_lower=(est - 1.96 * sd).tolist(), ci_upper=(est + 1.96 * sd).tolist())
    q = att["Q00"]; att.update(Q00=q, Q10=q, Q01=q, Q11=q, A_E=0.0, A_C=0.0)
    return post, pr, att, shr


def diagonal_chi2(ctx):
    sys.path.insert(0, os.path.join(HERE, "..", "..", "solution", "src"))
    import network as NW
    import bayes as BY
    post, pr, att, shr = load_oracle(ctx["oracle"])
    post = dict(post)
    NW.load(ctx["data"])
    rec = BY.Records(ctx["data"], ["E01", "E02", "E03", "E04", "E05", "E06"])
    pred = BY.predict_many([np.array(post["estimate"])], rec.episodes, ctx["data"], rec.stations, 4)[0]
    cv = cn = 0.0
    for e, s, kind, y, L, m in rec.blocks:
        if kind == "vapour":
            h = np.log(np.maximum(pred[(e, s)]["vapour"], BY.LOG_FLOOR_V))
            cv += float((((y - h) ** 2) / np.diag(NW.vapour_cov(s))).sum())
        else:
            h = np.log(np.maximum(pred[(e, s)]["counts"].ravel()[m], BY.LOG_FLOOR_N))
            cn += float((((y - h) ** 2) / np.diag(NW.counts_cov(s))[m]).sum())
    post["chi2_calibration"] = cv + cn; post["chi2_by_stream"] = dict(vapour=cv, counts=cn)
    return post, pr, att, shr


def uniform_shares(ctx):
    """Correct analysis but the region shares replaced by a plausible constant."""
    post, pr, att, _ = load_oracle(ctx["oracle"])
    return post, pr, att, guessed_shares(ctx["events"])


BASELINES = dict(climatology=climatology, nearest_station=nearest_station, zero_width=zero_width,
                 hedged=hedged, copied_attribution=copied_attribution, linearised=linearised,
                 false_sensitivity=false_sensitivity, shuffled=shuffled, wrong_operator=wrong_operator,
                 no_anomaly=no_anomaly, diagonal_chi2=diagonal_chi2, uniform_shares=uniform_shares)


def main():
    data_dir, tests_dir, oracle_dir, out_root = sys.argv[1:5]
    which = sys.argv[5:] or list(BASELINES)
    tj = json.load(open(os.path.join(tests_dir, "sealed", "truth.json")))
    ctx = dict(data=data_dir, tests=tests_dir, oracle=oracle_dir, events=list(tj["domain_region_shares"]),
               windex=rows(os.path.join(data_dir, "withheld_index.csv")),
               obs=observed(data_dir, ["E01", "E02", "E03", "E04", "E05", "E06"]))
    for name in which:
        write(os.path.join(out_root, name), *BASELINES[name](ctx))
        print("wrote", name)


if __name__ == "__main__":
    main()
