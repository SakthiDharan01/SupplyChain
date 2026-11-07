"""Revision experiments A: (1) calibration of the confidence-gate score,
(2) cost-ceiling-only control and supervised / hybrid forecasters under the
same execution policies, (3) guaranteed exposure bound, (4) stale-approval
handling.  Tuning seeds are used only to fit calibration maps; every reported
number is on the 50 test seeds (0-49).  Output: rev_a.json
"""
import json
import numpy as np
from scipy.stats import beta as bd
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sim import *  # noqa
from run_experiments import (forecasts, PRO, IND, DELTA, RHO, gbm_forecast,
                             train_gbm)

TS = np.arange(T0 - 1, T - 1)
TAUS = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.07, 0.05]
RHOS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def temper(P, A, B, tau):
    k = (A + B) * tau
    return P * k, (1 - P) * k


def collect(seeds, cfg=PRO):
    """Per-seed arrays for tier-1 suppliers on evaluation days."""
    out = []
    for s in seeds:
        w = simulate(s)
        y = w["y"]
        P, A, B = forecasts(w, cfg)
        L, c = w["L"], w["c"]
        qstar = (c / (L * E_EFF))[:, None]
        d = P[N2:] * (L * E_EFF)[:, None] > c[:, None]
        out.append(dict(w=w, P=P, A=A, B=B, qstar=qstar, d=d,
                        qn=np.concatenate([w["q"][N2:, 1:], np.zeros((N1, 1))], 1)))
    return out


def tail_score(rec, tau):
    A, B = temper(rec["P"][N2:], rec["A"][N2:], rec["B"][N2:], tau)
    return bd.sf(rec["qstar"], A, B)


def ece_binary(score, truth, nb=10):
    bins = np.minimum((score * nb).astype(int), nb - 1)
    e = 0.0
    for b in range(nb):
        m = bins == b
        if m.any():
            e += m.mean() * abs(score[m].mean() - truth[m].mean())
    return float(e)


def desired_arrays(recs, tau):
    S, Y = [], []
    for r in recs:
        sc = tail_score(r, tau)[:, TS]
        tr = (r["qn"][:, TS] > r["qstar"])
        m = r["d"][:, TS]
        S.append(sc[m]); Y.append(tr[m])
    return np.concatenate(S), np.concatenate(Y).astype(float)


def main():
    out = {}
    tune = collect(TUNE_SEEDS)
    test = collect(TEST_SEEDS)

    # ------------------------------------------------ (1) calibration of the gate score
    ece_tune = {}
    for tau in TAUS:
        s, yv = desired_arrays(tune, tau)
        ece_tune[tau] = ece_binary(s, yv)
    tau_star = min(ece_tune, key=ece_tune.get)
    s_tu, y_tu = desired_arrays(tune, 1.0)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(s_tu, y_tu)
    out["tau_star"] = tau_star
    out["ece_tune_by_tau"] = {str(k): v for k, v in ece_tune.items()}

    # credible-interval coverage of the hidden failure probability (all eval points)
    def coverage(recs, tau, lvl):
        hit = []
        for r in recs:
            A, B = temper(r["P"][N2:], r["A"][N2:], r["B"][N2:], tau)
            lo, hi = bd.ppf((1 - lvl) / 2, A, B), bd.ppf(1 - (1 - lvl) / 2, A, B)
            q = r["qn"]
            hit.append(((q >= lo) & (q <= hi))[:, TS].ravel())
        return float(np.concatenate(hit).mean())
    out["coverage"] = {lvl: dict(raw=coverage(test, 1.0, lvl), tempered=coverage(test, tau_star, lvl))
                       for lvl in (0.5, 0.8, 0.9)}

    # conformal intervals around the point forecast (normalised absolute error)
    def conf_scores(recs):
        z = []
        for r in recs:
            p = r["P"][N2:]; n = r["A"][N2:] + r["B"][N2:]
            sd = np.sqrt(p * (1 - p) / (n + 1))
            z.append((np.abs(r["qn"] - p) / sd)[:, TS].ravel())
        return np.concatenate(z)
    zt = conf_scores(tune)
    conf = {}
    for lvl in (0.8, 0.9):
        k = np.quantile(zt, min(1.0, np.ceil((len(zt) + 1) * lvl) / len(zt)))
        hits = []
        for r in test:
            p = r["P"][N2:]; n = r["A"][N2:] + r["B"][N2:]
            sd = np.sqrt(p * (1 - p) / (n + 1))
            hits.append((np.abs(r["qn"] - p) <= k * sd)[:, TS].ravel())
        conf[lvl] = dict(k=float(k), coverage=float(np.concatenate(hits).mean()))
    out["conformal"] = conf

    # reliability of the gate score and gate behaviour (test)
    def gate_eval(kind):
        rows = {r_: dict(share=[], spend=[], acts=[], cost=[], hit=[], maxc=0.0) for r_ in RHOS}
        eces = []
        for r in test:
            w = r["w"]; y = w["y"]
            P, A, B = r["P"], r["A"], r["B"]
            if kind == "raw":
                sc = tail_score(r, 1.0)
            elif kind == "temp":
                sc = tail_score(r, tau_star)
            else:
                sc = iso.predict(tail_score(r, 1.0).ravel()).reshape(N1, T)
            tr = (r["qn"] > r["qstar"])
            m = r["d"][:, TS]
            eces.append((sc[:, TS][m], tr[:, TS][m]))
            for rho in RHOS:
                pol = run_policy(P, A, B, y, w, "gov", DELTA, rho, score=sc)
                gate = r["d"] & (sc >= rho) & (w["c"] <= CMAX)[:, None]
                g = gate[:, TS]
                rows[rho]["share"].append((float((tr[:, TS] & g).sum()), float(g.sum())))
                rows[rho]["spend"].append(pol["auto_spend"])
                rows[rho]["acts"].append(pol["auto_acts"])
                rows[rho]["hit"].append((pol["auto_hits"], pol["auto_acts"]))
                rows[rho]["cost"].append(pol["cost"])
        S = np.concatenate([a for a, _ in eces]); Yv = np.concatenate([b for _, b in eces]).astype(float)
        res = dict(ece_score=ece_binary(S, Yv), rho={})
        for rho in RHOS:
            sh = rows[rho]["share"]
            res["rho"][str(rho)] = dict(
                share_worthwhile=sum(a for a, _ in sh) / max(sum(b for _, b in sh), 1),
                spend=float(np.mean(rows[rho]["spend"])),
                cost=float(np.mean(rows[rho]["cost"])),
                precision=sum(a for a, _ in rows[rho]["hit"]) / max(sum(b for _, b in rows[rho]["hit"]), 1),
                coverage_desired=sum(b for _, b in sh) / max(sum(float(r["d"][:, TS].sum()) for r in test), 1))
        return res
    out["gate"] = {k: gate_eval(k) for k in ("raw", "temp", "iso")}
    print("tau*", tau_star, "tuning ECE", {k: round(v, 3) for k, v in ece_tune.items()})
    print("coverage", out["coverage"], "conformal", conf)
    for k, v in out["gate"].items():
        print(k, "ECE(score)", round(v["ece_score"], 3),
              {r_: (round(x["share_worthwhile"], 3), round(x["spend"]), round(x["cost"], 4)) for r_, x in v["rho"].items()})

    # ------------------------------------------------ (2) controls under the same execution policies
    print("training supervised models ...", flush=True)
    gbm = train_gbm()

    def hyb_feats(w):
        _, Wsup, Wreg = graph_weights(w, 0.5)
        p0 = static_forecast(w["y"])[0][:, 0]
        F = features(w["y"], p0, Wsup, Wreg)
        P, A, B = forecasts(w, PRO)
        return np.concatenate([F, P[..., None], (A + B)[..., None]], -1)
    X, Yt = [], []
    for s in TRAIN_SEEDS:
        w = simulate(s)
        F = hyb_feats(w)
        X.append(F[:, TS, :].reshape(-1, F.shape[-1])); Yt.append(w["y"][:, TS + 1].reshape(-1))
    hyb = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_depth=4, random_state=0)
    hyb.fit(np.concatenate(X), np.concatenate(Yt))

    names = ["ind", "trust", "gbm", "hybrid"]
    modes = ["human", "human_reval", "ceil", "ceil_reval", "gov", "gov_reval", "naive"]
    acc = {n: {m: [] for m in modes} for n in names}
    fm = {n: [] for n in names}
    bound = []
    for r in test:
        w = r["w"]; y = w["y"]
        Pi, Ai, Bi = forecasts(w, IND)
        Pp, Ap, Bp = r["P"], r["A"], r["B"]
        Pg = gbm_forecast(gbm, w)
        F = hyb_feats(w)
        Ph = np.full((N, T), 0.1)
        Ph[:, TS] = hyb.predict_proba(F[:, TS, :].reshape(-1, F.shape[-1]))[:, 1].reshape(N, -1)
        fc = dict(ind=(Pi, Ai, Bi), trust=(Pp, Ap, Bp), gbm=(Pg, Ap, Bp), hybrid=(Ph, Ap, Bp))
        bound.append(float(w["c"][w["c"] <= CMAX].sum() * len(TS)))
        for n in names:
            P, A, B = fc[n]
            fm[n].append(forecast_metrics(P, y))
            acc[n]["human"].append(run_policy(P, A, B, y, w, "human", DELTA))
            acc[n]["human_reval"].append(run_policy(P, A, B, y, w, "human", DELTA, reval=True))
            acc[n]["ceil"].append(run_policy(P, A, B, y, w, "ceil", DELTA))
            acc[n]["ceil_reval"].append(run_policy(P, A, B, y, w, "ceil", DELTA, reval=True))
            acc[n]["naive"].append(run_policy(P, A, B, y, w, "naive"))
            if n in ("ind", "trust"):
                acc[n]["gov"].append(run_policy(P, A, B, y, w, "gov", DELTA, RHO))
                acc[n]["gov_reval"].append(run_policy(P, A, B, y, w, "gov", DELTA, RHO, reval=True))
    summ = {}
    for n in names:
        summ[n] = dict(brier=ci95([a for a, _ in fm[n]]), ece=ci95([b for _, b in fm[n]]))
        for m in modes:
            rows = acc[n][m]
            if not rows:
                continue
            summ[n][m] = dict(cost=ci95([x["cost"] for x in rows]),
                              spend=float(np.mean([x["auto_spend"] for x in rows])),
                              waste=float(np.mean([x["auto_waste"] for x in rows])),
                              precision=sum(x["auto_hits"] for x in rows) / max(sum(x["auto_acts"] for x in rows), 1),
                              onset=float(np.mean([x["onset"] for x in rows])))
    out["controls"] = summ
    out["bound_per_network"] = ci95(bound)

    def pdiff(n1, m1, n2, m2):
        a = np.array([x["cost"] for x in acc[n1][m1]]); b = np.array([x["cost"] for x in acc[n2][m2]])
        return ci95(a - b)
    out["paired_cost"] = {
        "trust_gov_minus_trust_human": pdiff("trust", "gov", "trust", "human"),
        "trust_ceil_minus_trust_human": pdiff("trust", "ceil", "trust", "human"),
        "trust_gov_minus_trust_ceil": pdiff("trust", "gov", "trust", "ceil"),
        "gbm_ceil_minus_gbm_human": pdiff("gbm", "ceil", "gbm", "human"),
        "gbm_ceil_minus_trust_gov": pdiff("gbm", "ceil", "trust", "gov"),
        "gbm_human_minus_trust_human": pdiff("gbm", "human", "trust", "human"),
        "hybrid_human_minus_gbm_human": pdiff("hybrid", "human", "gbm", "human"),
        "hybrid_ceil_minus_gbm_ceil": pdiff("hybrid", "ceil", "gbm", "ceil"),
        "hybrid_human_minus_trust_human": pdiff("hybrid", "human", "trust", "human"),
        "trust_gov_reval_minus_trust_gov": pdiff("trust", "gov_reval", "trust", "gov"),
        "trust_human_reval_minus_trust_human": pdiff("trust", "human_reval", "trust", "human"),
    }
    br = lambda n: np.array([a for a, _ in fm[n]])
    out["paired_brier"] = dict(hybrid_minus_gbm=ci95(br("hybrid") - br("gbm")),
                               hybrid_minus_trust=ci95(br("hybrid") - br("trust")),
                               gbm_minus_trust=ci95(br("gbm") - br("trust")))
    for n in names:
        print(n, "Brier %.4f ECE %.4f" % (summ[n]["brier"][0], summ[n]["ece"][0]),
              {m: (round(summ[n][m]["cost"][0], 4), round(summ[n][m]["spend"]), round(summ[n][m]["precision"], 3))
               for m in modes if m in summ[n]})
    print("bound per network", out["bound_per_network"])
    for k, v in out["paired_cost"].items():
        print(" ", k, "%+.4f +- %.4f" % v)
    for k, v in out["paired_brier"].items():
        print(" ", k, "%+.4f +- %.4f" % v)
    with open("rev_a.json", "w") as fh:
        json.dump(out, fh, indent=1, default=lambda o: float(o))


if __name__ == "__main__":
    main()
