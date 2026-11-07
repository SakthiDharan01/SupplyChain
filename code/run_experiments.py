import json
import sys
import numpy as np
from sim import *  # noqa

# ---- hyper-parameters selected on the tuning seeds (see explore_surprise.py)
IND = dict(gamma=0.90, lam=0.0, wr=0.0)       # independent Beta (tuned gamma)
PRO = dict(gamma=0.85, lam=2.0, wr=0.5)       # proposed trust-graph filter
DELTA = 1                                     # human approval latency (days)
RHO = 0.8                                     # confidence gate
rng_boot = np.random.default_rng(0)


def trust_filter(y, W, lam, gamma):
    """Proposed: Beta belief per supplier; neighbours' *surprise* (outcome
    minus own forecast) is transferred as discounted pseudo-evidence."""
    af = np.zeros(N); as_ = np.zeros(N)
    P = np.zeros((N, T)); A = np.zeros((N, T)); B = np.zeros((N, T))
    pprev = np.full(N, A0 / (A0 + B0))
    for t in range(T):
        yt = y[:, t]
        s = W @ (yt - pprev)
        af = gamma * af + yt + lam * np.maximum(s, 0)
        as_ = gamma * as_ + (1 - yt) + lam * np.maximum(-s, 0)
        A[:, t] = af + A0; B[:, t] = as_ + B0
        P[:, t] = A[:, t] / (A[:, t] + B[:, t])
        pprev = P[:, t]
    return P, A, B


def forecasts(world, cfg, raw=False):
    W, _, _ = graph_weights(world, cfg["wr"])
    f = beta_filter if raw else trust_filter
    return f(world["y"], W, cfg["lam"], cfg["gamma"])


# ---- supervised baseline -------------------------------------------------
def train_gbm():
    X, Y = [], []
    for s in TRAIN_SEEDS:
        w = simulate(s)
        _, Wsup, Wreg = graph_weights(w, 0.5)
        p0 = static_forecast(w["y"])[0][:, 0]
        F = features(w["y"], p0, Wsup, Wreg)
        ts = np.arange(T0 - 1, T - 1)
        X.append(F[:, ts, :].reshape(-1, F.shape[-1]))
        Y.append(w["y"][:, ts + 1].reshape(-1))
    X = np.concatenate(X); Y = np.concatenate(Y)
    m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08,
                                       max_depth=4, random_state=0)
    m.fit(X, Y)
    return m


def gbm_forecast(model, world):
    _, Wsup, Wreg = graph_weights(world, 0.5)
    p0 = static_forecast(world["y"])[0][:, 0]
    F = features(world["y"], p0, Wsup, Wreg)
    ts = np.arange(T0 - 1, T - 1)
    P = np.full((N, T), 0.1)
    P[:, ts] = model.predict_proba(F[:, ts, :].reshape(-1, F.shape[-1]))[:, 1].reshape(N, -1)
    return P


def pooled(rows, key_num, key_den):
    return sum(r[key_num] for r in rows) / max(sum(r[key_den] for r in rows), 1)


def summarize(name, fm, pol):
    b = [x[0] for x in fm]; e = [x[1] for x in fm]
    c = [p["cost"] for p in pol]
    return dict(name=name,
                brier=ci95(b), ece=ci95(e), cost=ci95(c),
                protected=ci95([p["protected"] for p in pol]),
                onset=ci95([p["onset"] for p in pol]),
                auto_hit=pooled(pol, "auto_hits", "auto_acts"),
                auto_cov=pooled(pol, "auto_acts", "desired"),
                max_auto=max(p["max_auto_cost"] for p in pol))


def main(out):
    print("training GBM baseline ...", flush=True)
    gbm = train_gbm()
    out_all = {}

    def evaluate(seeds, shock_scale=1.0, full=True):
        R_ = {k: dict(fm=[], pol=[]) for k in
              ["none", "static_h", "ind_h", "gbm_h", "pro_h", "ind_g",
               "pro_g", "pro_n", "oracle"]}
        extra = dict(raw_h=[], raw_fm=[], supply_h=[], supply_fm=[],
                     region_h=[], region_fm=[], raw_g=[])
        sweeps = dict(delta={d: dict(h=[], g=[], n=[]) for d in [0, 1, 2, 3]},
                      rho={r: dict(ind=[], pro=[]) for r in
                           [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]})
        for s in seeds:
            w = simulate(s, shock_scale=shock_scale)
            y = w["y"]
            Ps, As, Bs = static_forecast(y)
            Pi, Ai, Bi = forecasts(w, IND)
            Pp, Ap, Bp = forecasts(w, PRO)
            fm_s = forecast_metrics(Ps, y); fm_i = forecast_metrics(Pi, y)
            fm_p = forecast_metrics(Pp, y)
            R_["none"]["fm"].append(fm_s); R_["none"]["pol"].append(run_policy(Ps, As, Bs, y, w, "none"))
            R_["oracle"]["fm"].append((0.0, 0.0)); R_["oracle"]["pol"].append(run_policy(Ps, As, Bs, y, w, "oracle"))
            R_["static_h"]["fm"].append(fm_s); R_["static_h"]["pol"].append(run_policy(Ps, As, Bs, y, w, "human", DELTA))
            R_["ind_h"]["fm"].append(fm_i); R_["ind_h"]["pol"].append(run_policy(Pi, Ai, Bi, y, w, "human", DELTA))
            R_["pro_h"]["fm"].append(fm_p); R_["pro_h"]["pol"].append(run_policy(Pp, Ap, Bp, y, w, "human", DELTA))
            R_["ind_g"]["fm"].append(fm_i); R_["ind_g"]["pol"].append(run_policy(Pi, Ai, Bi, y, w, "gov", DELTA, RHO))
            R_["pro_g"]["fm"].append(fm_p); R_["pro_g"]["pol"].append(run_policy(Pp, Ap, Bp, y, w, "gov", DELTA, RHO))
            R_["pro_n"]["fm"].append(fm_p); R_["pro_n"]["pol"].append(run_policy(Pp, Ap, Bp, y, w, "naive"))
            if full:
                Pg = gbm_forecast(gbm, w)
                R_["gbm_h"]["fm"].append(forecast_metrics(Pg, y))
                R_["gbm_h"]["pol"].append(run_policy(Pg, Ap, Bp, y, w, "human", DELTA))
                # ablations of the propagation channel
                for tag, cfg, raw in [("raw", dict(gamma=0.9, lam=0.25, wr=0.75), True),
                                      ("supply", dict(PRO, wr=0.0), False),
                                      ("region", dict(PRO, wr=1.0), False)]:
                    Pa, Aa, Ba = forecasts(w, cfg, raw=raw)
                    extra[tag + "_fm"].append(forecast_metrics(Pa, y))
                    extra[tag + "_h"].append(run_policy(Pa, Aa, Ba, y, w, "human", DELTA))
                    if tag == "raw":
                        extra["raw_g"].append(run_policy(Pa, Aa, Ba, y, w, "gov", DELTA, RHO))
                for d in sweeps["delta"]:
                    sweeps["delta"][d]["h"].append(run_policy(Pp, Ap, Bp, y, w, "human", d)["cost"])
                    sweeps["delta"][d]["g"].append(run_policy(Pp, Ap, Bp, y, w, "gov", d, RHO)["cost"])
                    sweeps["delta"][d]["n"].append(run_policy(Pp, Ap, Bp, y, w, "naive")["cost"])
                for r in sweeps["rho"]:
                    sweeps["rho"][r]["ind"].append(run_policy(Pi, Ai, Bi, y, w, "gov", DELTA, r))
                    sweeps["rho"][r]["pro"].append(run_policy(Pp, Ap, Bp, y, w, "gov", DELTA, r))
        return R_, extra, sweeps

    # ------------------------------------------------ main experiment
    R_, extra, sweeps = evaluate(TEST_SEEDS)
    names = dict(none="No mitigation", static_h="Static scorecard + approval",
                 ind_h="Independent Beta + approval", gbm_h="Supervised GBM + approval",
                 pro_h="Trust graph + approval", ind_g="Independent Beta + governed autonomy",
                 pro_g="Trust graph + governed autonomy", pro_n="Trust graph + unrestricted autonomy",
                 oracle="Oracle (perfect foresight)")
    main_tab = [summarize(names[k], R_[k]["fm"], R_[k]["pol"]) for k in names]
    out_all["main"] = main_tab
    print("\nMAIN TABLE")
    for r in main_tab:
        print(f"{r['name']:<42} Brier {r['brier'][0]:.4f}±{r['brier'][1]:.4f} ECE {r['ece'][0]:.4f} "
              f"cost {r['cost'][0]:.3f}±{r['cost'][1]:.3f} prot {r['protected'][0]:.3f} "
              f"onset {r['onset'][0]:.3f} hit {r['auto_hit']:.3f} cov {r['auto_cov']:.3f} maxA {r['max_auto']:.2f}")

    # paired differences
    def paired(a, b):
        d = np.array(a) - np.array(b)
        return ci95(d)
    br = lambda k: [x[0] for x in R_[k]["fm"]]
    co = lambda k: [p["cost"] for p in R_[k]["pol"]]
    pair = dict(
        brier_pro_minus_ind=paired(br("pro_h"), br("ind_h")),
        brier_pro_minus_gbm=paired(br("pro_h"), br("gbm_h")),
        brier_ind_minus_static=paired(br("ind_h"), br("static_h")),
        cost_pro_h_minus_ind_h=paired(co("pro_h"), co("ind_h")),
        cost_pro_g_minus_pro_h=paired(co("pro_g"), co("pro_h")),
        cost_ind_g_minus_ind_h=paired(co("ind_g"), co("ind_h")),
        cost_pro_g_minus_ind_g=paired(co("pro_g"), co("ind_g")),
        cost_pro_h_minus_gbm_h=paired(co("pro_h"), co("gbm_h")),
        cost_pro_h_minus_static_h=paired(co("pro_h"), co("static_h")),
        cost_pro_g_minus_static_h=paired(co("pro_g"), co("static_h")),
    )
    out_all["paired"] = pair
    print("\nPAIRED (mean, 95% half-width)")
    for k, v in pair.items():
        print(f"  {k}: {v[0]:+.4f} ± {v[1]:.4f}")
    # retained fraction of unrestricted-autonomy benefit
    ch, cg, cn, c0 = (np.mean(co(k)) for k in ["pro_h", "pro_g", "pro_n", "none"])
    out_all["retained_frac"] = float((ch - cg) / (ch - cn))
    out_all["savings_vs_none"] = {k: float((c0 - np.mean(co(k))) / c0) for k in names}
    print("retained fraction of unrestricted-autonomy saving:", out_all["retained_frac"])
    print("savings vs none:", {k: round(v, 3) for k, v in out_all["savings_vs_none"].items()})

    # ablation table
    abl = []
    abl.append(("Independent Beta (lambda=0)", R_["ind_h"]["fm"], R_["ind_h"]["pol"], R_["ind_g"]["pol"]))
    abl.append(("Raw-outcome propagation", extra["raw_fm"], extra["raw_h"], extra["raw_g"]))
    abl.append(("Surprise propagation, supply edges only", extra["supply_fm"], extra["supply_h"], None))
    abl.append(("Surprise propagation, region edges only", extra["region_fm"], extra["region_h"], None))
    abl.append(("Surprise propagation, both (proposed)", R_["pro_h"]["fm"], R_["pro_h"]["pol"], R_["pro_g"]["pol"]))
    out_all["ablation"] = []
    print("\nABLATION")
    for nm, fm, ph, pg in abl:
        row = dict(name=nm, brier=ci95([x[0] for x in fm]), ece=ci95([x[1] for x in fm]),
                   cost_h=ci95([p["cost"] for p in ph]),
                   cost_g=ci95([p["cost"] for p in pg]) if pg else None)
        out_all["ablation"].append(row)
        print(f"{nm:<44} Brier {row['brier'][0]:.4f} ECE {row['ece'][0]:.4f} costH {row['cost_h'][0]:.3f}"
              + (f" costG {row['cost_g'][0]:.3f}" if pg else ""))

    # sweeps
    out_all["delta"] = {d: {k: ci95(v) for k, v in sweeps["delta"][d].items()} for d in sweeps["delta"]}
    print("\nDELTA SWEEP (cost per supplier-day)")
    for d, v in out_all["delta"].items():
        print(d, {k: round(x[0], 3) for k, x in v.items()})
    out_all["rho"] = {}
    print("\nRHO SWEEP")
    for r in sweeps["rho"]:
        row = {}
        for tag in ["ind", "pro"]:
            pol = sweeps["rho"][r][tag]
            row[tag] = dict(hit=pooled(pol, "auto_hits", "auto_acts"),
                            cov=pooled(pol, "auto_acts", "desired"),
                            cost=ci95([p["cost"] for p in pol]),
                            waste=float(np.mean([p["auto_waste"] for p in pol])))
        out_all["rho"][r] = row
        print(r, {t: (round(row[t]["hit"], 3), round(row[t]["cov"], 3), round(row[t]["cost"][0], 3)) for t in row})

    # correlation-strength sweep (hyper-parameters NOT re-tuned)
    out_all["corr"] = {}
    print("\nCORRELATION SWEEP (Brier; shock_scale)")
    for sc in [0.0, 0.5, 1.0, 1.5, 2.0]:
        bi, bp = [], []
        for s in TEST_SEEDS:
            w = simulate(s, shock_scale=sc)
            bi.append(forecast_metrics(forecasts(w, IND)[0], w["y"])[0])
            bp.append(forecast_metrics(forecasts(w, PRO)[0], w["y"])[0])
        d = np.array(bp) - np.array(bi)
        out_all["corr"][sc] = dict(ind=ci95(bi), pro=ci95(bp), diff=ci95(d),
                                   rel=float(d.mean() / np.mean(bi)))
        print(sc, "ind", round(np.mean(bi), 4), "pro", round(np.mean(bp), 4),
              "diff", ci95(d), "rel", round(d.mean() / np.mean(bi), 4))

    with open(out, "w") as fh:
        json.dump(out_all, fh, indent=1, default=lambda o: float(o))
    print("saved", out)


if __name__ == "__main__":
    main(sys.argv[1])
