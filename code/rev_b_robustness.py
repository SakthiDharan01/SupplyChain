"""Revision experiments B: robustness of the trust graph (hyper-parameters are NOT
re-tuned in any condition).  (1) misspecified graphs, (2) missing and delayed
observations, (3) different disruption generators.  Test seeds 0-49.
Output: rev_b.json
"""
import json
import numpy as np
from sim import *  # noqa
from run_experiments import IND, PRO, DELTA, RHO

TS = np.arange(T0 - 1, T - 1)


def trust_filter_obs(y, mask, W, lam, gamma):
    """Surprise-propagation filter that tolerates unobserved supplier-days."""
    af = np.zeros(N); as_ = np.zeros(N)
    P = np.zeros((N, T)); A = np.zeros((N, T)); B = np.zeros((N, T))
    pprev = np.full(N, A0 / (A0 + B0))
    for t in range(T):
        yt = y[:, t]; mt = mask[:, t].astype(float)
        s = W @ ((yt - pprev) * mt)
        af = gamma * af + yt * mt + lam * np.maximum(s, 0)
        as_ = gamma * as_ + (1 - yt) * mt + lam * np.maximum(-s, 0)
        A[:, t] = af + A0; B[:, t] = as_ + B0
        P[:, t] = A[:, t] / (A[:, t] + B[:, t])
        pprev = P[:, t]
    return P, A, B


def observed(y, kind, par, rng):
    mask = np.ones_like(y, bool)
    if kind == "missing":
        mask = rng.random(y.shape) >= par
        return y, mask
    if kind == "delay":
        d = int(par)
        ys = np.zeros_like(y); ys[:, d:] = y[:, :-d]
        mk = np.zeros_like(mask); mk[:, d:] = True
        return ys, mk
    return y, mask


def corrupt_graph(w, kind, rng):
    w2 = dict(w); up = w["up"].copy(); region = w["region"].copy()
    if kind in ("drop50",):
        up = up * (rng.random(up.shape) >= 0.5)
    if kind in ("rewire", "random"):
        up = np.zeros_like(up)
        for i in range(N2, N):
            k = int(w["up"][i].sum())
            up[i, rng.choice(N2, size=max(k, 1), replace=False)] = 1.0
    if kind in ("region50",):
        m = rng.random(N) < 0.5
        region = np.where(m, rng.integers(0, R, N), region)
    if kind == "random":
        region = rng.integers(0, R, N)
    w2["up"] = up; w2["region"] = region
    return graph_weights(w2, PRO["wr"])[0]


def simulate_variant(seed, kind):
    """Generators different from the one used to design the method."""
    if kind == "independent":           # no shared shocks, no dependency
        return simulate(seed, shock_scale=0.0, dep=0.0)
    rng = np.random.default_rng(seed + 90000)
    base_w = simulate(seed)             # reuse graph, economics (same draw order)
    region, up, L, c = base_w["region"], base_w["up"], base_w["L"], base_w["c"]
    base = rng.beta(2, 28, N)
    kappa = rng.uniform(0, 0.6, N)
    upn = up / np.maximum(up.sum(1, keepdims=True), 1.0)
    outage = np.zeros((N, T), bool)
    so = rng.random(N) < 0.07
    for t in range(T):
        u = rng.random(N)
        so = np.where(so, u > 0.10, u < 0.008)
        outage[:, t] = so
    if kind == "global":                # one network-wide shock, not regional
        g = np.zeros(T, bool); st = False
        for t in range(T):
            u = rng.random()
            st = (u > 0.15) if st else (u < 0.01)
            g[t] = st
        shock_t = np.repeat(g[None, :], N, 0)
        base_t = np.repeat(base[:, None], T, 1)
        qlin = base_t + 0.5 * kappa[:, None] * shock_t + 0.35 * (upn @ outage.astype(float))
        q = np.where(outage, 0.95, np.clip(qlin, 0, 0.97))
        y = (rng.random((N, T)) < q).astype(float)
    elif kind == "drift":               # slow random-walk reliability, sticky shocks
        walk = np.cumsum(rng.normal(0, 0.06, (N, T)), 1)
        base_t = np.clip(base[:, None] * np.exp(walk), 0, 0.6)
        shock = np.zeros((R, T), bool); st = rng.random(R) < 0.14
        for t in range(T):
            u = rng.random(R)
            st = np.where(st, u > 0.04, u < 0.02)
            shock[:, t] = st
        qlin = base_t + kappa[:, None] * shock[region, :] + 0.35 * (upn @ outage.astype(float))
        q = np.where(outage, 0.95, np.clip(qlin, 0, 0.97))
        y = (rng.random((N, T)) < q).astype(float)
    elif kind == "contagion":           # failures spread over a hidden graph
        H = np.zeros((N, N))
        for i in range(N):
            js = rng.choice([j for j in range(N) if j != i], size=3, replace=False)
            H[i, js] = 1 / 3
        y = np.zeros((N, T)); q = np.zeros((N, T)); prev = np.zeros(N)
        for t in range(T):
            qt = base + 0.30 * (H @ prev) + 0.35 * (upn @ outage[:, t].astype(float))
            qt = np.where(outage[:, t], 0.95, np.clip(qt, 0, 0.97))
            yt = (rng.random(N) < qt).astype(float)
            y[:, t] = yt; q[:, t] = qt; prev = yt
    else:
        raise ValueError(kind)
    return dict(y=y, region=region, up=up, L=L, c=c, q=q)


def evaluate(world_fn, make_obs=None, make_W=None, seeds=TEST_SEEDS):
    rows = dict(static=[], ind=[], trust=[], c_human_ind=[], c_human_trust=[], c_gov_trust=[],
                spend_trust=[], spend_unres=[], c_unres_trust=[])
    rng = np.random.default_rng(7)
    for s in seeds:
        w = world_fn(s); y = w["y"]
        W = graph_weights(w, PRO["wr"])[0] if make_W is None else make_W(w, rng)
        Wi = graph_weights(w, 0.0)[0]
        yo, mk = (y, np.ones_like(y, bool)) if make_obs is None else make_obs(y, rng)
        Ps, As, Bs = static_forecast(y)
        Pi, Ai, Bi = trust_filter_obs(yo, mk, Wi, 0.0, IND["gamma"])
        Pp, Ap, Bp = trust_filter_obs(yo, mk, W, PRO["lam"], PRO["gamma"])
        rows["static"].append(forecast_metrics(Ps, y)[0])
        rows["ind"].append(forecast_metrics(Pi, y)[0])
        rows["trust"].append(forecast_metrics(Pp, y)[0])
        rows["c_human_ind"].append(run_policy(Pi, Ai, Bi, y, w, "human", DELTA)["cost"])
        rows["c_human_trust"].append(run_policy(Pp, Ap, Bp, y, w, "human", DELTA)["cost"])
        g = run_policy(Pp, Ap, Bp, y, w, "gov", DELTA, RHO)
        n_ = run_policy(Pp, Ap, Bp, y, w, "naive")
        rows["c_gov_trust"].append(g["cost"]); rows["spend_trust"].append(g["auto_spend"])
        rows["spend_unres"].append(n_["auto_spend"]); rows["c_unres_trust"].append(n_["cost"])
    a = {k: np.array(v) for k, v in rows.items()}
    d = a["trust"] - a["ind"]
    return dict(brier_static=float(a["static"].mean()), brier_ind=float(a["ind"].mean()),
                brier_trust=float(a["trust"].mean()),
                diff=ci95(d), rel=float(d.mean() / a["ind"].mean()),
                cost_human_ind=float(a["c_human_ind"].mean()),
                cost_human_trust=float(a["c_human_trust"].mean()),
                cost_diff=ci95(a["c_human_trust"] - a["c_human_ind"]),
                cost_gov_trust=float(a["c_gov_trust"].mean()),
                spend_gov=float(a["spend_trust"].mean()), spend_unres=float(a["spend_unres"].mean()),
                cost_unres_trust=float(a["c_unres_trust"].mean()))


def main():
    base = lambda s: simulate(s)
    out = {}
    out["baseline"] = evaluate(base)
    print("baseline", out["baseline"])
    out["graph"] = {}
    for kind in ["drop50", "rewire", "region50", "random"]:
        out["graph"][kind] = evaluate(base, make_W=lambda w, rng, k=kind: corrupt_graph(w, k, rng))
        r = out["graph"][kind]
        print("graph", kind, round(r["brier_trust"], 4), "vs ind", round(r["brier_ind"], 4), "rel %.3f" % r["rel"])
    out["missing"] = {}
    for p in [0.1, 0.3, 0.5]:
        out["missing"][str(p)] = evaluate(base, make_obs=lambda y, rng, p=p: observed(y, "missing", p, rng))
        r = out["missing"][str(p)]
        print("missing", p, round(r["brier_trust"], 4), "ind", round(r["brier_ind"], 4), "rel %.3f" % r["rel"],
              "cost trust %.3f ind %.3f" % (r["cost_human_trust"], r["cost_human_ind"]))
    out["delay"] = {}
    for dly in [1, 3, 7]:
        out["delay"][str(dly)] = evaluate(base, make_obs=lambda y, rng, d=dly: observed(y, "delay", d, rng))
        r = out["delay"][str(dly)]
        print("delay", dly, round(r["brier_trust"], 4), "ind", round(r["brier_ind"], 4), "static", round(r["brier_static"], 4),
              "rel %.3f" % r["rel"], "cost trust %.3f ind %.3f" % (r["cost_human_trust"], r["cost_human_ind"]))
    out["generator"] = {}
    for kind in ["independent", "global", "drift", "contagion"]:
        out["generator"][kind] = evaluate(lambda s, k=kind: simulate_variant(s, k))
        r = out["generator"][kind]
        print("gen", kind, "static %.4f ind %.4f trust %.4f" % (r["brier_static"], r["brier_ind"], r["brier_trust"]),
              "diff %+.4f+-%.4f rel %.3f" % (r["diff"][0], r["diff"][1], r["rel"]),
              "cost h-ind %.3f h-trust %.3f gov %.3f spend %.0f/%.0f" % (
                  r["cost_human_ind"], r["cost_human_trust"], r["cost_gov_trust"], r["spend_gov"], r["spend_unres"]))
    with open("rev_b.json", "w") as fh:
        json.dump(out, fh, indent=1, default=lambda o: float(o))


if __name__ == "__main__":
    main()
