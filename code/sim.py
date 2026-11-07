"""
Simulation study: uncertainty-aware supplier trust propagation and
confidence-gated autonomous mitigation.

All simulator parameters were fixed BEFORE any results were inspected.
Seeds: tuning 1000-1019, supervised-baseline training 2000-2029, test 0-49.
"""
import json
import sys
import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats import t as t_dist
from sklearn.ensemble import HistGradientBoostingClassifier

# ---------------------------------------------------------------- constants
N2, N1 = 16, 24          # tier-2 (raw) suppliers, tier-1 suppliers
N = N1 + N2              # indices 0..N2-1 tier-2 ; N2..N-1 tier-1
R = 6                    # regions
T = 250                  # days
T0 = 30                  # warm-up window (static scorecard is fitted here)
E_EFF = 0.85             # backup effectiveness
CMAX = 1.5               # max cost of an action allowed to run unattended
A0, B0 = 1.0, 9.0        # Beta prior (mean failure prob 0.1)

TUNE_SEEDS = list(range(1000, 1020))
TRAIN_SEEDS = list(range(2000, 2030))
TEST_SEEDS = list(range(0, 50))


# ---------------------------------------------------------------- simulator
def simulate(seed, shock_scale=1.0, dep=0.35):
    rng = np.random.default_rng(seed)
    region = rng.integers(0, R, N)
    up = np.zeros((N, N))
    for i in range(N2, N):
        k = rng.integers(1, 4)
        js = rng.choice(N2, size=k, replace=False)
        up[i, js] = 1.0
    base = rng.beta(2, 28, N)
    kappa = rng.uniform(0, 0.6, N) * shock_scale
    # regime changes (supplier instability)
    changes = rng.random(N) < 0.3
    tau = rng.integers(40, 200, N)
    fac = np.where(rng.random(N) < 0.5, 3.0, 0.3)
    # economics of tier-1 suppliers
    L = rng.uniform(6, 20, N1)
    c = rng.uniform(0.5, 3.0, N1)

    shock = np.zeros((R, T), bool)
    st = rng.random(R) < 0.14
    for t in range(T):
        u = rng.random(R)
        st = np.where(st, u > 0.12, u < 0.02)
        shock[:, t] = st
    outage = np.zeros((N, T), bool)
    so = rng.random(N) < 0.07
    for t in range(T):
        u = rng.random(N)
        so = np.where(so, u > 0.10, u < 0.008)
        outage[:, t] = so

    upn = up / np.maximum(up.sum(1, keepdims=True), 1.0)
    tt = np.arange(T)[None, :]
    base_t = base[:, None] * np.where(changes[:, None] & (tt >= tau[:, None]),
                                      fac[:, None], 1.0)
    q = (base_t + kappa[:, None] * shock[region, :]
         + dep * (upn @ outage.astype(float)))
    q = np.clip(q, 0, 0.97)
    q = np.where(outage, 0.95, q)
    y = rng.random((N, T)) < q
    return dict(y=y.astype(float), region=region, up=up, L=L, c=c, q=q)


# ------------------------------------------------------------ graph weights
def graph_weights(world, w_region):
    up, region = world["up"], world["region"]
    Wsup = up / np.maximum(up.sum(1, keepdims=True), 1.0)
    Wreg = np.zeros((N, N))
    for r in range(R):
        idx = np.where(region == r)[0]
        if len(idx) > 1:
            for i in idx:
                for j in idx:
                    if i != j:
                        Wreg[i, j] = 1.0 / (len(idx) - 1)
    W = (1 - w_region) * Wsup + w_region * Wreg
    return W, Wsup, Wreg


# ----------------------------------------------------------------- filters
def beta_filter(y, W, lam, gamma):
    """Evolving Beta belief per supplier with evidence propagated over the
    graph (discount lam) and exponential forgetting (gamma).
    Returns forecast mean for day t+1 plus Beta posterior parameters."""
    af = np.zeros(N)
    as_ = np.zeros(N)
    ws = W.sum(1)
    P = np.zeros((N, T)); A = np.zeros((N, T)); B = np.zeros((N, T))
    for t in range(T):
        yt = y[:, t]
        s = W @ yt
        af = gamma * af + yt + lam * s
        as_ = gamma * as_ + (1 - yt) + lam * (ws - s)
        A[:, t] = af + A0
        B[:, t] = as_ + B0
        P[:, t] = A[:, t] / (A[:, t] + B[:, t])
    return P, A, B


def static_forecast(y):
    fails = y[:, :T0].sum(1)
    p0 = (fails + A0) / (T0 + A0 + B0)
    P = np.repeat(p0[:, None], T, 1)
    A = np.repeat((fails + A0)[:, None], T, 1)
    B = np.repeat((T0 - fails + B0)[:, None], T, 1)
    return P, A, B


def features(y, p0, Wsup, Wreg):
    cs = np.concatenate([np.zeros((N, 1)), np.cumsum(y, 1)], 1)

    def roll(k):
        out = np.zeros((N, T))
        for t in range(T):
            lo = max(0, t - k + 1)
            out[:, t] = (cs[:, t + 1] - cs[:, lo]) / (t + 1 - lo)
        return out
    y2 = np.concatenate([np.zeros((N, 1)), y[:, :-1]], 1)
    y3 = np.concatenate([np.zeros((N, 2)), y[:, :-2]], 1)
    m3, m7, m30 = roll(3), roll(7), roll(30)
    tier1 = np.repeat((np.arange(N) >= N2).astype(float)[:, None], T, 1)
    feats = [y, y2, y3, m3, m7, m30, Wsup @ y, Wsup @ m3, Wreg @ y, Wreg @ m3,
             np.repeat(p0[:, None], T, 1), tier1]
    return np.stack(feats, -1)  # (N,T,F)


# ------------------------------------------------------------- evaluation
def forecast_metrics(P, y):
    """Brier / ECE on tier-1 suppliers, forecasts made at t for day t+1."""
    ts = np.arange(T0 - 1, T - 1)
    p = P[N2:, ts].ravel()
    o = y[N2:, ts + 1].ravel()
    brier = float(np.mean((p - o) ** 2))
    bins = np.minimum((p * 10).astype(int), 9)
    ece = 0.0
    for b in range(10):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - o[m].mean())
    return brier, float(ece)


def run_policy(P, A, B, y, world, mode, delta=1, rho=0.8, cmax=CMAX,
               gate_type="tail", ratio_m=2.0, reval=False, score=None):
    """mode: none | oracle | human | naive | gov | ceil.
    ceil  = cost-ceiling-only gate (no confidence test), added in revision.
    reval = approval path re-checks that the action is still desired when it
            takes effect (stale approvals are cancelled).
    score = optional precomputed gate score in [0,1] (calibrated tail score)."""
    L, c = world["L"], world["c"]
    P1, A1, B1 = P[N2:], A[N2:], B[N2:]
    y1 = y[N2:]
    ts = np.arange(T0 - 1, T - 1)
    d = np.zeros((N1, T), bool)
    d[:, ts] = P1[:, ts] * (L * E_EFF)[:, None] > c[:, None]

    def shift(x, k):
        if k == 0:
            return x
        out = np.zeros_like(x)
        out[:, k:] = x[:, :-k]
        return out

    gate = np.zeros_like(d)
    if mode == "none":
        act = np.zeros_like(d)
    elif mode == "oracle":
        act = np.zeros_like(d)
        nxt = np.zeros((N1, T), bool)
        nxt[:, :-1] = y1[:, 1:] > 0
        act = nxt & ((E_EFF * L)[:, None] > c[:, None])
    elif mode == "human":
        act = shift(d, delta)
        if reval:
            act = act & d
    elif mode == "naive":
        act = d.copy()
        gate = d.copy()
    elif mode == "gov":
        qstar = c / (L * E_EFF)
        if score is not None:
            ok = score >= rho
        elif gate_type == "tail":
            ok = beta_dist.sf(qstar[:, None], A1, B1) >= rho
        else:
            ok = (P1 / qstar[:, None]) >= ratio_m
        gate = d & ok & (c <= cmax)[:, None]
        appr = shift(d, delta)
        if reval:
            appr = appr & d
        act = gate | appr
    elif mode == "ceil":
        gate = d & (c <= cmax)[:, None]
        appr = shift(d, delta)
        if reval:
            appr = appr & d
        act = gate | appr
    else:
        raise ValueError(mode)

    yn = np.zeros((N1, T), bool); yn[:, :-1] = y1[:, 1:] > 0
    yc = y1 > 0
    a = act[:, ts]; f = yn[:, ts]; g = gate[:, ts]
    onset = f & ~yc[:, ts]
    cost = (c[:, None] * a + L[:, None] * (1 - E_EFF * a) * f).mean()
    out = dict(
        cost=float(cost),
        protected=float((a & f).sum() / max(f.sum(), 1)),
        onset=float((a & onset).sum() / max(onset.sum(), 1)),
        auto_acts=float(g.sum()),
        auto_hits=float((g & f).sum()),
        auto_waste=float((c[:, None] * (g & ~f)).sum()),
        auto_spend=float((c[:, None] * g).sum()),
        desired=float(d[:, ts].sum()),
        max_auto_cost=float((c[:, None] * g).max()) if g.any() else 0.0,
    )
    return out


def ci95(x):
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return float(x.mean()), 0.0
    h = t_dist.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return float(x.mean()), float(h)


# ------------------------------------------------------------------ tuning
def tune():
    worlds = [simulate(s) for s in TUNE_SEEDS]
    res = {}
    for gamma in [0.8, 0.85, 0.9, 0.95, 0.97, 1.0]:
        for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for wr in ([0.0] if lam == 0 else [0.0, 0.25, 0.5, 0.75, 1.0]):
                bs = []
                for w in worlds:
                    W, _, _ = graph_weights(w, wr)
                    P, _, _ = beta_filter(w["y"], W, lam, gamma)
                    bs.append(forecast_metrics(P, w["y"])[0])
                res[(gamma, lam, wr)] = float(np.mean(bs))
    ind = min((k for k in res if k[1] == 0.0), key=lambda k: res[k])
    prop = min(res, key=lambda k: res[k])
    return res, ind, prop


if __name__ == "__main__":
    import pickle
    res, ind, prop = tune()
    print("independent best (gamma,lam,wr):", ind, res[ind])
    print("propagated  best (gamma,lam,wr):", prop, res[prop])
    with open(sys.argv[1], "wb") as fh:
        pickle.dump(dict(res=res, ind=ind, prop=prop), fh)
