"""Cold-start experiment and autonomous-spend accounting (test seeds only)."""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sim import *  # noqa
from run_experiments import (forecasts, trust_filter, IND, PRO, DELTA, RHO,
                             gbm_forecast, train_gbm)

EVAL0 = 120


def brier_window(P, y, t0=EVAL0):
    ts = np.arange(t0, T - 1)
    p = P[N2:, ts].ravel(); o = y[N2:, ts + 1].ravel()
    return float(np.mean((p - o) ** 2))


def main():
    out = {}
    # ---------------- cold start: supervised models trained on in-world history
    Hs = [20, 40, 60, 90]
    rec = {h: dict(gbm=[], lr=[]) for h in Hs}
    ref = dict(static=[], ind=[], pro=[], gbm_pre=[])
    gbm_pre = train_gbm()
    for s in TEST_SEEDS:
        w = simulate(s); y = w["y"]
        _, Wsup, Wreg = graph_weights(w, 0.5)
        p0 = static_forecast(y)[0][:, 0]
        F = features(y, p0, Wsup, Wreg)
        ref["static"].append(brier_window(static_forecast(y)[0], y))
        ref["ind"].append(brier_window(forecasts(w, IND)[0], y))
        ref["pro"].append(brier_window(forecasts(w, PRO)[0], y))
        ref["gbm_pre"].append(brier_window(gbm_forecast(gbm_pre, w), y))
        ts_eval = np.arange(EVAL0, T - 1)
        Xe = F[:, ts_eval, :].reshape(-1, F.shape[-1])
        for h in Hs:
            ts_tr = np.arange(T0 - 1, T0 - 1 + h)
            Xt = F[:, ts_tr, :].reshape(-1, F.shape[-1])
            Yt = y[:, ts_tr + 1].reshape(-1)
            for name, model in [
                ("gbm", HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08,
                                                       max_depth=4, random_state=0)),
                ("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)))]:
                model.fit(Xt, Yt)
                P = np.full((N, T), 0.1)
                P[:, ts_eval] = model.predict_proba(Xe)[:, 1].reshape(N, -1)
                rec[h][name].append(brier_window(P, y))
    out["cold"] = dict(
        ref={k: ci95(v) for k, v in ref.items()},
        H={h: {k: ci95(v) for k, v in rec[h].items()} for h in Hs})
    print("COLD START (Brier, eval days 120+)")
    print({k: round(v[0], 4) for k, v in out["cold"]["ref"].items()})
    for h in Hs:
        print(h, {k: (round(v[0], 4), round(v[1], 4)) for k, v in out["cold"]["H"][h].items()})
    # paired: pro vs best small-data supervised at each H
    for h in Hs:
        d = np.array(ref["pro"]) - np.array(rec[h]["gbm"])
        d2 = np.array(ref["pro"]) - np.array(rec[h]["lr"])
        print(f"H={h} pro-gbm {ci95(d)}  pro-lr {ci95(d2)}")

    # ---------------- autonomous spend accounting
    rows = {"gov": [], "naive": [], "gov_ind": []}
    for s in TEST_SEEDS:
        w = simulate(s); y = w["y"]
        Pp, Ap, Bp = forecasts(w, PRO)
        Pi, Ai, Bi = forecasts(w, IND)
        rows["gov"].append(run_policy(Pp, Ap, Bp, y, w, "gov", DELTA, RHO))
        rows["naive"].append(run_policy(Pp, Ap, Bp, y, w, "naive"))
        rows["gov_ind"].append(run_policy(Pi, Ai, Bi, y, w, "gov", DELTA, RHO))
    acc = {}
    for k, v in rows.items():
        spend = np.array([r["auto_spend"] for r in v]); waste = np.array([r["auto_waste"] for r in v])
        acc[k] = dict(spend=ci95(spend), waste=ci95(waste),
                      acts=ci95([r["auto_acts"] for r in v]),
                      maxc=max(r["max_auto_cost"] for r in v))
        print(k, {kk: (np.round(vv, 1) if not isinstance(vv, float) else vv) for kk, vv in acc[k].items()})
    out["autonomy"] = acc
    with open("extra.json", "w") as fh:
        json.dump(out, fh, indent=1, default=lambda o: float(o))


if __name__ == "__main__":
    main()
