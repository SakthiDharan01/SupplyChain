"""Governance frontier: unreviewed autonomous spend vs. cost, tail-probability
gate vs. point-estimate gate, over a grid of gate strictness and action-cost
ceilings. Test seeds only; same forecasts for both gates."""
import json
import numpy as np
from sim import *  # noqa
from run_experiments import forecasts, PRO, DELTA

RHOS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
MS = [1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0]
CMAXS = [1.0, 1.5, 2.0, 3.0]


def main():
    acc = {}
    def add(key, r):
        acc.setdefault(key, []).append(r)
    for s in TEST_SEEDS:
        w = simulate(s); y = w["y"]
        P, A, B = forecasts(w, PRO)
        add(("human",), run_policy(P, A, B, y, w, "human", DELTA))
        add(("naive",), run_policy(P, A, B, y, w, "naive"))
        for cm in CMAXS:
            for rho in RHOS:
                add(("tail", cm, rho), run_policy(P, A, B, y, w, "gov", DELTA, rho, cm, "tail"))
            for m in MS:
                add(("ratio", cm, m), run_policy(P, A, B, y, w, "gov", DELTA, 0.0, cm, "ratio", m))
    out = {}
    for k, rows in acc.items():
        cost = np.mean([r["cost"] for r in rows])
        spend = np.mean([r["auto_spend"] for r in rows])
        waste = np.mean([r["auto_waste"] for r in rows])
        hit = sum(r["auto_hits"] for r in rows) / max(sum(r["auto_acts"] for r in rows), 1)
        out["|".join(map(str, k))] = dict(cost=float(cost), spend=float(spend),
                                           waste=float(waste), hit=float(hit))
    with open("frontier.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("human", out["human"]); print("naive", out["naive"])
    for cm in CMAXS:
        print("cmax", cm)
        for rho in RHOS:
            o = out[f"tail|{cm}|{rho}"]; print(f"  tail rho={rho}: spend {o['spend']:.0f} cost {o['cost']:.4f} hit {o['hit']:.3f}")
        for m in MS:
            o = out[f"ratio|{cm}|{m}"]; print(f"  ratio m={m}: spend {o['spend']:.0f} cost {o['cost']:.4f} hit {o['hit']:.3f}")


if __name__ == "__main__":
    main()
