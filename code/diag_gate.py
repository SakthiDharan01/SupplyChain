"""Does posterior uncertainty help? (1) risk-coverage of tail-probability gate
vs point-estimate gate; (2) credible-interval coverage of the hidden true q."""
import json
import numpy as np
from scipy.stats import beta as bd
from sim import *  # noqa
from run_experiments import forecasts, PRO, IND

COVS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]


def collect(cfg, seeds):
    S_tail, S_ratio, S_rand, HIT, CI80, CI90, QERR = [], [], [], [], [], [], []
    rng = np.random.default_rng(1)
    for s in seeds:
        w = simulate(s); y = w["y"]
        P, A, B = forecasts(w, cfg)
        L, c = w["L"], w["c"]
        ts = np.arange(T0 - 1, T - 1)
        P1, A1, B1 = P[N2:, ts], A[N2:, ts], B[N2:, ts]
        qn = w["q"][N2:, ts + 1]
        yn = y[N2:, ts + 1]
        qstar = (c / (L * E_EFF))[:, None]
        low = (c <= CMAX)[:, None] & (P1 > qstar)     # low-risk class, desired
        tail = bd.sf(qstar, A1, B1)
        ratio = P1 / qstar
        m = low
        S_tail.append(tail[m]); S_ratio.append(ratio[m]); S_rand.append(rng.random(m.sum()))
        HIT.append(yn[m])
        lo80, hi80 = bd.ppf(0.1, A1, B1), bd.ppf(0.9, A1, B1)
        lo90, hi90 = bd.ppf(0.05, A1, B1), bd.ppf(0.95, A1, B1)
        CI80.append(((qn >= lo80) & (qn <= hi80)).ravel())
        CI90.append(((qn >= lo90) & (qn <= hi90)).ravel())
        QERR.append(np.abs(P1 - qn).ravel())
    cat = lambda x: np.concatenate(x)
    return (cat(S_tail), cat(S_ratio), cat(S_rand), cat(HIT),
            cat(CI80).mean(), cat(CI90).mean(), cat(QERR).mean())


def prec_at_cov(score, hit, cov):
    k = int(len(score) * cov)
    idx = np.argsort(-score)[:k]
    return float(hit[idx].mean())


def main():
    out = {}
    for name, cfg in [("pro", PRO), ("ind", IND)]:
        tail, ratio, rnd, hit, c80, c90, qerr = collect(cfg, TEST_SEEDS)
        out[name] = dict(
            base_hit=float(hit.mean()),
            ci80_cov=float(c80), ci90_cov=float(c90), mae_q=float(qerr),
            tail={str(cv): prec_at_cov(tail, hit, cv) for cv in COVS},
            ratio={str(cv): prec_at_cov(ratio, hit, cv) for cv in COVS},
            rand={str(cv): prec_at_cov(rnd, hit, cv) for cv in COVS})
        print(name, "N desired low-risk:", len(hit), "base hit", round(hit.mean(), 3),
              "CI80 cov", round(c80, 3), "CI90 cov", round(c90, 3), "MAE(q)", round(qerr, 4))
        for cv in COVS:
            print(f"  cov {cv:.2f}  tail {out[name]['tail'][str(cv)]:.3f}  ratio {out[name]['ratio'][str(cv)]:.3f}"
                  f"  rand {out[name]['rand'][str(cv)]:.3f}")
    with open("diag.json", "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
