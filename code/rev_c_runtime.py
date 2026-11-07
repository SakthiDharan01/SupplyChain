"""Revision experiments C: computational cost of one daily update of the trust
graph with sparse operators (single process, this machine).  Output: rev_c.json
"""
import json
import platform
import time
import numpy as np
import scipy.sparse as sp
from scipy.special import betainc


def build(N, seed=0, R=50):
    rng = np.random.default_rng(seed)
    tier1 = rng.random(N) < 0.6
    rows, cols = [], []
    t2 = np.where(~tier1)[0]
    for i in np.where(tier1)[0]:
        k = rng.integers(1, 4)
        js = rng.choice(t2, size=k, replace=False)
        rows += [i] * k; cols += list(js)
    deg = np.bincount(rows, minlength=N).astype(float)
    data = [1.0 / deg[i] for i in rows]
    Wsup = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    region = rng.integers(0, R, N)
    return tier1, Wsup, region, R


def step(state, y, tier1, Wsup, region, R, gamma=0.85, lam=2.0, omega=0.5, qstar=None):
    af, as_, pprev = state
    u = y - pprev
    s_sup = Wsup @ u
    S = np.bincount(region, weights=u, minlength=R)
    n = np.bincount(region, minlength=R).astype(float)
    s_reg = (S[region] - u) / np.maximum(n[region] - 1, 1)
    s = (1 - omega) * s_sup + omega * s_reg
    af = gamma * af + y + lam * np.maximum(s, 0)
    as_ = gamma * as_ + (1 - y) + lam * np.maximum(-s, 0)
    A = af + 1.0; B = as_ + 9.0
    p = A / (A + B)
    desired = tier1 & (p > qstar)
    tail = 1.0 - betainc(A[desired], B[desired], qstar[desired])      # Pr(q > q*)
    return (af, as_, p), int(desired.sum()), tail


def main():
    out = dict(machine=platform.processor() or platform.machine(), python=platform.python_version(),
               numpy=np.__version__, scipy=__import__("scipy").__version__, results={})
    for N in [40, 1000, 10000, 100000]:
        tier1, Wsup, region, R = build(N)
        rng = np.random.default_rng(1)
        qstar = rng.uniform(0.04, 0.35, N)
        state = (np.zeros(N), np.zeros(N), np.full(N, 0.1))
        times = []
        for t in range(120):
            y = (rng.random(N) < 0.08).astype(float)
            t0 = time.perf_counter()
            state, nd, _ = step(state, y, tier1, Wsup, region, R, qstar=qstar)
            times.append(time.perf_counter() - t0)
        ms = float(np.median(times[20:]) * 1e3)
        out["results"][str(N)] = dict(median_ms=ms, p95_ms=float(np.percentile(times[20:], 95) * 1e3))
        print(N, "median %.3f ms/day  p95 %.3f ms" % (ms, out["results"][str(N)]["p95_ms"]))
    with open("rev_c.json", "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
