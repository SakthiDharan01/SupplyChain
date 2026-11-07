import numpy as np, pickle
import sim
from sim import *

def surprise_filter(y, W, lam, gamma):
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

if __name__ == "__main__":
    worlds = [simulate(s) for s in TUNE_SEEDS]
    res = {}
    for gamma in [0.8, 0.85, 0.9, 0.95, 0.97]:
        for lam in [0, 0.5, 1, 2, 4, 8]:
            for wr in ([0.0] if lam == 0 else [0.0, 0.5, 1.0]):
                bs = []
                for w in worlds:
                    W, _, _ = graph_weights(w, wr)
                    P, _, _ = surprise_filter(w["y"], W, lam, gamma)
                    bs.append(forecast_metrics(P, w["y"])[0])
                res[(gamma, lam, wr)] = float(np.mean(bs))
    ks = sorted(res, key=lambda k: res[k])[:8]
    for k in ks: print(k, round(res[k], 5))
    print("indep best:", min((k for k in res if k[1]==0), key=lambda k: res[k]), )
    k=min((k for k in res if k[1]==0), key=lambda k: res[k]); print(res[k])
