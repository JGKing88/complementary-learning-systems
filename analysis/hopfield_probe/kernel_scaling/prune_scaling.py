"""Stochastic pruning: survivors, shape and aliases against pool size and width. L = 1716.

Pool: P random units, frequency n uniform over the L x L torus. Keep each with probability
exp(-r^2 |omega_n|^2 / 2) (lattice form exp(-r^2 (2 - cos - cos))), weight 1.
Predictions: N = P / (2 pi r^2); far sd = 1/sqrt(2N); largest alias = sqrt(2 ln (L/r)^2) / sqrt(2N).

Docs: EXPERIMENTS_IDEAL_ENCODER.md §25.4. Run on a compute node, from the repo root:
    python analysis/hopfield_probe/kernel_scaling/prune_scaling.py
"""
import numpy as np
L = 1716
d = np.arange(L); d = np.where(d > L // 2, d - L, d)
RAD = np.hypot(d[:, None], d[None, :])

print(f"{'r':>3s} {'pool':>9s} | {'N':>6s} pred {'':>4s} | half-width (target) | far sd  pred  | largest alias  pred")
for r in (8.0, 16.0, 32.0):
    for pool in (100_000, 400_000, 1_600_000, 6_400_000):
        rows = []
        for seed in range(3):
            g = np.random.default_rng(seed)
            n = g.integers(-858, 859, (pool, 2)); o = 2 * np.pi * n / L
            keep = g.random(pool) < np.exp(-r * r * (2 - np.cos(o[:, 0]) - np.cos(o[:, 1])))
            nk = n[keep]
            A = np.zeros((L, L), complex)
            np.add.at(A, (nk[:, 0] % L, nk[:, 1] % L), 1.0)
            k = np.fft.ifft2(A).real * L * L / len(nk)
            row = k[0, :int(4 * r)]
            half = np.interp(0.5, row[::-1], np.arange(len(row))[::-1])
            far = k[RAD > 3 * r]
            rows.append((len(nk), half, far.std(), far.max()))
            del n, o, keep
        R = np.array(rows); N = R[:, 0].mean()
        Np = pool / (2 * np.pi * r * r)
        print(f"{r:3.0f} {pool:9,d} | {N:6.0f} {Np:6.0f}     | {R[:,1].mean():5.2f} ({r*np.sqrt(2*np.log(2)):5.2f})       | "
              f"{R[:,2].mean():.3f} {1/np.sqrt(2*N):.3f} | {R[:,3].min():.3f}-{R[:,3].max():.3f}  "
              f"{np.sqrt(2*np.log((L/r)**2))/np.sqrt(2*N):.3f}", flush=True)
