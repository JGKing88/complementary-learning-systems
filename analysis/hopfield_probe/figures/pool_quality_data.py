"""Pool size vs kernel quality for the depression learner (frequency level, no grid code).

    python analysis/hopfield_probe/figures/pool_quality_data.py   # writes pool_quality.json

A pool of P units with frequencies n uniform over the L x L torus (L = 1716). The depression
rule on one random walk of T steps (kappa = 0.005, kappa T = r^2 / 2) gives each unit
w^2 = (1 - kappa d1)^(2 T1) (1 - kappa d2)^(2 T2), with d_i = 2 - 2 cos(omega_i) and T1 + T2 = T
the steps along each axis (binomial). Kernel over the whole arena by one 2D FFT.
Measured: effective units n_eff = (sum w^2)^2 / sum w^4, largest similarity beyond 3r
(largest alias), and half-width along an axis.
"""
import json
from pathlib import Path

import numpy as np

L, KAPPA = 1716, 0.005
POOLS = [25_000, 50_000, 100_000, 200_000, 400_000, 800_000, 1_600_000, 3_200_000, 6_400_000]
RS = [8.0, 16.0, 32.0]
SEEDS = 3
d = np.arange(L); d = np.where(d > L // 2, d - L, d)
RAD = np.hypot(d[:, None], d[None, :])

out = []
for r in RS:
    T = int(round(r * r / (2 * KAPPA)))
    for pool in POOLS:
        for seed in range(SEEDS):
            g = np.random.default_rng(seed)
            n = g.integers(-858, 859, (pool, 2))
            T1 = g.binomial(T, 0.5)
            dd = 2 - 2 * np.cos(2 * np.pi * n / L)
            logw2 = 2 * T1 * np.log1p(-KAPPA * dd[:, 0]) + 2 * (T - T1) * np.log1p(-KAPPA * dd[:, 1])
            w2 = np.exp(logw2)
            m = w2 > 1e-12
            n, w2 = n[m], w2[m]
            A = np.zeros((L, L), complex)
            np.add.at(A, (n[:, 0] % L, n[:, 1] % L), w2)
            k = np.fft.ifft2(A).real * L * L / w2.sum()
            row = k[0, : int(4 * r)]
            half = float(np.interp(0.5, row[::-1], np.arange(len(row))[::-1]))
            rec = dict(r=r, pool=pool, seed=seed, T=T, n_eff=float(w2.sum() ** 2 / (w2 ** 2).sum()),
                       alias=float(k[RAD > 3 * r].max()), half_width=half)
            out.append(rec)
            print(rec, flush=True)
            del n, w2, A, k

Path(__file__).with_name("pool_quality.json").write_text(json.dumps(out, indent=1))
print("written")
