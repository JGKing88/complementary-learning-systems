"""Lenient pruning (width r1 < r) plus graded weights to restore r. L = 1716, r = 16, pool 400k.

Keep each unit w.p. exp(-r1^2 (2 - cos - cos)); give survivors w^2 = exp(-(r^2 - r1^2)(2 - cos - cos)).
Predicted: N = P / (2 pi r1^2), N_eff = P (2 r^2 - r1^2) / (2 pi r^4).

Docs: EXPERIMENTS_IDEAL_ENCODER.md §25.5. Run on a compute node, from the repo root:
    python analysis/hopfield_probe/kernel_scaling/lenient_prune.py
"""
import numpy as np
L, r, POOL = 1716, 16.0, 400_000
d = np.arange(L); d = np.where(d > L // 2, d - L, d)
RAD = np.hypot(d[:, None], d[None, :])
print(f"target half-width {r*np.sqrt(2*np.log(2)):.2f}")
for r1 in (16.0, 12.0, 8.0, 4.0, 0.0):
    rows = []
    for seed in range(3):
        g = np.random.default_rng(seed)
        n = g.integers(-858, 859, (POOL, 2)); o = 2 * np.pi * n / L
        v = 2 - np.cos(o[:, 0]) - np.cos(o[:, 1])
        keep = g.random(POOL) < np.exp(-r1 * r1 * v)
        w2 = np.exp(-(r * r - r1 * r1) * v[keep])
        A = np.zeros((L, L), complex)
        np.add.at(A, (n[keep, 0] % L, n[keep, 1] % L), w2)
        k = np.fft.ifft2(A).real * L * L / w2.sum()
        row = k[0, :64]
        rows.append((keep.sum(), w2.sum() ** 2 / (w2 ** 2).sum(),
                     np.interp(0.5, row[::-1], np.arange(64)[::-1]), k[RAD > 3 * r].max()))
    R = np.array(rows)
    Np = POOL / (2 * np.pi * r1 * r1) if r1 > 0 else POOL
    print(f"r1 {r1:4.0f} | survivors {R[:,0].mean():7.0f} (pred {Np:7.0f}) | n_eff {R[:,1].mean():5.0f} "
          f"(pred {POOL*(2*r*r - r1*r1)/(2*np.pi*r**4):5.0f}) | half-width {R[:,2].mean():5.2f} | alias {R[:,3].min():.3f}-{R[:,3].max():.3f}",
          flush=True)
