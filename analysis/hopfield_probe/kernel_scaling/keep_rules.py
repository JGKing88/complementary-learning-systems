"""Equal output size (512 units, M = 1024): how the keep rule changes the kernel. L = 1716, r = 16.

Docs: EXPERIMENTS_IDEAL_ENCODER.md §25.5. Run on a compute node, from the repo root:
    python analysis/hopfield_probe/kernel_scaling/keep_rules.py
"""
import numpy as np
L, r, K = 1716, 16.0, 512
d = np.arange(L); d = np.where(d > L // 2, d - L, d)
RAD = np.hypot(d[:, None], d[None, :])
hw_t = r * np.sqrt(2 * np.log(2))


def stats(n, w2):
    A = np.zeros((L, L), complex)
    np.add.at(A, (n[:, 0] % L, n[:, 1] % L), w2)
    k = np.fft.ifft2(A).real * L * L / w2.sum()
    row = k[0, :60]
    half = np.interp(0.5, row[::-1], np.arange(60)[::-1])
    neff = w2.sum() ** 2 / (w2 ** 2).sum()
    return len(n), neff, half, k[RAD > 3 * r].max()


def pool_units(pool, g):
    n = g.integers(-858, 859, (pool, 2)); o = 2 * np.pi * n / L
    return n, np.exp(-256 * (2 - np.cos(o[:, 0]) - np.cos(o[:, 1])))


rules = {}
for seed in range(3):
    g = np.random.default_rng(seed)
    n = np.rint(g.normal(0, L / (2 * np.pi * r), (K, 2))).astype(int)
    rules.setdefault("ideal: 512 drawn from P, equal weights", []).append(stats(n, np.ones(K)))
    pool = int(round(K * 2 * np.pi * r * r / np.log(200)))          # top 512 hold ~99.5%
    n, w2 = pool_units(pool, g); top = np.argsort(-w2)[:K]
    rules.setdefault(f"graded, pool {pool // 1000}k sized so top 512 hold 99.5%", []).append(
        stats(n[top], w2[top]) + (w2[top].sum() / w2.sum(),))
    n, w2 = pool_units(400_000, g); top = np.argsort(-w2)[:K]
    rules.setdefault("graded, pool 400k, keep top 512", []).append(
        stats(n[top], w2[top]) + (w2[top].sum() / w2.sum(),))
    pool = int(round(K * 2 * np.pi * r * r))                          # about 512 survive
    n, w2 = pool_units(pool, g); keep = g.random(pool) < w2
    rules.setdefault(f"stochastic pruning, pool {pool // 1000}k (keep w.p. w^2, weight 1)", []).append(
        stats(n[keep], np.ones(keep.sum())))
print(f"target half-width {hw_t:.2f}")
for name, rows in rules.items():
    R = np.array([row[:4] for row in rows]); mass = [row[4] for row in rows if len(row) > 4]
    print(f"{name:58s} kept {R[:,0].mean():5.0f}  n_eff {R[:,1].mean():5.0f}  half-width {R[:,2].mean():5.2f}  "
          f"max alias {R[:,3].min():.3f}-{R[:,3].max():.3f}" + (f"  weight held {np.mean(mass):.3f}" if mass else ""))
