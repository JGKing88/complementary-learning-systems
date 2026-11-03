"""Sec 20.2: stochastic pruning vs graded gains, from uniform random frequencies (pure waves).

    python -m analysis.hopfield_probe.gwn_prune_vs_grade

Graded: weight w = exp(-r^2 v / 2), kernel sum w^2 cos / sum w^2.
Stochastic pruning: keep unit with probability w^2 (= the Gaussian spectrum shape), equal weights.
Deterministic cut: keep if |w r| inside the 2-sigma disc, equal weights."""
import numpy as np
N, r = 1716, 16.0
rng = np.random.default_rng(0)
def kernel_stats(om, wt, rng):
    wt = wt / wt.sum()
    d = np.arange(0, 81)
    k = (wt[None, :] * np.cos(np.outer(d, om[:, 0]))).sum(1)
    half = d[np.argmax(k < 0.5)]
    D = rng.integers(-858, 859, (3000, 2)); D = D[np.hypot(*D.T) > 100]
    far = (wt[None, :] * np.cos(D @ om.T)).sum(1)
    return half, far.std(), k.min()
for nu in (400_000, 820_000):
    n = rng.integers(-858, 859, (nu, 2)); om = 2 * np.pi * n / N
    v = 2 - np.cos(om[:, 0]) - np.cos(om[:, 1])
    w2 = np.exp(-r * r * v)                       # = w^2, the Gaussian spectrum
    m = w2 > 1e-8
    h, sd, kmin = kernel_stats(om[m], w2[m], rng)
    print(f"{nu:>7d} units  graded:      n_eff {w2.sum()**2/(w2**2).sum():6.0f}  half-width {h}  far sd {sd:.3f}")
    keep = rng.random(nu) < w2
    h, sd, kmin = kernel_stats(om[keep], np.ones(keep.sum()), rng)
    print(f"{'':>7s}        stochastic:  kept {keep.sum():6d}  half-width {h}  far sd {sd:.3f}")
    cut = np.hypot(n[:, 0], n[:, 1]) < 2 * N / (2 * np.pi * r)
    h, sd, kmin = kernel_stats(om[cut], np.ones(cut.sum()), rng)
    print(f"{'':>7s}        hard cut:    kept {cut.sum():6d}  half-width {h}  far sd {sd:.3f}  most negative {kmin:.3f}")
n = np.rint(rng.normal(0, N / (2 * np.pi * r), (512, 2))); om = 2 * np.pi * n / N
h, sd, _ = kernel_stats(om, np.ones(512), rng)
print(f"ideal (512 drawn from the Gaussian): half-width {h}  far sd {sd:.3f}")
