"""Biasing the whole-number weights j toward small values. L = 1716, r = 16.

Per axis, each module's centred weight j_l is drawn from a discrete Gaussian of sd s (or
uniform); n = 156 j1 + 143 j2 + 132 j3 mod 1716. q(n) is computed exactly per axis.
Two ways to build the code from a biased pool:
  prune:  keep each unit w.p. P_n / P_0, weight 1   (survivors follow q * P, not P)
  import: keep all units, weight w^2 = P_n / q_n     (unbiased while q > 0 where P is)
Reports, per unit built: chance a unit survives (prune) or effective units per unit (import),
and for a pool sized to give ~512 effective units: half-width and largest alias beyond 3r.

Docs: EXPERIMENTS_IDEAL_ENCODER.md §25.6. Run on a compute node, from the repo root:
    python analysis/hopfield_probe/kernel_scaling/small_j_bias.py
"""
import itertools
import numpy as np
L, r = 1716, 16.0
LAMS = (11, 12, 13)
CF = (156, 143, 132)
d = np.arange(L); d = np.where(d > L // 2, d - L, d)
RAD = np.hypot(d[:, None], d[None, :])
om = 2 * np.pi * d / L
P1 = np.exp(-r * r * (1 - np.cos(om)))           # per-axis Gaussian spectrum (lattice form)
P1 /= P1.sum()


def j_probs(lam, s):
    jc = np.arange(lam); jc = np.where(jc > lam // 2, jc - lam, jc)
    p = np.ones(lam) if s is None else np.exp(-jc ** 2 / (2 * s * s))
    return jc, p / p.sum()


def q_axis(s):
    q = np.zeros(L)
    tabs = [j_probs(l, s) for l in LAMS]
    for (j1, p1), (j2, p2), (j3, p3) in itertools.product(*[list(zip(*t)) for t in tabs]):
        q[(CF[0] * j1 + CF[1] * j2 + CF[2] * j3) % L] += p1 * p2 * p3
    return q


def sample_n(q, size, g):
    return np.stack([g.choice(L, size, p=q), g.choice(L, size, p=q)], 1)


def kernel_stats(n, w2):
    A = np.zeros((L, L), complex)
    np.add.at(A, (n[:, 0], n[:, 1]), w2)
    k = np.fft.ifft2(A).real * L * L / w2.sum()
    row = k[0, :64]
    half = np.interp(0.5, row[::-1], np.arange(64)[::-1])
    far = RAD > 3 * r
    i = np.argmax(np.where(far, k, -9))
    return half, k[far].max(), (int(d[i // L]), int(d[i % L]))


print(f"target half-width {r*np.sqrt(2*np.log(2)):.2f}")
for s in (None, 3.0, 2.0, 1.0):
    q1 = q_axis(s)
    surv = (q1 * P1 / P1.max()).sum() ** 2                 # 2D: product of axes
    pos = q1 > 0
    neff_per = 1 / (P1[pos] ** 2 / q1[pos]).sum() ** 2     # importance weighting, per unit built
    miss = P1[~pos].sum()
    q = q1
    g = np.random.default_rng(0)
    # prune: pool so ~512 survive
    pool = min(int(512 / surv), 4_000_000)
    n = sample_n(q, pool, g)
    keep = g.random(pool) < (P1[n[:, 0]] / P1.max()) * (P1[n[:, 1]] / P1.max())
    hp, ap, wp = kernel_stats(n[keep], np.ones(keep.sum()))
    nk = int(keep.sum())
    del n, keep
    # importance: pool so ~512 effective, capped at 4M units
    pool_i = min(int(512 / neff_per), 4_000_000)
    n = sample_n(q, pool_i, g)
    w2 = (P1[n[:, 0]] * P1[n[:, 1]]) / (q1[n[:, 0]] * q1[n[:, 1]])
    hi, ai, wi = kernel_stats(n, w2)
    ne = w2.sum() ** 2 / (w2 ** 2).sum()
    del n, w2
    label = "uniform j" if s is None else f"j ~ Gaussian, sd {s:.0f}"
    print(f"{label:20s} | prune: 1 survivor per {1/surv:7.0f} units ({nk} kept), half-width {hp:5.2f}, alias {ap:.3f} at {wp} | "
          f"import: 1 effective per {1/neff_per:7.0f} units (pool {pool_i:,}, n_eff {ne:.0f}), half-width {hi:5.2f}, alias {ai:.3f} at {wi}"
          f" | spectrum unreachable {miss:.3f}", flush=True)
