"""Is the similarity strictly decreasing out to 2r (and 3r)? r = 16, L = 1716.

Docs: EXPERIMENTS_IDEAL_ENCODER.md §25.2. Run on a compute node, from the repo root:
    python analysis/hopfield_probe/kernel_scaling/monotone_rays.py
"""
import numpy as np
L, r = 1716, 16.0
DIRS = [(1, 0), (0, 1), (1, 1), (1, -1), (2, 1), (1, 2), (2, -1), (1, -2)]
rng = np.random.default_rng(0)


def first_rise(k):
    """First step index where k goes up (None if strictly decreasing) and the largest rise."""
    dk = np.diff(k)
    up = np.nonzero(dk >= 0)[0]
    return (None if len(up) == 0 else int(up[0]) + 1), float(dk.max())


def rays(n, w2, reach):
    """Kernel along each direction, at integer multiples of the direction, out to |delta| <= reach."""
    out = []
    for a, b in DIRS:
        s = np.arange(0, int(reach / np.hypot(a, b)) + 1)
        D = np.stack([s * a, s * b], 1)
        k = (w2[None, :] * np.cos(2 * np.pi * (D @ n.T) / L)).sum(1) / w2.sum()
        out.append((np.hypot(a, b), k))
    return out


def report(tag, n, w2):
    for mult in (2, 3):
        bad = []
        for step, k in rays(n, w2, mult * r):
            i, rise = first_rise(k)
            if i is not None:
                bad.append(f"{i * step:.0f}")
        print(f"  {tag:46s} to {mult}r: {'strictly decreasing on all 8 rays' if not bad else 'first rise at ' + ', '.join(bad) + ' cells'}")


# (a) the exact depression kernel, every frequency present (separable in x and y)
nn = np.arange(L); nn = np.where(nn > L // 2, nn - L, nn); om = 2 * np.pi * nn / L
d = 2 - 2 * np.cos(om)
print("(a) exact depression kernel, all frequencies, 300 walks per kappa")
for kap in (0.005, 0.05, 0.5, 2.0, 5.0):
    T = round(r * r / (2 * kap))
    worst = {2: 0, 3: 0}
    for s in range(300):
        for ax in range(2):
            Tx = rng.binomial(T, 0.5)
            w2 = np.clip(1 - kap * d, 0, None) ** (2 * Tx)
            for mult in (2, 3):
                D = np.arange(0, int(mult * r) + 1)
                k = (w2[None, :] * np.cos(np.outer(D, om))).sum(1) / w2.sum()
                if first_rise(k)[0] is not None:
                    worst[mult] += 1
    print(f"  kappa {kap:5.3f} T {T:6d}: axis kernels that rise before 2r: {worst[2]}/600, before 3r: {worst[3]}/600")

# (b) finite networks
print("(b) finite networks (8 directions, integer steps)")
for seed in range(3):
    g = np.random.default_rng(seed)
    n = np.rint(g.normal(0, L / (2 * np.pi * r), (512, 2))).astype(int)
    report(f"ideal, 512 frequencies drawn from P, seed {seed}", n, np.ones(512))
for pool in (400_000, 1_600_000):
    for seed in range(2):
        g = np.random.default_rng(10 + seed)
        n = g.integers(-858, 859, (pool, 2))
        o = 2 * np.pi * n / L
        w2 = np.exp(-2 * 128 * (2 - np.cos(o[:, 0]) - np.cos(o[:, 1])))
        order = np.argsort(-w2); keep = order[: np.searchsorted(np.cumsum(w2[order]) / w2.sum(), 0.995) + 1]
        report(f"uniform pool {pool:,}, depression weights, {len(keep)} kept, seed {seed}", n[keep], w2[keep])
