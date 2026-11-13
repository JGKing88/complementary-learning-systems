"""Stress test: fewer angle units, and phase noise. Periods 11, 12, 13 (L = 1716), r = 16.

Each unit: frequency n, centred whole-number weights j (CRT), angle sum_l,i j phi;
z = (w cos, w sin) / sqrt(sum w^2). Phase noise: each evaluation adds N(0, s^2) cells to
each of the six phases independently. Measures, around a goal at P0:
  traps: cells with 0 < |delta| <= 2r whose (noisy) similarity is >= all 4 neighbours';
  walk: greedy steps to the most similar 4-neighbour (fresh noise every read) from 16
        starts at distance 2r; success = within 2 cells of the goal by 100 steps.

Docs: EXPERIMENTS_IDEAL_ENCODER.md §25.3. Run on a compute node, from the repo root:
    python analysis/hopfield_probe/kernel_scaling/stress_units_noise.py
"""
import numpy as np
L, r = 1716, 16.0
LAMS = np.array([11, 12, 13])
INV = np.array([6, -1, 7])                         # j = INV * n mod lambda
P0 = np.array([700, 900])
R2 = int(2 * r)


def centred(v, lam):
    v = v % lam
    return np.where(v > lam // 2, v - lam, v)


def units_ideal(m, g):
    n = np.rint(g.normal(0, L / (2 * np.pi * r), (m, 2))).astype(int)
    return n, np.ones(m)


def units_uniform(pool, g):
    n = g.integers(-858, 859, (pool, 2))
    o = 2 * np.pi * n / L
    w2 = np.exp(-256 * (2 - np.cos(o[:, 0]) - np.cos(o[:, 1])))
    order = np.argsort(-w2)
    keep = order[: np.searchsorted(np.cumsum(w2[order]) / w2.sum(), 0.995) + 1]
    return n[keep], w2[keep]


def weights_j(n):
    """(U, 2 axes, 3 modules) centred CRT weights."""
    return np.stack([np.stack([centred(INV[l] * n[:, i], LAMS[l]) for l in range(3)], 1)
                     for i in range(2)], 1)


def codes(X, J, w, s, g):
    """z at locations X (N, 2) with phase noise s cells."""
    ph = 2 * np.pi * ((X[:, :, None] % LAMS) + s * g.standard_normal((len(X), 2, 3))) / LAMS
    theta = np.einsum("xia,uia->xu", ph, J)
    a = np.sqrt(w / w.sum())
    return np.concatenate([a * np.cos(theta), a * np.sin(theta)], 1)


gx, gy = np.meshgrid(np.arange(-R2 - 1, R2 + 2), np.arange(-R2 - 1, R2 + 2), indexing="ij")
GRID = np.stack([gx.ravel(), gy.ravel()], 1)
RAD = np.hypot(gx, gy)[1:-1, 1:-1]
STEPS = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
STARTS = np.rint(np.stack([R2 * np.cos(np.linspace(0, 2 * np.pi, 16, endpoint=False)),
                           R2 * np.sin(np.linspace(0, 2 * np.pi, 16, endpoint=False))], 1)).astype(int)


def measure(n, w, s, g):
    J = weights_j(n)
    zg = codes(P0[None], J, w, s, g)[0]
    S = (codes(P0 + GRID, J, w, s, g) @ zg).reshape(gx.shape)
    c = S[1:-1, 1:-1]
    nb = np.max(np.stack([S[2:, 1:-1], S[:-2, 1:-1], S[1:-1, 2:], S[1:-1, :-2]]), 0)
    traps = int(((c >= nb) & (RAD > 0) & (RAD <= 2 * r)).sum())
    ok = 0
    for st in STARTS:
        pos = st.copy()
        for _ in range(100):
            nbrs = P0 + pos + STEPS
            pos = pos + STEPS[np.argmax(codes(nbrs, J, w, s, g) @ zg)]
            if np.hypot(*pos) <= 2:
                ok += 1
                break
    return traps, ok


print(f"{'network':38s} {'units':>6s} {'noise s':>8s}   traps within 2r (3 seeds)   walks from 2r reaching the goal")
for kind, sizes in (("ideal", (512, 128, 32)), ("uniform + depression", (400_000, 100_000, 25_000))):
    for size in sizes:
        for s in (0.0, 0.02, 0.05, 0.1):
            tr, ok, nu = [], [], []
            for seed in range(3):
                g = np.random.default_rng(1000 * seed + size % 997)
                n, w = units_ideal(size, g) if kind == "ideal" else units_uniform(size, g)
                t, k = measure(n, w, s, g)
                tr.append(t); ok.append(k); nu.append(len(n))
            label = kind if kind == "ideal" else f"{kind}, pool {size // 1000}k"
            print(f"{label:38s} {int(np.mean(nu)):6d} {s:8.2f}   {str(tr):26s} {sum(ok)}/48", flush=True)
