"""Is position identifiable from a single view? And where does it fail?

Prediction under test: a 120 deg cone at distance d from the wall it faces spans
~3.5*d cells of that wall, so an agent standing near a wall sees very little of
it and cannot localise along it. Ambiguity should therefore concentrate at cells
CLOSE to the faced wall, and finer wall segments should relieve it.
"""
import sys
import numpy as np

from analysis.scaffold_experiments.wall_resolution import (
    hires_codebook, wall_pattern,
)

SIZE, OBS = 8, 60


def views_north(res, sigma, seed):
    wc = wall_pattern(np.random.RandomState(seed), res, sigma)
    cb = hires_codebook(wc, SIZE, res, OBS)
    V = cb[:, :, 0, :].reshape(SIZE * SIZE, OBS)          # heading North
    return V / np.linalg.norm(V, axis=1, keepdims=True)


def report(label, res, sigma, n_seeds=8, thresh=0.99):
    """Mean #confusable partners per cell, bucketed by distance to North wall."""
    by_d = {d: [] for d in range(SIZE)}
    exact = []
    for seed in range(n_seeds):
        V = views_north(res, sigma, seed)
        C = V @ V.T
        np.fill_diagonal(C, -1.0)
        conf = (C > thresh).sum(axis=1)
        exact.append((C > 0.9999).any(axis=1).mean())
        for idx in range(SIZE * SIZE):
            y = idx % SIZE                                 # cb is [x, y]
            by_d[SIZE - 1 - y].append(conf[idx])            # distance to N wall
    print(f"\n  {label}")
    print(f"    cells with an exact twin: {np.mean(exact):.1%}")
    print(f"    {'dist to faced wall':>20} " +
          " ".join(f"{d:>5}" for d in range(SIZE)))
    print(f"    {'mean confusable':>20} " +
          " ".join(f"{np.mean(by_d[d]):5.2f}" for d in range(SIZE)))


print("Confusable partners per cell (cosine > 0.99), heading North, 8 envs")
print("distance 0 = standing against the North wall, 7 = furthest from it")
report("res=1  sigma=0   (current env)", 1, 0.0)
report("res=4  sigma=0", 4, 0.0)
report("res=16 sigma=8", 16, 8.0)
report("res=16 sigma=0", 16, 0.0)

print("\n\nHow much wall a 120 deg cone actually spans, by distance:")
print(f"  {'d':>4} {'cells of wall in view':>24} {'segments @res=1':>17} "
      f"{'@res=16':>9}")
for d in range(1, SIZE):
    span = 2 * d * np.tan(np.deg2rad(60))
    print(f"  {d:4d} {min(span, SIZE):24.1f} {min(span, SIZE):17.1f} "
          f"{min(span, SIZE) * 16:9.0f}")
