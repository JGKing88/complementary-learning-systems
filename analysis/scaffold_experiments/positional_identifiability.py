"""Can a single view say where the agent is? And what wall_resolution buys.

Read-only probe against the production env -- no prototypes, no local copy of
the ray-caster.

THE DEFECT
    At ``wall_resolution=1`` each wall carries one +/-1 segment per grid cell,
    and segment boundaries sit exactly on cell boundaries. So every ray landing
    anywhere inside a cell reads the same value, and the wall carries nothing
    about where *within* a cell the agent stands. Some cells then produce
    bit-identical observations, which is an information-theoretic wall: there is
    no difference for any readout to find, learned or otherwise.

    It concentrates where the geometry says it must. A 120 deg cone at distance
    d from the wall it faces spans only ~2*d*tan(60) ~ 3.5*d cells of that wall,
    so against a wall the agent sees ~3.5 segments -- against the ~6 bits needed
    to name one of 64 cells.

THE FIX
    ``wall_resolution=R`` subdivides each cell into R segments, so a stripe edge
    can fall inside a cell and two positions within one cell can read
    differently. This measures how fast that closes the gap, and confirms the
    requirement is near-independent of room size: the cone sees a fixed extent
    of wall however big the room, so this is a property of the sensor.

Usage:
    python -m analysis.scaffold_experiments.positional_identifiability
"""
from __future__ import annotations

import numpy as np

from hopfield_nav.config import EnvConfig
from hopfield_nav.world.env import FOVEAL_HALF_ANGLE_DEG, make_env

OBS = 60


def _north_views(size, res, obs, seed):
    """Every cell's North-facing view: (size*size, obs), unit-normalised."""
    env = make_env(EnvConfig(size=size, observation_size=obs,
                             wall_resolution=res), "discrete", seed=seed)
    V = env._codebook[:, :, 0, :].reshape(size * size, obs)
    return V / np.linalg.norm(V, axis=1, keepdims=True)


def twin_fraction(size, res, obs=OBS, n_seeds=6):
    """Fraction of cells sharing a bit-identical view with some other cell."""
    out = []
    for seed in range(n_seeds):
        C = _north_views(size, res, obs, seed)
        C = C @ C.T
        np.fill_diagonal(C, -1.0)
        out.append((C > 0.9999).any(axis=1).mean())
    return float(np.mean(out))


def by_distance(size, res, obs=OBS, n_seeds=6, thresh=0.99):
    """Mean confusable partners per cell, bucketed by distance to the N wall."""
    buckets = {d: [] for d in range(size)}
    for seed in range(n_seeds):
        C = _north_views(size, res, obs, seed)
        C = C @ C.T
        np.fill_diagonal(C, -1.0)
        conf = (C > thresh).sum(axis=1)
        for idx in range(size * size):
            y = idx % size                       # codebook is indexed [x, y]
            buckets[size - 1 - y].append(conf[idx])
    return [float(np.mean(buckets[d])) for d in range(size)]


def main() -> int:
    print("Exact observational twins -- cells no readout can ever separate")
    print(f"({OBS} rays, heading North; 0% means position is identifiable)")
    print()
    sizes, resolutions = [6, 8, 12, 16, 24], [1, 2, 4, 8]
    print(f"  {'size':>5} {'cells':>6} " +
          " ".join(f"{'res=' + str(r):>8}" for r in resolutions))
    print("  " + "-" * (14 + 9 * len(resolutions)))
    for size in sizes:
        row = " ".join(f"{twin_fraction(size, r):7.1%} " for r in resolutions)
        print(f"  {size:5d} {size * size:6d} {row}")
    print("\n  Flat in size: a cone sees a fixed extent of wall however big the")
    print("  room, so the requirement is set by the sensor, not the geometry.")

    size = 8
    print(f"\n\nWhere the failures sit ({size}x{size}, confusable partners per cell)")
    print("  distance 0 = standing against the North wall it is facing")
    print(f"  {'res':>5} " + " ".join(f"{d:>6}" for d in range(size)))
    print("  " + "-" * (8 + 7 * size))
    for res in (1, 2, 8):
        print(f"  {res:5d} " +
              " ".join(f"{v:6.2f}" for v in by_distance(size, res)))

    print("\n\nWhy: how much wall a 120 deg cone spans, by distance")
    print(f"  {'d':>4} {'cells of wall in view':>24} {'segments @res=1':>17}")
    for d in range(1, size):
        span = min(2 * d * np.tan(np.deg2rad(FOVEAL_HALF_ANGLE_DEG)), size)
        print(f"  {d:4d} {span:24.1f} {span:17.1f}")
    print("\n  Against a wall you see ~3.5 segments; naming one of 64 cells")
    print("  needs ~6 bits. That is the whole defect.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
