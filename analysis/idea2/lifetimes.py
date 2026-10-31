"""Random-walk lifetimes for meta-training idea 2's memory offline (plan §6.4).

An env is its wall code and panorama; views are ray-cast on demand, so a pool
of thousands of envs costs a few MB. A lifetime places each env at a random
scaffold offset with a random goal. A visit is a random walk whose views,
labelled with that env's goal phases, are what a write trains on. Test views
are drawn at uniform random cells and headings.
"""
from __future__ import annotations

import numpy as np

from hopfield_nav.config import EnvConfig
from hopfield_nav.memory.plastic import canvas, goal_phases
from hopfield_nav.world.env import make_env, raycast_codes

SIZE, OBS, WALL_RES, LAMBDAS, NPOS = 20, 60, 4, (11, 12, 13), 1716


class EnvPool:
    """Wall codes and panoramas of ``n`` envs drawn from ``seed``'s stream."""

    def __init__(self, n: int, seed: int, distal_amp: float = 1.0):
        cfg = EnvConfig(size=SIZE, observation_size=OBS, wall_resolution=WALL_RES,
                        distal_amp=distal_amp)
        seeds = np.random.RandomState(seed).randint(0, 2 ** 31 - 1, size=n)
        self.walls, self.panos = [], []
        for s in seeds:
            e = make_env(cfg, "continuous", seed=int(s))
            self.walls.append(e._wall_code.copy())
            self.panos.append(e._panorama.copy())

    def __len__(self):
        return len(self.walls)

    def views(self, i: int, xs, ys, psi) -> np.ndarray:
        return raycast_codes(self.walls[i], SIZE, xs, ys, psi, OBS, WALL_RES,
                             self.panos[i])


class Placed:
    """One env in one lifetime: pool index, scaffold offset, goal cell."""

    def __init__(self, pool: EnvPool, idx: int, rng: np.random.RandomState):
        self.pool, self.idx = pool, idx
        self.offset = rng.randint(0, NPOS - SIZE, size=2)
        self.goal = rng.randint(0, SIZE, size=2)
        g = self.goal + self.offset
        self.target = goal_phases([g[0]], [g[1]], LAMBDAS)            # (1, 12)
        gx, gy = np.meshgrid(np.arange(SIZE), np.arange(SIZE), indexing="ij")
        self.cells_local = np.stack([gx.ravel(), gy.ravel()], 1)
        self.cells_global = self.cells_local + self.offset

    def visit(self, rng, steps: int) -> np.ndarray:
        """Canvas inputs along a random walk: heading follows the move, +- 45 deg."""
        pos = rng.randint(0, SIZE, size=2).astype(int)
        xs, ys, ps = [], [], []
        moves = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])
        for _ in range(steps):
            k = rng.randint(4)
            pos = np.clip(pos + moves[k], 0, SIZE - 1)
            xs.append(pos[0]); ys.append(pos[1])
            ps.append(k * np.pi / 2 + rng.uniform(-np.pi / 4, np.pi / 4))
        psi = np.array(ps)
        return canvas(self.pool.views(self.idx, np.array(xs, float), np.array(ys, float), psi), psi)

    def test_views(self, rng, n: int):
        """``n`` views at uniform random cells and headings, plus the cells."""
        cells = rng.randint(0, SIZE, size=(n, 2))
        psi = rng.uniform(-np.pi, np.pi, n)
        v = self.pool.views(self.idx, cells[:, 0].astype(float), cells[:, 1].astype(float), psi)
        return canvas(v, psi), cells
