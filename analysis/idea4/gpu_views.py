"""Training views for idea 4 on the GPU (GRID_MLP_NAV_PLAN §7b.2).

A torch port of ``hopfield_nav.world.env.raycast_codes`` followed by
``hopfield_nav.memory.plastic.canvas``, batched over envs of an
``analysis.idea2.lifetimes.EnvPool``. Geometry runs in float64, as the numpy
version does, so the two agree view for view (``python -m
analysis.idea4.gpu_views`` checks this). The retention test does not use this
file. It keeps idea 2's numpy path, so its envs, goals and views are the ones
§10.9 scored.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from hopfield_nav.world.env import PANORAMA_BIN_DEG, PANORAMA_BINS, cone_offsets
from analysis.idea2.lifetimes import LAMBDAS, NPOS, OBS, SIZE, WALL_RES

MOVES = ((0, 1), (1, 0), (0, -1), (-1, 0))


class GpuPool:
    """An ``EnvPool``'s wall codes and panoramas as device tensors."""

    def __init__(self, pool, device):
        self.walls = torch.from_numpy(np.stack(pool.walls).astype(np.float32)).to(device)
        self.panos = torch.from_numpy(np.stack(pool.panos).astype(np.float32)).to(device)
        self.offsets = torch.as_tensor(cone_offsets(OBS), dtype=torch.float64, device=device)
        self.device = device

    def __len__(self):
        return len(self.walls)

    def canvas(self, env, xs, ys, psi):
        """Canvas inputs (values + mask, ``2 * PANORAMA_BINS``) for views from
        cells (xs, ys) facing psi in envs ``env``. All args (...,); returns
        (..., 360) float32."""
        shape = env.shape
        env, xs, ys, psi = (t.reshape(-1) for t in (env, xs, ys, psi))
        ang = psi.double()[:, None] + self.offsets[None, :]            # (B, n)
        dx, dy = torch.sin(ang), torch.cos(ang)
        cx, cy = xs.double()[:, None], ys.double()[:, None]
        hi = SIZE - 0.5
        inf = torch.tensor(math.inf, dtype=torch.float64, device=ang.device)

        def plane(num, den, keep):
            t = num / torch.where(keep, den, torch.ones_like(den))
            return torch.where(keep & (t >= 0.0), t, inf)

        t_n, t_e = plane(hi - cy, dy, dy > 0.0), plane(hi - cx, dx, dx > 0.0)
        t_s, t_w = plane(-0.5 - cy, dy, dy < 0.0), plane(-0.5 - cx, dx, dx < 0.0)
        ts, hits = [], []
        for t, h in ((t_n, cx + t_n * dx), (t_e, cy + t_e * dy),
                     (t_s, cx + t_s * dx), (t_w, cy + t_w * dy)):
            ts.append(torch.where(torch.isfinite(t) & ((h < -0.5) | (h > hi)), inf, t))
            hits.append(h)
        ts, hits = torch.stack(ts, -1), torch.stack(hits, -1)         # (B, n, 4)
        wall = ts.argmin(-1)
        h = hits.gather(-1, wall[..., None])[..., 0]
        fine = (torch.where(torch.isfinite(h), h, torch.zeros_like(h)) + 0.5) * WALL_RES
        seg = fine.floor().clamp(0, SIZE * WALL_RES - 1).long()
        codes = self.walls[env[:, None], wall, seg]
        codes = torch.where(torch.isfinite(ts).any(-1), codes, torch.zeros_like(codes))
        deg = torch.remainder(torch.rad2deg(ang), 360.0)
        bins = torch.remainder(torch.floor(deg / PANORAMA_BIN_DEG).long(), PANORAMA_BINS)
        codes = codes + self.panos[env[:, None], bins]
        out = torch.zeros(len(env), 2 * PANORAMA_BINS, device=ang.device)
        out.scatter_(1, bins, codes)
        out.scatter_(1, bins + PANORAMA_BINS, torch.ones_like(codes))
        return out.reshape(*shape, 2 * PANORAMA_BINS)


def walk(B: int, steps: int, device, gen=None):
    """``B`` random walks of ``steps`` moves, as ``Placed.visit`` makes them:
    a uniform start cell, then a uniform move each step clipped at the walls,
    heading along the move +- 45 deg. Returns xs, ys, psi, each (steps, B)."""
    moves = torch.tensor(MOVES, device=device)
    pos = torch.randint(0, SIZE, (B, 2), device=device, generator=gen)
    k = torch.randint(0, 4, (steps, B), device=device, generator=gen)
    xy = []
    for s in range(steps):
        pos = (pos + moves[k[s]]).clamp(0, SIZE - 1)
        xy.append(pos)
    xy = torch.stack(xy)
    jitter = torch.rand(steps, B, device=device, generator=gen) * (math.pi / 2) - math.pi / 4
    return xy[..., 0], xy[..., 1], k * (math.pi / 2) + jitter


def random_cells(shape, device, gen=None):
    """Uniform cells and headings, as ``Placed.test_views`` draws them."""
    c = torch.randint(0, SIZE, (*shape, 2), device=device, generator=gen)
    psi = (torch.rand(*shape, device=device, generator=gen) * 2 - 1) * math.pi
    return c[..., 0], c[..., 1], psi


def goal_phases(g):
    """(..., 2) global cells -> (..., 12) unit phase pairs, module-major, as
    ``hopfield_nav.memory.plastic.goal_phases``."""
    cols = []
    for lam in LAMBDAS:
        for k in (0, 1):
            a = 2 * math.pi * torch.remainder(g[..., k].double(), lam) / lam
            cols += [torch.cos(a), torch.sin(a)]
    return torch.stack(cols, -1).float()


def place(B: int, K: int, device, gen=None):
    """Scaffold offsets and goal cells for B lifetimes of K envs, as ``Placed``
    draws them. Returns (offset (B, K, 2), goal (B, K, 2), target (B, K, 12))."""
    off = torch.randint(0, NPOS - SIZE, (B, K, 2), device=device, generator=gen)
    goal = torch.randint(0, SIZE, (B, K, 2), device=device, generator=gen)
    return off, goal, goal_phases(off + goal)


def _check(n_envs=20, n_views=4000, seed=0):
    """Compare the torch canvas with the numpy one on random views."""
    from hopfield_nav.memory.plastic import canvas, goal_phases as np_phases
    from analysis.idea2.lifetimes import EnvPool
    pool = EnvPool(n_envs, seed=seed + 1)
    gp = GpuPool(pool, torch.device("cpu"))
    rng = np.random.RandomState(seed)
    env = rng.randint(0, n_envs, n_views)
    xs, ys = rng.randint(0, SIZE, n_views), rng.randint(0, SIZE, n_views)
    psi = rng.uniform(-np.pi, np.pi, n_views)
    psi[:50] = np.arange(50) * (np.pi / 2)              # exact cardinal headings
    ref = np.concatenate([canvas(pool.views(e, [x], [y], [p]), [p])
                          for e, x, y, p in zip(env, xs, ys, psi)])
    got = gp.canvas(torch.from_numpy(env), torch.from_numpy(xs), torch.from_numpy(ys),
                    torch.from_numpy(psi)).numpy()
    bad = np.abs(got - ref).max(axis=1) > 1e-5
    print(f"canvas: {bad.sum()} of {n_views} views differ (max abs diff "
          f"{np.abs(got - ref).max():.2e})")
    g = rng.randint(0, NPOS, (100, 2))
    d = np.abs(goal_phases(torch.from_numpy(g)).numpy() - np_phases(g[:, 0], g[:, 1], LAMBDAS))
    print(f"goal phases: max abs diff {d.max():.2e}")


if __name__ == "__main__":
    _check()
