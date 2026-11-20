"""Agent-like movement for idea 4's memory (GRID_MLP_NAV_PLAN §7b.14).

The navigation agent moves continuously: unit steps in any direction from a
float position, snapped to the nearest cell (``ContinuousGridEnv.step``),
facing the direction travelled. Along its own trajectories in the continual
protocol about a quarter of its steps change the cell diagonally and about
a tenth leave it unchanged; the memory was trained on walks that move one
cell along an axis every step, and fails on the agent's (§7b.14 stage 3).

Batched paths, (L, B) per step, with each row's number of moves n (B,):
row b is active at steps t <= n[b].
- ``approach``: from a float start toward the goal, unit steps in the goal's
  direction plus Gaussian angle noise, until within ``radius`` of the goal
  (the agent's at-goal test, goal radius 1).
- ``wander``: a correlated random walk (the heading drifts by Gaussian angle
  noise each step), the agent exploring or detouring.
- ``revisit``: an optional wander, then an approach.
- ``search``: a wander from a random start, then an approach; the first
  arrival is the last step.
"""
from __future__ import annotations

import math

import torch


def _snap(pf, size):
    return pf.round().clamp(0, size - 1).long()


def _walk(pf, theta, steps, size, noise, gen, goal=None, radius=1.0):
    """Shared stepping. Returns float positions (steps + 1, B, 2), headings
    (steps + 1, B) and the step each row first came within ``radius`` of
    ``goal`` (or ``steps`` if never / no goal)."""
    B = len(pf)
    dev = pf.device
    out_p, out_h = [pf.clone()], [theta.clone()]
    arrive = torch.full((B,), steps, dtype=torch.long, device=dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    if goal is not None:
        hit = (_snap(pf, size).float() - goal.float()).norm(dim=-1) <= radius
        arrive = torch.where(hit, torch.zeros_like(arrive), arrive)
        done |= hit
    for t in range(1, steps + 1):
        if goal is not None:
            to = goal.float() - pf
            theta = torch.atan2(to[:, 0], to[:, 1])          # clockwise from North, as the env
        theta = theta + noise * torch.randn(B, device=dev, generator=gen)
        step = torch.stack([torch.sin(theta), torch.cos(theta)], -1)
        new = (pf + step).clamp(0, size - 1)
        disp = new - pf
        moved = disp.norm(dim=-1) > 1e-9
        head = torch.where(moved, torch.atan2(disp[:, 0], disp[:, 1]), out_h[-1])
        pf = torch.where(done[:, None], pf, new)
        head = torch.where(done, out_h[-1], head)
        out_p.append(pf.clone())
        out_h.append(head)
        if goal is not None:
            hit = (_snap(pf, size).float() - goal.float()).norm(dim=-1) <= radius
            arrive = torch.where(hit & ~done, torch.full_like(arrive, t), arrive)
            done |= hit
            if bool(done.all()):
                break
    return torch.stack(out_p), torch.stack(out_h), arrive


def approach(start_f, goal, size, gen, noise=0.3, radius=1.0, max_steps=60):
    """Cells (L, B, 2), headings (L, B), moves n (B,)."""
    theta = torch.rand(len(start_f), device=start_f.device, generator=gen) * 2 * math.pi
    p, h, n = _walk(start_f, theta, max_steps, size, noise, gen, goal=goal, radius=radius)
    return _snap(p, size), h, n


def wander(start_f, steps, size, gen, noise=0.5):
    theta = torch.rand(len(start_f), device=start_f.device, generator=gen) * 2 * math.pi
    p, h, _ = _walk(start_f, theta, steps, size, noise, gen)
    return p, h


def _join(pw, hw, goal, size, gen, noise, radius, max_steps):
    """A wander (float positions pw, headings hw) followed by an approach
    from where it ended."""
    cells_a, h_a, n_a = approach(pw[-1], goal, size, gen, noise, radius, max_steps)
    cells = torch.cat([_snap(pw, size), cells_a[1:]])
    heads = torch.cat([hw, h_a[1:]])
    return cells, heads, n_a + (len(pw) - 1)


def revisit(goal, size, gen, wander_max=0, p_wander=0.0, noise=0.3, radius=1.0, max_steps=60):
    """A path from a random cell to within ``radius`` of the goal, after a
    detour of up to ``wander_max`` steps with probability ``p_wander``."""
    B, dev = len(goal), goal.device
    start = torch.randint(0, size, (B, 2), device=dev, generator=gen).float()
    L = int(torch.randint(1, wander_max + 1, (1,), generator=gen)) if wander_max else 0
    if L and float(torch.rand(1, generator=gen)) < p_wander:
        pw, hw = wander(start, L, size, gen)
        return _join(pw, hw, goal, size, gen, noise, radius, max_steps)
    return approach(start, goal, size, gen, noise, radius, max_steps)


def search(goal, size, gen, wander_min=5, wander_max=60, noise=0.3, radius=1.0, max_steps=60):
    """Exploring from a random cell, then finding the goal; the first
    arrival is step n (the store step)."""
    B, dev = len(goal), goal.device
    start = torch.randint(0, size, (B, 2), device=dev, generator=gen).float()
    L = int(torch.randint(wander_min, wander_max + 1, (1,), generator=gen))
    pw, hw = wander(start, L, size, gen)
    cells, heads, n = _join(pw, hw, goal, size, gen, noise, radius, max_steps)
    # the wander may already pass within the radius; the store fires at the first arrival
    near = (cells.float() - goal[None].float()).norm(dim=-1) <= radius
    t = torch.arange(len(cells), device=dev)[:, None].expand_as(near)
    first = torch.where(near, t, torch.full_like(t, len(cells))).min(0).values
    return cells, heads, torch.minimum(first, n)
