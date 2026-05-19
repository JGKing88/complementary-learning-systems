"""Shortest-path oracle for the open grid.

The current GridEnv has no obstacles, so shortest-path = greedy Manhattan
toward the goal with random tie-breaking. The module name is `oracle_bfs`
because the BFS extension to obstacle-aware envs is a drop-in replacement
that preserves the same call signature.
"""
from __future__ import annotations

import numpy as np

from .env import CARDINAL_ACTIONS


def bfs_action_batch_discrete(
    positions: np.ndarray,
    goal: tuple[int, int],
    size: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Per-env shortest-path cardinal action toward `goal`.

    positions: (B, 2) int — current cells.
    goal:      (gx, gy) int.
    size:      grid size for boundary clipping.

    Returns (B,) int32 indices into CARDINAL_ACTIONS. Ties (multiple actions
    that achieve the same min Manhattan distance to goal) broken uniformly at
    random via `rng`. At goal, returns 0 — caller is responsible for masking
    the loss at at-goal steps.
    """
    B = positions.shape[0]
    out = np.zeros(B, dtype=np.int32)
    gx, gy = int(goal[0]), int(goal[1])
    for b in range(B):
        cx, cy = int(positions[b, 0]), int(positions[b, 1])
        if (cx, cy) == (gx, gy):
            out[b] = 0
            continue
        best_dist = None
        best_actions: list[int] = []
        for a_idx, (dx, dy) in enumerate(CARDINAL_ACTIONS):
            nx = max(0, min(size - 1, cx + dx))
            ny = max(0, min(size - 1, cy + dy))
            dist = abs(nx - gx) + abs(ny - gy)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_actions = [a_idx]
            elif dist == best_dist:
                best_actions.append(a_idx)
        out[b] = best_actions[rng.randint(len(best_actions))]
    return out


def bfs_action_batch_continuous(
    positions: np.ndarray,
    goal: tuple[int, int],
    rng: np.random.RandomState,
) -> np.ndarray:
    """Unit (dx, dy) toward goal per env. positions: (B, 2) int (snapped).

    At-goal: returns a random unit vector (no preferred direction). Caller
    masks the loss at at-goal steps so this is never supervised.
    """
    B = positions.shape[0]
    out = np.zeros((B, 2), dtype=np.float32)
    gx, gy = float(goal[0]), float(goal[1])
    for b in range(B):
        dx = gx - float(positions[b, 0])
        dy = gy - float(positions[b, 1])
        n = float(np.sqrt(dx * dx + dy * dy))
        if n < 1e-8:
            theta = rng.uniform(0.0, 2 * np.pi)
            out[b] = [np.sin(theta), np.cos(theta)]
        else:
            out[b] = [dx / n, dy / n]
    return out
