"""Teacher actions for DAgger: shortest-path and novelty-seeking.

The current GridEnv has no obstacles, so shortest-path = greedy Manhattan
toward the goal with random tie-breaking. The module name is `oracle_bfs`
because the BFS extension to obstacle-aware envs is a drop-in replacement
that preserves the same call signature.

The novelty oracles below were in `bc.py`, which made `rollout.collector` --
the module that *calls* them, once per step, to label the explore phase --
import from `updates`, a layer above it. They are teachers, not losses;
`bc_update` consumes the labels they produced but never calls them.
"""
from __future__ import annotations

import numpy as np

from ..world.env import CARDINAL_ACTIONS


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


# ---------------------------------------------------------------------------
# Novelty oracles
# ---------------------------------------------------------------------------

def novelty_action_batch_discrete(
    positions: np.ndarray,
    visited_cells: np.ndarray,
    size: int,
    rng: np.random.RandomState,
    fallback: str = "random",
) -> np.ndarray:
    """Pick a cardinal action per env that lands on an unvisited cell.

    positions:     (B, 2) int — current cell.
    visited_cells: (B, size, size) bool — cells visited this rollout.
    fallback:      "random" → uniform over all 4 actions when no neighbor is
                   unvisited; "stay" → pick action 0 (agent still nudges).

    Returns (B,) int32 indices into CARDINAL_ACTIONS.
    """
    B = positions.shape[0]
    out = np.zeros(B, dtype=np.int32)
    for b in range(B):
        cx, cy = int(positions[b, 0]), int(positions[b, 1])
        unvisited_cands = []
        all_cands = []
        for a_idx, (dx, dy) in enumerate(CARDINAL_ACTIONS):
            nx = max(0, min(size - 1, cx + dx))
            ny = max(0, min(size - 1, cy + dy))
            all_cands.append(a_idx)
            if not visited_cells[b, nx, ny]:
                unvisited_cands.append(a_idx)
        if unvisited_cands:
            out[b] = unvisited_cands[rng.randint(len(unvisited_cands))]
        elif fallback == "stay":
            out[b] = 0
        else:  # "random"
            out[b] = all_cands[rng.randint(len(all_cands))]
    return out


def novelty_action_batch_continuous(
    positions: np.ndarray,
    visited_cells: np.ndarray,
    size: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Unit vector toward the nearest unvisited cell per env.

    positions: (B, 2) float — continuous position (snapping tolerated).
    Returns (B, 2) float32 unit vectors. Falls back to a random unit vector
    if every cell has been visited.
    """
    B = positions.shape[0]
    out = np.zeros((B, 2), dtype=np.float32)
    # Precompute (X, Y) grid once.
    xs, ys = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    for b in range(B):
        mask = ~visited_cells[b]  # (size, size) bool
        if not mask.any():
            theta = rng.uniform(0.0, 2 * np.pi)
            out[b] = [np.sin(theta), np.cos(theta)]
            continue
        dx = xs[mask] - positions[b, 0]
        dy = ys[mask] - positions[b, 1]
        dist2 = dx * dx + dy * dy
        k = int(np.argmin(dist2))
        v = np.array([dx[k], dy[k]], dtype=np.float32)
        n = float(np.linalg.norm(v))
        if n < 1e-8:
            theta = rng.uniform(0.0, 2 * np.pi)
            out[b] = [np.sin(theta), np.cos(theta)]
        else:
            out[b] = v / n
    return out


__all__ = [
    "bfs_action_batch_continuous",
    "bfs_action_batch_discrete",
    "novelty_action_batch_continuous",
    "novelty_action_batch_discrete",
]
