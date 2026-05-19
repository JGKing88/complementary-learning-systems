"""Core navigation primitives: Gram-Schmidt projection + one-step trajectory advancement.

Shared by:
  - cls/eval/nav_eval.py  (aggregate batched evaluation)
  - encoder_training/trajectory.py  (single-trajectory viz)

Grid convention: positions are (gx, gy) with gx = dim0 (axis 0 of encoded_Phi),
gy = dim1. "East" = +gx, "North" = +gy.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def compute_projection_matrix(
    encoded_Phi: np.ndarray,
    gx: int,
    gy: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Gram-Schmidt projection matrix at (gx, gy).

    Returns:
        W: (2, embed_dim) — row 0 = East basis, row 1 = North basis.
        current: (embed_dim,) — encoded state at (gx, gy).
    """
    Nx, Ny = encoded_Phi.shape[:2]
    gx = int(np.clip(gx, 1, Nx - 2))
    gy = int(np.clip(gy, 1, Ny - 2))

    current = encoded_Phi[gx, gy]
    # North = +gy (+dim1), East = +gx (+dim0)
    d_forward = encoded_Phi[gx, gy + 1] - current
    d_right = encoded_Phi[gx + 1, gy] - current

    # Gram-Schmidt: e1 = normalize(d_forward), e2 = orthogonalize(d_right, e1)
    e1 = d_forward / max(np.linalg.norm(d_forward), 1e-12)
    e2 = d_right - np.dot(d_right, e1) * e1
    e2 = e2 / max(np.linalg.norm(e2), 1e-12)

    W = np.stack([e2, e1], axis=0)  # row 0 = East, row 1 = North
    return W, current


def continuous_step(
    W: np.ndarray,
    current_np: np.ndarray,
    next_state_np: np.ndarray,
    scale: float = 1.0,
    normalize: bool = True,
) -> Tuple[Tuple[float, float], float]:
    """Project (next - current) onto W to get a grid-space step.

    Returns:
        (dgx, dgy): step in grid coordinates (dim0 = East, dim1 = North).
        magnitude: raw magnitude of projected vector.
    """
    d = next_state_np - current_np
    q = W @ d                          # q[0] = East, q[1] = North
    magnitude = float(np.linalg.norm(q))
    if magnitude < 1e-8:
        return (0.0, 0.0), 0.0

    q_scaled = (q / magnitude) * scale if normalize else q * scale
    return (float(q_scaled[0]), float(q_scaled[1])), magnitude


# ---------------------------------------------------------------------------
# Single-trajectory simulation
# ---------------------------------------------------------------------------

def simulate_trajectory(
    encoded_Phi: np.ndarray,
    hopfield,
    start_gx: int, start_gy: int,
    goal_loc: Tuple[int, int],
    gain: float,
    max_steps: int = 100,
    scale: float = 1.0,
    normalize: bool = True,
    platform_radius: float = 1.0,
    recompute_interval: int = 1,
    alpha: float = 0.8,
) -> np.ndarray:
    """Roll out one continuous trajectory. Returns (T, 2) positions [gx, gy]."""
    Nx, Ny = encoded_Phi.shape[:2]
    min_gx, max_gx = 1, Nx - 2
    min_gy, max_gy = 1, Ny - 2

    position = np.array([float(start_gx), float(start_gy)])
    trajectory = [position.copy()]

    grid_pos = np.clip(np.round(position).astype(int),
                       [min_gx, min_gy], [max_gx, max_gy])
    W, _ = compute_projection_matrix(encoded_Phi, grid_pos[0], grid_pos[1])
    steps_since = 0

    for _ in range(max_steps):
        grid_pos = np.clip(np.round(position).astype(int),
                           [min_gx, min_gy], [max_gx, max_gy])
        state = torch.from_numpy(encoded_Phi[grid_pos[0], grid_pos[1]].copy()).float()
        current_np = state.numpy()

        next_state, _ = hopfield.recall(
            state, steps=1, beta=gain, alpha=alpha,
            use_tanh=True, normalize_each=True,
        )

        if recompute_interval > 0 and steps_since >= recompute_interval:
            W, _ = compute_projection_matrix(encoded_Phi, grid_pos[0], grid_pos[1])
            steps_since = 0

        (dgx, dgy), mag = continuous_step(
            W, current_np, next_state.numpy(), scale=scale, normalize=normalize,
        )
        steps_since += 1
        if mag < 1e-6:
            break

        new_position = position + np.array([dgx, dgy])
        new_position[0] = float(np.clip(new_position[0], min_gx, max_gx))
        new_position[1] = float(np.clip(new_position[1], min_gy, max_gy))
        if np.linalg.norm(new_position - position) < 1e-6:
            break
        position = new_position
        trajectory.append(position.copy())
        if np.linalg.norm(position - np.array(goal_loc, dtype=float)) < platform_radius:
            break

    return np.array(trajectory)
