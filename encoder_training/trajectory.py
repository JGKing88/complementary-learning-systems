"""Trajectory simulation + plotting helpers.

Minimal clean version of what agent_testing.ipynb does:
- place evaluation environments (each with a goal)
- store goal embeddings in a Hopfield network
- simulate agent trajectories using Gram-Schmidt projection of the
  recalled-state difference onto local grid axes
- plot the resulting trajectory

This is for exploratory visualization. The training loop uses the more
complete `cls.eval.nav_eval.run_navigation_eval` for aggregate metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch

from cls.hopfield import Hopfield
from cls.nav import compute_projection_matrix, continuous_step
from cls.nav import simulate_trajectory as _cls_simulate_trajectory


# ---------------------------------------------------------------------------
# Environment + Hopfield setup
# ---------------------------------------------------------------------------

@dataclass
class EvalEnv:
    """A single evaluation environment."""
    y0: int
    x0: int
    size: int
    goal_gx: int   # global y coord of goal
    goal_gy: int   # global x coord of goal

    @property
    def bounds(self) -> Tuple[int, int, int, int]:
        return (self.y0, self.y0 + self.size, self.x0, self.x0 + self.size)


def place_random_envs(
    full_Npos: int,
    n_envs: int,
    env_size: int,
    avoid: List[Tuple[int, int, int]] | None = None,
    rng: np.random.RandomState | None = None,
    touch_ok: bool = False,
) -> List[Tuple[int, int]]:
    """Place `n_envs` non-overlapping square environments on a grid.

    `avoid` is a list of (y0, x0, size) patches to also avoid (e.g. training
    patches). Returns list of (y0, x0) top-left corners.

    If rng is None, uses fresh numpy random state → different every call.
    """
    if rng is None:
        rng = np.random.RandomState()
    avoid = avoid or []

    def overlaps(y0a, x0a, sa, y0b, x0b, sb):
        if touch_ok:
            return not (y0a + sa < y0b or y0b + sb < y0a or
                        x0a + sa < x0b or x0b + sb < x0a)
        return not (y0a + sa <= y0b or y0b + sb <= y0a or
                    x0a + sa <= x0b or x0b + sb <= x0a)

    placed: list[tuple[int, int]] = []
    tries = 0
    while len(placed) < n_envs and tries < 10_000:
        y0 = int(rng.randint(1, full_Npos - env_size - 1))
        x0 = int(rng.randint(1, full_Npos - env_size - 1))
        if any(overlaps(y0, x0, env_size, py, px, env_size) for py, px in placed):
            tries += 1; continue
        if any(overlaps(y0, x0, env_size, py, px, ps)
               for py, px, ps in avoid):
            tries += 1; continue
        placed.append((y0, x0))
        tries += 1
    if len(placed) < n_envs:
        raise RuntimeError(f"Could only place {len(placed)}/{n_envs} envs")
    return placed


def build_eval_envs(
    placements: List[Tuple[int, int]],
    env_size: int,
    rng: np.random.RandomState | None = None,
) -> List[EvalEnv]:
    """Pick a random goal location inside each placement.

    If rng is None, uses fresh numpy random state → different every call.
    """
    if rng is None:
        rng = np.random.RandomState()
    out: list[EvalEnv] = []
    for y0, x0 in placements:
        gy = int(rng.randint(y0, y0 + env_size))
        gx = int(rng.randint(x0, x0 + env_size))
        out.append(EvalEnv(y0=y0, x0=x0, size=env_size, goal_gx=gy, goal_gy=gx))
    return out


def build_hopfield(encoded_grid: np.ndarray, envs: List[EvalEnv],
                   gain: float) -> Hopfield:
    """Seed a Hopfield network with the encoded goal state of each env."""
    embed_dim = encoded_grid.shape[-1]
    hop = Hopfield(num_units=embed_dim, beta=gain, device="cpu")
    for env in envs:
        g = torch.from_numpy(encoded_grid[env.goal_gx, env.goal_gy]).float()
        hop.input_memory(g)
    return hop


# ---------------------------------------------------------------------------
# Gram-Schmidt projection for direction extraction
# ---------------------------------------------------------------------------

def projection_matrix_at(encoded_grid: np.ndarray, gx: int, gy: int,
                         full_Npos: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Thin wrapper around `cls.nav.compute_projection_matrix`."""
    return compute_projection_matrix(encoded_grid, gx, gy)


def projected_step(
    W: np.ndarray,
    z_cur: np.ndarray,
    z_next: np.ndarray,
    scale: float = 1.0,
    normalize: bool = True,
    floor: float = 0.0,
) -> Tuple[Tuple[float, float], float, float]:
    """Project displacement to a grid step and report angle + magnitude.

    Delegates to `cls.nav.continuous_step` for (dgx, dgy, magnitude),
    adds the angle-from-forward (from raw q) for plotting.
    """
    d = z_next - z_cur
    q = W @ d  # q[0] = East (dgx), q[1] = North (dgy)
    mag = float(np.linalg.norm(q))
    if mag < 1e-8:
        return (0.0, 0.0), 0.0, 0.0
    (dgx, dgy), _ = continuous_step(W, z_cur, z_next, scale=scale, normalize=normalize)
    if floor > 0:
        step_mag = float(np.hypot(dgx, dgy))
        if 0 < step_mag < floor:
            scale_up = floor / step_mag
            dgx *= scale_up; dgy *= scale_up
    angle = float(np.arctan2(q[0], q[1]))  # 0 = forward (North), positive = East-of-forward
    return (dgx, dgy), angle, mag


# ---------------------------------------------------------------------------
# Simulate a trajectory
# ---------------------------------------------------------------------------

def simulate_trajectory(
    encoded_grid: np.ndarray,
    hop: Hopfield,
    env: EvalEnv,
    start_gx: int, start_gy: int,
    gain: float,
    max_steps: int = 100,
    scale: float = 1.0,
    normalize: bool = True,
    platform_radius: float = 1.0,
    alpha: float = 0.8,
    recompute_interval: int = 1,
) -> Tuple[np.ndarray, List[float], List[float]]:
    """Simulate a continuous trajectory from (start_gx, start_gy).

    Delegates the core simulation to `cls.nav.simulate_trajectory`. Also
    collects per-step angles and magnitudes for downstream plotting by
    re-projecting along the saved trajectory (cheap: just two matmuls/step).

    Returns (trajectory[T, 2], angles[T-1], magnitudes[T-1]).
    """
    traj = _cls_simulate_trajectory(
        encoded_grid, hop, start_gx, start_gy,
        goal_loc=(env.goal_gx, env.goal_gy),
        gain=gain, max_steps=max_steps,
        scale=scale, normalize=normalize,
        platform_radius=platform_radius,
        recompute_interval=recompute_interval, alpha=alpha,
    )

    # Reconstruct per-step angles and magnitudes for plotting.
    angles: list[float] = []
    mags: list[float] = []
    H, W, _ = encoded_grid.shape
    proj_W, _ = compute_projection_matrix(encoded_grid,
                                          int(round(traj[0, 0])),
                                          int(round(traj[0, 1])))
    steps_since_W = 0
    for i in range(len(traj) - 1):
        gx = int(np.clip(round(traj[i, 0]), 1, H - 2))
        gy = int(np.clip(round(traj[i, 1]), 1, W - 2))
        z_cur = torch.from_numpy(encoded_grid[gx, gy]).float()
        z_next, _ = hop.recall(z_cur, steps=1, beta=gain, alpha=alpha,
                               use_tanh=True, normalize_each=True)
        if recompute_interval > 0 and steps_since_W >= recompute_interval:
            proj_W, _ = compute_projection_matrix(encoded_grid, gx, gy)
            steps_since_W = 0
        q = proj_W @ (z_next.numpy() - z_cur.numpy())
        mags.append(float(np.linalg.norm(q)))
        angles.append(float(np.arctan2(q[0], q[1])))
        steps_since_W += 1
    return traj, angles, mags


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_trajectory(
    env: EvalEnv,
    traj: np.ndarray,
    angles: List[float] | None = None,
    ax=None,
    show_arrows: bool = True,
    title: str = "",
):
    """Plot a single trajectory within its environment."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    # Env bounds
    ax.add_patch(plt.Rectangle(
        (env.x0 - 0.5, env.y0 - 0.5), env.size, env.size,
        fill=False, edgecolor="steelblue", linewidth=2,
        linestyle="--", label="env"))

    # All env cells (faint)
    gys, gxs = np.meshgrid(np.arange(env.size), np.arange(env.size), indexing="ij")
    ax.scatter(env.x0 + gxs.ravel(), env.y0 + gys.ravel(),
               c="lightgray", s=15, alpha=0.5, zorder=1)

    # Trajectory: gradient over time, plotted as (x, y) = (col, row)
    T = len(traj)
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, max(T, 2)))
    xs = traj[:, 1]; ys = traj[:, 0]
    for i in range(T - 1):
        ax.plot(xs[i:i+2], ys[i:i+2], color=colors[i],
                linewidth=2.5, alpha=0.85, zorder=2)
    ax.scatter(xs, ys, c=np.arange(T), cmap="plasma", s=50,
               edgecolors="white", linewidths=0.5, zorder=3)

    # Start / end markers
    ax.scatter(xs[0], ys[0], c="limegreen", s=180, marker="o",
               edgecolors="darkgreen", linewidths=2, zorder=4, label="start")
    ax.scatter(xs[-1], ys[-1], c="orange", s=180, marker="s",
               edgecolors="darkorange", linewidths=2, zorder=4, label="end")
    ax.add_patch(plt.Circle((env.goal_gy, env.goal_gx), 1.0,
                            fill=False, edgecolor="red", linewidth=2,
                            zorder=5, label="goal"))
    ax.scatter(env.goal_gy, env.goal_gx, c="red", s=140, marker="*",
               edgecolors="darkred", linewidths=1, zorder=6)

    ax.set_aspect("equal")
    pad = 2
    ax.set_xlim(env.x0 - pad, env.x0 + env.size + pad)
    ax.set_ylim(env.y0 + env.size + pad, env.y0 - pad)  # y flipped (image coord)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    return ax
