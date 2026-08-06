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
from typing import List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
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

    # Start / end: small dots (env-sized panels are tight)
    ax.scatter(xs[0], ys[0], c="limegreen", s=12, marker="o",
               edgecolors="darkgreen", linewidths=0.45, zorder=4, label="start")
    ax.scatter(xs[-1], ys[-1], c="orange", s=12, marker="o",
               edgecolors="darkorange", linewidths=0.45, zorder=4, label="end")
    ax.add_patch(plt.Circle((env.goal_gy, env.goal_gx), 1.0,
                            fill=False, edgecolor="red", linewidth=1.25,
                            zorder=5, label="goal"))
    ax.scatter(env.goal_gy, env.goal_gx, c="red", s=28, marker="o",
               edgecolors="darkred", linewidths=0.5, zorder=6)

    ax.set_aspect("equal")
    # Minimal margin: goal ring has radius 1 in grid units, so pad≈1 avoids
    # clipping when the goal sits on the env edge.
    pad = 1.0
    ax.set_xlim(env.x0 - pad, env.x0 + env.size + pad - 1)
    ax.set_ylim(env.y0 + env.size + pad - 1, env.y0 - pad)  # y flipped (image coord)
    ax.margins(0)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    return ax


def plot_all_envs_global(
    full_npos: int,
    env_trajs: Sequence[Tuple[EvalEnv, np.ndarray]],
    *,
    method_label: str = "Continuous Method",
    figsize: Tuple[float, float] = (10.0, 9.0),
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Single full-grid figure: env regions, goals, trajectories colored by time.

    Matches the ``All Environments`` overview style: light-gray grid (sparse
    major ticks via ``MaxNLocator``), semi-transparent env boxes with dashed edges, per-step
    scatter along each path (``plasma``), one **Time Step** colorbar, legend
    (start / end / goal), and a small compass in the lower-left (axes coords).

    Trajectory arrays use the same layout as ``simulate_trajectory`` output:
    ``[:, 0]`` = row (gx), ``[:, 1]`` = column (gy). Axes follow the notebook /
    heatmap convention: **y increases downward** (row 0 at the top), x is
    column index left to right.
    """
    own_fig = ax is None
    if own_fig:
        _, ax = plt.subplots(figsize=figsize, layout="constrained")

    n_env = len(env_trajs)
    vmax_t = 1
    for _, tr in env_trajs:
        vmax_t = max(vmax_t, len(tr) - 1)

    norm = Normalize(vmin=0, vmax=float(vmax_t))

    ax.set_xlim(-0.5, full_npos - 0.5)
    ax.set_ylim(full_npos - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title(
        f"All Environments ({method_label})\n"
        f"{n_env} environment{'s' if n_env != 1 else ''} in "
        f"{full_npos}×{full_npos} space",
        fontsize=12,
        fontweight="bold",
    )
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))
    ax.grid(True, which="major", color="0.85", linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)

    # Distinct env + trajectory colors (tab10-like, readable on white).
    base = plt.cm.tab10(np.linspace(0, 0.9, max(n_env, 3)))

    for k, (env, traj) in enumerate(env_trajs):
        ec = base[k % len(base)]
        fc = (*ec[:3], 0.22)
        ax.add_patch(
            plt.Rectangle(
                (env.x0 - 0.5, env.y0 - 0.5),
                env.size,
                env.size,
                facecolor=fc,
                edgecolor=ec,
                linewidth=1.4,
                linestyle="--",
                zorder=1,
            )
        )
        ax.text(
            env.x0 + env.size * 0.5,
            env.y0 - max(2.0, env.size * 0.08),
            f"Env {k}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color=ec,
            zorder=5,
        )

        xs = traj[:, 1]
        ys = traj[:, 0]
        T = len(traj)
        ax.scatter(
            xs,
            ys,
            c=np.arange(T),
            cmap="plasma",
            norm=norm,
            s=14,
            linewidths=0.2,
            edgecolors="face",
            zorder=4,
        )
        ax.scatter(
            xs[0],
            ys[0],
            c="limegreen",
            s=28,
            marker="o",
            edgecolors="darkgreen",
            linewidths=0.55,
            zorder=6,
        )
        ax.scatter(
            xs[-1],
            ys[-1],
            c="orange",
            s=26,
            marker="s",
            edgecolors="darkorange",
            linewidths=0.55,
            zorder=6,
        )
        ax.add_patch(
            plt.Circle(
                (env.goal_gy, env.goal_gx),
                1.0,
                fill=False,
                edgecolor="gold",
                linewidth=1.8,
                zorder=3,
            )
        )

    sm = ScalarMappable(norm=norm, cmap="plasma")
    sm.set_array([])
    cbar = ax.figure.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Time Step")

    leg = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="limegreen",
            markeredgecolor="darkgreen",
            markersize=6,
            linestyle="None",
            label="Start",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="orange",
            markeredgecolor="darkorange",
            markersize=6,
            linestyle="None",
            label="End",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="none",
            markeredgecolor="gold",
            markeredgewidth=1.4,
            markersize=6,
            linestyle="None",
            label="Goal (r=1)",
        ),
    ]
    ax.legend(handles=leg, loc="upper right", fontsize=9, framealpha=0.95)

    # Decorative compass (axes fraction, lower-left; matches reference layout).
    cx, cy = 0.045, 0.08
    ax.text(cx, cy + 0.035, "N", transform=ax.transAxes, fontsize=11,
            color="tab:green", fontweight="bold", ha="center", va="center",
            zorder=20, clip_on=False)
    ax.text(cx, cy - 0.035, "S", transform=ax.transAxes, fontsize=11,
            color="tab:red", fontweight="bold", ha="center", va="center",
            zorder=20, clip_on=False)
    ax.text(cx - 0.032, cy, "W", transform=ax.transAxes, fontsize=11,
            color="tab:purple", fontweight="bold", ha="center", va="center",
            zorder=20, clip_on=False)
    ax.text(cx + 0.032, cy, "E", transform=ax.transAxes, fontsize=11,
            color="tab:blue", fontweight="bold", ha="center", va="center",
            zorder=20, clip_on=False)

    return ax
