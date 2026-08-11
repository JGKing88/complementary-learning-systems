"""Visualize the foveal sensory input.

What this draws:
  1. The four wall codes (one row per wall, columns = segments along the wall).
  2. A grid of (top-down map, observation vector) pairs for several positions.
     The observation vector is taken from env.obs_at(pos) so the figure exercises
     the same ray-cast code that training and eval consume.

Self-check: the cone overlays each ray colored by its ±1 code as returned by
env.obs_at(pos), and the wall segments are colored the same way. A correct
implementation places each ray's endpoint on a wall segment whose color matches
the ray color.

Run:  python -m analysis.schematics.sensory_input  [--size 8 --obs 60 --seed 0 --out path.png]
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle

from cls_paths import figures_dir
from hopfield_nav.world.env import FOVEAL_HALF_ANGLE_DEG, GridEnv

NEG_COLOR = "#222222"
POS_COLOR = "#dddddd"
ROOM_BG = "#f7f5f0"
CONE_FILL = "#5577aa"
AGENT_COLOR = "#a35454"


def code_color(v: float) -> str:
    return NEG_COLOR if v < 0 else POS_COLOR


def ray_endpoint(cx: float, cy: float, sin_a: float, cos_a: float, size: int):
    """Where the ray from (cx, cy) hits the room boundary. Mirrors env geometry
    (walls at x=-0.5, x=size-0.5, y=-0.5, y=size-0.5) so the drawn endpoints
    match what _raycast_segment_code in env.py uses."""
    candidates = []
    if cos_a > 0:
        candidates.append((size - 0.5 - cy) / cos_a)
    if cos_a < 0:
        candidates.append((-0.5 - cy) / cos_a)
    if sin_a > 0:
        candidates.append((size - 0.5 - cx) / sin_a)
    if sin_a < 0:
        candidates.append((-0.5 - cx) / sin_a)
    t = min(t for t in candidates if t > 1e-9)
    return cx + t * sin_a, cy + t * cos_a


def draw_room(ax, env: GridEnv, pos=None, draw_rays: bool = True, psi: float = 0.0):
    """Draw the room, and the foveal cone from ``pos`` facing ``psi`` radians.

    ``psi`` is clockwise from North, matching the env: 0 is the North-facing
    cone this schematic drew before headings could turn.
    """
    size = env.size
    wall = env._wall_code  # (4, size): 0=N, 1=E, 2=S, 3=W

    pad = 1.1
    ax.set_xlim(-pad - 0.6, size - 0.5 + 0.6 + 0.1)
    ax.set_ylim(-pad - 0.6, size - 0.5 + 0.6 + 0.1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Floor
    ax.add_patch(Rectangle(
        (-0.5, -0.5), size, size, color=ROOM_BG, ec="#bbb", lw=0.6, zorder=0,
    ))

    strip = 0.45
    for k in range(size):
        ax.add_patch(Rectangle(
            (k - 0.5, size - 0.5), 1, strip,
            facecolor=code_color(wall[0, k]), edgecolor="#888", lw=0.4, zorder=2,
        ))  # N
        ax.add_patch(Rectangle(
            (size - 0.5, k - 0.5), strip, 1,
            facecolor=code_color(wall[1, k]), edgecolor="#888", lw=0.4, zorder=2,
        ))  # E
        ax.add_patch(Rectangle(
            (k - 0.5, -0.5 - strip), 1, strip,
            facecolor=code_color(wall[2, k]), edgecolor="#888", lw=0.4, zorder=2,
        ))  # S
        ax.add_patch(Rectangle(
            (-0.5 - strip, k - 0.5), strip, 1,
            facecolor=code_color(wall[3, k]), edgecolor="#888", lw=0.4, zorder=2,
        ))  # W

    for i in range(size + 1):
        ax.plot([i - 0.5, i - 0.5], [-0.5, size - 0.5], color="#dcdcdc", lw=0.3, zorder=1)
        ax.plot([-0.5, size - 0.5], [i - 0.5, i - 0.5], color="#dcdcdc", lw=0.3, zorder=1)

    if pos is None:
        return

    cx, cy = pos
    obs = env.obs_at(pos, psi)  # same ray-cast the env observes through
    n = obs.shape[0]
    half = np.deg2rad(FOVEAL_HALF_ANGLE_DEG)
    # Facing psi rotates the whole cone: theta is clockwise from forward, so
    # the world angle of each ray is just psi + theta.
    angles = psi + (-half + (np.arange(n) + 0.5) * (2 * half / n))
    sin_a = np.sin(angles)
    cos_a = np.cos(angles)

    # Cone wedge background
    L = 2.5 * size
    cone_pts = [(cx, cy)]
    for s, c in zip(sin_a, cos_a):
        cone_pts.append((cx + L * s, cy + L * c))
    ax.add_patch(Polygon(cone_pts, color=CONE_FILL, alpha=0.10, ec="none", zorder=3))

    if draw_rays:
        for s, c, v in zip(sin_a, cos_a, obs):
            ex, ey = ray_endpoint(cx, cy, s, c, env.size)
            ax.plot(
                [cx, ex], [cy, ey],
                color=code_color(float(v)), lw=0.5, alpha=0.85, zorder=4,
            )

    ax.plot([cx], [cy], marker="o", color=AGENT_COLOR, ms=7, zorder=5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=8)
    ap.add_argument("--obs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    out = args.out or os.path.join(
        str(figures_dir(ensure=True) / "schematics"),
        f"sensory_input_size{args.size}_obs{args.obs}_seed{args.seed}.png",
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)

    env = GridEnv(
        size=args.size,
        observation_size=args.obs,
        seed=args.seed,
        goals_active=False,
    )

    s = args.size
    rng = np.random.RandomState(args.seed)
    n_positions = 6
    flat = rng.choice(s * s, size=n_positions, replace=False)
    positions = [(int(i // s), int(i % s)) for i in flat]

    n_rows = len(positions)
    fig = plt.figure(figsize=(12, 2.0 + 2.4 * n_rows))
    gs = fig.add_gridspec(
        n_rows + 1, 2,
        height_ratios=[0.7] + [1.0] * n_rows,
        width_ratios=[1.0, 1.3],
        hspace=0.55, wspace=0.20,
    )

    ax_top = fig.add_subplot(gs[0, :])
    ax_top.imshow(env._wall_code, cmap="gray", vmin=-1, vmax=1, aspect="auto")
    ax_top.set_yticks([0, 1, 2, 3])
    ax_top.set_yticklabels(["N", "E", "S", "W"])
    ax_top.set_xticks(range(s))
    ax_top.set_xlabel("segment index along wall")
    ax_top.set_title(
        "Wall codes (black = -1, white = +1). Ray colors below should match the wall segment they hit.",
        fontsize=10,
    )

    for r, pos in enumerate(positions, start=1):
        ax_map = fig.add_subplot(gs[r, 0])
        draw_room(ax_map, env, pos=pos)
        ax_map.set_title(f"position = {pos}", fontsize=10)

        ax_obs = fig.add_subplot(gs[r, 1])
        obs = env.obs_at(pos)
        ax_obs.imshow(
            np.broadcast_to(obs, (8, len(obs))),
            cmap="gray", vmin=-1, vmax=1, aspect="auto",
        )
        ax_obs.set_yticks([])
        ax_obs.set_xticks([0, len(obs) // 2, len(obs) - 1])
        ax_obs.set_xticklabels(["-60deg (W of N)", "0deg (N)", "+60deg (E of N)"])
        ax_obs.set_title(
            f"env.obs_at({pos}) - {len(obs)} rays",
            fontsize=10,
        )

    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
