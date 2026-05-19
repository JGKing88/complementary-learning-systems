"""Progression schematic: storing memories carves basins.

Four 3D panels, horizontal strip: 0, 1, 2, 3 memories stored. Each panel
shares the same three env outlines; basins appear as goals are written.
A dashed vertical 'memory write' indicator shows the new memory being
deposited into its environment. Output is vector PDF + SVG.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["pdf.fonttype"] = 42
rcParams["svg.fonttype"] = "none"
rcParams["font.family"] = "sans-serif"

OUT_DIR = "/orcd/home/002/jackking/cls/hopfield_nav/figures"

env_centers = np.array([[-8.0, 5.5], [7.5, 3.8], [0.5, -8.0]])
env_radii = np.array([3.6, 3.3, 3.7])
goal_offsets = np.array([[0.4, -0.5], [-0.3, 0.4], [0.2, 0.3]])
goals = env_centers + goal_offsets

GOAL_DEPTH = 0.42
GOAL_SIGMA = 1.15
ENV_DEPTH = 0.035
BOWL = 0.0008
MEM_COLOR = "#c0392b"

def energy(x, y, stored_count):
    z = BOWL * (x**2 + y**2)
    for (cx, cy), r in zip(env_centers, env_radii):
        z += -ENV_DEPTH * np.exp(
            -((x - cx) ** 2 + (y - cy) ** 2) / (2 * (r * 0.95) ** 2)
        )
    for k in range(stored_count):
        gx, gy = goals[k]
        z += -GOAL_DEPTH * np.exp(
            -((x - gx) ** 2 + (y - gy) ** 2) / (2 * GOAL_SIGMA ** 2)
        )
    return z

N = 280
xs = np.linspace(-14.0, 14.0, N)
ys = np.linspace(-14.0, 14.0, N)
X, Y = np.meshgrid(xs, ys)

# shared z-limits so panels are visually comparable
Z_full = energy(X, Y, 3)
zmin, zmax = float(Z_full.min()) - 0.02, 0.05

fig = plt.figure(figsize=(16.5, 5.2))

theta = np.linspace(0, 2 * np.pi, 220)

def agent_path_in_env(k, n=60, seed=0):
    """Meandering path inside env k that ends at goal k."""
    rng = np.random.default_rng(seed + k)
    cx, cy = env_centers[k]
    r = env_radii[k]
    # start somewhere inside, away from goal
    start_angle = rng.uniform(0, 2 * np.pi)
    start = np.array([cx + r * 0.75 * np.cos(start_angle),
                      cy + r * 0.75 * np.sin(start_angle)])
    end = goals[k]
    t = np.linspace(0, 1, n)
    # base: straight line start -> end
    pts = (1 - t)[:, None] * start + t[:, None] * end
    # add damped sinusoidal wiggle perpendicular to the line so it looks like exploration
    direction = end - start
    perp = np.array([-direction[1], direction[0]])
    perp = perp / (np.linalg.norm(perp) + 1e-9)
    wiggle = np.sin(t * np.pi * 3.0) * (1 - t) * r * 0.35
    pts = pts + perp[None, :] * wiggle[:, None]
    return pts

for panel in range(4):
    ax = fig.add_subplot(1, 4, panel + 1, projection="3d", computed_zorder=False)
    Z = energy(X, Y, panel)
    ax.plot_surface(
        X, Y, Z, cmap="Blues_r",
        linewidth=0, antialiased=True,
        rstride=2, cstride=2, alpha=0.92, edgecolor="none", zorder=1,
    )

    # env outlines
    for (cx, cy), r in zip(env_centers, env_radii):
        bx = cx + r * np.cos(theta)
        by = cy + r * np.sin(theta)
        bz = energy(bx, by, panel) + 0.006
        ax.plot(bx, by, bz, color="#2a2a2a", lw=1.6, zorder=5)

    # env labels
    for i, ((cx, cy), r) in enumerate(zip(env_centers, env_radii)):
        lx, ly = cx, cy + r + 0.8
        lz = float(energy(np.array([lx]), np.array([ly]), panel)[0]) + 0.08
        ax.text(lx, ly, lz, f"Env {i+1}", fontsize=9, ha="center", zorder=25)

    # "memory write" indicator on panels where a new memory was just stored
    if panel >= 1:
        k = panel - 1  # memory index just stored
        gx, gy = goals[k]
        gz_bottom = float(energy(np.array([gx]), np.array([gy]), panel)[0])
        gz_top = 0.22
        # dashed vertical line from up high down into the basin
        ax.plot([gx, gx], [gy, gy], [gz_top, gz_bottom + 0.01],
                color=MEM_COLOR, lw=1.5, linestyle=(0, (3, 2)), zorder=18)
        # memory marker at top
        ax.scatter([gx], [gy], [gz_top], color=MEM_COLOR, s=45,
                   edgecolor="white", linewidth=0.9, depthshade=False, zorder=19)
        ax.text(gx + 0.4, gy + 0.4, gz_top + 0.02,
                f"m$_{k+1}$", fontsize=10, color=MEM_COLOR, zorder=25)

    # panel title
    titles = ["0 memories stored", "store m$_1$", "store m$_2$", "store m$_3$"]
    ax.set_title(titles[panel], fontsize=11, pad=-4)

    # view / styling
    ax.view_init(elev=42, azim=-60)
    ax.grid(False)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlim(-14, 14); ax.set_ylim(-14, 14); ax.set_zlim(zmin, zmax)
    ax.set_box_aspect((1, 1, 0.25))

plt.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=-0.05)

# tiny top-down agent insets above panels 2..4 (indices 1..3)
AGENT_COLOR = "#d55e00"
for panel in range(1, 4):
    k = panel - 1  # which env/memory
    # figure-coord position: above the corresponding 3D panel
    panel_width = (0.99 - 0.01) / 4
    cx_fig = 0.01 + panel_width * (panel + 0.5)
    iw, ih = 0.075, 0.15
    ax_in = fig.add_axes([cx_fig - iw / 2, 0.82, iw, ih])

    cx, cy = env_centers[k]
    r = env_radii[k]
    # env boundary
    ax_in.add_patch(plt.Circle((cx, cy), r, fill=False, edgecolor="#2a2a2a", lw=1.2))
    # agent path
    path = agent_path_in_env(k)
    ax_in.plot(path[:, 0], path[:, 1], color=AGENT_COLOR, lw=1.2,
               solid_capstyle="round", zorder=3)
    # start dot
    ax_in.scatter(path[0, 0], path[0, 1], color=AGENT_COLOR, s=10,
                  edgecolor="white", linewidth=0.6, zorder=4)
    # agent at goal (triangle)
    gx, gy = goals[k]
    ax_in.scatter([gx], [gy], color=AGENT_COLOR, marker="^", s=40,
                  edgecolor="white", linewidth=0.7, zorder=5)
    # crimson "write" ring around the goal to suggest storage event
    ax_in.add_patch(plt.Circle((gx, gy), 0.9, fill=False,
                               edgecolor=MEM_COLOR, lw=1.4, linestyle=(0, (2, 1.5)),
                               zorder=6))
    pad = r + 0.9
    ax_in.set_xlim(cx - pad, cx + pad)
    ax_in.set_ylim(cy - pad, cy + pad)
    ax_in.set_aspect("equal")
    ax_in.set_xticks([]); ax_in.set_yticks([])
    for spine in ax_in.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#888888")

plt.savefig(f"{OUT_DIR}/memory_storage_schematic.pdf", bbox_inches="tight")
plt.savefig(f"{OUT_DIR}/memory_storage_schematic.svg", bbox_inches="tight")
print(f"wrote {OUT_DIR}/memory_storage_schematic.pdf and .svg")
