"""Schematic of Hopfield energy landscape over VH space.

3D energy surface with three environment basins and a trajectory descending
into one goal. Top-down panel on the right.  Style matches the encoder
schematic (warm off-white, ink/muted/burgundy palette).
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch

import os

from cls_paths import figures_dir

# ---------- style ----------
rcParams["pdf.fonttype"] = 42
rcParams["svg.fonttype"] = "none"
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Helvetica Neue", "Helvetica", "Arial",
                                "DejaVu Sans"]
rcParams["mathtext.fontset"] = "dejavusans"
rcParams["font.size"] = 13
rcParams["axes.labelcolor"] = "#333333"
rcParams["text.color"] = "#222222"

INK = "#2a2a2a"
MUTED = "#6b6f76"
FAINT = "#b8bcc4"
FIG_BG = "white"
PANEL_BG = "#fbfaf6"           # subtle warm tint
TRAJ_COLOR = "#c25a3a"         # muted terracotta
START_COLOR = "#c25a3a"
GOAL_COLOR = "#8e3b3b"         # deeper burgundy for the destination
ENV_OUTLINE = "#2a2a2a"
TITLE_COLOR = "#333333"
TITLE_SIZE = 15
LABEL_SIZE = 12

# Schematics render to the shared figures root on pool, not into the
# source tree. Override the root with the CLS_RUNS env var.
OUT_DIR = str(figures_dir(ensure=True) / "schematics")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- landscape ----------
env_centers = np.array([[-7.5, 5.5], [7.5, 4.0], [0.5, -7.8]])
env_radii = np.array([3.6, 3.3, 3.7])
goal_offsets = np.array([[0.4, -0.5], [-0.3, 0.4], [0.2, 0.3]])
goals = env_centers + goal_offsets

GOAL_DEPTH = 0.42
GOAL_SIGMA = 1.15
ENV_DEPTH = 0.035
BOWL = 0.0008


def energy(x, y):
    z = BOWL * (x ** 2 + y ** 2)
    for (cx, cy), r in zip(env_centers, env_radii):
        z += -ENV_DEPTH * np.exp(-((x - cx) ** 2 + (y - cy) ** 2)
                                  / (2 * (r * 0.95) ** 2))
    for gx, gy in goals:
        z += -GOAL_DEPTH * np.exp(-((x - gx) ** 2 + (y - gy) ** 2)
                                   / (2 * GOAL_SIGMA ** 2))
    return z


def energy_grad(x, y):
    gx_ = 2 * BOWL * x
    gy_ = 2 * BOWL * y
    for (cx, cy), r in zip(env_centers, env_radii):
        s2 = (r * 0.95) ** 2
        w = -ENV_DEPTH * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s2))
        gx_ += w * (-(x - cx) / s2)
        gy_ += w * (-(y - cy) / s2)
    for gxc, gyc in goals:
        s2 = GOAL_SIGMA ** 2
        w = -GOAL_DEPTH * np.exp(-((x - gxc) ** 2 + (y - gyc) ** 2) / (2 * s2))
        gx_ += w * (-(x - gxc) / s2)
        gy_ += w * (-(y - gyc) / s2)
    return gx_, gy_


N = 380
xs = np.linspace(-13.0, 13.0, N)
ys = np.linspace(-13.0, 13.0, N)
X, Y = np.meshgrid(xs, ys)
Z = energy(X, Y)

# trajectory: gradient descent inside env 0
env_i = 0
rim_dir = np.array([0.55, 0.75])
rim_dir = rim_dir / np.linalg.norm(rim_dir)
pos = env_centers[env_i] + rim_dir * (env_radii[env_i] * 0.82)
path = [pos.copy()]
lr = 4.0
for _ in range(500):
    gx_, gy_ = energy_grad(pos[0], pos[1])
    step = -lr * np.array([gx_, gy_])
    nrm = np.linalg.norm(step)
    if nrm > 0.12:
        step = step / nrm * 0.12
    pos = pos + step
    path.append(pos.copy())
path = np.array(path)
tx, ty = path[:, 0], path[:, 1]
tz = energy(tx, ty) + 0.006

# desaturated cool→warm gradient: deep basins blue-gray, rim a warm cream
landscape_cmap = LinearSegmentedColormap.from_list("landscape", [
    (0.00, "#4f6f95"),    # deep blue (lowest energy = goal basins)
    (0.45, "#94a4bd"),    # mid blue-gray
    (0.80, "#d6d4ca"),    # warm gray
    (1.00, "#ece9df"),    # warm cream (high energy / rim)
])

# ---------- figure ----------
FIG_W, FIG_H = 11.0, 5.4
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=FIG_BG)


def panel_title(ax, text, y=0.97):
    ax.text2D(0.5, y, text, transform=ax.transAxes, ha="center",
              va="top", fontsize=TITLE_SIZE, color=TITLE_COLOR)


# ---------- main 3D panel ----------
ax = fig.add_axes([-0.02, -0.05, 0.74, 1.05], projection="3d",
                  computed_zorder=False)
ax.set_facecolor(FIG_BG)

surf = ax.plot_surface(
    X, Y, Z,
    cmap=landscape_cmap,
    linewidth=0, antialiased=True,
    rstride=2, cstride=2,
    alpha=0.95, edgecolor="none", zorder=1,
)

# environment rims
theta = np.linspace(0, 2 * np.pi, 220)
for (cx, cy), r in zip(env_centers, env_radii):
    bx = cx + r * np.cos(theta)
    by = cy + r * np.sin(theta)
    bz = energy(bx, by) + 0.008
    ax.plot(bx, by, bz, color=ENV_OUTLINE, lw=1.4, zorder=5,
            solid_capstyle="round")

# trajectory
ax.plot(tx, ty, tz, color=TRAJ_COLOR, lw=2.6, solid_capstyle="round",
        solid_joinstyle="round", zorder=15)
ax.scatter([tx[0]], [ty[0]], [tz[0] + 0.004], color=START_COLOR, s=44,
           edgecolor="white", linewidth=1.2, depthshade=False, zorder=16)
# trajectory end coincides with the goal at the basin minimum
ax.scatter([tx[-1]], [ty[-1]], [tz[-1] + 0.003], color=GOAL_COLOR, s=52,
           edgecolor="white", linewidth=1.2, depthshade=False, zorder=16)

# labels
for i, ((cx, cy), r) in enumerate(zip(env_centers, env_radii)):
    lx, ly = cx, cy + r + 0.55
    lz = float(energy(np.array([lx]), np.array([ly]))[0]) + 0.05
    ax.text(lx, ly, lz, f"env {i + 1}", fontsize=LABEL_SIZE,
            ha="center", color=INK, zorder=25)

ax.text(tx[0] + 0.5, ty[0] + 0.7, tz[0] + 0.06, "start",
        fontsize=LABEL_SIZE - 1, color=START_COLOR, fontweight="bold",
        zorder=25)
gx0, gy0 = goals[env_i]
gz0 = float(energy(np.array([gx0]), np.array([gy0]))[0])
ax.text(gx0 + 0.55, gy0 - 0.55, gz0 - 0.04, "goal",
        fontsize=LABEL_SIZE - 1, color=GOAL_COLOR, fontweight="bold",
        zorder=25)

# axes: hide everything (labels floated awkwardly in 3D); rely on title
ax.view_init(elev=40, azim=-60)
ax.grid(False)
for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
    pane.set_visible(False)
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.line.set_color("none")
ax.set_box_aspect((1, 1, 0.27))

panel_title(ax, "Hopfield energy over $h$", y=0.92)

# ---------- top-down panel ----------
ax2 = fig.add_axes([0.72, 0.18, 0.26, 0.66])
ax2.set_facecolor(FIG_BG)
ax2.contourf(X, Y, Z, levels=24, cmap=landscape_cmap)
# subtle level lines for crispness
ax2.contour(X, Y, Z, levels=10, colors=MUTED, linewidths=0.35,
            alpha=0.45)

for (cx, cy), r in zip(env_centers, env_radii):
    ax2.add_patch(plt.Circle((cx, cy), r, fill=False, edgecolor=ENV_OUTLINE,
                              lw=1.4))
ax2.plot(tx, ty, color=TRAJ_COLOR, lw=2.2, solid_capstyle="round", zorder=6)
ax2.scatter([tx[0]], [ty[0]], color=START_COLOR, s=42,
            edgecolor="white", linewidth=1.0, zorder=7)
ax2.scatter([tx[-1]], [ty[-1]], color=GOAL_COLOR, s=52,
            edgecolor="white", linewidth=1.0, zorder=7)

# arrowhead near end of trajectory to suggest direction
mid = max(len(tx) - 12, 0)
arr = FancyArrowPatch((tx[mid], ty[mid]), (tx[-3], ty[-3]),
                       arrowstyle="-|>", mutation_scale=12,
                       color=TRAJ_COLOR, linewidth=0, zorder=8)
ax2.add_patch(arr)

ax2.set_xticks([]); ax2.set_yticks([])
ax2.set_aspect("equal")
ax2.set_xlim(xs.min(), xs.max())
ax2.set_ylim(ys.min(), ys.max())
for spine in ax2.spines.values():
    spine.set_linewidth(0.8)
    spine.set_edgecolor(MUTED)
ax2.set_title("top-down", fontsize=TITLE_SIZE - 1, color=TITLE_COLOR, pad=8)

# small legend below the inset
leg_ax = fig.add_axes([0.72, 0.06, 0.26, 0.10])
leg_ax.set_xlim(0, 1); leg_ax.set_ylim(0, 1)
leg_ax.set_xticks([]); leg_ax.set_yticks([])
for s in leg_ax.spines.values():
    s.set_visible(False)

# legend entries
def legend_dot(x, color):
    leg_ax.scatter([x], [0.78], s=46, c=color, edgecolor="white",
                   linewidth=1.0, zorder=3)


legend_dot(0.04, START_COLOR)
leg_ax.text(0.10, 0.78, "start", va="center", ha="left",
            fontsize=10, color=INK)
legend_dot(0.36, GOAL_COLOR)
leg_ax.text(0.42, 0.78, "goal", va="center", ha="left",
            fontsize=10, color=INK)
leg_ax.plot([0.62, 0.74], [0.78, 0.78], color=TRAJ_COLOR, lw=2.2,
            solid_capstyle="round")
leg_ax.text(0.78, 0.78, "trajectory", va="center", ha="left",
            fontsize=10, color=INK)

plt.savefig(f"{OUT_DIR}/hopfield_schematic.pdf", bbox_inches="tight",
            facecolor=FIG_BG)
plt.savefig(f"{OUT_DIR}/hopfield_schematic.svg", bbox_inches="tight",
            facecolor=FIG_BG)
plt.savefig(f"{OUT_DIR}/hopfield_schematic_preview.png", bbox_inches="tight",
            facecolor=FIG_BG, dpi=140)
print(f"wrote {OUT_DIR}/hopfield_schematic.pdf and .svg")
