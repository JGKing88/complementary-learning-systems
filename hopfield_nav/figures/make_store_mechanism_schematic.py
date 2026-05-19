"""Companion schematic: how a single memory gets stored.

Three frames, single environment:
  1. agent explores within the env (no basin yet)
  2. agent reaches the goal; 'write' event flashes
  3. basin is now carved into VH space around the stored memory

Output is vector PDF + SVG.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Circle

rcParams["pdf.fonttype"] = 42
rcParams["svg.fonttype"] = "none"
rcParams["font.family"] = "sans-serif"

OUT_DIR = "/orcd/home/002/jackking/cls/hopfield_nav/figures"

env_center = np.array([0.0, 0.0])
env_radius = 3.6
goal = env_center + np.array([0.6, -0.4])

GOAL_DEPTH = 0.42
GOAL_SIGMA = 1.15
ENV_DEPTH = 0.035
BOWL = 0.0008
AGENT_COLOR = "#d55e00"
MEM_COLOR = "#c0392b"

def energy(x, y, stored):
    z = BOWL * (x**2 + y**2)
    z += -ENV_DEPTH * np.exp(
        -((x - env_center[0]) ** 2 + (y - env_center[1]) ** 2)
        / (2 * (env_radius * 0.95) ** 2)
    )
    if stored:
        z += -GOAL_DEPTH * np.exp(
            -((x - goal[0]) ** 2 + (y - goal[1]) ** 2) / (2 * GOAL_SIGMA ** 2)
        )
    return z

# agent exploration path ending at goal
def build_path(n=90):
    rng = np.random.default_rng(3)
    start = env_center + np.array([-2.4, 1.9])
    t = np.linspace(0, 1, n)
    pts = (1 - t)[:, None] * start + t[:, None] * goal
    direction = goal - start
    perp = np.array([-direction[1], direction[0]])
    perp = perp / (np.linalg.norm(perp) + 1e-9)
    # multi-frequency wiggle to look like exploration
    wig = (np.sin(t * np.pi * 2.6) * (1 - t) * 1.2
           + np.sin(t * np.pi * 5.1 + 0.5) * (1 - t) * 0.5)
    pts = pts + perp[None, :] * wig[:, None]
    # small noise
    pts = pts + rng.normal(0, 0.05, pts.shape)
    return pts

path = build_path()

fig = plt.figure(figsize=(13.5, 4.4))

# ---------- Frame 1: agent exploring ----------
ax1 = fig.add_subplot(1, 3, 1)
ax1.add_patch(Circle(env_center, env_radius, fill=True,
                     facecolor="#eaf2fb", edgecolor="#2a2a2a", lw=1.5, zorder=1))
# show goal as dashed target ring (not yet stored)
ax1.add_patch(Circle(goal, 0.55, fill=False, edgecolor="#7a7a7a",
                     lw=1.0, linestyle=(0, (2, 2)), zorder=2))
# partial path (agent still exploring, so draw up to ~70%)
p_idx = int(len(path) * 0.7)
ax1.plot(path[:p_idx, 0], path[:p_idx, 1], color=AGENT_COLOR, lw=1.8,
         solid_capstyle="round", zorder=3)
ax1.scatter(path[0, 0], path[0, 1], color=AGENT_COLOR, s=20,
            edgecolor="white", linewidth=0.8, zorder=4)
ax1.scatter(path[p_idx - 1, 0], path[p_idx - 1, 1], color=AGENT_COLOR,
            marker="^", s=90, edgecolor="white", linewidth=0.9, zorder=5)
ax1.text(env_center[0], env_center[1] + env_radius + 0.5,
         "agent explores", ha="center", fontsize=11)
ax1.text(goal[0] + 0.1, goal[1] - 0.9, "goal", fontsize=9, color="#555555")

# ---------- Frame 2: arrived + write event ----------
ax2 = fig.add_subplot(1, 3, 2)
ax2.add_patch(Circle(env_center, env_radius, fill=True,
                     facecolor="#eaf2fb", edgecolor="#2a2a2a", lw=1.5, zorder=1))
# full path
ax2.plot(path[:, 0], path[:, 1], color=AGENT_COLOR, lw=1.8,
         solid_capstyle="round", zorder=3)
ax2.scatter(path[0, 0], path[0, 1], color=AGENT_COLOR, s=20,
            edgecolor="white", linewidth=0.8, zorder=4)
# agent at goal
ax2.scatter(goal[0], goal[1], color=AGENT_COLOR, marker="^", s=110,
            edgecolor="white", linewidth=1.0, zorder=6)
# write flash: expanding dashed rings + filled glow
for rr, alpha in [(0.6, 0.45), (1.1, 0.25), (1.7, 0.12)]:
    ax2.add_patch(Circle(goal, rr, fill=True, facecolor=MEM_COLOR,
                         alpha=alpha, edgecolor="none", zorder=5))
ax2.add_patch(Circle(goal, 1.9, fill=False, edgecolor=MEM_COLOR,
                     lw=1.4, linestyle=(0, (3, 2)), zorder=7))
ax2.text(env_center[0], env_center[1] + env_radius + 0.5,
         "write memory", ha="center", fontsize=11, color=MEM_COLOR)

for ax in (ax1, ax2):
    pad = env_radius + 1.0
    ax.set_xlim(env_center[0] - pad, env_center[0] + pad)
    ax.set_ylim(env_center[1] - pad, env_center[1] + pad)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

# ---------- Frame 3: basin carved in VH space ----------
ax3 = fig.add_subplot(1, 3, 3, projection="3d", computed_zorder=False)
N = 220
xs = np.linspace(-env_radius - 1.2, env_radius + 1.2, N)
ys = np.linspace(-env_radius - 1.2, env_radius + 1.2, N)
X, Y = np.meshgrid(xs, ys)
Z = energy(X, Y, stored=True)
ax3.plot_surface(X, Y, Z, cmap="Blues_r", linewidth=0, antialiased=True,
                 rstride=2, cstride=2, alpha=0.92, edgecolor="none", zorder=1)
# env outline on surface
theta = np.linspace(0, 2 * np.pi, 220)
bx = env_center[0] + env_radius * np.cos(theta)
by = env_center[1] + env_radius * np.sin(theta)
bz = energy(bx, by, stored=True) + 0.006
ax3.plot(bx, by, bz, color="#2a2a2a", lw=1.6, zorder=5)
ax3.view_init(elev=42, azim=-60)
ax3.grid(False)
for pane in (ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane):
    pane.set_visible(False)
ax3.set_xticks([]); ax3.set_yticks([]); ax3.set_zticks([])
ax3.set_box_aspect((1, 1, 0.28))
ax3.set_title("basin in VH space", fontsize=11, pad=-2)

plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02, wspace=0.05)
plt.savefig(f"{OUT_DIR}/store_mechanism_schematic.pdf", bbox_inches="tight")
plt.savefig(f"{OUT_DIR}/store_mechanism_schematic.svg", bbox_inches="tight")
print(f"wrote {OUT_DIR}/store_mechanism_schematic.pdf and .svg")
