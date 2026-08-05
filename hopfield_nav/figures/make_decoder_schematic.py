"""Schematic: decoder takes two encoded grid states (x_t, x_hat) and
produces a continuous movement direction by

  1. building a local plane in embedding space from the N-neighbor and
     E-neighbor encoder displacements (d_N, d_E),
  2. projecting the recall displacement (x_hat - x_t) onto that plane,
  3. interpreting the projection's components as an (E, N) world direction.

Panels:
    left   — world grid: agent at x_t, recalled cell x_hat
    middle — embedding: plane spanned by d_N, d_E; projection of x_hat - x_t
    right  — world grid: resulting movement-direction arrow at the agent
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
from matplotlib.patches import FancyArrowPatch, Rectangle
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from mpl_toolkits.mplot3d.proj3d import proj_transform

import os

from cls_paths import figures_dir

# ---------- style (matches encoder_schematic.py) ----------
rcParams["pdf.fonttype"] = 42
rcParams["svg.fonttype"] = "none"
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Helvetica Neue", "Helvetica", "Arial",
                                "DejaVu Sans"]
rcParams["mathtext.fontset"] = "dejavusans"
rcParams["font.size"] = 14
rcParams["axes.labelcolor"] = "#333333"
rcParams["xtick.color"] = "#333333"
rcParams["ytick.color"] = "#333333"
rcParams["text.color"] = "#222222"

INK = "#2a2a2a"
MUTED = "#6b6f76"
FIG_BG = "white"
SPHERE_FACE = "#eef1f6"
SPHERE_WIRE = "#c6cfdc"
NEIGHBOR_COLOR = "#5577aa"
BASIS_COLOR = "#7a5fa8"        # purple — d_N, d_E
PLANE_FACE = "#cbb8e0"         # lavender — plane fill
PLANE_EDGE = "#9377b8"
DISPLACEMENT = "#c47e4f"       # terracotta — D = x_hat - x_t
PROJECTION = "#a3434a"         # red — D_proj (in-plane part)
PERP_COLOR = "#888888"         # gray — perpendicular drop (dashed)
GOAL_COLOR = "#a35454"
AGENT_COLOR = "#2a2a2a"
GRID_LINE = "#c6cfdc"

LABEL_SIZE = 14
TITLE_SIZE = 15
TITLE_COLOR = "#333333"

# Schematics render to the shared figures root on pool, not into the
# source tree. Override the root with the CLS_RUNS env var.
OUT_DIR = str(figures_dir(ensure=True) / "schematics")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------- 3D arrow patch with constant-size arrowhead ----------
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return float(min(zs))


def add_arrow3d(ax, p, v, color, lw=2.0, scale=18, ls="-",
                zorder=15, alpha=1.0, label=None, label_offset=None):
    end = p + v
    a = Arrow3D([p[0], end[0]], [p[1], end[1]], [p[2], end[2]],
                arrowstyle="-|>", mutation_scale=scale,
                color=color, lw=lw, ls=ls, alpha=alpha,
                capstyle="round", joinstyle="round", zorder=zorder)
    ax.add_artist(a)
    return end


# ---------- state space lattice + encoder ----------
n = 17
xs = np.linspace(-1.1, 1.1, n)
ys = np.linspace(-1.1, 1.1, n)
X2d, Y2d = np.meshgrid(xs, ys)
pts2d = np.stack([X2d.ravel(), Y2d.ravel()], axis=1)


def pos_to_color(pts):
    p = (pts - pts.min(0)) / (pts.max(0) - pts.min(0) + 1e-9)
    r = 0.35 + 0.50 * p[:, 0]
    g = 0.55 + 0.30 * p[:, 1]
    b = 0.80 - 0.35 * p[:, 0]
    return np.clip(np.stack([r, g, b], axis=1), 0, 1)


colors = pos_to_color(pts2d)


def encode(pts):
    """Inverse stereographic — same encoder as encoder_schematic.py."""
    x, y = pts[:, 0], pts[:, 1]
    r2 = x * x + y * y
    denom = 1.0 + r2
    return np.stack([2 * x / denom, 2 * y / denom, (1 - r2) / denom], axis=1)


pts3d_raw = encode(pts2d)

# Camera: pitched up enough that the tangent plane at the north pole is
# clearly visible from above. We do NOT rot-align (i.e. don't drag the
# encoder pole onto the camera axis) — that would make the tangent plane
# face-on to the viewer and erase the perpendicular drop.
ELEV_DEG, AZIM_DEG = 38, -55
elev_r = np.deg2rad(ELEV_DEG)
azim_r = np.deg2rad(AZIM_DEG)
view_dir = np.array([np.cos(elev_r) * np.cos(azim_r),
                     np.cos(elev_r) * np.sin(azim_r),
                     np.sin(elev_r)])

pts3d = pts3d_raw
vis = pts3d @ view_dir


def idx_at(target):
    return int(np.argmin(np.linalg.norm(pts2d - np.array(target), axis=1)))


# Pick a current state at the encoder pole (no rot_align), so its tangent
# plane is the horizontal plane viewed from above-camera at a useful tilt.
# Recalled state is offset NE so the projection has both E and N components.
i_t = idx_at([0.0, 0.0])
i_hat = idx_at([0.42, 0.42])

# For visualization, sample the basis at a coarser step than the literal
# grid neighbors — gives a much cleaner figure where d_N, d_E aren't
# microscopic next to D. The math (Gram-Schmidt + projection) is identical;
# the chosen neighbors are still encoder samples on the cap.
agent2d = pts2d[i_t]
dx_step = xs[1] - xs[0]
NEIGHBOR_STEP = 2
nb_idx = {
    "N": idx_at(agent2d + np.array([0.0, NEIGHBOR_STEP * dx_step])),
    "E": idx_at(agent2d + np.array([NEIGHBOR_STEP * dx_step, 0.0])),
}

# ---------- compute basis, projection, plane corners ----------
p_t = pts3d[i_t]
p_hat = pts3d[i_hat]
p_N = pts3d[nb_idx["N"]]
p_E = pts3d[nb_idx["E"]]
d_N = p_N - p_t
d_E = p_E - p_t

# Gram-Schmidt for projection
e_N_unit = d_N / np.linalg.norm(d_N)
d_E_perp = d_E - np.dot(d_E, e_N_unit) * e_N_unit
e_E_unit = d_E_perp / np.linalg.norm(d_E_perp)

# Displacement and its projection onto span(d_N, d_E)
D = p_hat - p_t
qN_3d = float(np.dot(D, e_N_unit))
qE_3d = float(np.dot(D, e_E_unit))
D_proj = qN_3d * e_N_unit + qE_3d * e_E_unit

# Plane visualization extent — a generous fixed size so the plane visibly
# extends well beyond the spherical cap and contains the shadow.
nN_len = float(np.linalg.norm(d_N))
nE_len = float(np.linalg.norm(d_E))
PLANE_EXT = 1.20
plane_N_pos = max(PLANE_EXT, qN_3d + 0.45)
plane_N_neg = max(PLANE_EXT, -qN_3d + 0.45)
plane_E_pos = max(PLANE_EXT, qE_3d + 0.45)
plane_E_neg = max(PLANE_EXT, -qE_3d + 0.45)

corners = np.array([
    p_t - plane_N_neg * e_N_unit - plane_E_neg * e_E_unit,
    p_t + plane_N_pos * e_N_unit - plane_E_neg * e_E_unit,
    p_t + plane_N_pos * e_N_unit + plane_E_pos * e_E_unit,
    p_t - plane_N_neg * e_N_unit + plane_E_pos * e_E_unit,
])

# ---------- figure ----------
FIG_W, FIG_H = 16.0, 5.5
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=FIG_BG)


def fig_arrow(x0, x1, y=0.5, color=INK, lw=1.6, scale=22):
    a = FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure,
                        arrowstyle="-|>", mutation_scale=scale,
                        linewidth=lw, color=color,
                        capstyle="round", joinstyle="round")
    fig.patches.append(a)


def strip_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def panel_title(ax, text):
    ax.set_title(text, fontsize=TITLE_SIZE, color=TITLE_COLOR,
                 pad=12, fontweight="regular")


def add_compass(ax, x0, y0, length=0.9, color=MUTED, fontsize=11):
    """Small N/E compass at world-grid position (x0, y0)."""
    ax.annotate("", xy=(x0, y0 + length), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.2, mutation_scale=10),
                zorder=6)
    ax.annotate("", xy=(x0 + length, y0), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.2, mutation_scale=10),
                zorder=6)
    ax.text(x0 - 0.05, y0 + length + 0.10, "N",
            fontsize=fontsize, color=color, fontweight="bold",
            ha="center", va="bottom")
    ax.text(x0 + length + 0.15, y0 - 0.05, "E",
            fontsize=fontsize, color=color, fontweight="bold",
            ha="left", va="center")


# ---------- Panel 1: world grid (input) ----------
ax1 = fig.add_axes([0.012, 0.12, 0.21, 0.78])
ax1.set_facecolor(FIG_BG)
GS = 7
ax1.set_xlim(-0.5, GS - 0.5)
ax1.set_ylim(-0.5, GS - 0.5)
ax1.set_aspect("equal")
for k in range(GS + 1):
    ax1.axhline(k - 0.5, color=GRID_LINE, lw=0.7, zorder=1)
    ax1.axvline(k - 0.5, color=GRID_LINE, lw=0.7, zorder=1)

agent_cell = (3, 3)
goal_cell = (5, 5)
n_cell = (agent_cell[0], agent_cell[1] + 1)
e_cell = (agent_cell[0] + 1, agent_cell[1])
# N and E neighbor cells used to form the local basis
ax1.scatter([n_cell[0], e_cell[0]], [n_cell[1], e_cell[1]],
            s=140, facecolor=NEIGHBOR_COLOR, edgecolor=INK, lw=0.8,
            zorder=5)
ax1.scatter([agent_cell[0]], [agent_cell[1]], s=180, color=AGENT_COLOR,
            zorder=6)
ax1.text(agent_cell[0] + 0.40, agent_cell[1] + 0.42, "$x_t$",
         fontsize=LABEL_SIZE, color=INK, fontweight="bold")
ax1.scatter([goal_cell[0]], [goal_cell[1]], s=180, color=GOAL_COLOR,
            zorder=6)
ax1.text(goal_cell[0] + 0.40, goal_cell[1] + 0.42, "$\\hat{x}$",
         fontsize=LABEL_SIZE, color=GOAL_COLOR, fontweight="bold")
add_compass(ax1, 0.0, 0.0)
strip_axes(ax1)
panel_title(ax1, "world grid")

fig_arrow(0.226, 0.260)

# ---------- Panel 2: embedding (plane + projection) ----------
ax2 = fig.add_axes([0.225, -0.06, 0.55, 1.12], projection="3d",
                   computed_zorder=False)
ax2.set_facecolor(FIG_BG)

# Show only a spherical cap around x_t — clearly a *patch* of the encoder
# manifold under a larger tangent plane.
CAP_V_MAX = np.deg2rad(62.0)
CAP_Z_MIN = float(np.cos(CAP_V_MAX))    # ~0.574
CAP_R_MAX = float(np.sin(CAP_V_MAX))    # ~0.819

u = np.linspace(0, 2 * np.pi, 160)
v = np.linspace(0, CAP_V_MAX, 50)
SX = np.outer(np.cos(u), np.sin(v))
SY = np.outer(np.sin(u), np.sin(v))
SZ = np.outer(np.ones_like(u), np.cos(v))

# Disk closing the bottom of the cap so it reads as a solid dome.
disk_r = np.linspace(0, CAP_R_MAX, 16)
disk_theta = np.linspace(0, 2 * np.pi, 80)
DX = np.outer(disk_r, np.cos(disk_theta))
DY = np.outer(disk_r, np.sin(disk_theta))
DZ = np.zeros_like(DX) + CAP_Z_MIN
ax2.plot_surface(DX, DY, DZ, color="#dde3ec", alpha=0.55,
                 linewidth=0, edgecolor="none", shade=False, zorder=2)

cap_mask = pts3d[:, 2] >= CAP_Z_MIN
front_mask = (vis >= 0) & cap_mask

# Faint encoded samples inside the cap — keeps the manifold context light.
alpha_dots = 0.40 + 0.25 * np.clip(vis, 0, 1)
rgba_dots = np.concatenate([colors, alpha_dots[:, None]], axis=1)
if front_mask.any():
    ax2.scatter(pts3d[front_mask, 0] * 1.012,
                pts3d[front_mask, 1] * 1.012,
                pts3d[front_mask, 2] * 1.012,
                c=rgba_dots[front_mask], s=22,
                edgecolor="none", linewidth=0,
                depthshade=False, zorder=10)

ax2.plot_surface(SX, SY, SZ, color=SPHERE_FACE, alpha=0.95,
                 linewidth=0, edgecolor="none",
                 rstride=2, cstride=3, zorder=3, shade=False)
ax2.plot_wireframe(SX, SY, SZ, color=SPHERE_WIRE, linewidth=0.4,
                   rcount=6, ccount=14, alpha=0.55, zorder=4)

# --- Plane spanned by d_N, d_E ---
plane = Poly3DCollection([corners], facecolor=PLANE_FACE, alpha=0.32,
                         edgecolor=PLANE_EDGE, lw=1.2, zorder=6)
ax2.add_collection3d(plane)

# Faint parallelogram outline for d_N, d_E "generators"
para = np.array([
    p_t,
    p_t + d_N,
    p_t + d_N + d_E,
    p_t + d_E,
    p_t,
])
ax2.plot(para[:, 0], para[:, 1], para[:, 2],
         color=PLANE_EDGE, lw=0.8, ls="--", alpha=0.55, zorder=7)

# --- d_N and d_E basis arrows ---
add_arrow3d(ax2, p_t, d_N, color=BASIS_COLOR, lw=2.4, scale=16, zorder=12)
add_arrow3d(ax2, p_t, d_E, color=BASIS_COLOR, lw=2.4, scale=16, zorder=12)
# Labels offset so they don't clash with the chord/projection arrows.
lab_N = p_t + d_N + np.array([-0.18, 0.02, 0.10])
lab_E = p_t + d_E + np.array([0.02, -0.16, -0.05])
ax2.text(lab_N[0], lab_N[1], lab_N[2], "$d_N$",
         fontsize=14, fontweight="bold", color=BASIS_COLOR, zorder=22)
ax2.text(lab_E[0], lab_E[1], lab_E[2], "$d_E$",
         fontsize=14, fontweight="bold", color=BASIS_COLOR, zorder=22)

# Neighbor encoding dots (small, blue)
for d_lab, p_n in [("N", p_N), ("E", p_E)]:
    ax2.scatter([p_n[0] * 1.005], [p_n[1] * 1.005], [p_n[2] * 1.005],
                s=70, facecolor=NEIGHBOR_COLOR, edgecolor=INK, lw=0.8,
                depthshade=False, zorder=13)

# --- projection D_proj (in plane), labeled q ---
add_arrow3d(ax2, p_t, D_proj, color=PROJECTION, lw=2.8, scale=18, zorder=15)
proj_lab = p_t + 0.62 * D_proj + np.array([-0.05, 0.16, 0.18])
ax2.text(proj_lab[0], proj_lab[1], proj_lab[2], "$q$",
         fontsize=22, fontweight="bold", color=PROJECTION, zorder=22)

# --- perpendicular drop (dashed) from x_hat up to its shadow on the plane ---
end_D = p_t + D
shadow = p_t + D_proj
ax2.plot([end_D[0], shadow[0]], [end_D[1], shadow[1]], [end_D[2], shadow[2]],
         color=PERP_COLOR, lw=1.6, ls="--", alpha=0.95, zorder=16)

# --- shadow of x_hat: small faded dot sitting on the plane ---
ax2.scatter([shadow[0]], [shadow[1]], [shadow[2]],
            s=70, facecolor=GOAL_COLOR, edgecolor=GOAL_COLOR, lw=0.6,
            alpha=0.55, depthshade=False, zorder=18)

# --- x_hat: small filled dot on the cap surface ---
ax2.scatter([p_hat[0]], [p_hat[1]], [p_hat[2]],
            s=70, facecolor=GOAL_COLOR, edgecolor=INK, lw=0.6,
            depthshade=False, zorder=19)

# --- x_t: small filled dot at the top of the cap ---
ax2.scatter([p_t[0]], [p_t[1]], [p_t[2]],
            s=70, facecolor=INK, edgecolor=INK, lw=0.6,
            depthshade=False, zorder=21)

# Labels for x_t and x_hat (rings removed; small filled dots above sit on
# the surface and don't need a ring).
xt_lab = np.asarray(p_t) + np.array([-0.20, -0.05, 0.18])
ax2.text(xt_lab[0], xt_lab[1], xt_lab[2], "$x_t$",
         fontsize=LABEL_SIZE, fontweight="bold", color=INK, zorder=22)
xhat_lab = np.asarray(p_hat) + np.array([0.16, 0.05, -0.05])
ax2.text(xhat_lab[0], xhat_lab[1], xhat_lab[2], "$\\hat{x}$",
         fontsize=LABEL_SIZE, fontweight="bold", color=GOAL_COLOR, zorder=22)

ax2.view_init(elev=ELEV_DEG, azim=AZIM_DEG)
# Tight zoom so the cap + plane fill the panel — sphere reads as bigger.
ax2.set_box_aspect((1, 1, 0.45))
ax2.set_xlim(-1.18, 1.18)
ax2.set_ylim(-1.18, 1.18)
ax2.set_zlim(CAP_Z_MIN - 0.05, 1.20)
ax2.set_axis_off()
panel_title(ax2, "project displacement onto local plane")

fig_arrow(0.755, 0.788)

# ---------- Panel 3: world grid (output direction) ----------
ax3 = fig.add_axes([0.785, 0.12, 0.21, 0.78])
ax3.set_facecolor(FIG_BG)
ax3.set_xlim(-0.5, GS - 0.5)
ax3.set_ylim(-0.5, GS - 0.5)
ax3.set_aspect("equal")
for k in range(GS + 1):
    ax3.axhline(k - 0.5, color=GRID_LINE, lw=0.7, zorder=1)
    ax3.axvline(k - 0.5, color=GRID_LINE, lw=0.7, zorder=1)

# Faded recall cell for context
ax3.scatter([goal_cell[0]], [goal_cell[1]], s=130, color=GOAL_COLOR,
            alpha=0.30, zorder=4)
ax3.text(goal_cell[0] + 0.40, goal_cell[1] + 0.42, "$\\hat{x}$",
         fontsize=12, color=GOAL_COLOR, alpha=0.45, fontweight="bold")

# Agent
ax3.scatter([agent_cell[0]], [agent_cell[1]], s=180, color=AGENT_COLOR,
            zorder=5)
ax3.text(agent_cell[0] - 0.10, agent_cell[1] - 0.55, "$x_t$",
         fontsize=LABEL_SIZE, color=INK, fontweight="bold")

# Movement direction arrow: q points along (qE, qN) in world (E, N)
q_norm = np.sqrt(qE_3d ** 2 + qN_3d ** 2)
dir_E = qE_3d / q_norm
dir_N = qN_3d / q_norm
arrow_len = 1.7
ex = agent_cell[0] + dir_E * arrow_len
ey = agent_cell[1] + dir_N * arrow_len
ax3.annotate("", xy=(ex, ey), xytext=(agent_cell[0], agent_cell[1]),
             arrowprops=dict(arrowstyle="-|>", color=PROJECTION,
                             lw=3.2, mutation_scale=24))
ax3.text(ex + 0.10, ey - 0.05, "$q$",
         fontsize=18, fontweight="bold", color=PROJECTION)

add_compass(ax3, 0.0, 0.0)
strip_axes(ax3)
panel_title(ax3, "movement direction")

plt.savefig(f"{OUT_DIR}/decoder_schematic.pdf",
            bbox_inches="tight", facecolor=FIG_BG)
plt.savefig(f"{OUT_DIR}/decoder_schematic.svg",
            bbox_inches="tight", facecolor=FIG_BG)
plt.savefig(f"{OUT_DIR}/decoder_schematic_preview.png",
            bbox_inches="tight", facecolor=FIG_BG, dpi=130)
print(f"wrote {OUT_DIR}/decoder_schematic.pdf and .svg")
print(f"  qE={qE_3d:+.3f}  qN={qN_3d:+.3f}  "
      f"|D|={np.linalg.norm(D):.3f}  |D_proj|={np.linalg.norm(D_proj):.3f}")
