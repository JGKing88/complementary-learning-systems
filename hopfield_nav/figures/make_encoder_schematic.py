"""Schematic: encoder maps 2D state → grid code → MLP → hypersphere
where cosine similarity reflects real-space distance.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
from matplotlib.patches import FancyArrowPatch, Circle
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import os

from cls_paths import figures_dir

# ---------- style ----------
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

INK = "#2a2a2a"            # near-black for strokes / labels
MUTED = "#6b6f76"          # secondary gray
BG = "#f7f5f0"             # warm off-white (matches many paper figures)
FIG_BG = "white"
PAIR_CLOSE = "#4a8c5e"     # muted green (A-B)
PAIR_FAR   = "#a35454"     # muted burgundy (A-C)
MOD_COLORS = ["#5577aa",   # muted blue
              "#4a9b6b",   # muted green
              "#c47e4f"]   # muted terracotta
SPHERE_FACE = "#eef1f6"
SPHERE_WIRE = "#c6cfdc"
NODE_FACE = "#fafafa"
NODE_EDGE = "#555555"
EDGE_COLOR = "#a6adb7"
TITLE_SIZE = 15
LABEL_SIZE = 15
TITLE_COLOR = "#333333"

# Schematics render to the shared figures root on pool, not into the
# source tree. Override the root with the CLS_RUNS env var.
OUT_DIR = str(figures_dir(ensure=True) / "schematics")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- data ----------
n = 17
xs = np.linspace(-1.1, 1.1, n)
ys = np.linspace(-1.1, 1.1, n)
X2d, Y2d = np.meshgrid(xs, ys)
pts2d = np.stack([X2d.ravel(), Y2d.ravel()], axis=1)

# Soft 2D→RGB gradient (desaturated), readable but not garish
def pos_to_color(pts):
    p = (pts - pts.min(0)) / (pts.max(0) - pts.min(0) + 1e-9)
    # anchor points in a pastel palette
    r = 0.35 + 0.50 * p[:, 0]
    g = 0.55 + 0.30 * p[:, 1]
    b = 0.80 - 0.35 * p[:, 0]
    return np.clip(np.stack([r, g, b], axis=1), 0, 1)

colors = pos_to_color(pts2d)

# grid code: 3 concatenated one-hot modules with distinct periods/directions
MODULE_LENGTHS = [5, 7, 11]
MODULE_ANGLES = [0.0, np.pi / 4, np.pi / 2]
MODULE_PERIODS = [0.6, 1.2, 2.2]

def grid_code_active(pos):
    out = []
    for L, ang, per in zip(MODULE_LENGTHS, MODULE_ANGLES, MODULE_PERIODS):
        phase = ((pos[0] * np.cos(ang) + pos[1] * np.sin(ang)) / per) % 1.0
        out.append(int(phase * L) % L)
    return out

# encoder: inverse stereographic (conformal, preserves local distances)
def encode(pts):
    x, y = pts[:, 0], pts[:, 1]
    r2 = x * x + y * y
    denom = 1.0 + r2
    return np.stack([2 * x / denom, 2 * y / denom, (1 - r2) / denom], axis=1)

pts3d = encode(pts2d)

ELEV_DEG, AZIM_DEG = 22, -48
elev_r = np.deg2rad(ELEV_DEG); azim_r = np.deg2rad(AZIM_DEG)
view_dir = np.array([np.cos(elev_r) * np.cos(azim_r),
                     np.cos(elev_r) * np.sin(azim_r),
                     np.sin(elev_r)])

def rot_align(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b); s = np.linalg.norm(v); c = float(np.dot(a, b))
    if s < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s ** 2))

pts3d = (rot_align(np.array([0, 0, 1.0]), view_dir) @ pts3d.T).T
vis = pts3d @ view_dir

def idx_at(target):
    return int(np.argmin(np.linalg.norm(pts2d - np.array(target), axis=1)))

iA = idx_at([0.0, 0.0])
iB = idx_at([0.28, 0.18])
iC = idx_at([-0.95, 0.85])

# ---------- figure ----------
FIG_W, FIG_H = 15.5, 6.0
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=FIG_BG)

def fig_arrow(x0, x1, y=0.5, color=INK, lw=1.6, scale=22):
    a = FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure,
                        arrowstyle="-|>", mutation_scale=scale,
                        linewidth=lw, color=color,
                        capstyle="round", joinstyle="round")
    fig.patches.append(a)

def strip_axes(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

def panel_title(ax, text):
    ax.set_title(text, fontsize=TITLE_SIZE, color=TITLE_COLOR,
                 pad=12, fontweight="regular")

# ---------- Panel 1: state space ----------
ax1 = fig.add_axes([0.010, 0.14, 0.195, 0.76])
ax1.set_facecolor(FIG_BG)
ax1.scatter(pts2d[:, 0], pts2d[:, 1], c=colors, s=58,
            edgecolor="none", linewidth=0, zorder=2)
# highlight rings on A, B, C
for i, lbl in [(iA, "A"), (iB, "B"), (iC, "C")]:
    ax1.scatter(pts2d[i, 0], pts2d[i, 1], s=220, facecolor="none",
                edgecolor=INK, lw=1.3, zorder=3)
    dx = 0.09 if lbl != "C" else -0.05
    dy = 0.11
    ax1.text(pts2d[i, 0] + dx, pts2d[i, 1] + dy, lbl,
             fontsize=LABEL_SIZE, color=INK, fontweight="bold", zorder=4)
# reference distances
for (i, j), col in [((iA, iB), PAIR_CLOSE), ((iA, iC), PAIR_FAR)]:
    ax1.plot([pts2d[i, 0], pts2d[j, 0]], [pts2d[i, 1], pts2d[j, 1]],
             linestyle=(0, (3, 2)), color=col, lw=1.2, zorder=1)
ax1.set_aspect("equal")
strip_axes(ax1)
panel_title(ax1, "state space (2D)")

fig_arrow(0.208, 0.230)

# ---------- Panel 2: grid code ----------
ax_gc = fig.add_axes([0.238, 0.14, 0.078, 0.76])
ax_gc.set_facecolor(FIG_BG)
D_grid = sum(MODULE_LENGTHS)
img = np.full((D_grid, 3, 3), 0.96)  # warm off-white background
inactive = np.array([0.93, 0.92, 0.89])
for col_i, pt_i in enumerate([iA, iB, iC]):
    active = grid_code_active(pts2d[pt_i])
    row = 0
    for k, (L, mcol) in enumerate(zip(MODULE_LENGTHS, MOD_COLORS)):
        rgb = np.array(mpl.colors.to_rgb(mcol))
        for j in range(L):
            img[row + j, col_i] = rgb if j == active[k] else inactive
        row += L
ax_gc.imshow(img, aspect="auto", interpolation="nearest")
# A/B/C column labels, positioned just below the grid (no visible axes)
ax_gc.set_xticks([])
ax_gc.set_yticks([])
for col_i, lbl in enumerate(["A", "B", "C"]):
    ax_gc.text(col_i, D_grid - 0.35, lbl,
               ha="center", va="top", fontsize=LABEL_SIZE,
               fontweight="bold", color=INK,
               transform=ax_gc.transData)
# module separators (thin, ink) and column separators (thin)
cum = np.cumsum(MODULE_LENGTHS)
for yb in cum[:-1]:
    ax_gc.axhline(yb - 0.5, color=INK, lw=0.7, alpha=0.6)
for xb in (0.5, 1.5):
    ax_gc.axvline(xb, color=INK, lw=0.7, alpha=0.6)
# module labels on right in matching colors
row = 0
for k, (L, mcol) in enumerate(zip(MODULE_LENGTHS, MOD_COLORS)):
    ax_gc.text(2.95, row + L / 2 - 0.5, f"$m_{k+1}$",
               va="center", ha="left", fontsize=LABEL_SIZE,
               color=mcol, fontweight="bold")
    row += L
# extend xlim so right-side labels aren't clipped
ax_gc.set_xlim(-0.5, 3.9)
for s in ax_gc.spines.values():
    s.set_visible(False)
panel_title(ax_gc, "grid code $g$")

fig_arrow(0.328, 0.348)

# ---------- Panel 3: MLP ----------
ax_ffn = fig.add_axes([0.342, 0.10, 0.215, 0.82])
ax_ffn.set_facecolor(FIG_BG)
ax_ffn.set_xlim(0, 1); ax_ffn.set_ylim(0, 1)
strip_axes(ax_ffn)
ax_ffn.set_aspect("auto")

layer_sizes = [7, 9, 9, 5]
layer_labels = ["$g$", "", "", "$f_\\theta(g)$"]
layer_x = np.linspace(0.10, 0.90, len(layer_sizes))
node_r = 0.032

node_positions = []
for lx, n_nodes in zip(layer_x, layer_sizes):
    ys = np.linspace(0.26, 0.78, n_nodes)
    node_positions.append(np.column_stack([np.full(n_nodes, lx), ys]))

# soft connection lines: alpha proportional to "weight" magnitude
rng = np.random.default_rng(0)
for i in range(len(layer_sizes) - 1):
    for p in node_positions[i]:
        for q in node_positions[i + 1]:
            w = 0.32 + 0.40 * rng.random()
            ax_ffn.plot([p[0], q[0]], [p[1], q[1]],
                        color=EDGE_COLOR, lw=0.7, alpha=w, zorder=1,
                        solid_capstyle="round")

# nodes — no hard outline, subtle fill + faint ring
for nodes in node_positions:
    for p in nodes:
        ax_ffn.add_patch(Circle(p, node_r, facecolor=NODE_FACE,
                                edgecolor=NODE_EDGE, lw=1.0, zorder=3))

# bottom labels only for input/output layers
for lx, lbl in zip(layer_x, layer_labels):
    if lbl:
        ax_ffn.text(lx, 0.14, lbl, ha="center", va="top",
                    fontsize=LABEL_SIZE, color=INK)
panel_title(ax_ffn, "encoder $f_\\theta$")

fig_arrow(0.560, 0.582)

# ---------- Panel 4: sphere ----------
_sphere_frac = FIG_H / FIG_W  # square bbox in inches → sphere fills axes
ax2 = fig.add_axes([0.592, 0.00, _sphere_frac, 1.00], projection="3d",
                   computed_zorder=False)
ax2.set_facecolor(FIG_BG)

u = np.linspace(0, 2 * np.pi, 120)
v = np.linspace(0, np.pi, 70)
SX = np.outer(np.cos(u), np.sin(v))
SY = np.outer(np.sin(u), np.sin(v))
SZ = np.outer(np.ones_like(u), np.cos(v))

front_mask = vis >= 0
back_mask = ~front_mask

# back dots: desaturated, very faint
col_back = 0.55 * colors + 0.45
alpha_back = 0.15 + 0.12 * np.clip(vis + 1, 0, 1)
rgba_back = np.concatenate([col_back, alpha_back[:, None]], axis=1)

# front dots: full color, slight silhouette fade
alpha_front = 0.90 + 0.10 * np.clip(vis, 0, 1)
rgba_front = np.concatenate([colors, alpha_front[:, None]], axis=1)

# back dots first
if back_mask.any():
    ax2.scatter(pts3d[back_mask, 0] * 0.998,
                pts3d[back_mask, 1] * 0.998,
                pts3d[back_mask, 2] * 0.998,
                c=rgba_back[back_mask], s=26, edgecolor="none",
                depthshade=False, zorder=1)

# sphere — cleaner: opaque-ish pastel surface, sparse latitude/longitude
ax2.plot_surface(SX, SY, SZ, color=SPHERE_FACE, alpha=0.92,
                 linewidth=0, edgecolor="none",
                 rstride=3, cstride=3, zorder=3)
# sparse wireframe (fewer lines, thinner) for subtle depth
ax2.plot_wireframe(SX, SY, SZ, color=SPHERE_WIRE, linewidth=0.35,
                   rcount=10, ccount=10, alpha=0.55, zorder=4)

# front dots
if front_mask.any():
    ax2.scatter(pts3d[front_mask, 0] * 1.015,
                pts3d[front_mask, 1] * 1.015,
                pts3d[front_mask, 2] * 1.015,
                c=rgba_front[front_mask], s=40,
                edgecolor="none", linewidth=0,
                depthshade=False, zorder=10)

# A, B, C highlights + vectors
for i, lbl in [(iA, "A"), (iB, "B"), (iC, "C")]:
    on_front = vis[i] >= 0
    a = 1.0 if on_front else 0.28
    ring_edge = INK if on_front else MUTED
    z_dot = 11 if on_front else 0
    ax2.scatter(pts3d[i, 0] * 1.02, pts3d[i, 1] * 1.02, pts3d[i, 2] * 1.02,
                s=180, facecolor="none", edgecolor=ring_edge, lw=1.3,
                alpha=a, depthshade=False, zorder=z_dot)
    ax2.plot([0, pts3d[i, 0]], [0, pts3d[i, 1]], [0, pts3d[i, 2]],
             color=INK, lw=0.9, alpha=a, zorder=z_dot - 1)
    ax2.text(pts3d[i, 0] * 1.22, pts3d[i, 1] * 1.22, pts3d[i, 2] * 1.22,
             lbl, fontsize=LABEL_SIZE, fontweight="bold",
             color=INK if on_front else MUTED, zorder=20)

# geodesic arcs with per-segment visibility fade
def sphere_arc(p, q, n=80):
    p = p / np.linalg.norm(p); q = q / np.linalg.norm(q)
    omega = np.arccos(np.clip(np.dot(p, q), -1, 1))
    t = np.linspace(0, 1, n)[:, None]
    return (np.sin((1 - t) * omega) * p + np.sin(t * omega) * q) / np.sin(omega)

for (i, j), col in [((iA, iB), PAIR_CLOSE), ((iA, iC), PAIR_FAR)]:
    arc = sphere_arc(pts3d[i], pts3d[j]) * 1.012
    arc_vis = arc @ view_dir
    segs = np.stack([arc[:-1], arc[1:]], axis=1)
    seg_vis = 0.5 * (arc_vis[:-1] + arc_vis[1:])
    seg_alpha = np.where(seg_vis >= 0, 0.92, 0.18)
    rgba = np.zeros((len(segs), 4))
    rgba[:, :3] = np.array(mpl.colors.to_rgb(col))
    rgba[:, 3] = seg_alpha
    lc = Line3DCollection(segs, colors=rgba, linewidths=2.0, zorder=9)
    ax2.add_collection3d(lc)

ax2.view_init(elev=ELEV_DEG, azim=AZIM_DEG)
ax2.set_box_aspect((1, 1, 1))
ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_zticks([])
ax2.grid(False)
for pane in (ax2.xaxis.pane, ax2.yaxis.pane, ax2.zaxis.pane):
    pane.set_visible(False)
panel_title(ax2, "hypersphere output")

plt.savefig(f"{OUT_DIR}/encoder_schematic.pdf", bbox_inches="tight",
            facecolor=FIG_BG)
plt.savefig(f"{OUT_DIR}/encoder_schematic.svg", bbox_inches="tight",
            facecolor=FIG_BG)
plt.savefig(f"{OUT_DIR}/encoder_schematic_preview.png", bbox_inches="tight",
            facecolor=FIG_BG, dpi=130)
print(f"wrote {OUT_DIR}/encoder_schematic.pdf and .svg")
