"""The ideal encoder as a network on one-hot grid modules (one axis), toy periods.

    python analysis/hopfield_probe/figures/phase_network.py

Modules of period 3, 4, 5 (L = 60). Grid cells (one-hot) -> phase units (weight
2 pi a / lambda on cell a) -> angle units (whole-number weights j from the Chinese
remainder theorem) -> cos and sin. Active cells shown for x = 7.
Writes phase_network.pdf (core Helvetica/Symbol) and phase_network.png here.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

import style
from style import INK_2, rich

style.apply()

LAMS = (3, 4, 5)
L = 3 * 4 * 5
X = 7                                     # the location whose active cells are shown
MOD_Y = (6.0, 3.7, 1.4)
MOD_COL = ("#2a78d6", "#eb6834", "#1baf7a")
ANGLES = ((1, 4.6, "#e34948"), (2, 0.85, "#6250d6"))
PITCH, SIDE = 0.72, 0.64
PX, PR = 6.6, 0.6                         # phase-node x and radius
AX, AR = 12.0, 0.65                       # angle-node x and radius
LIGHT = "#c9c8c4"


def crt_weights(n):
    """j_l in 0..lambda_l - 1 with sum_l (L / lambda_l) j_l = n (mod L)."""
    return tuple(next(j for j in range(lam) if ((L // lam) * j - n) % lam == 0) for lam in LAMS)


def edge_point(x0, y0, x1, y1, r):
    """The point at distance r from (x1, y1) toward (x0, y0)."""
    dx, dy = x0 - x1, y0 - y1
    d = (dx * dx + dy * dy) ** 0.5
    return x1 + r * dx / d, y1 + r * dy / d


fig = plt.figure(figsize=(4.6, 2.0))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(-1.9, 19.4)
ax.set_ylim(-0.75, 7.7)
ax.set_aspect("equal")
ax.axis("off")

for x, label in ((1.8, "grid cells"), (PX, "phase units"), (AX, "angle units")):
    rich(ax, x, 7.35, [(label, "")], size=8)

phase_xy = []
for lam, y, col in zip(LAMS, MOD_Y, MOD_COL):
    act, py = X % lam, y - 1.15
    phase_xy.append((PX, py))
    rich(ax, -0.3, y, [("λ", "g"), (" = %d" % lam, "")], ha="right", size=8, color=col)
    for a in range(lam):
        cx, on = a * PITCH + SIDE / 2, a == act
        ex, ey = edge_point(cx, y - SIDE / 2, PX, py, PR)
        ax.plot([cx, ex], [y - SIDE / 2, ey], color=col if on else LIGHT,
                lw=1.2 if on else 0.5, zorder=1.5 if on else 1)
        ax.add_patch(Rectangle((cx - SIDE / 2, y - SIDE / 2), SIDE, SIDE,
                               facecolor=col if on else "white",
                               edgecolor=col if on else LIGHT, lw=0.6, zorder=2))
        ax.text(cx, y, str(a), ha="center", va="center", fontsize=6,
                color="white" if on else INK_2, fontweight="bold" if on else "normal", zorder=3)
    ax.add_patch(Circle((PX, py), PR, facecolor="white", edgecolor=col, lw=1.1, zorder=3))
    rich(ax, PX, py, [("φ", "g"), (str(lam), "_")], size=8.5)

for n, y, col in ANGLES:
    for (x0, y0), j in zip(phase_xy, crt_weights(n)):
        sx, sy = edge_point(AX, y, x0, y0, PR)
        ex, ey = edge_point(x0, y0, AX, y, AR)
        ax.plot([sx, ex], [sy, ey], color=col, lw=0.9, zorder=1)
        t = 0.7
        ax.text(x0 + t * (AX - x0), y0 + t * (y - y0), str(j), ha="center", va="center",
                fontsize=7.5, color=col, fontweight="bold", zorder=4,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))
    ax.add_patch(Circle((AX, y), AR, facecolor="white", edgecolor=col, lw=1.2, zorder=3))
    rich(ax, AX, y, [("θ", "g"), (str(n), "_")], size=8.5)
    rich(ax, AX, y + 1.0, [("n", "i"), (" = %d" % n, "")], size=7.5, color=col)
    ax.add_artist(FancyArrowPatch((AX + AR + 0.1, y), (AX + AR + 1.1, y), arrowstyle="-|>",
                                  mutation_scale=6, color=INK_2, lw=0.7))
    rich(ax, AX + AR + 1.3, y, [("cos ", ""), ("θ", "g"), (str(n), "_"), (", sin ", ""),
                                ("θ", "g"), (str(n), "_")], ha="left", size=8)

for k in range(3):   # more angle units
    ax.add_patch(Circle((AX, 2.4 + 0.33 * k), 0.06, color=INK_2))

bx = 17.6
ax.plot([bx, bx + 0.3, bx + 0.3, bx], [5.2, 5.2, 0.25, 0.25], color=INK_2, lw=0.7)
rich(ax, bx + 0.6, 2.72, [("z", "i"), ("(", ""), ("x", "i"), (")", "")], ha="left", size=8.5)

style.save(fig, "phase_network", pad=0.02)
print("saved; weights", {n: crt_weights(n) for n, _, _ in ANGLES}, "active", [X % l for l in LAMS])
