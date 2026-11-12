"""The ideal encoder as a network on one-hot grid modules (one axis).

    python analysis/hopfield_probe/figures/phase_network.py

Grid cells (one-hot per module) -> phase units (weight 2 pi a / lambda on cell a)
-> angle units (whole-number weights j from the Chinese remainder theorem)
-> cos and sin. Worked example at x = 20 for n = 1 and n = 5.
Writes phase_network.pdf (core Helvetica/Symbol) and phase_network.png here.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

import style
from style import INK, INK_2, rich

style.apply()

X = 20                                    # the example location
MODS = [(11, 8.4, "#2a78d6"), (12, 5.4, "#eb6834"), (13, 2.4, "#1baf7a")]
ANGLES = [(1, (6, 11, 7), 6.7, "#e34948"), (5, (8, 7, 9), 2.3, "#6250d6")]
PITCH, SIDE = 0.72, 0.64
PX, PR = 12.6, 0.62                       # phase-node x and radius
AX, AR = 20.0, 0.70                       # angle-node x and radius
LIGHT = "#c9c8c4"

fig = plt.figure(figsize=(6.6, 3.55))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(-3.2, 32.9)
ax.set_ylim(-3.1, 10.5)
ax.set_aspect("equal")
ax.axis("off")


def edge_point(x0, y0, x1, y1, r):
    """The point at distance r from (x1, y1) toward (x0, y0)."""
    dx, dy = x0 - x1, y0 - y1
    d = (dx * dx + dy * dy) ** 0.5
    return x1 + r * dx / d, y1 + r * dy / d


# column headers
rich(ax, 4.6, 10.15, [("grid cells, one axis, one-hot", "")], size=8.5)
rich(ax, PX, 10.15, [("phase units", "")], size=8.5)
rich(ax, AX, 10.15, [("angle units, one per drawn ", ""), ("n", "i")], size=8.5)
rich(ax, 26.6, 10.15, [("output", "")], size=8.5)
rich(ax, PX, 9.45, [("φ", "g"), (" = ", ""), ("Σ", "g"), ("a", "i_"), (" (2", ""), ("π", "g"),
                    ("a", "i"), ("/", ""), ("λ", "g"), (") ", ""), ("g", "i"), ("a", "i_")],
     size=7.5, color=INK_2)
rich(ax, AX, 9.45, [("θ", "g"), (" = ", ""), ("j", "i"), ("1", "_"), ("φ", "g"), ("11", "_"),
                    (" + ", ""), ("j", "i"), ("2", "_"), ("φ", "g"), ("12", "_"), (" + ", ""),
                    ("j", "i"), ("3", "_"), ("φ", "g"), ("13", "_")],
     size=7.5, color=INK_2)

phase_xy = {}
for lam, y, col in MODS:
    act = X % lam
    py = y - 1.6
    phase_xy[lam] = (PX, py)
    rich(ax, -0.35, y, [("λ", "g"), (" = %d" % lam, "")], ha="right", size=8.5, color=col)
    rich(ax, 0.0, y + 0.72, [("weight on cell ", ""), ("a", "i"), (": 2", ""), ("π", "g"),
                             ("a", "i"), ("/%d" % lam, "")], ha="left", size=7, color=INK_2)
    for a in range(lam):
        cx = a * PITCH + SIDE / 2
        on = a == act
        if not on:   # fan line from the cell to its phase unit
            ex, ey = edge_point(cx, y - SIDE / 2, PX, py, PR)
            ax.plot([cx, ex], [y - SIDE / 2, ey], color=LIGHT, lw=0.45, zorder=1)
        ax.add_patch(Rectangle((cx - SIDE / 2, y - SIDE / 2), SIDE, SIDE,
                               facecolor=col if on else "white",
                               edgecolor=col if on else LIGHT, lw=0.6, zorder=2))
        ax.text(cx, y, str(a), ha="center", va="center", fontsize=5.5,
                color="white" if on else INK_2, fontweight="bold" if on else "normal",
                zorder=3)
    cx = act * PITCH + SIDE / 2
    ex, ey = edge_point(cx, y - SIDE / 2, PX, py, PR)
    ax.plot([cx, ex], [y - SIDE / 2, ey], color=col, lw=1.3, zorder=1.5)
    ax.add_patch(Circle((PX, py), PR, facecolor="white", edgecolor=col, lw=1.2, zorder=3))
    rich(ax, PX, py, [("φ", "g"), (str(lam), "_")], size=8.5)
    rich(ax, PX, py - 1.05, [("= 2", ""), ("π", "g"), ("·%d/%d" % (act, lam), "")],
         size=7, color=col)

for n, js, y, col in ANGLES:
    for (lam, _, _), j in zip(MODS, js):
        x0, y0 = phase_xy[lam]
        sx, sy = edge_point(AX, y, x0, y0, PR)
        ex, ey = edge_point(x0, y0, AX, y, AR)
        ax.plot([sx, ex], [sy, ey], color=col, lw=0.9, zorder=1)
        t = 0.72
        ax.text(x0 + t * (AX - x0), y0 + t * (y - y0), str(j), ha="center", va="center",
                fontsize=7.5, color=col, fontweight="bold", zorder=4,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))
    ax.add_patch(Circle((AX, y), AR, facecolor="white", edgecolor=col, lw=1.3, zorder=3))
    rich(ax, AX, y, [("θ", "g"), (str(n), "_")], size=9)
    rich(ax, AX, y + 1.15, [("n", "i"), (" = %d" % n, "")], size=7.5, color=col)
    ax.add_artist(FancyArrowPatch((AX + AR + 0.1, y), (AX + AR + 1.3, y), arrowstyle="-|>",
                                  mutation_scale=6, color=INK_2, lw=0.7))
    rich(ax, AX + AR + 1.5, y, [("cos ", ""), ("θ", "g"), (str(n), "_"), (",  sin ", ""),
                                ("θ", "g"), (str(n), "_")], ha="left", size=8)

for k in range(3):   # more angle units
    ax.add_patch(Circle((AX, 4.05 + 0.38 * k), 0.07, color=INK_2))
rich(ax, AX, 0.9, [("M", "i"), ("/2 in all", "")], size=7, color=INK_2)

# bracket and z(x)
bx = 27.4
ax.plot([bx, bx + 0.35, bx + 0.35, bx], [7.3, 7.3, 1.7, 1.7], color=INK_2, lw=0.7)
rich(ax, bx + 0.8, 4.9, [("z", "i"), ("(", ""), ("x", "i"), (")", "")], ha="left", size=9)
rich(ax, bx + 0.8, 4.05, [("scaled by ", ""), ("√", "g"), ("(2/", ""), ("M", "i"), (")", "")],
     ha="left", size=7, color=INK_2)

# worked example
L = [[("Example at ", ""), ("x", "i"), (" = 20: the active cells are ", ""), ("a", "i"),
      (" = 9, 8, 7, so the phases are 2", ""), ("π", "g"), ("·9/11, 2", ""), ("π", "g"),
      ("·8/12 and 2", ""), ("π", "g"), ("·7/13.", "")],
     [("n", "i"), (" = 1:  ", ""), ("θ", "g"), (" = 2", ""), ("π", "g"),
      ("(6·9/11 + 11·8/12 + 7·7/13) = 2", ""), ("π", "g"), ("(16 + 20/1716), the same angle as 2", ""),
      ("π", "g"), ("·1·20/1716.", "")],
     [("n", "i"), (" = 5:  ", ""), ("θ", "g"), (" = 2", ""), ("π", "g"),
      ("(8·9/11 + 7·8/12 + 9·7/13) = 2", ""), ("π", "g"), ("(16 + 100/1716), the same angle as 2", ""),
      ("π", "g"), ("·5·20/1716.", "")]]
for k, runs in enumerate(L):
    rich(ax, -3.0, -1.25 - 0.72 * k, runs, ha="left", size=7.5,
         color=[INK, ANGLES[0][3], ANGLES[1][3]][k])

style.save(fig, "phase_network", pad=0.02)
print("saved")
