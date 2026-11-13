"""The ideal encoder as a network on one-hot grid modules (one axis), toy periods.

    python analysis/hopfield_probe/figures/phase_network.py

Modules of period 3, 4, 5 (L = 60), each a column of one-hot cells. Grid cells ->
phase units phi_l (weight 2 pi a / lambda_l on cell a) -> angle units omega_n x
(whole-number weights j from the Chinese remainder theorem) -> cos and sin.
Active cells shown for x = 7. Writes phase_network.pdf (core Helvetica/Symbol)
and phase_network.png here.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

import style
from style import INK_2, rich

style.apply()

LAMS = (3, 4, 5)
L = 3 * 4 * 5
X = 7                                     # the location whose active cells are shown
MOD_COL = ("#2a78d6", "#eb6834", "#1baf7a")
ANGLES = ((1, 9.3, "#e34948"), (2, 4.6, "#6250d6"))
PITCH, SIDE, GAP = 0.72, 0.64, 0.95
CX = 0.32                                 # cell column centre
PX, PR = 4.6, 0.62                        # phase-node x and radius
AX, AR = 10.2, 0.72                       # angle-node x and radius
LIGHT = "#c9c8c4"


def crt_weights(n):
    """j_l in 0..lambda_l - 1 with sum_l (L / lambda_l) j_l = n (mod L)."""
    return tuple(next(j for j in range(lam) if ((L // lam) * j - n) % lam == 0) for lam in LAMS)


def edge_point(x0, y0, x1, y1, r):
    """The point at distance r from (x1, y1) toward (x0, y0)."""
    dx, dy = x0 - x1, y0 - y1
    d = (dx * dx + dy * dy) ** 0.5
    return x1 + r * dx / d, y1 + r * dy / d


fig = plt.figure(figsize=(4.6, 2.6))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(-2.4, 19.2)
ax.set_ylim(1.0, 12.9)
ax.set_aspect("equal")
ax.axis("off")

for x, label in ((CX, "grid cells"), (PX, "phase units"), (AX, "angle units")):
    rich(ax, x, 12.45, [(label, "")], size=8)

phase_xy, top = [], 11.5
for ell, (lam, col) in enumerate(zip(LAMS, MOD_COL), start=1):
    ys = [top - a * PITCH for a in range(lam)]
    yc = (ys[0] + ys[-1]) / 2
    phase_xy.append((PX, yc))
    act = X % lam
    rich(ax, CX - 0.75, yc, [("λ", "g"), (str(ell), "_"), (" = %d" % lam, "")], ha="right",
         size=8, color=col)
    for a, y in enumerate(ys):
        on = a == act
        sx, sy = CX + SIDE / 2, y
        ex, ey = edge_point(sx, sy, PX, yc, PR)
        ax.plot([sx, ex], [sy, ey], color=col if on else LIGHT, lw=1.3 if on else 0.5,
                zorder=1.5 if on else 1)
        ax.add_patch(Rectangle((CX - SIDE / 2, y - SIDE / 2), SIDE, SIDE,
                               facecolor=col if on else "white",
                               edgecolor=col if on else LIGHT, lw=0.6, zorder=2))
        ax.text(CX, y, str(a), ha="center", va="center", fontsize=6,
                color="white" if on else INK_2, fontweight="bold" if on else "normal", zorder=3)
        if on:   # label the active connection with its weight
            t = 0.45
            lx, ly = sx + t * (ex - sx), sy + t * (ey - sy)
            ax.add_patch(FancyBboxPatch((lx - 0.78, ly - 0.27), 1.56, 0.54,
                                        boxstyle="round,pad=0.02", fc="white", ec="none",
                                        zorder=3.5))
            rich(ax, lx, ly, [("2", ""), ("π", "g"), ("·%d/%d" % (a, lam), "")], size=7,
                 color=col, zorder=4)
    ax.add_patch(Circle((PX, yc), PR, facecolor="white", edgecolor=col, lw=1.1, zorder=3))
    rich(ax, PX, yc, [("φ", "g"), (str(ell), "_")], size=8.5)
    top = ys[-1] - PITCH - GAP

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
    rich(ax, AX, y, [("ω", "g"), (str(n), "_"), ("x", "i")], size=8.5)
    rich(ax, AX, y + 1.08, [("n", "i"), (" = %d" % n, "")], size=7.5, color=col)
    ax.add_artist(FancyArrowPatch((AX + AR + 0.1, y), (AX + AR + 1.0, y), arrowstyle="-|>",
                                  mutation_scale=6, color=INK_2, lw=0.7))
    rich(ax, AX + AR + 1.2, y, [("cos ", ""), ("ω", "g"), (str(n), "_"), ("x", "i"),
                                (", sin ", ""), ("ω", "g"), (str(n), "_"), ("x", "i")],
         ha="left", size=8)

for k in range(3):   # more angle units
    ax.add_patch(Circle((AX, 6.62 + 0.33 * k), 0.06, color=INK_2))

bx = 17.3
ax.plot([bx, bx + 0.3, bx + 0.3, bx], [9.9, 9.9, 4.0, 4.0], color=INK_2, lw=0.7)
rich(ax, bx + 0.6, 6.95, [("z", "i"), ("(", ""), ("x", "i"), (")", "")], ha="left", size=8.5)

style.save(fig, "phase_network", pad=0.02)
print("saved; weights", {n: crt_weights(n) for n, _, _ in ANGLES}, "active", [X % l for l in LAMS])
