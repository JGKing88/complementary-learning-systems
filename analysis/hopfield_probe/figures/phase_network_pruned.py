"""Random whole-number weights, then stochastic pruning (one axis, toy periods).

    python analysis/hopfield_probe/figures/phase_network_pruned.py

Same network as phase_network.py (modules of period 3, 4, 5, L = 60), but each
angle unit's weights j_l are random whole numbers in 0..lambda_l - 1, so its
frequency n is uniform over all 60. Panel a: 12 such units, each with its output
cos(omega_n x) over the arena. Panel b: each unit kept with probability
exp(-r^2 omega_n^2 / 2) (r = 2.5 cells); the survivors are mostly slow.
Writes phase_network_pruned.pdf (core Helvetica/Symbol) and .png here.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

import style
from style import INK, INK_2, rich

style.apply()

LAMS = (3, 4, 5)
L = 3 * 4 * 5
X = 7
U, R, SEED = 12, 2.5, 25
MOD_COL = ("#2a78d6", "#eb6834", "#1baf7a")
KEEP_COL = "#e34948"
LIGHT, FADED = "#c9c8c4", "#e2e1dd"
PITCH, SIDE, GAP = 0.72, 0.64, 0.95
PHX, PR = 3.4, 0.5
ANX, AR = 7.0, 0.26
WX0, WX1, WAMP = 7.6, 11.4, 0.3
OFFSET_B = 14.8

g = np.random.default_rng(SEED)
J = np.stack([g.integers(0, lam, U) for lam in LAMS], 1)
N = (J @ np.array([L // lam for lam in LAMS])) % L
N = np.where(N > L // 2, N - L, N)
KEEP = g.random(U) < np.exp(-R * R * (2 * np.pi * N / L) ** 2 / 2)

fig = plt.figure(figsize=(6.6, 3.1))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(-2.4, OFFSET_B + 11.8)
ax.set_ylim(0.9, 13.5)
ax.set_aspect("equal")
ax.axis("off")


def edge_point(x0, y0, x1, y1, r):
    dx, dy = x0 - x1, y0 - y1
    d = (dx * dx + dy * dy) ** 0.5
    return x1 + r * dx / d, y1 + r * dy / d


def panel(ox, pruned):
    phase_xy, top = [], 11.5
    for ell, (lam, col) in enumerate(zip(LAMS, MOD_COL), start=1):
        ys = [top - a * PITCH for a in range(lam)]
        yc = (ys[0] + ys[-1]) / 2
        phase_xy.append((ox + PHX, yc))
        rich(ax, ox - 0.45, yc, [("λ", "g"), (str(ell), "_"), (" = %d" % lam, "")], ha="right",
             size=7.5, color=col)
        for a, y in enumerate(ys):
            on = a == X % lam
            sx = ox + 0.32 + SIDE / 2
            ex, ey = edge_point(sx, y, ox + PHX, yc, PR)
            ax.plot([sx, ex], [y, ey], color=col if on else LIGHT, lw=1.1 if on else 0.45,
                    zorder=1.5 if on else 1)
            ax.add_patch(Rectangle((ox + 0.32 - SIDE / 2, y - SIDE / 2), SIDE, SIDE,
                                   facecolor=col if on else "white",
                                   edgecolor=col if on else LIGHT, lw=0.6, zorder=2))
        ax.add_patch(Circle((ox + PHX, yc), PR, facecolor="white", edgecolor=col, lw=1.0,
                            zorder=3))
        rich(ax, ox + PHX, yc, [("φ", "g"), (str(ell), "_")], size=7.5)
        top = ys[-1] - PITCH - GAP

    ys = np.linspace(11.45, 1.75, U)
    xs = np.arange(L)
    for u, y in enumerate(ys):
        alive = KEEP[u] or not pruned
        node_col = KEEP_COL if (pruned and KEEP[u]) else (INK_2 if alive else FADED)
        if alive:
            for (x0, y0) in phase_xy:
                sx, sy = edge_point(ox + ANX, y, x0, y0, PR)
                ex, ey = edge_point(x0, y0, ox + ANX, y, AR)
                ax.plot([sx, ex], [sy, ey], color=KEEP_COL if pruned else LIGHT,
                        lw=0.8 if pruned else 0.4, zorder=1)
        ax.add_patch(Circle((ox + ANX, y), AR, facecolor=node_col if (pruned and KEEP[u]) else "white",
                            edgecolor=node_col, lw=0.9, zorder=3))
        wx = ox + WX0 + (WX1 - WX0) * xs / (L - 1)
        ax.plot(wx, y + WAMP * np.cos(2 * np.pi * N[u] * xs / L),
                color=node_col if alive else FADED, lw=0.6 if alive else 0.5, zorder=2)


panel(0.0, pruned=False)
panel(OFFSET_B, pruned=True)

for x, label in ((0.32, "grid cells"), (PHX, "phases"), (ANX, "angle units")):
    rich(ax, x, 12.35, [(label, "")], size=7.5)
rich(ax, (WX0 + WX1) / 2 + 0.3, 12.35, [("cos ", ""), ("ω", "g"), ("n", "i_"), ("x", "i")],
     size=7.5)
rich(ax, -2.3, 13.2, [("a", "b"), ("  random whole-number weights", "")], ha="left", size=8.5)
rich(ax, OFFSET_B - 2.3, 13.2, [("b", "b"), ("  keep each unit with probability ", ""),
                                ("∝", "g"), (" P", "i"), ("(", ""), ("n", "i"), (")", "")],
     ha="left", size=8.5)

style.save(fig, "phase_network_pruned", pad=0.02)
print("saved; n", N.tolist(), "kept", N[KEEP].tolist())
