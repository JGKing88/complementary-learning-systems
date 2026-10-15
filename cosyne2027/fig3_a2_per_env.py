"""Figure 3 a2, per-arena version: success on each arena against update.

Run: ~/.conda/envs/cls/bin/python cosyne2027/fig3_a2_per_env.py
Reads cosyne2027/data/cl_results.json (cl_results.py writes it).

Small multiples, one per model (Agent-HaSH, naive SGD, XdG + SI), shared y
axis. One line per arena in the arena colors of fig 1, starting when the
arena is introduced; mean over 8 seeds, smoothed within each 200-update
window (the smoothing never mixes two windows) and joined across window
boundaries, so a drop at a boundary shows as a step. Dashed lines mark the
boundaries. Each panel is 35 x 25 mm.
"""
import json

import matplotlib.pyplot as plt
import numpy as np

import style

style.apply()

DATA = style.HERE / "data" / "cl_results.json"
MODELS = ["Agent-HaSH", "Naive SGD", "XdG + SI"]
NAMES = "ABCDE"
SMOOTH = 20                     # updates in the moving average
WINDOW = 200                    # updates per arena

MM = 1 / 25.4
PANEL_W, PANEL_H = 35 * MM, 25 * MM
LEFT, GAP, KEY_W = 0.42, 0.2, 0.62
BOTTOM, TOP = 0.36, 0.2
FIG_W = LEFT + 3 * PANEL_W + 2 * GAP + KEY_W
FIG_H = BOTTOM + PANEL_H + TOP


def smooth_within_windows(y, start):
    """Moving average inside each window from ``start`` on; NaN before it."""
    out = np.full_like(y, np.nan)
    k = np.ones(SMOOTH)
    for w0 in range(start, len(y), WINDOW):
        seg = y[w0:w0 + WINDOW]
        out[w0:w0 + WINDOW] = (np.convolve(seg, k, mode="same")
                               / np.convolve(np.ones_like(seg), k, mode="same"))
    return out


def panel(ax, curves, title, first):
    n, n_env = curves.shape
    x = np.arange(1, n + 1)
    for b in range(WINDOW, n, WINDOW):
        ax.axvline(b + 0.5, color=style.INK_MUTED, lw=0.5, ls=(0, (3, 2)),
                   zorder=0)
    # Later arenas first, so arena A is drawn on top where lines coincide
    for j in reversed(range(n_env)):
        y = smooth_within_windows(curves[:, j], j * WINDOW)
        ax.plot(x, y, color=style.ARENA_COLORS[j], lw=1.0, zorder=2)
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1.04)
    ax.set_xticks([0, n])
    # 0 starts at its tick and n ends at its tick, so labels of
    # neighbouring panels do not run together
    lo, hi = ax.get_xticklabels()
    lo.set_horizontalalignment("left")
    hi.set_horizontalalignment("right")
    ax.set_yticks([0, 1])
    ax.set_title(title, fontsize=style.FONT_PT, pad=3)
    if first:
        ax.set_ylabel("Success")
    else:
        ax.tick_params(labelleft=False)


def key(fig):
    x0 = (LEFT + 3 * PANEL_W + 2 * GAP + 0.12) / FIG_W
    ax = fig.add_axes([x0, BOTTOM / FIG_H, 1 - x0, PANEL_H / FIG_H])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0, 1.0, "Arena", va="center", color=style.INK)
    for j, name in enumerate(NAMES):
        y = 0.8 - j * 0.19
        ax.plot([0, 0.3], [y, y], color=style.ARENA_COLORS[j], lw=1.5)
        ax.text(0.42, y, name, va="center", color=style.INK)


def main():
    models = json.load(open(DATA))["models"]
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    axes = []
    for i, name in enumerate(MODELS):
        x0 = (LEFT + i * (PANEL_W + GAP)) / FIG_W
        ax = fig.add_axes([x0, BOTTOM / FIG_H, PANEL_W / FIG_W,
                           PANEL_H / FIG_H])
        curves = np.array(models[name]["env_curves"], dtype=float)
        panel(ax, curves, name, first=(i == 0))
        axes.append(ax)
    axes[1].set_xlabel("Update", labelpad=1)
    key(fig)
    style.save(fig, "fig3_a2_per_env")


if __name__ == "__main__":
    main()
