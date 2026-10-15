"""Agent-HaSH alone at 20 arenas.

Run: ~/.conda/envs/cls/bin/python cosyne2027/fig_agenthash_n20.py
Reads the Agent-HaSH entry of cosyne2027/data/cl_results_n20.json
(cl_results.py --n_envs 20; it skips control families that have not
finished, so this figure does not wait for them).

a: success on each arena against update, one line per arena from its
   introduction on, mean over 8 seeds, smoothed within each 200-update
   window. Twenty arenas are too many for categorical colours, so a line's
   shade gives its arena's order (light = early, dark = late).
b: the success matrix a[k, j] (arena j, after training through arena k;
   last 20% of each window), the matrix P and ACC are computed from.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import style
from fig3_a2_per_env import WINDOW, smooth_within_windows

style.apply()

DATA = style.HERE / "data" / "cl_results_n20.json"
FIG_W, FIG_H = 5.58, 1.95
ORDER = LinearSegmentedColormap.from_list(
    "arena_order", [style.tint(style.ARENA_COLORS[0], 0.65),
                    style.ARENA_COLORS[0], "#0f2f5c"])


def panel_curves(ax, r):
    curves = np.array(r["env_curves"], dtype=float)
    n, n_env = curves.shape
    x = np.arange(1, n + 1)
    for b in range(WINDOW, n, WINDOW):
        ax.axvline(b + 0.5, color=style.GRID, lw=0.4, zorder=0)
    for j in range(n_env):
        y = smooth_within_windows(curves[:, j], j * WINDOW)
        ax.plot(x, y, color=ORDER(j / (n_env - 1)), lw=0.8, zorder=2)
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1.04)
    ax.set_xticks(np.arange(0, n + 1, 1000))
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("Update")
    ax.set_ylabel("Success, each arena")
    # Order key: a small gradient bar with its two ends labelled
    kax = ax.inset_axes([0.62, 0.12, 0.3, 0.05])
    kax.imshow(np.linspace(0, 1, 100)[None], aspect="auto", cmap=ORDER)
    kax.set_xticks([]); kax.set_yticks([])
    for sp in kax.spines.values():
        sp.set_visible(False)
    kax.text(-0.04, 0.5, "arena 1", transform=kax.transAxes, ha="right",
             va="center")
    kax.text(1.04, 0.5, str(n_env), transform=kax.transAxes, ha="left",
             va="center")


def panel_matrix(ax, r):
    a = np.array([[np.nan if v is None else v for v in row]
                  for row in r["matrix"]], dtype=float)
    n = a.shape[0]
    im = ax.imshow(a, cmap="viridis", vmin=0, vmax=1, origin="upper",
                   interpolation="nearest")
    ticks = [0, n - 1]
    ax.set_xticks(ticks, [1, n])
    ax.set_yticks(ticks, [1, n])
    ax.set_xlabel("Arena tested")
    ax.set_ylabel("After arena")
    for sp in ax.spines.values():
        sp.set_visible(False)
    cax = ax.inset_axes([1.06, 0.0, 0.05, 1.0])
    cb = plt.colorbar(im, cax=cax, ticks=[0, 1])
    cb.outline.set_visible(False)
    cb.set_label("Success", labelpad=-4)
    ax.text(0.97, 0.97, f"P = {r['P']:.2f}\nACC = {r['ACC']:.2f}",
            transform=ax.transAxes, ha="right", va="top", color=style.INK)


def main():
    r = json.load(open(DATA))["models"]["Agent-HaSH"]
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax1 = fig.add_axes([0.085, 0.21, 0.5, 0.72])
    ax2 = fig.add_axes([0.7, 0.21, 0.2, 0.72])
    panel_curves(ax1, r)
    panel_matrix(ax2, r)
    style.save(fig, "fig_agenthash_n20")


if __name__ == "__main__":
    main()
