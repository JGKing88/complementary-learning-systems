"""Figure 3: main results, continual learning across 5 arenas.

Run: ~/.conda/envs/cls/bin/python cosyne2027/fig3_main_results.py
Reads cosyne2027/data/cl_results.json (cl_results.py writes it).

a1: plasticity P against final success ACC for every model, with y = x.
    On the diagonal is no forgetting; the drop below it is the amount
    forgotten. Error bars are +-SEM over 8 seeds on both axes.
a2: success on all arenas seen so far against update, for Agent-HaSH, naive
    SGD and the strongest control (XdG + SI), mean +-SEM over 8 seeds.
    Smoothed within each arena's window, never across a boundary; dashed
    lines mark the window boundaries.
"""
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

import style

style.apply()

DATA = {5: style.HERE / "data" / "cl_results.json",
        20: style.HERE / "data" / "cl_results_n20.json"}
FIG_W, FIG_H = 5.58, 1.95
CURVES = ["Agent-HaSH", "XdG + SI", "Naive SGD"]
SMOOTH = 20                     # updates in the moving average
WINDOW = 200                    # updates per arena
SHORT = {"Agent-HaSH": "Agent-HaSH", "Naive SGD": "Naive SGD",
         "Experience Replay": "Replay", "CLEAR": "CLEAR",
         "Online EWC": "Online EWC", "XdG + SI": "XdG + SI",
         "Multi-head oracle": "Multi-head"}
# Label offsets in points from each a1 point: (dx, dy, ha)
LABEL_AT = {"Agent-HaSH": (-8, -3, "right"), "XdG + SI": (-9, 0, "right"),
            "Experience Replay": (8, 0, "left"), "CLEAR": (-9, 0, "right"),
            "Multi-head oracle": (8, 0, "left"), "Naive SGD": (0, -10, "center"),
            "Online EWC": (-8, 2, "right")}
# At N=20 the controls cluster low and left, so they need their own places
# (and EWC its short name, beside the y axis)
LABEL_AT_N20 = {"Agent-HaSH": (-8, -3, "right"), "XdG + SI": (-9, 0, "right"),
                "Experience Replay": (0, 8, "center"), "CLEAR": (8, 3, "left"),
                "Multi-head oracle": (8, -2, "left"),
                "Naive SGD": (-8, -2, "right"), "Online EWC": (-8, 1, "right")}
SHORT_N20 = {**SHORT, "Online EWC": "EWC"}


def color(name):
    return style.MODEL_COLORS.get(name, style.OTHER_MODEL)


def smooth_within_windows(y):
    """Moving average inside each window; edges use the points available."""
    out = np.empty_like(y)
    for w0 in range(0, len(y), WINDOW):
        seg = y[w0:w0 + WINDOW]
        k = np.ones(SMOOTH)
        num = np.convolve(seg, k, mode="same")
        den = np.convolve(np.ones_like(seg), k, mode="same")
        out[w0:w0 + WINDOW] = num / den
    return out


def panel_a1(ax, models, n_envs=5):
    label_at, short = ((LABEL_AT, SHORT) if n_envs == 5
                       else (LABEL_AT_N20, SHORT_N20))
    ax.plot([0, 1], [0, 1], color=style.INK_MUTED, lw=0.7,
            ls=(0, (3, 2)), zorder=0)
    # Key for the diagonal, in the empty upper-left corner: the label is
    # too long to run along the line without crossing a point's label.
    ax.plot([0.03, 0.13], [0.86, 0.86], color=style.INK_MUTED, lw=0.7,
            ls=(0, (3, 2)))
    ax.text(0.16, 0.86, "no forgetting", va="center", color=style.INK_2)
    # Featured models on top
    order = sorted(models, key=lambda m: m in style.MODEL_COLORS)
    for name in order:
        r = models[name]
        c = color(name)
        ax.errorbar(r["P"], r["ACC"], xerr=r["P_sem"], yerr=r["ACC_sem"],
                    fmt="o", ms=4.5, color=c, mec="white", mew=0.6,
                    elinewidth=0.7, capsize=0, zorder=3)
        dx, dy, ha = label_at.get(name, (5, 0, "left"))
        ax.annotate(short[name], (r["P"], r["ACC"]), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va="center",
                    color=style.INK, zorder=4,
                    # the diagonal passes behind, not through, a label
                    bbox=dict(fc="white", ec="none", pad=0.2))
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.set_xlabel("Plasticity  P")
    ax.set_ylabel("Final success  ACC")


def panel_a2(ax, models):
    n = len(next(models[m] for m in CURVES if m in models)["seen_curve"])
    x = np.arange(1, n + 1)
    for b in range(WINDOW, n, WINDOW):
        ax.axvline(b + 0.5, color=style.INK_MUTED, lw=0.6, ls=(0, (3, 2)),
                   zorder=0)
    for name in CURVES:
        if name not in models:
            continue
        r = models[name]
        y = smooth_within_windows(np.asarray(r["seen_curve"]))
        e = smooth_within_windows(np.asarray(r["seen_curve_sem"]))
        c = color(name)
        for w0 in range(0, n, WINDOW):          # no line across a boundary
            sl = slice(w0, w0 + WINDOW)
            ax.fill_between(x[sl], (y - e)[sl], (y + e)[sl], color=c,
                            alpha=0.15, lw=0, zorder=1)
            ax.plot(x[sl], y[sl], color=c, lw=1.3, zorder=2)
        ax.annotate(SHORT[name], (n, min(y[-1], 0.96)), xytext=(4, 0),
                    textcoords="offset points", va="center", color=style.INK)
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1.02)
    # A tick per window at N=5; every 1000 updates when windows are many
    ax.xaxis.set_major_locator(MultipleLocator(WINDOW if n <= 5 * WINDOW
                                               else 1000))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.set_xlabel("Update")
    ax.set_ylabel("Success, seen arenas")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_envs", type=int, default=5, choices=sorted(DATA))
    args = p.parse_args()
    models = json.load(open(DATA[args.n_envs]))["models"]
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax1 = fig.add_axes([0.085, 0.21, 0.27, 0.75])
    ax2 = fig.add_axes([0.5, 0.21, 0.36, 0.75])
    panel_a1(ax1, models, args.n_envs)
    panel_a2(ax2, models)
    style.save(fig, "fig3_main_results" + ("" if args.n_envs == 5
                                           else f"_n{args.n_envs}"))


if __name__ == "__main__":
    main()
