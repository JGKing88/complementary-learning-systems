"""Figure 4: Agent-HaSH represents explore versus exploit abstractly.

Run: ~/.conda/envs/cls/bin/python cosyne2027/fig4_abstraction.py
Reads cosyne2027/data/abstraction.json, written by
analysis/phase_decoding/cosyne_abstraction_data.py on branch
worktree-phase-decoding-audit (method and every number:
docs/PHASE_ABSTRACTION.md there).

Data: d0_base u725 in 100 arenas it never trained in. Explore = goal not in
memory, exploit = goal in memory; time since the trial began and distance to
goal matched per arena, so behaviour does not separate the two.

a: geometry. Each line is one arena, from its mean state while exploring
   (amber open circle) to its mean state while exploiting (green dot), projected onto the
   top two principal components of the 200 condition means. Left: the
   recurrent state; right: the policy's input. Parallelism = mean cosine
   between the arenas' explore-to-exploit vectors in the full space
   (shuffle 95%: 0.01).
b: cross-arena decoding. A linear decoder trained on k arenas, tested on
   held-out arenas; mean +-SD over 10 draws. Gray band: geometric random
   model null (Bernardi et al. 2020), up to the highest 95th percentile of
   the three nulls.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, NullLocator

import style

style.apply()

DATA = style.HERE / "data" / "abstraction.json"
FIG_W, FIG_H = 5.58, 1.75
TRAINED = style.MODEL_COLORS["Agent-HaSH"]
CONTROL = style.OTHER_MODEL
PT = style.FONT_PT
# Phase colors: dataviz reference palette slots 4 (yellow) and 6 (green), so
# neither is an arena or model color. OKLab dE x100: 30 normal; 16 / 25 / 33
# under protan / deutan / tritan simulation (Machado 2009). Explore points are
# also open and exploit points filled, so phase never rests on color alone.
EXPLORE = "#eda100"
EXPLOIT = "#008300"

# Axes rectangles in inches: (left, bottom, width, height)
GEO_H_AX = (0.32, 0.32, 1.10, 1.08)
GEO_X_AX = (1.92, 0.32, 1.10, 1.08)
DEC_AX = (3.74, 0.40, 1.50, 1.24)


def add_axes(fig, rect):
    l, b, w, h = rect
    return fig.add_axes([l / FIG_W, b / FIG_H, w / FIG_W, h / FIG_H])


def panel_geometry(ax, geo, line_color, title, ps, key):
    p0 = np.asarray(geo["explore"])
    p1 = np.asarray(geo["exploit"])
    for a, b in zip(p0, p1):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=line_color, lw=0.4,
                alpha=0.45, solid_capstyle="butt", zorder=1)
    ax.scatter(p0[:, 0], p0[:, 1], s=4, facecolor="white", edgecolor=EXPLORE,
               lw=0.6, zorder=3)
    ax.scatter(p1[:, 0], p1[:, 1], s=3, color=EXPLOIT, lw=0, zorder=3)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("PC1", labelpad=2)
    ax.set_ylabel("PC2", labelpad=2)
    # Two title lines: the model, then its parallelism score.
    ax.annotate(f"parallelism {ps:.2f}", (0.5, 1.0), xycoords="axes fraction",
                xytext=(0, 2), textcoords="offset points", ha="center",
                va="bottom", fontsize=PT, color=style.INK_2)
    ax.annotate(title, (0.5, 1.0), xycoords="axes fraction", xytext=(0, 12),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=PT)
    if key:
        # Colored mark, ink label.
        for y, face, edge, label in ((0.93, "white", EXPLORE, "explore"),
                                     (0.79, EXPLOIT, EXPLOIT, "exploit")):
            ax.scatter([0.05], [y], s=9, facecolor=face, edgecolor=edge,
                       lw=0.8, transform=ax.transAxes, zorder=4, clip_on=False)
            ax.text(0.12, y, label, transform=ax.transAxes, ha="left",
                    va="center", fontsize=PT, color=style.INK_2)


def panel_decoding(ax, dec):
    k = np.asarray(dec["trained"]["k"])
    null_top = np.max([dec[s]["null_q95"] for s in dec], axis=0)
    ax.fill_between(k, 0.5, null_top, color=style.tint(CONTROL, 0.6), lw=0,
                    zorder=0)
    series = {"trained": (TRAINED, "-", "o"),
              "input": (CONTROL, "-", "s"),
              "untrained": (CONTROL, (0, (2.5, 1.5)), "^")}
    for name, (col, ls, mk) in series.items():
        m = np.asarray(dec[name]["mean"])
        sd = np.asarray(dec[name]["sd"])
        ax.errorbar(k, m, yerr=sd, color=col, ls=ls, lw=1.0, marker=mk, ms=2.5,
                    mew=0, elinewidth=0.6, capsize=0, zorder=3)
    # Direct labels. The two controls end 0.01 apart, so they split vertically.
    m = {s: dec[s]["mean"][-1] for s in dec}
    ax.text(6, 0.815, "Agent-HaSH", ha="center", va="top", fontsize=PT)
    ax.annotate("untrained", (k[-1], m["untrained"]), xytext=(4, 4),
                textcoords="offset points", va="center", fontsize=PT)
    ax.annotate("input", (k[-1], m["input"]), xytext=(4, -5),
                textcoords="offset points", va="center", fontsize=PT)
    ax.annotate("null", (k[-1], 0.52), xytext=(4, 0), textcoords="offset points",
                va="center", fontsize=PT, color=style.INK_2)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([1, 10, 50]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["1", "10", "50"])
    ax.set_xlim(0.8, 62)
    ax.set_ylim(0.45, 1.0)
    ax.set_yticks([0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.5", "0.75", "1"])
    ax.set_xlabel("training arenas", labelpad=1)
    ax.set_ylabel("held-out decoding", labelpad=1)


def main():
    d = json.loads(DATA.read_text())
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    panel_geometry(add_axes(fig, GEO_H_AX), d["geometry"]["h"], style.INK_2,
                   "Agent-HaSH", d["decoding"]["trained"]["parallelism"], key=True)
    panel_geometry(add_axes(fig, GEO_X_AX), d["geometry"]["x"], style.INK_MUTED,
                   "input", d["decoding"]["input"]["parallelism"], key=False)
    panel_decoding(add_axes(fig, DEC_AX), d["decoding"])
    style.save(fig, "fig4_abstraction")


if __name__ == "__main__":
    main()
