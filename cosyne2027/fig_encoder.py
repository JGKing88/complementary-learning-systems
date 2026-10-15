"""Encoder figure: similarity against distance.

Run: ~/.conda/envs/cls/bin/python cosyne2027/fig_encoder.py
Reads cosyne2027/data/encoder_similarity.json (encoder_similarity.py).

Main plot: raw cosine similarity between two positions' representations
against their axis-aligned distance, for the trained encoder (thick, the
embedding net's hue in fig 2), the grid code it reads (medium) and an
untrained encoder of the same architecture (thin, gray, dashed). Bands are
+-1 SD over the 2000 position pairs at each distance; the grid code is
translation-invariant, so its curve is exact and has no band. Triangles mark
the grid periods (11, 12, 13 cells).

Side panel: six trained units over a 200 x 200-cell window, processed as in
encoder_training/notebooks/explore.ipynb (mean-subtract, ReLU, sigma-30
high-pass, sigma-2 blur) plus a final ReLU at 0.6 of each map's peak (see
encoder_similarity.unit_maps); viridis, each map normalised to its own peak,
with a 50-cell scale bar. It sits beside the plot rather than
inside it: along an axis the grid code oscillates up to ~0.65 at every
distance, so no empty region inside the axes is tall enough.
"""
import json

import matplotlib.pyplot as plt
import numpy as np

import style

style.apply()

DATA = style.HERE / "data" / "encoder_similarity.json"
FIG_W, FIG_H = 3.73, 1.78
AX = [0.43, 0.3, 2.2, 1.4]           # axes: left, bottom, w, h (in)
MAPS = style.HERE / "data" / "encoder_unit_maps.npz"
MM = 1 / 25.4
MAP, MAP_GAP = 7 * MM, 0.9 * MM      # map side and gap (in)
PANEL_X, PANEL_TOP = 2.8, 1.58       # side panel: left edge, top of maps (in)
SCALE_CELLS = 50
CURVES = {  # name: (color, lw, linestyle)
    "trained encoder": ("#a35f00", 2.0, "-"),
    "grid code": ("#3d5a9e", 1.2, "-"),
    "untrained encoder": ("#b5b4af", 0.9, (0, (3, 2))),
}
# Label anchors in data coordinates: (x, y, ha)
LABEL_AT = {"trained encoder": (98, 0.1, "right"),
            "grid code": (14.5, 0.92, "left"),
            "untrained encoder": (100, 0.91, "right")}


def main():
    d = json.load(open(DATA))
    x = np.array(d["distances"])
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes([AX[0] / FIG_W, AX[1] / FIG_H, AX[2] / FIG_W,
                       AX[3] / FIG_H])

    for z, (name, (color, lw, ls)) in enumerate(CURVES.items()):
        m = np.array(d["curves"][name]["mean"])
        sd = np.array(d["curves"][name]["sd"])
        ax.fill_between(x, np.clip(m - sd, 0, 1), np.clip(m + sd, 0, 1),
                        color=color, alpha=0.15, lw=0, zorder=1)
        ax.plot(x, m, color=color, lw=lw, ls=ls, zorder=4 - z,
                solid_capstyle="butt")
        lx, ly, ha = LABEL_AT[name]
        ax.text(lx, ly, name, ha=ha, va="center", color=style.INK, zorder=5)

    # Grid periods: triangles on the x axis, labelled once
    for lam in d["lambdas"]:
        ax.plot(lam, 0.012, marker="^", ms=3.5, color=style.INK_2,
                mec="none", clip_on=False, zorder=6)
    style.rich(ax, np.mean(d["lambdas"]), 0.085, [("λ", "g")])

    ax.set_xlim(0, x[-1])
    ax.set_ylim(0, 1.03)
    ax.set_xticks([0, 50, 100])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("Distance (cells)")
    ax.set_ylabel("Cosine similarity")

    unit_panel(fig)
    style.save(fig, "fig_encoder")


def unit_panel(fig):
    data = np.load(MAPS)
    maps, window = data["maps"], int(data["window"])
    for k, m in enumerate(maps):
        row, col = divmod(k, 3)
        x = PANEL_X + col * (MAP + MAP_GAP)
        y = PANEL_TOP - (row + 1) * MAP - row * MAP_GAP
        a = fig.add_axes([x / FIG_W, y / FIG_H, MAP / FIG_W, MAP / FIG_H])
        # maps are [x][y]; show x across and y up
        # Each map normalised to its own peak, as rate maps usually are
        a.imshow(m.T / m.max(), origin="lower", cmap="viridis", vmin=0,
                 vmax=1, interpolation="bilinear")
        a.set_xticks([])
        a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.4)
            sp.set_color(style.INK_2)
    width = 3 * MAP + 2 * MAP_GAP
    fig.text((PANEL_X + width / 2) / FIG_W, (PANEL_TOP + 0.03) / FIG_H,
             "example units", ha="center", va="bottom", fontsize=8,
             color=style.INK_2)
    # Scale bar under the bottom-left map, its label on the same line
    bar = SCALE_CELLS / window * MAP
    yb = PANEL_TOP - 2 * MAP - MAP_GAP - 0.08
    fig.add_artist(plt.Line2D([PANEL_X / FIG_W, (PANEL_X + bar) / FIG_W],
                              [yb / FIG_H] * 2, color=style.INK, lw=1.2,
                              solid_capstyle="butt"))
    fig.text((PANEL_X + bar + 0.04) / FIG_W, yb / FIG_H,
             f"{SCALE_CELLS} cells", ha="left", va="center", fontsize=8,
             color=style.INK_2)


if __name__ == "__main__":
    main()
