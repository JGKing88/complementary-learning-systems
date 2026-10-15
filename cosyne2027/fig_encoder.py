"""Encoder figure: similarity against distance.

Run: ~/.conda/envs/cls/bin/python cosyne2027/fig_encoder.py
Reads cosyne2027/data/encoder_similarity.json (encoder_similarity.py).

Main plot: raw cosine similarity between two positions' representations
against their Euclidean distance, for the trained encoder (thick, the
embedding net's hue in fig 2), the grid code it reads (medium) and an
untrained encoder of the same architecture (thin, gray, dashed). Bands are
+-1 SD over the 2000 position pairs at each distance. Triangles mark the grid
periods (11, 12, 13 cells).
"""
import json

import matplotlib.pyplot as plt
import numpy as np

import style

style.apply()

DATA = style.HERE / "data" / "encoder_similarity.json"
FIG_W, FIG_H = 2.72, 1.78
AX = [0.43, 0.3, 2.2, 1.4]           # axes: left, bottom, w, h (in)
CURVES = {  # name: (color, lw, linestyle)
    "trained encoder": ("#a35f00", 2.0, "-"),
    "grid code": ("#3d5a9e", 1.2, "-"),
    "untrained encoder": ("#b5b4af", 0.9, (0, (3, 2))),
}
# Label anchors in data coordinates: (x, y, ha)
LABEL_AT = {"trained encoder": (14, 0.78, "left"),
            "grid code": (100, 0.33, "right"),
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

    style.save(fig, "fig_encoder")


if __name__ == "__main__":
    main()
