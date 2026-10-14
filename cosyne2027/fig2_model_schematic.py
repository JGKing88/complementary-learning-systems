"""Figure 2: the model.

Run: ~/.conda/envs/cls/bin/python cosyne2027/fig2_model_schematic.py

Redraws model_schematic.pdf (the hand-made version in this folder) in the
style of fig 1, without the small network drawings, at 3:1. Box fill gives
the module type: frozen (pretrained), plastic (online Hebbian), fixed
operation (no weights) or environment.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

import style

style.apply()


def fx(f):
    """Runs for f(x), with italic letters and upright parentheses."""
    return [(f, "i"), ("(", ""), ("x", "i"), (")", "")]

# Layout, in inches (the axes spans the figure with 1 data unit = 1 inch)
FIG_W = 6.5
FIG_H = FIG_W / 3
BOX_H = 0.4
ROUND = 0.04
TOP, MID, BOT = 1.68, 1.02, 0.36     # row centers
TITLE_DY, SUB_DY = 0.075, -0.085     # title and subtitle offsets in a box
ARROW = "-|>,head_length=3.2,head_width=1.7"
ARROW_2 = "<|-|>,head_length=3.2,head_width=1.7"

# (cx, cy, w)
EC = (0.7, TOP, 1.22)
HPC = (0.7, MID, 1.22)
LEC = (0.7, BOT, 1.22)
EMB = (2.2, TOP, 1.1)
ATT = (3.72, TOP, 1.1)
DEC = (2.95, MID, 1.5)
GRU = (5.62, 1.13, 1.6)
ENV = (5.68, BOT, 1.48)


def edges(b):
    cx, cy, w = b
    return cx - w / 2, cx + w / 2, cy - BOX_H / 2, cy + BOX_H / 2


def box(ax, b, fill, title, sub, dashed=False, lower_fill=None):
    """Rounded module box with a bold title and a subtitle line.

    ``lower_fill`` paints the bottom half a second color, for a module that
    is part frozen and part plastic.
    """
    x0, x1, y0, y1 = edges(b)
    patch = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                           boxstyle=f"round,pad=0,rounding_size={ROUND}",
                           fc=fill, ec="none", zorder=2)
    ax.add_patch(patch)
    if lower_fill is not None:
        lower = Rectangle((x0, y0), x1 - x0, (y1 - y0) * 0.45,
                          fc=lower_fill, ec="none", zorder=2)
        ax.add_patch(lower)
        lower.set_clip_path(patch)
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle=f"round,pad=0,rounding_size={ROUND}", fc="none",
        ec=style.BOX_EDGE, lw=0.8, ls=(0, (3, 1.5)) if dashed else "-",
        zorder=3))
    cx, cy = b[0], b[1]
    style.rich(ax, cx, cy + TITLE_DY, [(title, "b")])
    style.rich(ax, cx, cy + SUB_DY, [(sub, "")] if isinstance(sub, str)
               else sub, color=style.INK_2)


def arrow(ax, p0, p1, style_=ARROW, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=style_, color=style.INK_2, lw=0.9,
        shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}", zorder=1))


def label(ax, x, y, runs, ha="center"):
    style.rich(ax, x, y, runs, ha=ha)


def draw_vector_hash(ax):
    x0, x1 = 0.02, 1.38
    y0, y1 = BOT - BOX_H / 2 - 0.08, FIG_H - 0.04
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle=f"round,pad=0,rounding_size={ROUND * 2}", fc="#f3f5f7",
        ec=style.INK_MUTED, lw=0.7, ls=(0, (3, 2)), zorder=0))
    style.rich(ax, (x0 + x1) / 2, y1 - 0.1, [("Vector-HaSH", "b")])

    box(ax, EC, style.FIXED_OP, "Entorhinal cortex", [("MEC  ", "")] + fx("g"))
    box(ax, HPC, style.FIXED_OP, "Hippocampus", [("HPC  ", ""), ("p", "i")],
        lower_fill=style.PLASTIC)
    box(ax, LEC, style.PLASTIC, "Sensory input", [("LEC  ", ""), ("s", "i")])
    for upper, lower in ((EC, HPC), (HPC, LEC)):
        arrow(ax, (upper[0], edges(lower)[3]), (upper[0], edges(upper)[2]),
              ARROW_2)


def draw_key(ax):
    x, pitch, sw = 4.86, 0.15, 0.13
    entries = [(style.FROZEN, "frozen (pretrained)"),
               (style.PLASTIC, "plastic (online Hebbian)"),
               (style.FIXED_OP, "fixed operation"),
               (style.ENV, "environment")]
    for k, (fill, text) in enumerate(entries):
        y = FIG_H - 0.1 - k * pitch
        ax.add_patch(FancyBboxPatch(
            (x, y - sw / 2 + 0.01), sw, sw - 0.02,
            boxstyle="round,pad=0,rounding_size=0.02", fc=fill,
            ec=style.BOX_EDGE, lw=0.6))
        style.rich(ax, x + sw + 0.07, y, [(text, "")], ha="left",
                   color=style.INK_2)


def main():
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.set_aspect("equal")
    ax.axis("off")

    draw_vector_hash(ax)
    box(ax, EMB, style.FROZEN, "Embedding net", [("f", "i"), ("θ", "g_")])
    box(ax, ATT, style.PLASTIC, "Attractor net", [("W", "i")])
    box(ax, DEC, style.FIXED_OP, "Direction decoder",
        [("q", "i"), (" = ", ""), ("J", "i"), ("x", "i_"), ("T", "^k"),
         ("(", ""), ("z", "i"), ("*", ""), (" ", ""), ("−", "g"), (" ", "")]
        + fx("z") + [(")", "")], dashed=True)
    box(ax, GRU, style.FROZEN, "GRU controller",
        [("heads  ", ""), ("π", "g"), ("move", "_"), (", ", ""),
         ("π", "g"), ("store", "_"), (", ", ""), ("V", "i")])
    box(ax, ENV, style.ENV, "Environment", "sequential water maze")

    # g(x): entorhinal cortex -> embedding net
    arrow(ax, (edges(EC)[1], TOP), (edges(EMB)[0], TOP))
    label(ax, (edges(EC)[1] + edges(EMB)[0]) / 2 + 0.03, TOP + 0.1,
          fx("g"))
    # z(x): embedding net -> attractor net, and down to the decoder
    arrow(ax, (edges(EMB)[1], TOP), (edges(ATT)[0], TOP))
    label(ax, (edges(EMB)[1] + edges(ATT)[0]) / 2, TOP + 0.1, fx("z"))
    arrow(ax, (EMB[0], edges(EMB)[2]), (EMB[0], edges(DEC)[3]))
    label(ax, EMB[0] - 0.05, (edges(EMB)[2] + edges(DEC)[3]) / 2, fx("z"),
          ha="right")
    # z*: attractor net -> decoder
    arrow(ax, (ATT[0] - 0.1, edges(ATT)[2]), (DEC[0] + 0.5, edges(DEC)[3]))
    label(ax, ATT[0] - 0.02, (edges(ATT)[2] + edges(DEC)[3]) / 2,
          [("z", "i"), ("*", "")], ha="left")
    # q: decoder -> GRU
    arrow(ax, (edges(DEC)[1], MID), (edges(GRU)[0], MID))
    label(ax, (edges(DEC)[1] + edges(GRU)[0]) / 2, MID + 0.1, [("q", "i")])
    # store: GRU -> attractor net
    arrow(ax, (edges(GRU)[0] + 0.12, edges(GRU)[3]),
          (edges(ATT)[1], TOP - 0.06))
    label(ax, edges(GRU)[0] + 0.2, edges(GRU)[3] + 0.1,
          [("store ", "")] + fx("z"),
          ha="left")
    style.rich(ax, DEC[0], edges(DEC)[2] - 0.12,
               [("local tangent projection, no learned weights", "i")],
               color=style.INK_2)

    # GRU <-> environment
    gy0, ey1 = edges(GRU)[2], edges(ENV)[3]
    xs = [5.0, 5.5, 5.9]
    arrow(ax, (xs[0], gy0), (xs[0], ey1))
    label(ax, xs[0] - 0.05, (gy0 + ey1) / 2, [("movement", "")], ha="right")
    arrow(ax, (xs[1], ey1), (xs[1], gy0))
    label(ax, xs[1] + 0.05, (gy0 + ey1) / 2, [("o", "i"), ("t", "i_")], ha="left")
    arrow(ax, (xs[2], ey1), (xs[2], gy0))
    label(ax, xs[2] + 0.05, (gy0 + ey1) / 2, [("reward ", ""), ("r", "i"), ("t", "i_")],
          ha="left")
    # s: environment -> sensory input
    arrow(ax, (edges(ENV)[0], BOT), (edges(LEC)[1], BOT))
    label(ax, (edges(LEC)[1] + edges(ENV)[0]) / 2, BOT + 0.1, [("s", "i")])

    draw_key(ax)
    style.save(fig, "fig2_model_schematic")


if __name__ == "__main__":
    main()
