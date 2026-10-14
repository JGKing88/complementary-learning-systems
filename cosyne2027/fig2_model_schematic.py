"""Figure 2: the model.

Run: ~/.conda/envs/cls/bin/python cosyne2027/fig2_model_schematic.py

Redraws model_schematic.pdf (the hand-made version in this folder) in the
style of fig 1, at 3:1, with words on the arrows and brain regions under
the module names. Box fill gives the module type: frozen (pretrained),
plastic (online Hebbian), fixed (not learned) or environment. The
Vector-HaSH modules do not learn, so they are gray; the Hebbian links
between hippocampus and sensory input are drawn in the plastic color.

Layout: Vector-HaSH on the left; the embedding net and attractor along the
top; the controller under the attractor and the environment under the
controller; the key under the embedding net. The observation leaves the
environment once and splits to sensory input and to the controller.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

import style

style.apply()

# Layout, in inches (the axes spans the figure with 1 data unit = 1 inch)
FIG_W = 5.58
FIG_H = FIG_W / 3
BOX_H = 0.36
ROUND = 0.04
TOP, MID, BOT = 1.43, 0.87, 0.31     # row centers
TITLE_DY, SUB_DY = 0.07, -0.08       # title and subtitle offsets in a box
LABEL_DY = 0.09                      # arrow label above its arrow
ARROW = "-|>,head_length=3.2,head_width=1.7"
ARROW_2 = "<|-|>,head_length=3.2,head_width=1.7"

# (cx, cy, w)
EC = (0.66, TOP, 1.14)
HPC = (0.66, MID, 1.14)
LEC = (0.66, BOT, 1.14)
EMB = (2.5, TOP, 1.1)
ATT = (4.42, TOP, 1.15)
GRU = (4.42, MID, 1.15)
ENV = (4.42, BOT, 1.4)
LANE_L, LANE_R = 4.27, 4.57          # x of the two vertical arrow lanes
JUNCTION_X = 3.3                     # where the observation splits


def edges(b):
    cx, cy, w = b
    return cx - w / 2, cx + w / 2, cy - BOX_H / 2, cy + BOX_H / 2


def box(ax, b, fill, title, sub):
    """Rounded module box with a bold title and a subtitle line."""
    x0, x1, y0, y1 = edges(b)
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle=f"round,pad=0,rounding_size={ROUND}", fc=fill,
        ec=style.BOX_EDGE, lw=0.8, zorder=2))
    style.rich(ax, b[0], b[1] + TITLE_DY, [(title, "b")])
    style.rich(ax, b[0], b[1] + SUB_DY, [(sub, "")] if isinstance(sub, str)
               else sub, color=style.INK_2)


def arrow(ax, p0, p1, style_=ARROW, color=style.INK_2, lw=0.9):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style_, color=color,
                                 lw=lw, shrinkA=0, shrinkB=0, zorder=1))


def line(ax, xs, ys):
    ax.plot(xs, ys, color=style.INK_2, lw=0.9, zorder=1,
            solid_capstyle="butt")


def label(ax, x, y, text, ha="center"):
    style.rich(ax, x, y, [(text, "")], ha=ha)


def draw_vector_hash(ax):
    x0, x1 = 0.02, 1.3
    y0, y1 = edges(LEC)[2] - 0.07, FIG_H - 0.02
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle=f"round,pad=0,rounding_size={ROUND * 2}", fc="#f3f5f7",
        ec=style.INK_MUTED, lw=0.7, ls=(0, (3, 2)), zorder=0))
    style.rich(ax, (x0 + x1) / 2, y1 - 0.11, [("Vector-HaSH", "b")])

    box(ax, EC, style.FIXED_OP, "Entorhinal cortex", "MEC")
    box(ax, HPC, style.FIXED_OP, "Hippocampus", "HPC")
    box(ax, LEC, style.FIXED_OP, "Sensory input", "LEC")
    x = EC[0]
    arrow(ax, (x, edges(HPC)[3]), (x, edges(EC)[2]), ARROW_2)
    arrow(ax, (x, edges(LEC)[3]), (x, edges(HPC)[2]), ARROW_2,
          color=style.PLASTIC_INK, lw=1.2)


def draw_key(ax):
    x, pitch, sw = 1.62, 0.15, 0.13
    entries = [(style.FROZEN, "frozen (pretrained)"),
               (style.PLASTIC, "plastic (Hebbian)"),
               (style.FIXED_OP, "fixed (not learned)"),
               (style.ENV, "environment")]
    for k, (fill, text) in enumerate(entries):
        y = edges(EMB)[2] - 0.2 - k * pitch
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
    box(ax, EMB, style.FROZEN, "Embedding net",
        [("MEC ", ""), ("→", "g"), (" CA1", "")])
    box(ax, ATT, style.PLASTIC, "Attractor net", "CA1")
    box(ax, GRU, style.FROZEN, "GRU controller", "PFC")
    box(ax, ENV, style.ENV, "Environment", "sequential water maze")

    # Along the top: grid code, then embedding
    arrow(ax, (edges(EC)[1], TOP), (edges(EMB)[0], TOP))
    label(ax, (edges(EC)[1] + edges(EMB)[0]) / 2 + 0.03, TOP + LABEL_DY,
          "grid code")
    arrow(ax, (edges(EMB)[1], TOP), (edges(ATT)[0], TOP))
    label(ax, (edges(EMB)[1] + edges(ATT)[0]) / 2, TOP + LABEL_DY,
          "embedding")

    # Attractor <-> controller, controller <-> environment: two lanes each,
    # down on the left, up on the right, labels on the outside
    for upper, lower, down, up in ((ATT, GRU, "recall direction",
                                    "store embedding"),
                                   (GRU, ENV, "movement", "reward")):
        y_hi, y_lo = edges(upper)[2], edges(lower)[3]
        arrow(ax, (LANE_L, y_hi), (LANE_L, y_lo))
        arrow(ax, (LANE_R, y_lo), (LANE_R, y_hi))
        label(ax, LANE_L - 0.05, (y_hi + y_lo) / 2, down, ha="right")
        label(ax, LANE_R + 0.05, (y_hi + y_lo) / 2, up, ha="left")

    # Observation: one arrow out of the environment, split to sensory input
    # and to the controller
    line(ax, [edges(ENV)[0], JUNCTION_X], [BOT, BOT])
    arrow(ax, (JUNCTION_X, BOT), (edges(LEC)[1], BOT))
    line(ax, [JUNCTION_X, JUNCTION_X], [BOT, MID])
    arrow(ax, (JUNCTION_X, MID), (edges(GRU)[0], MID))
    ax.add_patch(Circle((JUNCTION_X, BOT), 0.022, fc=style.INK_2, ec="none",
                        zorder=2))
    label(ax, (edges(LEC)[1] + JUNCTION_X) / 2, BOT + LABEL_DY,
          "observation")

    draw_key(ax)
    style.save(fig, "fig2_model_schematic")


if __name__ == "__main__":
    main()
