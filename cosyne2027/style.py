"""Shared style for the Cosyne 2027 figures.

Every figure in this folder takes its colors and fonts from here, so an arena
keeps the same color from the task schematic through the results panels.

Arena colors are the first three slots of the dataviz reference palette
(blue, orange, aqua). Those three are the only slots that stay
distinguishable in every pair under colorblind simulation; a fourth arena
should get a letter label and a neutral gray, not a fourth hue.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors

HERE = Path(__file__).resolve().parent

ARENA_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]

INK = "#0b0b0b"          # primary text
INK_2 = "#52514e"        # secondary text, keys
INK_MUTED = "#8a8985"    # search paths, rules
WATER = "#eef3f7"        # arena floor
GRID = "#d5dde4"         # cell lines

FONT_PT = 9


def tint(color, frac):
    """Mix ``color`` with white; ``frac`` = 0 is the color, 1 is white."""
    r, g, b = mcolors.to_rgb(color)
    return (r + (1 - r) * frac, g + (1 - g) * frac, b + (1 - b) * frac)


def apply():
    mpl.rcParams.update({
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": FONT_PT,
        "text.color": INK,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    })


def save(fig, stem):
    """Write ``stem``.pdf and ``stem``.png next to the scripts with a 0.01 in
    pad. The figure size is the output size, so size figures to their content:
    the abstract has no room for empty margins."""
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"{stem}.{ext}", dpi=600 if ext == "png" else None,
                    bbox_inches="tight", pad_inches=0.01)
