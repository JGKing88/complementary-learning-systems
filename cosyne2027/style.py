"""Shared style for the Cosyne 2027 figures.

Every figure in this folder takes its colors and fonts from here, so an arena
keeps the same color from the task schematic through the results panels.

Arena colors are the first three slots of the dataviz reference palette
(blue, orange, aqua). Those three are the only slots that stay
distinguishable in every pair under colorblind simulation; a fourth arena
should get a letter label and a neutral gray, not a fourth hue.

The font is Helvetica, as Nimbus Sans: URW's clone with Helvetica's design
and metrics (the face Ghostscript substitutes for Helvetica). Licensed
Helvetica is not on the cluster. Matplotlib embeds it as Type 42 subsets,
which Chrome's PDF engine renders correctly.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent

ARENA_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]

INK = "#0b0b0b"          # primary text
INK_2 = "#52514e"        # secondary text, keys
INK_MUTED = "#8a8985"    # search paths, rules
WATER = "#eef3f7"        # arena floor
GRID = "#d5dde4"         # cell lines

# Model-schematic fills, one per module type
FROZEN = "#f9e0a6"       # pretrained, fixed weights
PLASTIC = "#f5b39a"      # online Hebbian
FIXED_OP = "#e3e2de"     # no weights
ENV = "#dfe9f2"          # environment and perception
BOX_EDGE = INK_2

HELVETICA_DIR = Path("/usr/share/fonts/urw-base35")
HELVETICA_FACES = ["Regular", "Bold", "Italic", "BoldItalic"]

FONT_PT = 9


def tint(color, frac):
    """Mix ``color`` with white; ``frac`` = 0 is the color, 1 is white."""
    r, g, b = mcolors.to_rgb(color)
    return (r + (1 - r) * frac, g + (1 - g) * frac, b + (1 - b) * frac)


def apply():
    for face in HELVETICA_FACES:
        f = HELVETICA_DIR / f"NimbusSans-{face}.otf"
        if f.exists():
            font_manager.fontManager.addfont(str(f))
    mpl.rcParams.update({
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Helvetica", "Arial",
                            "Liberation Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Nimbus Sans",
        "mathtext.it": "Nimbus Sans:italic",
        "mathtext.bf": "Nimbus Sans:bold",
        "font.size": FONT_PT,
        "text.color": INK,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    })


def save(fig, stem):
    """Write ``stem``.pdf and ``stem``.png next to the scripts, with no pad.
    The figure size is the output size exactly, so size figures to their
    content: the abstract has no room for empty margins."""
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"{stem}.{ext}", dpi=600 if ext == "png" else None,
                    bbox_inches="tight", pad_inches=0)
