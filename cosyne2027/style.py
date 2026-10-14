"""Shared style for the Cosyne 2027 figures.

Every figure in this folder takes its colors and fonts from here, so an arena
keeps the same color from the task schematic through the results panels.

Arena colors are the first three slots of the dataviz reference palette
(blue, orange, aqua). Those three are the only slots that stay
distinguishable in every pair under colorblind simulation; a fourth arena
should get a letter label and a neutral gray, not a fourth hue.

Fonts. The PDFs use the PDF core fonts Helvetica and Symbol by name, not
embedded, so Illustrator opens them with the Helvetica installed on the Mac
and the text stays live (an embedded clone shows as a missing font). Core
fonts only encode Latin-1, so Greek letters and the minus sign come from
Symbol. The PNG previews render with URW's metric-identical clones (Nimbus
Sans for Helvetica), since licensed Helvetica is not on the cluster.

Math is set with ``rich``, not mathtext: mathtext places each glyph by hand
in its own font, which Illustrator cannot map to Helvetica.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib import font_manager
from matplotlib.artist import Artist
from matplotlib.font_manager import FontProperties
from matplotlib.text import Text
from matplotlib.transforms import Bbox, IdentityTransform

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

FONT_PT = 9

URW_DIR = Path("/usr/share/fonts/urw-base35")
URW_FACES = ["Regular", "Bold", "Italic", "BoldItalic"]

# Symbol-font codes for the non-Latin-1 characters the figures use
SYMBOL_CODES = {"θ": "q", "π": "p", "−": "-"}


def tint(color, frac):
    """Mix ``color`` with white; ``frac`` = 0 is the color, 1 is white."""
    r, g, b = mcolors.to_rgb(color)
    return (r + (1 - r) * frac, g + (1 - g) * frac, b + (1 - b) * frac)


def apply():
    for face in URW_FACES:
        f = URW_DIR / f"NimbusSans-{face}.otf"
        if f.exists():
            font_manager.fontManager.addfont(str(f))
    mpl.rcParams.update({
        "pdf.use14corefonts": True,
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        # PDF looks fonts up among the core-font metrics and finds Helvetica;
        # Agg looks among installed fonts and falls through to Nimbus Sans.
        "font.sans-serif": ["Helvetica", "Nimbus Sans"],
        "font.size": FONT_PT,
        "text.color": INK,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    })


def _core_pdf(renderer):
    base = getattr(renderer, "_renderer", renderer)
    return (type(base).__name__ == "RendererPdf"
            and mpl.rcParams["pdf.use14corefonts"])


class Rich(Artist):
    """One line of mixed text: roman, italic and Greek runs, with sub- and
    superscripts, on a shared baseline.

    ``runs`` is a list of ``(text, kind)``; kind is a string of flags:
    ``i`` italic, ``b`` bold, ``g`` Greek or math symbol, ``_`` subscript,
    ``^`` superscript, ``k`` stacked over the previous run (as in J_x^T).
    Layout runs at draw time with the active renderer's
    metrics, so the PDF (core Helvetica and Symbol) and the PNG (Nimbus Sans)
    are each spaced for their own fonts. ``va="center"`` centers the cap
    height on ``y``.
    """
    SCRIPT = 0.7               # sub/superscript size, relative
    SUB, SUP = -0.25, 0.42     # baseline shifts, in em of the main size
    CAP = 0.72                 # Helvetica cap height, in em

    def __init__(self, x, y, runs, transform, ha="center", size=FONT_PT,
                 color=INK, zorder=4):
        super().__init__()
        self.xy, self.runs, self.ha = (x, y), runs, ha
        self.size, self.color = size, color
        self.set_transform(transform)
        self.set_zorder(zorder)

    def _layout(self, renderer):
        core = _core_pdf(renderer)
        px = renderer.points_to_pixels(1.0)
        items, x, start = [], 0.0, 0.0
        for text, kind in self.runs:
            size = self.size * (self.SCRIPT if set(kind) & {"_", "^"} else 1)
            if "g" in kind and core:
                text = "".join(SYMBOL_CODES.get(c, c) for c in text)
                family = "Symbol"
            else:
                family = "sans-serif"
            prop = FontProperties(
                family=family, size=size,
                style="italic" if "i" in kind else "normal",
                weight="bold" if "b" in kind else "normal")
            w, h, d = renderer.get_text_width_height_descent(
                text, prop, ismath=False)
            dy = (self.SUB if "_" in kind else self.SUP if "^" in kind
                  else 0.0) * self.size * px
            if "k" in kind:
                items.append((start, dy, text, prop, w, h, d))
                x = max(x, start + w)
            else:
                items.append((x, dy, text, prop, w, h, d))
                start, x = x, x + w
        x0, y0 = self.get_transform().transform(self.xy)
        x0 -= {"left": 0, "center": x / 2, "right": x}[self.ha]
        y0 -= self.CAP / 2 * self.size * px
        return x0, y0, items

    def draw(self, renderer):
        if not self.get_visible():
            return
        x0, y0, items = self._layout(renderer)
        for x, dy, text, prop, *_ in items:
            t = Text(x0 + x, y0 + dy, text, fontproperties=prop,
                     color=self.color, ha="left", va="baseline",
                     transform=IdentityTransform())
            t.set_figure(self.figure)
            t.draw(renderer)

    def get_window_extent(self, renderer=None):
        if renderer is None:
            renderer = self.figure._get_renderer()
        x0, y0, items = self._layout(renderer)
        boxes = [Bbox.from_extents(x0 + x, y0 + dy - d, x0 + x + w,
                                   y0 + dy + h - d)
                 for x, dy, _, _, w, h, d in items]
        return Bbox.union(boxes)


def rich(ax, x, y, runs, **kw):
    """Add a ``Rich`` line at data coordinates (x, y)."""
    r = Rich(x, y, runs, ax.transData, **kw)
    ax.add_artist(r)
    return r


def save(fig, stem):
    """Write ``stem``.pdf and ``stem``.png next to the scripts, with no pad.
    The figure size is the output size exactly, so size figures to their
    content: the abstract has no room for empty margins."""
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"{stem}.{ext}", dpi=600 if ext == "png" else None,
                    bbox_inches="tight", pad_inches=0)
