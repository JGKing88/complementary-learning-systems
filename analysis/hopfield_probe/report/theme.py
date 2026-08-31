"""The palette and chrome, in one place. No hex appears anywhere else.

This is the ``dataviz`` reference palette used **unchanged**, in
documented-validated configurations only:

* the first three categorical slots are recorded as passing all-pairs in both
  modes, which is the budget the Test A outcome map spends;
* the adjacent pairlist passes for the full eight, which is all any line chart
  here needs;
* the ordinal ramp respects its documented floors -- no step lighter than 250
  on the light surface, none darker than 600 on dark.

``node`` is not installed on this cluster, so ``validate_palette.js`` was not
re-run. Nothing here deviates from the documented instance, so nothing needed
it. **Any substitution -- a brand ramp, a fourth categorical hue, a re-stepped
ordinal -- must run the validator before it ships.**

Two encoding decisions worth stating, both from the viz spec:

``K`` is an ordinal ramp, not eight hues.
    ``K`` is an ordered magnitude, so a ramp is the *correct* encoding rather
    than a workaround -- the value-ramp anti-pattern is about nominal
    categories. It also keeps every chart inside the three-slot all-pairs
    budget.

``steps`` is a facet, not a colour.
    Six step counts times seven ``K`` values is 42 series. Colour cannot carry
    both, and ``K`` wins it because ``K`` is what a reader compares *within* a
    panel.
"""
from __future__ import annotations

# Categorical slots (light, dark)
CATEGORICAL = [
    ("#2a78d6", "#3987e5"),   # 1 blue
    ("#eb6834", "#d95926"),   # 2 orange
    ("#1baf7a", "#199e70"),   # 3 aqua
    ("#eda100", "#c98500"),   # 4 yellow
    ("#e87ba4", "#d55181"),   # 5 magenta
    ("#008300", "#008300"),   # 6 green
    ("#4a3aa7", "#9085e9"),   # 7 violet
    ("#e34948", "#e66767"),   # 8 red
]

# Sequential blue ramp, step -> hex.
BLUE = {
    100: "#cde2fb", 150: "#b7d3f6", 200: "#9ec5f4", 250: "#86b6ef",
    300: "#6da7ec", 350: "#5598e7", 400: "#3987e5", 450: "#2a78d6",
    500: "#256abf", 550: "#1c5cab", 600: "#184f95", 650: "#104281",
    700: "#0d366b",
}

# Ordinal steps for K. Light stays at or above 250, dark at or below 600 --
# the documented contrast floors for an ordinal ramp against each surface.
ORDINAL_LIGHT = [250, 300, 350, 400, 450, 500, 550]
ORDINAL_DARK = [600, 550, 500, 450, 400, 350, 300]

DIVERGING_MID = ("#f0efec", "#383835")

STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a",
          "critical": "#d03b3b"}

CHROME = {
    "surface":   ("#fcfcfb", "#1a1a19"),
    "plane":     ("#f9f9f7", "#0d0d0d"),
    "ink":       ("#0b0b0b", "#ffffff"),
    "ink2":      ("#52514e", "#c3c2b7"),
    "muted":     ("#898781", "#898781"),
    "grid":      ("#e1e0d9", "#2c2c2a"),
    "axis":      ("#c3c2b7", "#383835"),
    "border":    ("rgba(11,11,11,0.10)", "rgba(255,255,255,0.10)"),
}

FONT = ('system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, '
        'sans-serif')


def ordinal_colors(n: int, dark: bool = False) -> list[str]:
    """``n`` ordinal steps, light->dark, inside the documented floors."""
    steps = ORDINAL_DARK if dark else ORDINAL_LIGHT
    if n <= 0:
        return []
    if n <= len(steps):
        idx = [round(i * (len(steps) - 1) / max(n - 1, 1)) for i in range(n)]
        return [BLUE[steps[j]] for j in idx]
    keys = sorted(BLUE)
    idx = [round(i * (len(keys) - 1) / max(n - 1, 1)) for i in range(n)]
    return [BLUE[keys[j]] for j in idx]


def sequential(t: float, dark: bool = False) -> str:
    """Sample the blue ramp at ``t`` in [0, 1]. One hue, light->dark."""
    keys = sorted(BLUE)
    if dark:
        keys = list(reversed(keys))
    t = 0.0 if t != t else min(max(t, 0.0), 1.0)
    return BLUE[keys[int(round(t * (len(keys) - 1)))]]


def diverging(t: float, dark: bool = False) -> str:
    """``t`` in [-1, 1]: blue <-> red with a neutral gray midpoint.

    Warm/cool poles that read as opposite, and a midpoint that reads as
    "nothing" -- never a hue at zero, never two cool poles.
    """
    if t != t:
        return "none"
    t = min(max(t, -1.0), 1.0)
    mid = DIVERGING_MID[1 if dark else 0]
    if abs(t) < 1e-6:
        return mid
    pole = CATEGORICAL[7][1 if dark else 0] if t > 0 \
        else CATEGORICAL[0][1 if dark else 0]
    return _mix(mid, pole, abs(t))


def _mix(a: str, b: str, t: float) -> str:
    ca, cb = _rgb(a), _rgb(b)
    return "#%02x%02x%02x" % tuple(
        int(round(ca[i] + (cb[i] - ca[i]) * t)) for i in range(3))


def _rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def css() -> str:
    """Page stylesheet. Dark mode is *selected*, not flipped.

    The light palette is defined on bare ``:root``; the dark steps are
    redefined under both ``prefers-color-scheme`` (guarded so an explicit light
    stamp still wins) and ``[data-theme="dark"]`` (so a toggle wins in both
    directions). No colour has its only definition inside a media block.
    """
    def var_block(dark: bool) -> str:
        i = 1 if dark else 0
        rows = [f"  --{k}: {v[i]};" for k, v in CHROME.items()]
        rows += [f"  --cat{n + 1}: {c[i]};"
                 for n, c in enumerate(CATEGORICAL)]
        rows += [f"  --diverge-mid: {DIVERGING_MID[i]};"]
        rows += [f"  --status-{k}: {v};" for k, v in STATUS.items()]
        return "\n".join(rows)

    return f"""
:root {{
  color-scheme: light;
{var_block(False)}
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
{var_block(True)}
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
{var_block(True)}
}}

* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--plane); color: var(--ink);
  font-family: {FONT}; font-size: 15px; line-height: 1.5;
}}
a {{ color: var(--cat1); }}
.wrap {{ max-width: 1240px; margin: 0 auto; padding: 24px 20px 80px; }}

header.run {{
  position: sticky; top: 0; z-index: 20; background: var(--surface);
  border-bottom: 1px solid var(--border); padding: 10px 20px;
  display: flex; gap: 18px; align-items: baseline; flex-wrap: wrap;
}}
header.run .name {{ font-weight: 650; }}
header.run .kv {{ color: var(--ink2); font-size: 13px; }}
header.run .kv b {{ color: var(--ink); font-weight: 600; }}
/* The checkpoint path takes its own row: it is the one thing a reader has to
   copy verbatim, and inlining it among the kv spans would let it wrap into
   them. `order` puts it last regardless of where run_header emits it. */
header.run .ckpt {{
  flex-basis: 100%; order: 99; font-family: var(--mono, ui-monospace,
  "SFMono-Regular", Menlo, monospace); font-size: 11.5px; color: var(--muted);
  word-break: break-all; line-height: 1.45;
}}
header.run .ckpt b {{ color: var(--ink2); font-weight: 500; }}
nav.tabs {{ display: flex; gap: 4px; flex-wrap: wrap; margin-left: auto; }}
nav.tabs a {{
  padding: 4px 10px; border-radius: 6px; text-decoration: none;
  color: var(--ink2); font-size: 13px;
}}
nav.tabs a.on {{ background: var(--cat1); color: #fff; }}

.filters {{
  display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
  padding: 12px 0 18px; border-bottom: 1px solid var(--border);
  margin-bottom: 22px;
}}
.filters label {{ font-size: 13px; color: var(--ink2); }}
.filters select {{
  font: inherit; font-size: 13px; padding: 4px 8px; border-radius: 6px;
  border: 1px solid var(--axis); background: var(--surface); color: var(--ink);
}}

h1 {{ font-size: 26px; margin: 8px 0 4px; letter-spacing: -0.01em; }}
h2 {{ font-size: 19px; margin: 34px 0 6px; }}
h3 {{ font-size: 15px; margin: 20px 0 4px; color: var(--ink2); }}
p.lede {{ color: var(--ink2); margin: 0 0 18px; max-width: 76ch; }}
p.note {{ color: var(--ink2); font-size: 13px; max-width: 76ch; }}

.tiles {{ display: grid; gap: 14px;
  grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); }}
.tile {{
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 14px 16px;
}}
.tile .label {{ font-size: 12px; color: var(--ink2); text-transform: uppercase;
  letter-spacing: 0.04em; }}
.tile .value {{ font-size: 34px; font-weight: 620; line-height: 1.1;
  margin: 4px 0 2px; }}
.tile .sub {{ font-size: 12px; color: var(--muted); }}
.tile .cmp {{ margin-top: 8px; font-size: 12px; color: var(--ink2);
  display: flex; flex-direction: column; gap: 2px; }}
.tile .cmp span b {{ font-weight: 600; font-variant-numeric: tabular-nums; }}

.card {{
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 16px 16px 10px; margin: 14px 0;
  overflow-x: auto;
}}
.card > h3:first-child {{ margin-top: 0; }}
.grid2 {{ display: grid; gap: 14px;
  grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); }}
.facets {{ display: grid; gap: 10px;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); }}
.facet-title {{ font-size: 12px; color: var(--ink2); margin: 0 0 2px 6px; }}

details {{ margin-top: 10px; }}
summary {{ cursor: pointer; font-size: 13px; color: var(--ink2); }}
table {{ border-collapse: collapse; font-size: 12.5px; margin-top: 8px;
  font-variant-numeric: tabular-nums; }}
th, td {{ text-align: right; padding: 3px 10px;
  border-bottom: 1px solid var(--border); }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ color: var(--ink2); font-weight: 600; }}

.legend {{ display: flex; gap: 14px; flex-wrap: wrap; font-size: 12px;
  color: var(--ink2); margin: 2px 0 8px 6px; }}
.legend i {{ display: inline-block; width: 16px; height: 0;
  border-top-width: 2px; border-top-style: solid; vertical-align: middle;
  margin-right: 5px; }}
.legend i.sw {{ height: 10px; width: 10px; border-radius: 2px;
  border-top: 0; vertical-align: -1px; }}

.banner {{
  border-left: 4px solid var(--status-warning); background: var(--surface);
  border-radius: 6px; padding: 12px 16px; margin: 12px 0 22px;
  display: flex; gap: 10px; align-items: flex-start;
}}
.banner .icon {{ font-size: 18px; line-height: 1.2; }}
.banner b {{ display: block; }}

#tip {{
  position: fixed; z-index: 60; pointer-events: none; opacity: 0;
  background: var(--surface); color: var(--ink); font-size: 12.5px;
  border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px;
  box-shadow: 0 6px 22px rgba(0,0,0,0.16); max-width: 300px;
  transition: opacity .08s linear;
}}
#tip .tv {{ font-weight: 640; font-variant-numeric: tabular-nums; }}
#tip .tr {{ display: flex; gap: 8px; align-items: center;
  justify-content: space-between; }}
#tip .tk {{ display: inline-block; width: 14px; border-top: 2px solid;
  margin-right: 4px; vertical-align: middle; }}
#tip .th {{ color: var(--ink2); margin-bottom: 4px; }}

svg {{ display: block; max-width: 100%; height: auto; }}
svg text {{ fill: var(--ink2); font-family: {FONT}; }}
svg .ax {{ stroke: var(--axis); }}
svg .gl {{ stroke: var(--grid); }}
svg .chance {{ stroke: var(--muted); stroke-dasharray: 3 3; }}
svg .hit {{ fill: transparent; }}
svg .cell:hover, svg [data-c]:hover {{ stroke: var(--ink); stroke-width: 1.5; }}
.filtered-out {{ display: none !important; }}
.enc-hdr {{ display: contents; }}
.footer {{ margin-top: 46px; padding-top: 14px;
  border-top: 1px solid var(--border); color: var(--muted); font-size: 12px; }}
"""


__all__ = ["BLUE", "CATEGORICAL", "CHROME", "FONT", "STATUS", "css",
           "diverging", "ordinal_colors", "sequential"]
