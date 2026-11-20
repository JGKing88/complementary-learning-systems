"""Ratio grids from the report rendered as heat maps (field width over grid spacing)."""
import html
import math
import re

import heatcolor as hc

LEVELS = ("Dorsal", "Intermediate", "Ventral")
NUM = r"(\d+(?:\.\d+)?)"


def split_range(text):
    """'0.59 to 1.9' -> (0.59, 1.9); '2.4' -> (2.4, 2.4); anything else -> None."""
    m = re.fullmatch(NUM + r"(?: to " + NUM + r")?", text.strip())
    if not m:
        return None
    lo = float(m.group(1))
    return lo, float(m.group(2)) if m.group(2) else lo


def split_label(text):
    """'Dorsal CA1, T maze (29 to 75 cm)' -> ('Dorsal CA1, T maze', '29 to 75 cm')."""
    m = re.fullmatch(r"(.*) \(([^()]*)\)", text.strip())
    return (m.group(1), m.group(2)) if m else (text.strip(), "")


def parse(lines, species):
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines]
    head, body = rows[0], rows[2:]
    cols = [split_label(h) for h in head[1:]]
    out = []
    for r in body:
        label, width = split_label(r[0])
        cells = []
        for (clab, cspan), c in zip(cols, r[1:]):
            cells.append({"text": c, "range": split_range(c), "col": clab, "col_span": cspan})
        out.append({"label": label, "width": width, "cells": cells})
    return {"species": species, "cols": cols, "rows": out}


def _cell(row, cell):
    e = html.escape
    if cell["range"] is None:
        return f'<td class="nc"><span>{e(cell["text"])}</span></td>'
    lo, hi = cell["range"]
    mid = math.sqrt(lo * hi)
    style = ";".join([
        f"--bl:{hc.ratio_color(mid, 'light')}", f"--tl:{hc.ink_for(hc.ratio_color(mid, 'light'), 'light')}",
        f"--bd:{hc.ratio_color(mid, 'dark')}", f"--td:{hc.ink_for(hc.ratio_color(mid, 'dark'), 'dark')}",
    ])
    mid_s = f"{mid:.2g}" if mid < 10 else f"{mid:.0f}"
    attrs = (f'data-r="{e(cell["text"])}" data-m="{e(mid_s)}" '
             f'data-row="{e(row["label"])}: {e(row["width"])}" '
             f'data-col="{e(cell["col"])}: {e(cell["col_span"])}"')
    return f'<td class="v" tabindex="0" style="{style}" {attrs}>{e(cell["text"])}</td>'


def table_html(grid):
    e = html.escape
    ncol = len(grid["cols"]) + 1
    head = ['<th class="hm-corner" scope="col">Hippocampal recording<span>field width</span></th>']
    for lab, sp in grid["cols"]:
        head.append(f'<th scope="col">{e(lab)}<span>{e(sp)}</span></th>')
    body, level = [], None
    for row in grid["rows"]:
        first = row["label"].split(" ", 1)[0]
        lvl = first if first in LEVELS else None
        if lvl and lvl != level:
            body.append(f'<tr class="hm-group"><th scope="rowgroup" colspan="{ncol}">{e(lvl)}</th></tr>')
            level = lvl
        region, _, env = row["label"].partition(", ")
        if lvl:
            region = region[len(lvl):].strip() or region
        rh = (f'<th scope="row"><b>{e(region)}</b>'
              f'<span>{e(env)}</span><span class="w">{e(row["width"])}</span></th>')
        body.append("<tr>" + rh + "".join(_cell(row, c) for c in row["cells"]) + "</tr>")
    return (f'<figure class="hm-fig"><figcaption>{e(grid["species"])}</figcaption>'
            f'<div class="hm-wrap"><table class="hm"><thead><tr>{"".join(head)}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div></figure>')


def legend_html():
    stops = {}
    for mode in ("light", "dark"):
        xs = [i / 20 for i in range(21)]
        cols = [hc.ratio_color(10 ** (2 * x - 1), mode) for x in xs]
        stops[mode] = "linear-gradient(90deg, " + ", ".join(
            f"{c} {x * 100:.0f}%" for c, x in zip(cols, xs)) + ")"
    ticks = "".join(
        f'<span style="left:{(math.log10(v) + 1) / 2 * 100:.1f}%">{t}</span>'
        for v, t in ((0.1, "0.1"), (0.3, "0.3"), (1, "1"), (3, "3"), (10, "10")))
    return (f'<div class="hm-legend" style="--gl:{stops["light"]};--gd:{stops["dark"]}">'
            '<div class="hm-scale"><div class="hm-bar" aria-hidden="true"></div>'
            f'<div class="hm-ticks" aria-hidden="true">{ticks}</div>'
            '<div class="hm-ends"><span>Field narrower than the grid spacing</span>'
            '<span>Field wider</span></div></div>'
            '<div class="hm-nckey"><i aria-hidden="true"></i>Not comparable</div></div>')


def block_html(grids, sources_href):
    if not grids:
        return ""
    figs = "".join(table_html(g) for g in grids)
    return (
        '<section class="hm-block" aria-labelledby="hm-title">'
        '<h2 id="hm-title">Place-field width against grid spacing</h2>'
        '<p class="hm-sub">Each cell divides a hippocampal place-field width by one entorhinal grid '
        "module's spacing, as a range from the narrowest field over the widest spacing to the widest "
        "field over the narrowest spacing. Colour is the range's geometric midpoint on a log scale "
        "and saturates beyond 0.1 and 10. Rows run from dorsal to ventral and name their environment. "
        "Fields grow with the environment and grids barely change, so compare levels only within "
        f'one environment. <a href="#{sources_href}">Sources, field rules and limits</a>.</p>'
        + legend_html() + f'<div class="hm-figs">{figs}</div>'
        '<div class="hm-tip" role="tooltip" hidden></div></section>')
