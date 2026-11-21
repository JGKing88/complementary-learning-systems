"""Ratio grids from the report rendered as heat maps (width over grid spacing).

Two kinds of grid exist, told apart by the first header cell: "Recording (field width)" divides a
place-field width by a module's spacing, and "Recording (kernel full width at 0.5)" divides the full
width at which population-vector correlation falls to 0.5. A trailing "Studies" column, if present,
is not drawn as a cell; it feeds the tooltip.
"""
import html
import math
import re

import heatcolor as hc

LEVELS = ("Dorsal", "Intermediate", "Ventral")
NUM = r"(\d+(?:\.\d+)?)"
KINDS = {
    "field": {
        "title_id": "hm-title",
        "title": "Place-field width against grid spacing",
        "corner": "field width",
        "ends": ("Field narrower than the grid spacing", "Field wider"),
        "sub": ("Each cell divides a hippocampal place-field width by one entorhinal grid module's "
                "spacing, as a range from the narrowest field over the widest spacing to the widest "
                "field over the narrowest spacing. Each row pools every audited width for one region, "
                "long-axis level and environment class, and names how many studies it holds; hover "
                "or focus a cell to list them. Colour is the range's geometric midpoint on a log "
                "scale and saturates beyond 0.1 and 10. Rows run from dorsal to ventral. Fields grow "
                "with the environment and grids barely change, so compare levels only within one "
                "environment class."),
        "link": "Sources, field rules and limits",
    },
    "kernel": {
        "title_id": "hm-kernel-title",
        "title": "Population similarity width against grid spacing",
        "corner": "kernel full width at r = 0.5",
        "ends": ("Kernel narrower than the grid spacing", "Kernel wider"),
        "sub": ("Each cell divides the full width at which hippocampal population-vector "
                "correlation falls to 0.5, both sides of a position, by one grid module's spacing. "
                "Each row is one study, region, level and condition. The colour scale is the same "
                "as above. No study measured a hippocampal kernel and grid spacing in the same "
                "animals, so every cell pairs different animals, and species without a measured "
                "grid spacing are left out."),
        "link": "Sources, measures and limits",
    },
}


def kind_of(header_cell):
    """'Recording (field width)' -> 'field'; 'Recording (kernel full width at 0.5)' -> 'kernel'."""
    return "kernel" if "kernel" in header_cell else "field"


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
    has_studies = head[-1] == "Studies"
    ncol = len(head) - 1 - has_studies
    cols = [split_label(h) for h in head[1:1 + ncol]]
    out = []
    for r in body:
        label, width = split_label(r[0])
        cells = []
        for (clab, cspan), c in zip(cols, r[1:1 + ncol]):
            cells.append({"text": c, "range": split_range(c), "col": clab, "col_span": cspan})
        studies = [s.strip() for s in r[-1].split(";") if s.strip()] if has_studies else []
        out.append({"label": label, "width": width, "cells": cells, "studies": studies})
    return {"species": species, "kind": kind_of(head[0]), "cols": cols, "rows": out}


def _studies_text(studies):
    if not studies:
        return ""
    n = len(studies)
    return f"{n} {'study' if n == 1 else 'studies'}: " + "; ".join(studies)


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
    if row["studies"] and not all(st in row["label"] for st in row["studies"]):
        attrs += f' data-s="{e(_studies_text(row["studies"]))}"'
    return f'<td class="v" tabindex="0" style="{style}" {attrs}>{e(cell["text"])}</td>'


def table_html(grid):
    e = html.escape
    ncol = len(grid["cols"]) + 1
    corner = KINDS[grid["kind"]]["corner"]
    head = [f'<th class="hm-corner" scope="col">Hippocampal recording<span>{e(corner)}</span></th>']
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


def legend_html(ends):
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
            f'<div class="hm-ends"><span>{html.escape(ends[0])}</span>'
            f'<span>{html.escape(ends[1])}</span></div></div>'
            '<div class="hm-nckey"><i aria-hidden="true"></i>Not comparable</div></div>')


def group_html(kind, grids, sources_href):
    k = KINDS[kind]
    figs = "".join(table_html(g) for g in grids)
    return (f'<div class="hm-group-block"><h2 id="{k["title_id"]}">{html.escape(k["title"])}</h2>'
            f'<p class="hm-sub">{html.escape(k["sub"])} '
            f'<a href="#{sources_href}">{html.escape(k["link"])}</a>.</p>'
            + legend_html(k["ends"]) + f'<div class="hm-figs">{figs}</div></div>')


def block_html(grids, sources):
    """grids: parsed grids in page order; sources: {kind: id of the section that holds them}."""
    if not grids:
        return ""
    parts, kinds = [], []
    for kind in ("field", "kernel"):
        mine = [g for g in grids if g["kind"] == kind]
        if mine:
            parts.append(group_html(kind, mine, sources[kind]))
            kinds.append(kind)
    return (f'<section class="hm-block" aria-labelledby="{KINDS[kinds[0]]["title_id"]}">' + "".join(parts)
            + '<div class="hm-tip" role="tooltip" hidden></div></section>')
