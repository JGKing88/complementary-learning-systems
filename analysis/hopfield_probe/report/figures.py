"""Every chart, as an inline SVG string.

No matplotlib and no external assets: a page has to be one file you can email,
and a strict CSP would block a CDN anyway. The interaction layer is one small
JS function in ``page.py`` that reads a JSON payload emitted beside each chart.

The rules from the viz spec that are enforced *here* rather than left to the
caller:

* **``n`` is always available.** Every binned curve emits its per-bin sample
  count into the tooltip payload and the table view, and draws the faint
  ``n``-strip under the axis. The near-goal bins are the sparsest and the most
  interesting, and a curve that hides its ``n`` there is misleading.
* **Chance is drawn, not described** -- a labelled hairline, not a caption.
* **A legend whenever there are two or more series**, never colour alone.
* **One axis.** There is no dual-axis entry point in this module.
* **Series labels are untrusted text.** They are escaped on the way into the
  SVG and inserted with ``textContent`` on the way into a tooltip.
"""
from __future__ import annotations

import html
import json
import math
from itertools import count

from .theme import CATEGORICAL, diverging, sequential

_uid = count(1)


def _esc(s) -> str:
    return html.escape(str(s), quote=True)


def _fmt(v, nd=2) -> str:
    if v is None or (isinstance(v, float) and (v != v or math.isinf(v))):
        return "--"
    if isinstance(v, (int,)) or (isinstance(v, float) and v == int(v)
                                 and abs(v) < 1e6):
        return f"{int(v)}"
    return f"{v:.{nd}f}"


def _n(v: float) -> str:
    """Shortest faithful number for an SVG coordinate.

    A trailing ".0" on every coordinate is a few hundred kilobytes across a
    page of heatmaps, and nothing reads it.
    """
    r = round(v, 1)
    return str(int(r)) if r == int(r) else f"{r:.1f}"


def _nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return [lo, hi]
    raw = (hi - lo) / max(n, 1)
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw / mag <= m:
            step = m * mag
            break
    else:
        step = 10 * mag
    start = math.ceil(lo / step) * step
    out, v = [], start
    while v <= hi + step * 1e-9:
        out.append(round(v, 10))
        v += step
    return out or [lo, hi]


def _centers(edges: list[float]) -> list[float]:
    return [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]


# ---------------------------------------------------------------------------
# Line chart
# ---------------------------------------------------------------------------

def line_chart(
    x: list[float],
    series: list[dict],
    *,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    ylim: tuple[float, float] | None = None,
    chance: float | None = None,
    chance_label: str = "chance",
    log_x: bool = False,
    n_per_bin: list[int] | None = None,
    width: int = 520,
    height: int = 300,
    x_is_category: bool = False,
) -> str:
    """One y-axis, crosshair tooltip listing every series at the hovered x.

    ``series``: ``{label, values, color, dash?, band?: (lo, hi), width?}``.
    ``values`` may contain ``None``; gaps break the path rather than
    interpolating across a bin that had no samples.
    """
    cid = f"c{next(_uid)}"
    ml, mr, mt, mb = 56, 14, 26 if title else 12, 46
    if n_per_bin:
        mb += 16
    iw, ih = width - ml - mr, height - mt - mb

    xs = [v for v in x if v is not None]
    if log_x:
        xs = [v for v in xs if v and v > 0]
    x0, x1 = (min(xs), max(xs)) if xs else (0.0, 1.0)
    if x1 <= x0:
        x1 = x0 + 1.0

    allv = [v for s in series for v in s["values"] if v is not None]
    for s in series:
        for key in ("band",):
            if s.get(key):
                allv += [v for pair in zip(*s[key]) for v in pair
                         if v is not None]
    if chance is not None:
        allv.append(chance)
    if ylim:
        y0, y1 = ylim
    elif allv:
        y0, y1 = min(allv), max(allv)
        pad = (y1 - y0) * 0.08 or (abs(y1) * 0.1 or 1.0)
        y0, y1 = y0 - pad, y1 + pad
    else:
        y0, y1 = 0.0, 1.0

    def px(v):
        if x_is_category:
            n = max(len(x) - 1, 1)
            return ml + (x.index(v) if v in x else 0) / n * iw
        if log_x:
            lv = math.log10(max(v, 1e-6))
            l0, l1 = math.log10(max(x0, 1e-6)), math.log10(max(x1, 1e-6))
            return ml + (lv - l0) / max(l1 - l0, 1e-9) * iw
        return ml + (v - x0) / (x1 - x0) * iw

    def py(v):
        return mt + ih - (v - y0) / max(y1 - y0, 1e-12) * ih

    o = [f'<svg viewBox="0 0 {width} {height}" role="img" '
         f'aria-label="{_esc(title or ylabel)}" data-chart="{cid}">']
    if title:
        o.append(f'<text x="{ml}" y="14" font-size="12" '
                 f'font-weight="600">{_esc(title)}</text>')

    for t in _nice_ticks(y0, y1):
        yy = py(t)
        if not (mt - 1 <= yy <= mt + ih + 1):
            continue
        o.append(f'<line class="gl" x1="{ml}" y1="{yy:.1f}" x2="{ml + iw}" '
                 f'y2="{yy:.1f}" stroke-width="1"/>')
        o.append(f'<text x="{ml - 7}" y="{yy + 3.5:.1f}" font-size="11" '
                 f'text-anchor="end">{_fmt(t)}</text>')

    xticks = x if x_is_category else _nice_ticks(x0, x1, 6)
    for t in xticks:
        xx = px(t)
        if not (ml - 1 <= xx <= ml + iw + 1):
            continue
        o.append(f'<line class="ax" x1="{xx:.1f}" y1="{mt + ih}" '
                 f'x2="{xx:.1f}" y2="{mt + ih + 4}" stroke-width="1"/>')
        o.append(f'<text x="{xx:.1f}" y="{mt + ih + 17}" font-size="11" '
                 f'text-anchor="middle">{_fmt(t)}</text>')

    o.append(f'<line class="ax" x1="{ml}" y1="{mt + ih}" x2="{ml + iw}" '
             f'y2="{mt + ih}" stroke-width="1"/>')

    if chance is not None:
        yy = py(chance)
        o.append(f'<line class="chance" x1="{ml}" y1="{yy:.1f}" '
                 f'x2="{ml + iw}" y2="{yy:.1f}" stroke-width="1"/>')
        o.append(f'<text x="{ml + iw - 2}" y="{yy - 4:.1f}" font-size="10.5" '
                 f'text-anchor="end">{_esc(chance_label)}</text>')

    # Bands first so lines sit on top.
    for s in series:
        band = s.get("band")
        if not band:
            continue
        lo, hi = band
        pts_hi, pts_lo = [], []
        for xv, a, b in zip(x, lo, hi):
            if a is None or b is None or xv is None:
                continue
            if log_x and (not xv or xv <= 0):
                continue
            pts_hi.append(f"{px(xv):.1f},{py(b):.1f}")
            pts_lo.append(f"{px(xv):.1f},{py(a):.1f}")
        if len(pts_hi) > 1:
            o.append(f'<polygon points="{" ".join(pts_hi + pts_lo[::-1])}" '
                     f'fill="{s["color"]}" opacity="0.13"/>')

    for s in series:
        seg, segs = [], []
        for xv, v in zip(x, s["values"]):
            if v is None or xv is None or (log_x and (not xv or xv <= 0)):
                if len(seg) > 1:
                    segs.append(seg)
                seg = []
                continue
            seg.append(f"{px(xv):.1f},{py(v):.1f}")
        if len(seg) > 1:
            segs.append(seg)
        elif len(seg) == 1:
            cx, cy = seg[0].split(",")
            o.append(f'<circle cx="{cx}" cy="{cy}" r="3" '
                     f'fill="{s["color"]}"/>')
        dash = f' stroke-dasharray="{s["dash"]}"' if s.get("dash") else ""
        for sg in segs:
            o.append(f'<polyline points="{" ".join(sg)}" fill="none" '
                     f'stroke="{s["color"]}" stroke-width="{s.get("width", 2)}"'
                     f'{dash} stroke-linejoin="round" stroke-linecap="round"/>')

    if n_per_bin:
        top = mt + ih + 26
        mx = max(n_per_bin) or 1
        bw = max(iw / max(len(n_per_bin), 1) - 1, 1)
        for i, nv in enumerate(n_per_bin):
            if i >= len(x) or x[i] is None:
                continue
            if log_x and (not x[i] or x[i] <= 0):
                continue
            hgt = 11 * (nv / mx)
            o.append(f'<rect x="{px(x[i]) - bw / 2:.1f}" '
                     f'y="{top + 11 - hgt:.1f}" width="{bw:.1f}" '
                     f'height="{hgt:.1f}" fill="var(--muted)" opacity="0.35"/>')
        o.append(f'<text x="{ml - 7}" y="{top + 10}" font-size="9.5" '
                 f'text-anchor="end">n</text>')

    o.append(f'<line id="{cid}-cross" x1="0" y1="{mt}" x2="0" '
             f'y2="{mt + ih}" stroke="var(--muted)" stroke-width="1" '
             f'opacity="0"/>')
    o.append(f'<rect class="hit" x="{ml}" y="{mt}" width="{iw}" '
             f'height="{ih}"/>')

    if ylabel:
        o.append(f'<text x="{-(mt + ih / 2):.1f}" y="13" font-size="11" '
                 f'transform="rotate(-90)" text-anchor="middle">'
                 f'{_esc(ylabel)}</text>')
    if xlabel:
        o.append(f'<text x="{ml + iw / 2:.1f}" y="{height - 4}" '
                 f'font-size="11" text-anchor="middle">{_esc(xlabel)}</text>')
    o.append("</svg>")

    payload = {
        "type": "line", "id": cid, "x": x, "logx": log_x, "cat": x_is_category,
        "ml": ml, "iw": iw, "mt": mt, "ih": ih, "w": width,
        "xlabel": xlabel, "ylabel": ylabel,
        "n": n_per_bin or [],
        "series": [{"label": s["label"], "color": s["color"],
                    "values": s["values"]} for s in series],
    }
    legend = ""
    if len(series) >= 2:
        items = "".join(
            f'<span><i style="border-top-color:{s["color"]};'
            f'{"border-top-style:dashed;" if s.get("dash") else ""}"></i>'
            f'{_esc(s["label"])}</span>' for s in series)
        legend = f'<div class="legend">{items}</div>'

    table = _table_from_series(x, series, xlabel, n_per_bin)
    return (f'<figure class="chartbox" style="margin:0">{legend}'
            f'{"".join(o)}'
            f'<script type="application/json" class="chartdata">'
            f'{json.dumps(payload)}</script>{table}</figure>')


def _table_from_series(x, series, xlabel, n_per_bin) -> str:
    head = "".join(f"<th>{_esc(s['label'])}</th>" for s in series)
    nhead = "<th>n</th>" if n_per_bin else ""
    rows = []
    for i, xv in enumerate(x):
        cells = "".join(
            f"<td>{_fmt(s['values'][i] if i < len(s['values']) else None)}</td>"
            for s in series)
        ncell = (f"<td>{n_per_bin[i]}</td>"
                 if n_per_bin and i < len(n_per_bin) else "")
        rows.append(f"<tr><td>{_fmt(xv)}</td>{cells}{ncell}</tr>")
    return (f'<details><summary>table</summary><table><thead><tr>'
            f'<th>{_esc(xlabel or "x")}</th>{head}{nhead}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></details>')


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------

# Above this many cells a heatmap is drawn as colour-bucketed run-length paths
# with a coarse hit grid, rather than one <rect> per cell. See _run_length_cells.
LARGE_MAP_CELLS = 4000
_BUCKETS = 24
_HIT_BLOCKS = 24


def _run_length_cells(matrix, nx, ny, cell, ml, mt, vmin, vmax, kind):
    """One path per colour bucket, horizontal runs merged.

    Quantising to `_BUCKETS` levels is what makes runs long enough to be worth
    merging; the eye cannot resolve more steps than that on a 3-pixel cell, and
    the colourbar is continuous either way.
    """
    span = max(vmax - vmin, 1e-12)
    paths: dict[int, list[str]] = {}
    for j in range(ny):
        yy = mt + (ny - 1 - j) * cell
        i = 0
        while i < nx:
            v = matrix[i][j]
            b = None if v is None else min(
                int((v - vmin) / span * _BUCKETS), _BUCKETS - 1)
            run = 1
            while i + run < nx:
                v2 = matrix[i + run][j]
                b2 = None if v2 is None else min(
                    int((v2 - vmin) / span * _BUCKETS), _BUCKETS - 1)
                if b2 != b:
                    break
                run += 1
            if b is not None:
                xx = ml + i * cell
                w = run * cell
                paths.setdefault(b, []).append(
                    f"M{xx:.1f} {yy:.1f}h{w:.1f}v{cell:.1f}h{-w:.1f}z")
            i += run

    out = []
    for b, segs in sorted(paths.items()):
        t = (b + 0.5) / _BUCKETS
        col = diverging(2 * t - 1) if kind == "diverging" else sequential(t)
        out.append(f'<path d="{"".join(segs)}" fill="{col}"/>')
    return out


def _hit_grid(matrix, counts, nx, ny, cell, ml, mt):
    """Transparent blocks carrying the mean of what they cover.

    A 3-pixel cell is not a hoverable target, so the hit target is a block --
    which is also what keeps the interaction rule ("the hit target is bigger
    than the mark") true rather than nominally satisfied.
    """
    bx = max(1, nx // _HIT_BLOCKS)
    by = max(1, ny // _HIT_BLOCKS)
    out = []
    for i0 in range(0, nx, bx):
        for j0 in range(0, ny, by):
            vals, ns = [], 0
            for i in range(i0, min(i0 + bx, nx)):
                for j in range(j0, min(j0 + by, ny)):
                    v = matrix[i][j]
                    if v is not None:
                        vals.append(v)
                        if counts:
                            ns += counts[i][j]
            if not vals:
                continue
            j1 = min(j0 + by, ny)
            xx = ml + i0 * cell
            yy = mt + (ny - 1 - (j1 - 1)) * cell
            mean = sum(vals) / len(vals)
            tip = (f"x {i0}-{min(i0 + bx, nx) - 1}, y {j0}-{j1 - 1}: "
                   f"{mean:.1f}" + (f" (n={ns})" if ns else ""))
            out.append(
                f'<rect class="hit" x="{xx:.1f}" y="{yy:.1f}" '
                f'width="{(min(i0 + bx, nx) - i0) * cell:.1f}" '
                f'height="{(j1 - j0) * cell:.1f}" '
                f'data-tip="{_esc(tip)}"/>')
    return out



def heatmap(
    matrix: list[list[float | None]],
    *,
    title: str = "",
    kind: str = "sequential",           # "sequential" | "diverging"
    vmin: float | None = None,
    vmax: float | None = None,
    unit: str = "",
    mark: tuple[int, int] | None = None,
    counts: list[list[int]] | None = None,
    x_origin: float = 0.0,
    y_origin: float = 0.0,
    cell: float = 9.0,
    xlabel: str = "",
    ylabel: str = "",
    overlay: list[dict] | None = None,
) -> str:
    """A grid of cells; the cell is the hit target.

    ``matrix[i][j]`` is drawn at column ``i`` (x, rightward) and row ``j``
    (y, **upward**) so the picture matches the world's axes rather than image
    convention -- getting this wrong flips every map vertically while leaving
    the aggregates untouched.
    """
    cid = f"c{next(_uid)}"
    nx, ny = len(matrix), len(matrix[0]) if matrix else 0
    ml, mt, mr, mb = 40, 22 if title else 8, 66, 34
    w, h = ml + nx * cell + mr, mt + ny * cell + mb

    vals = [v for row in matrix for v in row if v is not None]
    if vmin is None:
        vmin = min(vals) if vals else 0.0
    if vmax is None:
        vmax = max(vals) if vals else 1.0
    if kind == "diverging":
        m = max(abs(vmin), abs(vmax)) or 1.0
        vmin, vmax = -m, m
    if vmax <= vmin:
        vmax = vmin + 1e-9

    def color(v):
        if v is None:
            return "none"
        t = (v - vmin) / (vmax - vmin)
        return (diverging(2 * t - 1) if kind == "diverging"
                else sequential(t))

    o = [f'<svg viewBox="0 0 {w} {h}" role="img" '
         f'aria-label="{_esc(title)}" data-chart="{cid}">']
    if title:
        o.append(f'<text x="{ml}" y="13" font-size="12" '
                 f'font-weight="600">{_esc(title)}</text>')

    big = nx * ny > LARGE_MAP_CELLS
    if not big:
        for i in range(nx):
            for j in range(ny):
                v = matrix[i][j]
                xx = ml + i * cell
                yy = mt + (ny - 1 - j) * cell
                o.append(
                    f'<rect x="{_n(xx)}" y="{_n(yy)}" width="{_n(cell)}" '
                    f'height="{_n(cell)}" fill="{color(v)}" '
                    f'data-c="{i},{j}"/>')
    else:
        o.extend(_run_length_cells(matrix, nx, ny, cell, ml, mt, vmin, vmax,
                                   kind))
        o.extend(_hit_grid(matrix, counts, nx, ny, cell, ml, mt))

    if mark is not None:
        mi, mj = mark
        if 0 <= mi < nx and 0 <= mj < ny:
            o.append(f'<rect x="{ml + mi * cell:.1f}" '
                     f'y="{mt + (ny - 1 - mj) * cell:.1f}" '
                     f'width="{cell:.1f}" height="{cell:.1f}" fill="none" '
                     f'stroke="var(--ink)" stroke-width="2"/>')

    for ov in (overlay or []):
        oi, oj = ov["cell"]
        cx = ml + (oi + 0.5) * cell
        cy = mt + (ny - 1 - oj + 0.5) * cell
        r = ov.get("r", cell * 0.42)
        o.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" '
                 f'fill="{ov.get("fill", "var(--cat2)")}" '
                 f'stroke="var(--surface)" stroke-width="1.5" '
                 f'opacity="{ov.get("opacity", 0.9)}" '
                 f'data-tip="{_esc(ov.get("tip", ""))}"/>')

    bx = ml + nx * cell + 14
    o.append(f'<defs><linearGradient id="{cid}g" x1="0" y1="1" x2="0" y2="0">')
    for s in range(11):
        t = s / 10
        col = diverging(2 * t - 1) if kind == "diverging" else sequential(t)
        o.append(f'<stop offset="{t}" stop-color="{col}"/>')
    o.append("</linearGradient></defs>")
    bh = ny * cell
    o.append(f'<rect x="{bx}" y="{mt}" width="10" height="{bh}" '
             f'fill="url(#{cid}g)" stroke="var(--axis)" stroke-width="0.5"/>')
    for frac, val in ((0.0, vmin), (0.5, (vmin + vmax) / 2), (1.0, vmax)):
        yy = mt + bh - frac * bh
        o.append(f'<text x="{bx + 14}" y="{yy + 3.5:.1f}" font-size="10">'
                 f'{_fmt(val, 1)}</text>')
    if unit:
        o.append(f'<text x="{bx + 14}" y="{mt - 5}" font-size="10">'
                 f'{_esc(unit)}</text>')

    if xlabel:
        o.append(f'<text x="{ml + nx * cell / 2:.1f}" y="{h - 6}" '
                 f'font-size="11" text-anchor="middle">{_esc(xlabel)}</text>')
    if ylabel:
        o.append(f'<text x="{-(mt + bh / 2):.1f}" y="12" font-size="11" '
                 f'transform="rotate(-90)" text-anchor="middle">'
                 f'{_esc(ylabel)}</text>')
    o.append("</svg>")
    # The cell values ride in the payload and the tooltip is formatted in JS
    # from `data-c`, rather than a `data-tip` string per rect. At the spec's
    # 8-bins-per-cell sub-cell map that is 25 600 cells, where the difference
    # is megabytes of redundant text in a file that has to stay emailable.
    payload = {
        "type": "grid", "id": cid,
        "v": None if big else matrix, "n": None if big else counts,
        "unit": unit, "x0": x_origin, "y0": y_origin,
    }
    return (f'<figure class="chartbox" style="margin:0">{"".join(o)}'
            f'<script type="application/json" class="chartdata">'
            f'{json.dumps(payload)}</script></figure>')


# ---------------------------------------------------------------------------
# Stacked bars, grouped bars, polar
# ---------------------------------------------------------------------------

def stacked_bars(
    x: list[float], cats: list[str], frac: list[list[float | None]],
    colors: list[str], *, title: str = "", xlabel: str = "",
    counts: list[int] | None = None, width: int = 520, height: int = 280,
) -> str:
    """Composition at each x. ``frac[bin][cat]`` should sum to 1."""
    cid = f"c{next(_uid)}"
    ml, mr, mt, mb = 46, 14, 22 if title else 8, 42
    iw, ih = width - ml - mr, height - mt - mb
    nb = len(x)
    bw = iw / max(nb, 1)

    o = [f'<svg viewBox="0 0 {width} {height}" role="img" '
         f'aria-label="{_esc(title)}" data-chart="{cid}">']
    if title:
        o.append(f'<text x="{ml}" y="13" font-size="12" '
                 f'font-weight="600">{_esc(title)}</text>')
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        yy = mt + ih - t * ih
        o.append(f'<line class="gl" x1="{ml}" y1="{yy:.1f}" x2="{ml + iw}" '
                 f'y2="{yy:.1f}" stroke-width="1"/>')
        o.append(f'<text x="{ml - 7}" y="{yy + 3.5:.1f}" font-size="11" '
                 f'text-anchor="end">{t:.0%}</text>')

    for b in range(nb):
        row = frac[b] if b < len(frac) else []
        if not row or all(v is None for v in row):
            continue
        acc = 0.0
        xx = ml + b * bw
        for c, v in enumerate(row):
            v = v or 0.0
            hgt = v * ih
            if hgt <= 0:
                continue
            yy = mt + ih - (acc + v) * ih
            n = counts[b] if counts and b < len(counts) else None
            tip = (f"{_fmt(x[b])}: {cats[c]} {v:.1%}"
                   + (f" (n={n})" if n else ""))
            # 2px surface gap between adjacent fills.
            o.append(f'<rect class="cell" x="{xx + 0.5:.1f}" y="{yy:.1f}" '
                     f'width="{max(bw - 2, 1):.1f}" '
                     f'height="{max(hgt - 1, 0.5):.1f}" fill="{colors[c]}" '
                     f'data-tip="{_esc(tip)}"/>')
            acc += v

    step = max(1, nb // 10)
    for b in range(0, nb, step):
        o.append(f'<text x="{ml + (b + 0.5) * bw:.1f}" y="{mt + ih + 16}" '
                 f'font-size="11" text-anchor="middle">{_fmt(x[b])}</text>')
    if xlabel:
        o.append(f'<text x="{ml + iw / 2:.1f}" y="{height - 4}" '
                 f'font-size="11" text-anchor="middle">{_esc(xlabel)}</text>')
    o.append("</svg>")

    legend = "".join(
        f'<span><i class="sw" style="background:{colors[c]}"></i>'
        f'{_esc(cats[c])}</span>' for c in range(len(cats)))
    return (f'<figure class="chartbox" style="margin:0">'
            f'<div class="legend">{legend}</div>{"".join(o)}'
            f'<script type="application/json" class="chartdata">'
            f'{json.dumps({"type": "cells", "id": cid})}</script></figure>')


def grouped_bars(
    groups: list[str], series: list[dict], *, title: str = "",
    ylabel: str = "", chance: float | None = None,
    width: int = 520, height: int = 280,
) -> str:
    """Paired bars, e.g. a control against its default."""
    cid = f"c{next(_uid)}"
    ml, mr, mt, mb = 56, 14, 22 if title else 8, 46
    iw, ih = width - ml - mr, height - mt - mb
    ng, ns = len(groups), len(series)
    gw = iw / max(ng, 1)
    bw = gw / max(ns + 0.6, 1)

    vals = [v for s in series for v in s["values"] if v is not None]
    if chance is not None:
        vals.append(chance)
    y1 = max(vals) * 1.12 if vals else 1.0
    o = [f'<svg viewBox="0 0 {width} {height}" role="img" '
         f'aria-label="{_esc(title)}" data-chart="{cid}">']
    if title:
        o.append(f'<text x="{ml}" y="13" font-size="12" '
                 f'font-weight="600">{_esc(title)}</text>')
    for t in _nice_ticks(0, y1):
        yy = mt + ih - t / y1 * ih
        if yy < mt:
            continue
        o.append(f'<line class="gl" x1="{ml}" y1="{yy:.1f}" x2="{ml + iw}" '
                 f'y2="{yy:.1f}" stroke-width="1"/>')
        o.append(f'<text x="{ml - 7}" y="{yy + 3.5:.1f}" font-size="11" '
                 f'text-anchor="end">{_fmt(t)}</text>')
    if chance is not None:
        yy = mt + ih - chance / y1 * ih
        o.append(f'<line class="chance" x1="{ml}" y1="{yy:.1f}" '
                 f'x2="{ml + iw}" y2="{yy:.1f}" stroke-width="1"/>')

    for g in range(ng):
        for si, s in enumerate(series):
            v = s["values"][g] if g < len(s["values"]) else None
            if v is None:
                continue
            xx = ml + g * gw + 0.3 * bw + si * bw
            hgt = max(v / y1 * ih, 0.5)
            o.append(f'<rect class="cell" x="{xx + 1:.1f}" '
                     f'y="{mt + ih - hgt:.1f}" width="{max(bw - 2, 1):.1f}" '
                     f'height="{hgt:.1f}" rx="2" fill="{s["color"]}" '
                     f'data-tip="{_esc(f"{groups[g]} / {s["label"]}: "
                                       f"{_fmt(v)}")}"/>')
        o.append(f'<text x="{ml + (g + 0.5) * gw:.1f}" y="{mt + ih + 16}" '
                 f'font-size="11" text-anchor="middle">{_esc(groups[g])}</text>')
    o.append(f'<line class="ax" x1="{ml}" y1="{mt + ih}" x2="{ml + iw}" '
             f'y2="{mt + ih}" stroke-width="1"/>')
    if ylabel:
        o.append(f'<text x="{-(mt + ih / 2):.1f}" y="13" font-size="11" '
                 f'transform="rotate(-90)" text-anchor="middle">'
                 f'{_esc(ylabel)}</text>')
    o.append("</svg>")
    legend = ""
    if ns >= 2:
        legend = '<div class="legend">' + "".join(
            f'<span><i class="sw" style="background:{s["color"]}"></i>'
            f'{_esc(s["label"])}</span>' for s in series) + "</div>"
    return (f'<figure class="chartbox" style="margin:0">{legend}{"".join(o)}'
            f'<script type="application/json" class="chartdata">'
            f'{json.dumps({"type": "cells", "id": cid})}</script></figure>')


def polar_bars(
    values: list[float | None], *, title: str = "", unit: str = "",
    color: str = "var(--cat1)", size: int = 260,
    labels: tuple[str, ...] = ("E", "NE", "N", "NW", "W", "SW", "S", "SE"),
) -> str:
    """Sector values on their own angular axes.

    A radar is legitimate here precisely because the axes *are* angular -- this
    is a physical direction, not an arbitrary multi-attribute comparison.
    """
    cid = f"c{next(_uid)}"
    n = len(values)
    cx = cy = size / 2
    R = size / 2 - 34
    vals = [v for v in values if v is not None]
    mx = max(vals) if vals else 1.0
    mx = mx if mx > 0 else 1.0

    o = [f'<svg viewBox="0 0 {size} {size}" role="img" '
         f'aria-label="{_esc(title)}" data-chart="{cid}">']
    if title:
        o.append(f'<text x="6" y="13" font-size="12" '
                 f'font-weight="600">{_esc(title)}</text>')
    for f in (0.25, 0.5, 0.75, 1.0):
        o.append(f'<circle cx="{cx}" cy="{cy}" r="{R * f:.1f}" fill="none" '
                 f'class="gl" stroke-width="1"/>')
    o.append(f'<text x="{cx + 3}" y="{cy - R:.1f}" font-size="9.5">'
             f'{_fmt(mx, 1)}</text>')

    step = 2 * math.pi / n
    for i, v in enumerate(values):
        a0 = -math.pi + i * step
        a1 = a0 + step
        r = 0 if v is None else max(v, 0) / mx * R
        if r <= 0:
            continue
        x0 = cx + r * math.cos(a0)
        y0 = cy - r * math.sin(a0)
        x1 = cx + r * math.cos(a1)
        y1 = cy - r * math.sin(a1)
        large = 0
        o.append(
            f'<path class="cell" d="M {cx} {cy} L {x0:.1f} {y0:.1f} '
            f'A {r:.1f} {r:.1f} 0 {large} 0 {x1:.1f} {y1:.1f} Z" '
            f'fill="{color}" opacity="0.65" stroke="var(--surface)" '
            f'stroke-width="1.5" '
            f'data-tip="{_esc(f"{labels[i % len(labels)]}: {_fmt(v)} {unit}")}"/>')
        am = (a0 + a1) / 2
        o.append(f'<text x="{cx + (R + 13) * math.cos(am):.1f}" '
                 f'y="{cy - (R + 13) * math.sin(am) + 3.5:.1f}" '
                 f'font-size="10" text-anchor="middle">'
                 f'{_esc(labels[i % len(labels)])}</text>')
    o.append("</svg>")
    return (f'<figure class="chartbox" style="margin:0">{"".join(o)}'
            f'<script type="application/json" class="chartdata">'
            f'{json.dumps({"type": "cells", "id": cid})}</script></figure>')


def quiver_over_heat(
    q: list[list[float]], heat: list[list[float | None]], size: int,
    goal: tuple[int, int], *, title: str = "", cell: float = 16.0,
    sinks: list[dict] | None = None, unit: str = "deg",
) -> str:
    """The ``q`` field as arrows over a sequential background of ``|err|``.

    The only place in the report that draws the field itself. A sink is a
    *location* and no aggregate can point at one -- but the sink cells and
    basin sizes are recorded as numbers by ``flow.py``, and this illustrates
    them rather than being the means of discovering them.
    """
    cid = f"c{next(_uid)}"
    ml, mt, mr, mb = 34, 22 if title else 8, 60, 30
    w, h = ml + size * cell + mr, mt + size * cell + mb
    vals = [v for row in heat for v in row if v is not None]
    vmin, vmax = (min(vals), max(vals)) if vals else (0.0, 1.0)
    if vmax <= vmin:
        vmax = vmin + 1e-9

    o = [f'<svg viewBox="0 0 {w} {h}" role="img" '
         f'aria-label="{_esc(title)}" data-chart="{cid}">']
    if title:
        o.append(f'<text x="{ml}" y="13" font-size="12" '
                 f'font-weight="600">{_esc(title)}</text>')

    for i in range(size):
        for j in range(size):
            v = heat[i][j]
            xx, yy = ml + i * cell, mt + (size - 1 - j) * cell
            t = 0.0 if v is None else (v - vmin) / (vmax - vmin)
            o.append(f'<rect class="cell" x="{xx:.1f}" y="{yy:.1f}" '
                     f'width="{cell:.1f}" height="{cell:.1f}" '
                     f'fill="{sequential(t)}" opacity="0.55" '
                     f'data-tip="{_esc(f"({i}, {j}) — {_fmt(v)} {unit}")}"/>')

    mag = [math.hypot(*v) for v in q] or [1.0]
    mmax = max(mag) or 1.0
    for i in range(size):
        for j in range(size):
            qx, qy = q[i * size + j]
            m = math.hypot(qx, qy)
            if m < 1e-12:
                continue
            L = 0.42 * cell * (m / mmax) ** 0.5
            ux, uy = qx / m, qy / m
            cx = ml + (i + 0.5) * cell
            cy = mt + (size - 1 - j + 0.5) * cell
            x2, y2 = cx + ux * L, cy - uy * L
            o.append(f'<line x1="{cx - ux * L:.1f}" y1="{cy + uy * L:.1f}" '
                     f'x2="{x2:.1f}" y2="{y2:.1f}" stroke="var(--ink)" '
                     f'stroke-width="1.1" opacity="0.75"/>')
            o.append(f'<circle cx="{x2:.1f}" cy="{y2:.1f}" r="1.6" '
                     f'fill="var(--ink)" opacity="0.75"/>')

    gi, gj = goal
    o.append(f'<rect x="{ml + gi * cell:.1f}" '
             f'y="{mt + (size - 1 - gj) * cell:.1f}" width="{cell:.1f}" '
             f'height="{cell:.1f}" fill="none" stroke="var(--ink)" '
             f'stroke-width="2.5"/>')

    for sk in (sinks or []):
        for (si, sj) in sk["cells"]:
            o.append(f'<circle cx="{ml + (si + 0.5) * cell:.1f}" '
                     f'cy="{mt + (size - 1 - sj + 0.5) * cell:.1f}" '
                     f'r="{min(3 + sk["basin"] ** 0.5, cell * 0.45):.1f}" '
                     f'fill="var(--cat2)" stroke="var(--surface)" '
                     f'stroke-width="1.5" opacity="0.92" '
                     f'data-tip="{_esc(f"spurious sink at ({si}, {sj}), "
                                       f"basin {sk["basin"]} cells")}"/>')
    o.append("</svg>")
    return (f'<figure class="chartbox" style="margin:0">{"".join(o)}'
            f'<script type="application/json" class="chartdata">'
            f'{json.dumps({"type": "cells", "id": cid})}</script></figure>')


def stat_tile(label: str, value: str, sub: str = "",
              compare: list[tuple[str, str]] | None = None) -> str:
    """A number is the chart. A one-bar bar chart is the anti-pattern."""
    cmp_html = ""
    if compare:
        rows = "".join(f"<span>{_esc(k)} <b>{_esc(v)}</b></span>"
                       for k, v in compare)
        cmp_html = f'<div class="cmp">{rows}</div>'
    return (f'<div class="tile"><div class="label">{_esc(label)}</div>'
            f'<div class="value">{_esc(value)}</div>'
            f'<div class="sub">{_esc(sub)}</div>{cmp_html}</div>')


def table(headers: list[str], rows: list[list[str]],
          *, summary: str = "table") -> str:
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in r) + "</tr>"
                   for r in rows)
    return (f'<details open><summary>{_esc(summary)}</summary><table>'
            f'<thead><tr>{head}</tr></thead><tbody>{body}</tbody>'
            f'</table></details>')


__all__ = [
    "grouped_bars", "heatmap", "line_chart", "polar_bars", "quiver_over_heat",
    "stacked_bars", "stat_tile", "table",
]
