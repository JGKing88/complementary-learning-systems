"""Width / grid-spacing ratio tables: ratio_tables.py SPEC.json OUT.md

SPEC.json holds one entry per grid:
{"Rat": {"columns": [{"label": "M1", "lo_cm": 38, "hi_cm": 50}, ...],
         "rows": [{"label": "Dorsal CA1, 1.7 m track", "lo_cm": 42, "hi_cm": 52,
                   "studies": ["Diba 2008"]}, ...]},
 "Rat kernel": {"header": "Recording (kernel full width at 0.5)", "columns_from": "Rat", "rows": [...]}}
A cell spans (row lo / column hi) to (row hi / column lo), so it carries the
uncertainty of both ranges. A single value has lo == hi. "header" sets the first header
cell (default "Recording (field width)"), "columns_from" reuses another entry's columns, and
an optional per-row "studies" list adds a final Studies column, which the page turns into
the heat-map tooltip. An optional per-row "group" (e.g. "Retrosplenial cortex") prints a
header row, "| **group** | | ... |", before the first row of each run of that group; rows
without a group are grouped on the page by their leading Dorsal, Intermediate or Ventral.
"""
import json
import sys


def fmt(x):
    """Two significant figures: 0.998 prints as 1.0 and 0.0967 as 0.097."""
    x = float(f"{x:.2g}")
    if x >= 10:
        return f"{x:.0f}"
    if x >= 1:
        return f"{x:.1f}"
    if x >= 0.1:
        return f"{x:.2f}"
    return f"{x:.3f}"


def span(lo, hi):
    """One unit per range: centimetres below 1 m, metres to two decimals at 1 m and above."""
    if max(lo, hi) < 100:
        one = lambda v: f"{v:g}"
        u = " cm"
    else:
        one = lambda v: f"{v / 100:.2f}"
        u = " m"
    return f"{one(lo)}{u}" if lo == hi else f"{one(lo)} to {one(hi)}{u}"


def shown(d):
    """A spec entry may override its printed range, e.g. "about 3 m"."""
    return d.get("show") or span(d["lo_cm"], d["hi_cm"])


def table(spec, cols):
    rows = spec["rows"]
    studies = any(r.get("studies") for r in rows)
    head = f"| {spec.get('header', 'Recording (field width)')} | " + " | ".join(
        f"{c['label']} ({shown(c)})" for c in cols) + (" | Studies" if studies else "") + " |"
    lines = [head, "|---" * (len(cols) + 1 + studies) + "|"]
    group = None
    for r in rows:
        if r.get("group") and r["group"] != group:
            lines.append(f"| **{r['group']}** |" + " |" * (len(cols) + studies))
        group = r.get("group")
        cells = []
        for c in cols:
            only = c.get("only_rows_containing")
            if only and only not in r["label"]:
                cells.append("not comparable")
                continue
            lo = r["lo_cm"] / c["hi_cm"]
            hi = r["hi_cm"] / c["lo_cm"]
            cells.append(fmt(lo) if fmt(lo) == fmt(hi) else f"{fmt(lo)} to {fmt(hi)}")
        if studies:
            cells.append("; ".join(r.get("studies", [])))
        lines.append(f"| {r['label']} ({shown(r)}) | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    spec = json.load(open(sys.argv[1], encoding="utf-8"))
    out = []
    for name, s in spec.items():
        cols = spec[s["columns_from"]]["columns"] if "columns_from" in s else s["columns"]
        out.append(f"### {name}\n\n{table(s, cols)}\n")
    open(sys.argv[2], "w", encoding="utf-8").write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
