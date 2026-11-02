"""Idea 3's in-training evals as windowed tables (GRID_MLP_NAV_PLAN §7.5).

    python -m analysis.idea3.summarise name=<nav_p2_JOB.out> [...] [--window 500]

Reads the ``[navigate_uN] task=`` and ``lifetime=`` lines of each training log
and prints, per window of updates, the mean of each metric over the evals in
it, with the number of evals. One eval swings a lot, so a window is the unit to
read. Logs without lifetime lines (idea 1, Agent-HaSH) print task metrics only,
at distractor count 0.
"""
from __future__ import annotations

import argparse
import ast
import re
from collections import defaultdict

import numpy as np

TASK = ("found_rate", "steps_first", "revisit_found", "revisit_steps_first",
        "steps_per_reach")
LIFE = ("first_found", "known_reached", "known_steps", "unknown_found")
LINE = re.compile(r"\[navigate_u(\d+)\] (task|lifetime)=(.*)$")


def read(path):
    out = defaultdict(dict)
    with open(path, errors="replace") as f:
        for ln in f:
            m = LINE.search(ln.strip())
            if not m:
                continue
            u, kind, body = int(m.group(1)), m.group(2), m.group(3)
            d = ast.literal_eval(body.replace("nan", "None"))
            if kind == "task":
                d = d.get(0, {})
            out[u].update({k: v for k, v in d.items() if v is not None})
    return dict(sorted(out.items()))


def table(name, series, window):
    keys = [k for k in TASK + LIFE if any(k in v for v in series.values())]
    print(f"\n{name}  ({len(series)} evals, last u{max(series) if series else 0})")
    print("window      n  " + "  ".join(f"{k:>19s}" for k in keys))
    by = defaultdict(list)
    for u, v in series.items():
        by[(u - 1) // window].append(v)
    for w in sorted(by):
        rows = by[w]
        cells = []
        for k in keys:
            vals = [r[k] for r in rows if k in r]
            cells.append(f"{np.mean(vals):19.3f}" if vals else f"{'':>19s}")
        print(f"u{w * window:>4d}-{(w + 1) * window:<5d} {len(rows):2d}  " + "  ".join(cells))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("logs", nargs="+", help="name=path")
    p.add_argument("--window", type=int, default=500)
    a = p.parse_args()
    for item in a.logs:
        name, path = item.split("=", 1)
        table(name, read(path), a.window)


if __name__ == "__main__":
    main()
