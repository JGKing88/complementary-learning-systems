"""Task-eval series of two navigate runs side by side, windowed over updates.

    python -m analysis.sensory_key.compare_runs gmlp=<log> distal=<log>

Reads the ``[navigate_uN] task={...}`` lines the trainer prints and averages
each metric over update windows, so no claim rests on one eval point.
"""
from __future__ import annotations

import re
import sys

import numpy as np

KEYS = ["found_rate", "steps_first", "reaches_post", "steps_per_reach",
        "revisit_found", "revisit_steps_first", "revisit_steps_per_reach",
        "cos_aq_pre", "cos_aq_post"]
WINDOWS = [(0, 500), (500, 1000), (1000, 2000), (2000, 3000), (3000, 4000)]


def series(path):
    rows = []
    for ln in open(path):
        m = re.search(r"\[navigate_u(\d+)\] task=(\{.*\})", ln)
        if m:
            rows.append((int(m.group(1)), eval(m.group(2), {"nan": float("nan")})))
    return rows


def main():
    runs = dict(a.split("=", 1) for a in sys.argv[1:])
    S = {n: series(p) for n, p in runs.items()}
    for n, rows in S.items():
        print(f"{n}: {len(rows)} eval points, u{rows[0][0]}..u{rows[-1][0]}")
    dists = sorted(next(iter(S.values()))[0][1])
    for nd in dists:
        print(f"\n=== {nd} distractors: window means (n = eval points) ===")
        print(f"{'window':>11} {'run':7} {'n':>2} " + " ".join(f"{k[:13]:>13}" for k in KEYS))
        for lo, hi in WINDOWS:
            for n, rows in S.items():
                sel = [r[nd] for u, r in rows if lo < u <= hi]
                if not sel:
                    continue
                vals = [np.nanmean([s.get(k, np.nan) for s in sel]) for k in KEYS]
                print(f"u{lo:>4}-{hi:<5} {n:7} {len(sel):2} " + " ".join(f"{v:13.3f}" for v in vals))


if __name__ == "__main__":
    main()
