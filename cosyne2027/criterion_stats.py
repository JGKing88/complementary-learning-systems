"""Episodes to criterion and success on earlier arenas, per model (N=5).

Run: ~/.conda/envs/cls/bin/python cosyne2027/criterion_stats.py

Criterion: the first update u in an arena's 200-update window whose next 25
held-out evaluations on that arena (success = goal within T_eval = 100
steps) average >= 0.9. The suite's metrics.episodes_to_criterion uses the
same threshold and window but trailing, so it cannot report fewer than 25;
the forward window can report 1 and treats every model the same. "Never" is
an arena that does not reach criterion within its window. Success on
earlier arenas pools the held-out evaluations of arena j during windows
k > j. Medians are over the arenas that reach criterion.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cl_results as cr  # noqa: E402

W, THR, BLOCK = 25, 0.9, 200


def to_criterion(curve, j):
    """First update (1-based within arena j's window) whose next W held-out
    evaluations on arena j average >= THR; None if never within the window."""
    y = curve[j * BLOCK:(j + 1) * BLOCK, j]
    for u in range(BLOCK - W + 1):
        if y[u:u + W].mean() >= THR:
            return u + 1
    return None


def main():
    models = json.load(open(cr.HERE / "data" / "cl_results.json"))["models"]
    for name, m in models.items():
        paths = ([cr.AGENTHASH / f"agenthash_s{s}.json" for s in cr.SEEDS]
                 if name == "Agent-HaSH" else cr.control_paths(m["config"]))
        hits, revisit = [], []
        for p in paths:
            curve = cr.seed_record(json.load(open(p)))[0]
            n = curve.shape[1]
            hits += [to_criterion(curve, j) for j in range(n)]
            for k in range(1, n):
                revisit.extend(curve[k * BLOCK:(k + 1) * BLOCK, :k].ravel())
        got = np.array([h for h in hits if h is not None])
        never = 1 - len(got) / len(hits)
        at1 = np.mean([h == 1 for h in hits])
        q = np.percentile(got, [25, 50, 75]) if len(got) else [np.nan] * 3
        rv = np.asarray(revisit)
        print(f"{name:18s} median {q[1]:5.0f} (IQR {q[0]:.0f}-{q[2]:.0f})  "
              f"never in {BLOCK}: {never:4.0%}  at update 1: {at1:4.0%}  "
              f"earlier-arena success {np.nanmean(rv):.3f} "
              f"(n={np.sum(~np.isnan(rv))})")


if __name__ == "__main__":
    main()
