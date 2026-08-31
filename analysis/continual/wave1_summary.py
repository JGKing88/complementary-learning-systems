"""Aggregate Wave 1 into the tables the paper needs.

Four questions, in the order they have to be answered:

  A   What is the *best* the naive pretrained control can be made to do?
      Until that is settled, every method comparison is against a strawman.
  A2  Same for the from-scratch control, where `init_log_std` actually bites.
  B   Where does Experience Replay sit, as a function of stored bytes?
  C   Where does online EWC sit, as a function of lambda?

Every row is mean +/- SEM over seeds, and every row carries the cost axes
alongside the score, because the deliverable is a frontier and not a
leaderboard (plan section 0.1). `retained` -- the mean over the envs the stream
has already left -- is the headline; `current` is the plasticity check that
stops a method scoring well by freezing the network.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

from . import metrics as M


def _mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def _sem(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if len(xs) < 2:
        return float("nan")
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1) / len(xs))


def _load_group(d: str, pattern: str, strip_seed: str) -> dict[str, list[dict]]:
    """-> {config_label: [summary per seed]}"""
    groups: dict[str, list[dict]] = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(d, pattern))):
        try:
            hist = json.load(open(p))
        except Exception as e:
            print(f"  ! unreadable {os.path.basename(p)}: {e}")
            continue
        if not hist.get("blocks"):
            continue
        label = re.sub(strip_seed, "", os.path.splitext(os.path.basename(p))[0])
        groups[label].append(M.summarize(hist))
    return dict(groups)


def _table(title: str, groups: dict[str, list[dict]], sort_key="retained",
           extra_cols: tuple[str, ...] = ()) -> list[tuple[str, dict]]:
    print(f"\n{title}")
    if not groups:
        print("  (nothing found)")
        return []
    hdr = (f"  {'config':<34} {'n':>3} {'retained':>16} {'current':>16} "
           f"{'forget':>8} {'stab.gap':>9} {'eps/crit':>9} {'bytes':>12}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    rows = []
    for label, rs in groups.items():
        row = {k: _mean([r[k] for r in rs]) for k in
               ("retained", "current_env", "forgetting", "stability_gap",
                "episodes_to_criterion", "state_bytes")}
        row["retained_sem"] = _sem([r["retained"] for r in rs])
        row["current_sem"] = _sem([r["current_env"] for r in rs])
        row["n"] = len(rs)
        rows.append((label, row))
    rows.sort(key=lambda kv: (-kv[1][sort_key]
                              if not math.isnan(kv[1][sort_key]) else 0))
    for label, r in rows:
        print(f"  {label:<34} {r['n']:>3} "
              f"{r['retained']:>9.4f} +/-{r['retained_sem']:<5.4f} "
              f"{r['current_env']:>9.4f} +/-{r['current_sem']:<5.4f} "
              f"{r['forgetting']:>8.3f} {r['stability_gap']:>9.3f} "
              f"{r['episodes_to_criterion']:>9.1f} {r['state_bytes']/1e6:>10.1f}MB")
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", required=True)
    args = p.parse_args()
    d = args.dir

    print("=" * 118)
    print("WAVE 1 SUMMARY   docs/CONTINUAL_CONTROLS_PLAN.md sections 3 and 4")
    print("=" * 118)

    a = _table("A. Tier-1 tuning, PRETRAINED control (naive SGD, no method)",
               _load_group(d, "A_*.json", r"_s\d+$"))
    ab = _table("A-batch. W1 sensitivity: batch_envs=16 against the headline regime",
                _load_group(d, "Abatch_*.json", r"_s\d+$"))
    a2 = _table("A2. Tier-1 tuning, FROM-SCRATCH control (init_log_std, W5)",
                _load_group(d, "A2_*.json", r"_s\d+$"))
    r = _table("R. Matched reference: method=none at B/C's exact configuration",
               _load_group(d, "R_*.json", r"_s\d+$"))
    b = _table("B. Experience Replay", _load_group(d, "B_*.json", r"_s\d+$"))
    c = _table("C. Online EWC", _load_group(d, "C_*.json", r"_s\d+$"))

    print("\n" + "-" * 118)
    print("READING")
    ref = r[0][1]["retained"] if r else float("nan")
    if r:
        print(f"  Reference (no method, pretrained, lr=1e-3): "
              f"retained {ref:.4f}, current {r[0][1]['current_env']:.4f}")
    if a:
        best_label, best = a[0]
        print(f"  Best naive control: {best_label} -> retained {best['retained']:.4f} "
              f"(+/-{best['retained_sem']:.4f}), current {best['current_env']:.4f}")
        print("    ^ this is what every method must be compared against, not the")
        print("      recorded default -- an untuned control is a strawman.")
    for name, rows in (("Experience Replay", b), ("Online EWC", c)):
        if not rows:
            continue
        label, top = rows[0]
        delta = top["retained"] - ref if ref == ref else float("nan")
        print(f"  Best {name}: {label} -> retained {top['retained']:.4f} "
              f"(+/-{top['retained_sem']:.4f}), current {top['current_env']:.4f}, "
              f"{top['state_bytes']/1e6:.1f} MB stored")
        if delta == delta:
            print(f"    vs matched reference: {delta:+.4f} retained")
    print("-" * 118)
    print("  Cost axes not in this table, because they are constants of the")
    print("  method rather than measurements: the Hopfield store acquires an env")
    print("  in 1 episode and 0 gradient steps; every row above took 200 of each.")
    print("-" * 118)


if __name__ == "__main__":
    main()
