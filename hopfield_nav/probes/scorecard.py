"""Rank every evaluated checkpoint in the line on all three target metrics.

The deliverable is one policy that scores well on `mean_coverage` (up),
`success_rate` (up) and `mean_steps` (down) at once. Two facts make "read the
last eval of the best run" the wrong way to find it:

- §3x: coverage peaks partway through an explore run and then falls, so the
  final checkpoint is routinely worse than one from the middle.
- §3r: `success_rate` and `mean_steps` decouple in composite policies, so a run
  can look better on one and worse on the other at the same update.

So this scans the eval logs, treats every (run, update) as a candidate, and
reports the Pareto front -- the candidates no other candidate beats on all
three at once. Ranking by a weighted sum is deliberately not the default: the
brief never gave weights, and inventing them would hide the tradeoff rather
than show it.
"""
import argparse
import glob
import os

# From training/, not from the sibling probe: probes/ is the CLI layer and
# test_layering rule 5 forbids a probe importing a probe.
from ..training.eval_log import parse_log


def candidates(paths, n_dist):
    out = []
    for p in paths:
        try:
            rows = parse_log(p)
        except Exception:  # noqa: BLE001 -- a truncated log should not stop the scan
            continue
        by_update = {}
        for r in rows:
            # A run's final `[after_navigate]` eval has no update number; it
            # duplicates the last numbered eval, so dropping it loses nothing
            # and keeps `update` an int for sorting and formatting.
            if r.get("n_dist") != n_dist or r.get("update") is None:
                continue
            by_update.setdefault(r["update"], {}).update(r)
        name = os.path.basename(p).replace("ee_", "").replace(".out", "")
        for upd, r in by_update.items():
            cov, suc, stp = (r.get("mean_coverage"), r.get("success_rate"),
                             r.get("mean_steps"))
            spd = r.get("mean_speed")
            if None in (cov, suc, stp):
                continue
            out.append({"run": name, "update": upd, "mean_coverage": cov,
                        "success_rate": suc, "mean_steps": stp,
                        "mean_speed": spd if spd is not None else float("nan")})
    return out


def dominates(a, b):
    """True if a is at least as good on all three and strictly better on one."""
    ge = (a["mean_coverage"] >= b["mean_coverage"]
          and a["success_rate"] >= b["success_rate"]
          and a["mean_steps"] <= b["mean_steps"])
    gt = (a["mean_coverage"] > b["mean_coverage"]
          or a["success_rate"] > b["success_rate"]
          or a["mean_steps"] < b["mean_steps"])
    return ge and gt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", nargs="+", default=None)
    p.add_argument("--n_dist", type=int, default=0)
    p.add_argument("--all", action="store_true",
                   help="list every candidate, not just the Pareto front")
    a = p.parse_args()
    paths = a.logs or sorted(glob.glob(
        "/orcd/pool/003/jackking/cls_runs/logs/ee_*.out"))

    cands = candidates(paths, a.n_dist)
    if not cands:
        print("no candidates found")
        return
    front = [c for c in cands if not any(dominates(o, c) for o in cands)]
    show = sorted(cands if a.all else front,
                  key=lambda c: (-c["mean_coverage"], c["mean_steps"]))

    print(f"{len(cands)} evaluated checkpoints at n_dist={a.n_dist}; "
          f"{len(front)} on the Pareto front\n")
    print(f"{'run':<22}{'update':>7}{'coverage':>10}{'success':>9}{'steps':>9}"
          f"{'speed':>8}")
    for c in show:
        star = " *" if c in front else "  "
        # `mean_steps` averages successes only (metrics.py:350), so a policy
        # that only solves nearby goals reports a flattering number. Flag it
        # rather than silently ranking on it -- see §3ab.
        warn = " <- steps unreliable, low success" if c["success_rate"] < 0.9 else ""
        print(f"{star}{c['run']:<20}{c['update']:>7}{c['mean_coverage']:>10.3f}"
              f"{c['success_rate']:>9.3f}{c['mean_steps']:>9.1f}"
              f"{c['mean_speed']:>8.3f}{warn}")
    print("\n* = Pareto-optimal: nothing else beats it on all three at once.")
    print("mean_speed = mean(start_dist / steps), the distance-normalized "
          "efficiency; prefer it to mean_steps when success_rate differs.")


if __name__ == "__main__":
    main()
