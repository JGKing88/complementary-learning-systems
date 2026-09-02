"""Swept coverage from an `explore_traj` dump, for MATCHED comparisons.

The problem this exists for: `swept_coverage` (§19) postdates several arms, so
`p20_e` has no swept number in its training log while `p26_abspos` logs one
every eval. Comparing the two would cross protocols -- different env draws,
different trial counts, different rollout code -- and the P2 doc has already
been bitten once by a number that looked comparable and was not (§18.4).

`explore_traj` scores several checkpoints on the SAME envs, the SAME starts and
the SAME seed in one process. Reducing its dump here means every arm's swept
number comes off identical trajectories, whatever its training log happened to
record.

Uses `hopfield_nav.evaluation.swept.SweptArea` -- the same class the online
evaluator calls, not a reimplementation -- so the reduction cannot drift from
what training reports.

    python -m analysis.nav_tri.swept_from_traj --json traj.json --radius 1.0
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from hopfield_nav.evaluation.swept import SweptArea


def swept_for(paths, size: int, radius: float, at_step: int | None = None):
    """(per_trial, union) for a list of (T, 2) continuous paths.

    ``at_step`` truncates every path, which is how the 200-step and 1000-step
    numbers in §23 were separated: swept is cumulative, so the horizon is part
    of the metric and quoting one without it is meaningless.
    """
    arr = [np.asarray(p, dtype=np.float64) for p in paths]
    T = min(len(p) for p in arr)
    if at_step is not None:
        T = min(T, int(at_step))
    sa = SweptArea(size, radius, len(arr))
    stack = np.stack([p[:T] for p in arr])          # (B, T, 2)
    for t in range(T):
        sa.add(stack[:, t])
    r = sa.result()
    return r.per_trial, r.union, T


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", required=True, help="an explore_traj dump")
    p.add_argument("--size", type=int, default=20)
    p.add_argument("--radius", type=float, required=True,
                   help="goal_radius; swept area is defined against it, so it "
                        "must match the value the arms were scored under.")
    p.add_argument("--at_step", type=int, default=None)
    p.add_argument("--labels", nargs="*", default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    d = json.load(open(a.json))
    labels = a.labels or d["labels"]
    rows = {}
    print("  %-28s %8s %8s %8s %8s"
          % ("checkpoint", "steps", "swept", "sd", "union"))
    for lab in labels:
        paths = [t["by_ckpt"][lab]["path"] for t in d["trials"]]
        per, union, T = swept_for(paths, a.size, a.radius, a.at_step)
        rows[lab] = {"swept": float(per.mean()), "sd": float(per.std()),
                     "union": float(union), "n": len(paths), "steps": T}
        print("  %-28s %8d %8.3f %8.3f %8.3f"
              % (lab[-28:], T, per.mean(), per.std(), union))
    print("  swept is CUMULATIVE, so the step count is part of the number.")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(rows, fh, indent=2)
        print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
