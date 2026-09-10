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
    p.add_argument("--vs_billiard", action="store_true",
                   help="also report the billiard swept at each model's OWN "
                        "realized speed, and the ratio. Use this whenever "
                        "comparing models that do not share a world -- two "
                        "arms on different encoders cannot be rolled on the "
                        "same trajectories, but both can be divided by a "
                        "billiard, which does not depend on the encoder.")
    p.add_argument("--labels", nargs="*", default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    d = json.load(open(a.json))
    labels = a.labels or d["labels"]
    rows = {}
    hdr = ("  %-24s %6s %7s %7s %7s %7s"
           % ("checkpoint", "steps", "swept", "sd", "union", "speed"))
    if a.vs_billiard:
        hdr += " %8s %8s" % ("bill@sp", "swept_eff")
    print(hdr)
    for lab in labels:
        paths = [t["by_ckpt"][lab]["path"] for t in d["trials"]]
        per, union, T = swept_for(paths, a.size, a.radius, a.at_step)
        # Realized speed, from the same truncated paths the swept used, so the
        # billiard reference is matched to what was actually scored.
        arr = [np.asarray(p, dtype=np.float64)[:T] for p in paths]
        speed = float(np.mean([
            np.linalg.norm(np.diff(p, axis=0), axis=-1).mean() for p in arr]))
        rows[lab] = {"swept": float(per.mean()), "sd": float(per.std()),
                     "union": float(union), "n": len(paths), "steps": T,
                     "realized_speed": speed}
        line = ("  %-24s %6d %7.3f %7.3f %7.3f %7.3f"
                % (lab[-24:], T, per.mean(), per.std(), union, speed))
        if a.vs_billiard:
            from analysis.nav_tri.coverage_baselines import swept_billiard
            ref = swept_billiard(speed, a.size, T, a.radius)
            eff = float(per.mean()) / max(ref, 1e-9)
            rows[lab]["billiard_at_speed"] = ref
            rows[lab]["swept_efficiency"] = eff
            line += " %8.3f %8.3f" % (ref, eff)
        print(line)
    print("  swept is CUMULATIVE, so the step count is part of the number.")
    if a.vs_billiard:
        print("  swept_eff = swept / billiard at THIS model's own realized "
              "speed.\n"
              "  It exists because swept is monotone in speed (§19.2), so a "
              "model that\n"
              "  sweeps more by moving faster has not explored better -- and "
              "because two\n"
              "  models trained on different encoders cannot share "
              "trajectories at all,\n"
              "  while a billiard in an empty box does not depend on the "
              "encoder.\n"
              "  The reference is coverage_baselines' billiard, which differs "
              "from the\n"
              "  table in swept.py's docstring; see swept_billiard's "
              "docstring. Ratios\n"
              "  against one reference are safe; the absolute is "
              "implementation-named.")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(rows, fh, indent=2)
        print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
