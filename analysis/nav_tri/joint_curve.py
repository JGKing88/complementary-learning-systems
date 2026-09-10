"""Coverage and navigation for one run, per update, in one table.

The two live in different eval dicts, so reading them apart invites picking a
checkpoint that is good at one and bad at the other -- which is precisely the
failure mode of the whole combined-model problem. This prints them together
and scores each update, so "the best checkpoint" is a defined quantity rather
than an impression.

The score is deliberately explicit about its trade-off:

    score = coverage / COV_REF  +  STEP_REF / mean_steps_all

with `COV_REF` the explore-only best (0.385) and `STEP_REF` the readout-limited
oracle (10.0). Each term is 1.0 when that metric matches the best a *specialist*
achieved, so a score of 2.0 would be a combined model that gave up nothing, and
the two halves are directly comparable. `mean_steps_all` rather than
`mean_steps`, so a policy cannot score by failing its hard trials.

Usage:
    python -m analysis.nav_tri.joint_curve <jobid> [n_dist]
"""
from __future__ import annotations

import ast
import re
import sys

LOG = "/orcd/pool/003/jackking/cls_runs/logs/nav_tri_%s.out"
COV_REF = 0.385     # best explore-only, w2_e_long2 u1150
STEP_REF = 10.0     # oracle mean_steps at cos 1.0, |a| = 1


def main() -> None:
    job = sys.argv[1]
    nd = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    pat = re.compile(r"\[navigate_u(\d+)\] (nav|expl)=(\{.*\})\s*$")
    cov: dict[int, float] = {}
    nav: dict[int, tuple] = {}

    for line in open(LOG % job, errors="replace"):
        m = pat.search(line)
        if not m:
            continue
        u, kind = int(m.group(1)), m.group(2)
        d = ast.literal_eval(re.sub(r"\b(nan|inf)\b", "None", m.group(3)))
        b = d.get(nd)
        if not b:
            continue
        if kind == "expl":
            cov[u] = b["mean_coverage"]
        else:
            tt, ts, ms = b["total_trials"], b["total_successes"], b["mean_steps"]
            allv = (ts * ms + (tt - ts) * 200) / tt if tt else float("nan")
            nav[u] = (b["success_rate"], ms, allv)

    print(f"job {job}, n_dist={nd}   "
          f"score = cov/{COV_REF} + {STEP_REF}/steps_all")
    print(f"{'update':>7s} {'cov':>7s} {'sr':>6s} {'steps':>7s} "
          f"{'all':>7s} {'score':>7s}")
    best = (None, -1.0)
    for u in sorted(set(cov) & set(nav)):
        c = cov[u]
        sr, ms, allv = nav[u]
        score = c / COV_REF + (STEP_REF / allv if allv > 0 else 0.0)
        if score > best[1]:
            best = (u, score)
        print(f"u{u:<6d} {c:7.4f} {sr:6.3f} {ms:7.1f} {allv:7.1f} {score:7.3f}")
    if best[0] is not None:
        print(f"\nbest joint checkpoint: u{best[0]}  score {best[1]:.3f}")
        print("(a specialist-matching combined model would score 2.0)")


if __name__ == "__main__":
    main()
