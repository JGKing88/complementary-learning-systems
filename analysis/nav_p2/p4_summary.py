"""Best checkpoint per P4 arm, selected on the metric that matters.

Reads the training logs directly. Selection is the lowest `mean_steps` at ten
distractors **subject to a success floor**, because the two trade off: a policy
that abandons its distant starts posts a wonderful mean_steps over the easy ones
it kept, and phase 2's whole time-penalty bracket was designed to risk exactly
that. Reporting the pair, plus the speed, is what makes the trade visible.

`mean_steps` is over successful trials only and reads 0.0 rather than NaN at
zero successes (phase-1 finding 8), so zero is excluded rather than treated as
the best possible score.

    python -m analysis.nav_p2.p4_summary
"""
from __future__ import annotations

import argparse
import ast
import re

ARMS = [
    ("p4_x    sigma 0.50  tp 0.05", 21102411),
    ("p4_s12  sigma 0.30  tp 0.05", 21102413),
    ("p4_s18  sigma 0.165 tp 0.05", 21118288),
    ("p4_tp10 sigma 0.50  tp 0.10", 21118290),
    ("p4_tp15 sigma 0.50  tp 0.15", 21118291),
]
PAT = re.compile(r"navigate_u(\d+)\] nav=(\{.*\})")


def rows_for(jid: int, logdir: str):
    out = []
    path = f"{logdir}/nav_p2_{jid}.out"
    try:
        fh = open(path, errors="ignore")
    except OSError:
        return out
    with fh:
        for line in fh:
            m = PAT.search(line)
            if not m:
                continue
            try:
                d = ast.literal_eval(m.group(2))
            except Exception:
                continue
            if 0 in d and 10 in d:
                out.append((int(m.group(1)), d[0], d[10]))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--logdir", default="/orcd/pool/003/jackking/cls_runs/logs")
    p.add_argument("--floor", type=float, default=0.85,
                   help="minimum success_rate at ten distractors")
    args = p.parse_args()

    hdr = (f"{'arm':<28}{'u':>6}{'succ@0':>8}{'steps@0':>9}"
           f"{'succ@10':>9}{'steps@10':>10}{'spd@10':>8}{'evals':>7}")
    print(hdr)
    print("-" * len(hdr))
    for name, jid in ARMS:
        rows = rows_for(jid, args.logdir)
        ok = [r for r in rows
              if r[2]["success_rate"] >= args.floor and r[2]["mean_steps"] > 0]
        if not ok:
            print(f"{name:<28}{'no eval cleared the success floor':>40}")
            continue
        u, d0, d10 = min(ok, key=lambda r: r[2]["mean_steps"])
        print(f"{name:<28}{u:>6}{d0['success_rate']:>8.3f}{d0['mean_steps']:>9.2f}"
              f"{d10['success_rate']:>9.3f}{d10['mean_steps']:>10.2f}"
              f"{d10['mean_speed']:>8.2f}{len(rows):>7}")

    print(f"\nselection: lowest mean_steps at d=10 with success >= {args.floor}")
    print("reference: ideal mean_steps is (10.85 - 1) / speed; at the 2.0 cap "
          "that is 4.9")


if __name__ == "__main__":
    main()
