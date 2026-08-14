"""Turn the slurm logs of a wave into the markdown tables the tracker wants.

`nav_tri_status.sh` answers "what is running"; this answers "what happened",
in the form that gets pasted into `docs/EXPERIMENTS_NAV_TRI.md`. Kept as code
rather than done by hand each wave because the eval lines are python dict
reprs embedded in a log, transcribing them is error-prone, and the resulting
table is the thing every conclusion in that document rests on.

Emits two tables:

  curve    mean_coverage at n_dist=0 against update index, one row per run.
           Runs are compared at a MATCHED update index, never at each run's
           own end -- pi_fiete is ~1.6x slower than mit_normal_gpu through node
           contention, so equal wall-clock is not equal training.
  final    the last eval of each run: coverage at each distractor level, and
           nav (success_rate / mean_steps) where the run evaluated it.

Usage:
    python -m analysis.nav_tri.collect_results --prefix w1
    python -m analysis.nav_tri.collect_results --at 100 200 300 400
"""
from __future__ import annotations

import argparse
import ast
import glob
import os
import re

LOGDIR = "/orcd/pool/003/jackking/cls_runs/logs"

_VARIANT = re.compile(r"=== nav_tri variant=(\S+)")
_EVAL = re.compile(r"\[(\S+)\] (nav|disc|expl)=(\{.*\})\s*$")
_UPDATE = re.compile(r"navigate_u(\d+)")
_SPU = re.compile(r"s/u=([0-9.]+)")


def _parse(path):
    """(variant, {update: {kind: {n_dist: metrics}}}, s_per_update, last_u)."""
    variant, evals, spu, last_u = None, {}, None, 0
    with open(path, errors="replace") as fh:
        for line in fh:
            m = _VARIANT.search(line)
            if m:
                variant = m.group(1)
                continue
            m = _SPU.search(line)
            if m:
                spu = float(m.group(1))
            m = re.search(r"  u(\d+)\(", line)
            if m:
                last_u = max(last_u, int(m.group(1)))
            m = _EVAL.search(line)
            if m:
                tag, kind, body = m.groups()
                u = _UPDATE.search(tag)
                key = int(u.group(1)) if u else -1      # -1 = after_navigate
                try:
                    evals.setdefault(key, {})[kind] = ast.literal_eval(body)
                except (ValueError, SyntaxError):
                    pass
    return variant, evals, spu, last_u


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prefix", default="",
                   help="only variants starting with this, e.g. w1")
    p.add_argument("--at", type=int, nargs="+", default=None,
                   help="update indices for the curve table; default = every "
                        "index that all matched runs share")
    p.add_argument("--logdir", default=LOGDIR)
    args = p.parse_args()

    runs = []
    for path in sorted(glob.glob(os.path.join(args.logdir, "nav_tri_*.out"))):
        variant, evals, spu, last_u = _parse(path)
        if not variant or not variant.startswith(args.prefix):
            continue
        if not evals:
            continue
        runs.append({"job": os.path.basename(path)[8:-4], "variant": variant,
                     "evals": evals, "spu": spu, "last_u": last_u})
    if not runs:
        print(f"no runs matching prefix {args.prefix!r} in {args.logdir}")
        return

    # --- curve -------------------------------------------------------------
    if args.at:
        cols = args.at
    else:
        shared = None
        for r in runs:
            us = {u for u in r["evals"] if u > 0}
            shared = us if shared is None else (shared & us)
        cols = sorted(shared or [])
    print("#### coverage curve — `mean_coverage` at `n_dist=0`\n")
    print("| variant | " + " | ".join(f"u{c}" for c in cols) + " | s/u | last |")
    print("|---" * (len(cols) + 3) + "|")
    for r in sorted(runs, key=lambda r: r["variant"]):
        cells = []
        for c in cols:
            e = r["evals"].get(c, {}).get("expl", {})
            v = e.get(0, e.get("0", {})).get("mean_coverage")
            cells.append(f"{v:.4f}" if v is not None else "—")
        print(f"| `{r['variant']}` | " + " | ".join(cells)
              + f" | {r['spu'] or float('nan'):.1f} | u{r['last_u']} |")

    # --- final -------------------------------------------------------------
    print("\n#### last eval of each run\n")
    print("| variant | update | cov d0 | cov d10 | cells/step d0 "
          "| nav d0 (sr / steps) | nav d10 (sr / steps) |")
    print("|---|---|---|---|---|---|---|")
    for r in sorted(runs, key=lambda r: r["variant"]):
        u = max(r["evals"]) if -1 not in r["evals"] else -1
        block = r["evals"][u]
        expl, nav = block.get("expl", {}), block.get("nav", {})

        def _e(nd, k):
            d = expl.get(nd, expl.get(str(nd), {}))
            v = d.get(k)
            return f"{v:.4f}" if isinstance(v, (int, float)) else "—"

        def _n(nd):
            d = nav.get(nd, nav.get(str(nd), {}))
            if not d:
                return "—"
            return f"{d.get('success_rate', float('nan')):.3f} / " \
                   f"{d.get('mean_steps', float('nan')):.1f}"
        label = "final" if u == -1 else f"u{u}"
        print(f"| `{r['variant']}` | {label} | {_e(0,'mean_coverage')} "
              f"| {_e(10,'mean_coverage')} | {_e(0,'cells_per_step')} "
              f"| {_n(0)} | {_n(10)} |")

    print("\nReference lines (docs §3.1 / §3.3.1): coverage ceiling 0.5025, "
          "lawnmower 0.478 (unreachable — position is not decodable),\n"
          "billiard at the instructed wall_resolution **0.352**, "
          "run-and-tumble 0.274, uniform random walk 0.178.\n"
          "mean_steps reference at |a|=1: 10.1 at cos(q,goal)=0.99 "
          "(n_dist<=3), 15.3 at cos=0.70 (n_dist=10).")


if __name__ == "__main__":
    main()
