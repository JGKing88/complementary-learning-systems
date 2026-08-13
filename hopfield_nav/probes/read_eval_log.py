"""Turn `do_eval`'s printed dicts back into a table.

Every run prints, per eval, one line per evaluator:

    [navigate_u25] nav={0: {'success_rate': ..., ...}, 5: {...}, 10: {...}}
    [navigate_u25] expl={0: {'mean_coverage': ..., ...}, ...}

which is a Python dict repr and so is exactly recoverable. Reading them back
here rather than pulling from wandb keeps the comparison runnable on a login
node with no network, and keeps it working for a run that was killed at the
wall-clock limit -- which is most of them, and precisely the case where the
last eval is the one that matters.

    python -m hopfield_nav.probes.read_eval_log \\
        --logs /path/ee_X1_*.out --metric mean_coverage --at last

`--at last` gives one row per run at its final eval, which is the comparison
table. `--at all` gives the whole curve.
"""
from __future__ import annotations

import argparse
import ast
import glob
import os
import re

LINE = re.compile(r"\[(?P<tag>[^\]]+)\]\s+(?P<kind>nav|disc|expl)=(?P<body>\{.*\})\s*$")
UPDATE = re.compile(r"u(\d+)$")

# Which metrics are worth a column, per evaluator, in the order they are shown.
DEFAULT_METRICS = {
    "expl": ("mean_coverage", "union_coverage", "goal_find_rate"),
    "nav": ("success_rate", "mean_steps", "mean_speed"),
    "disc": ("store_success_rate", "store_efficiency"),
}


def parse_log(path: str) -> list[dict]:
    """One record per (eval, evaluator, distractor count)."""
    out: list[dict] = []
    with open(path, errors="replace") as f:
        for line in f:
            m = LINE.search(line)
            if not m:
                continue
            try:
                body = ast.literal_eval(m.group("body"))
            except (ValueError, SyntaxError):
                continue
            tag = m.group("tag")
            um = UPDATE.search(tag)
            update = int(um.group(1)) if um else None
            for n_dist, metrics in body.items():
                if not isinstance(metrics, dict):
                    continue
                out.append({
                    "run": os.path.basename(path),
                    "tag": tag, "update": update,
                    "kind": m.group("kind"), "n_dist": n_dist, **metrics,
                })
    return out


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return "nan" if v != v else f"{v:.3f}"
    return str(v)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logs", nargs="+", required=True,
                   help="log files or globs")
    p.add_argument("--kind", default="expl", choices=("expl", "nav", "disc"))
    p.add_argument("--metrics", nargs="*", default=None,
                   help="default: a per-evaluator set, see DEFAULT_METRICS")
    p.add_argument("--at", default="last", choices=("last", "all"))
    p.add_argument("--label", default=None,
                   help="regex with one group, applied to the filename, used "
                        "as the run label. Default: the whole filename.")
    args = p.parse_args()

    paths: list[str] = []
    for pattern in args.logs:
        paths.extend(sorted(glob.glob(pattern)) or [pattern])

    metrics = args.metrics or DEFAULT_METRICS[args.kind]
    label_re = re.compile(args.label) if args.label else None

    rows = []
    for path in paths:
        recs = [r for r in parse_log(path) if r["kind"] == args.kind]
        if not recs:
            continue
        if args.at == "last":
            # The last eval that actually produced this evaluator. A run killed
            # at the wall has no `after_navigate` line, so "last" has to mean
            # the highest update seen, not a fixed tag.
            last = max(r["update"] for r in recs if r["update"] is not None) \
                if any(r["update"] is not None for r in recs) else None
            recs = [r for r in recs if r["update"] == last]
        name = os.path.basename(path)
        if label_re:
            m = label_re.search(name)
            if m:
                name = m.group(1)
        for r in recs:
            r["run"] = name
            rows.append(r)

    if not rows:
        print("no eval lines found")
        return

    header = ["run", "update", "n_dist", *metrics]
    widths = [max(len(h), *(len(_fmt(r.get(h))) for r in rows))
              for h in header]
    print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("  ".join("-" * w for w in widths))
    for r in sorted(rows, key=lambda r: (r["run"], r["update"] or 0,
                                         r["n_dist"])):
        print("  ".join(_fmt(r.get(h)).ljust(w) for h, w in zip(header, widths)))


if __name__ == "__main__":
    main()
