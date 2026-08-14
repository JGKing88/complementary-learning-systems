"""Parse `do_eval`'s printed dicts back into records.

`world_setup.do_eval` prints, per eval, one line per evaluator:

    [navigate_u25] nav={0: {'success_rate': ..., ...}, 5: {...}, 10: {...}}
    [navigate_u25] expl={0: {'mean_coverage': ..., ...}, ...}

which is a Python dict repr and so is exactly recoverable. Reading those back
rather than pulling from wandb keeps every comparison runnable on a login node
with no network, and keeps it working for a run killed at the wall-clock limit
-- which is most of them, and precisely the case where the last eval matters.

This lives beside `do_eval` rather than in `probes/` because two probes need it
(`read_eval_log`, `scorecard`) and `probes/` is the CLI layer: a probe importing
a probe is what `test_layering.test_nothing_imports_a_cli` forbids.
"""
from __future__ import annotations

import ast
import os
import re

LINE = re.compile(r"\[(?P<tag>[^\]]+)\]\s+(?P<kind>nav|disc|expl)=(?P<body>\{.*\})\s*$")
UPDATE = re.compile(r"u(\d+)$")


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
