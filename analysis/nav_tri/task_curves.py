"""The task-regime held-out series, several runs on one figure.

`docs/EXPERIMENTS_TASK_FAITHFUL.md`. A task-regime run prints, every eval,

    [navigate_uN] task={0: {'found_rate': ..., 'revisit_steps_first': ...,
                            'steps_per_reach': ..., 'cos_aq_post': ...}, 10: {...}}

from `evaluate_task` (six held-out arenas, sampled, visits=2). Four panels:
search (found_rate: did visit 1 touch the goal within 200 steps), revisit
steps (steps to the goal from a fresh state with the goal already in
memory -- the continual protocol's number), steps per touch after the
store, and follow_q (cos of the action with the recall) after the store.
One colour per run, d=0 dotted and d=10 solid.

    python -m analysis.nav_tri.task_curves \\
        --log "3 redrawn arenas, wave-1 rule=$CLS_RUNS/logs/nav_p2_22883646.out" \\
        --log "1 arena K=4, wave-1 rule=$CLS_RUNS/logs/nav_p2_22883645.out" \\
        --out $CLS_RESULTS/nav_tri_probe/task_by_update.png
"""
from __future__ import annotations

import argparse
import ast
import re

import numpy as np

_LINE = re.compile(r"^\s*\[navigate_u(\d+)\] task=(\{.*\})\s*$")

PANELS = (
    ("found_rate", "search: goal found within 200 steps", (0.0, 1.03)),
    ("revisit_steps_first", "revisit: steps to goal, fresh state", (0.0, 90.0)),
    ("steps_per_reach", "steps per touch after the store", (0.0, 90.0)),
    ("cos_aq_post", "follow_q after the store", (-0.5, 1.0)),
)


def parse_task_log(path: str) -> dict[int, dict[int, dict[str, float]]]:
    """{update: {n_dist: metrics}} for every `[navigate_uN] task=` line."""
    out: dict[int, dict[int, dict[str, float]]] = {}
    with open(path) as f:
        for line in f:
            m = _LINE.match(line)
            if not m:
                continue
            # `nan` is not a literal for ast; the dict is otherwise plain.
            body = m.group(2).replace("nan", "None")
            d = ast.literal_eval(body)
            out[int(m.group(1))] = {
                int(k): {kk: (float("nan") if vv is None else float(vv))
                         for kk, vv in v.items()}
                for k, v in d.items()
            }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", action="append", required=True,
                   help="label=path; repeatable (split at the last =)")
    p.add_argument("--out", required=True)
    p.add_argument("--n_dist", type=int, nargs="+", default=[0, 10])
    p.add_argument("--xmax", type=int, default=None)
    p.add_argument("--smooth", type=int, default=1,
                   help="centred moving-average window in evals (1 = none)")
    p.add_argument("--title", default=None)
    a = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = []
    for spec in a.log:
        label, path = spec.rsplit("=", 1)
        log = parse_task_log(path)
        if not log:
            raise SystemExit(f"{path}: no [navigate_uN] task= lines")
        runs.append((label, log))

    def _smooth(v):
        if a.smooth <= 1:
            return v
        k = a.smooth
        out = np.full_like(v, np.nan)
        for i in range(len(v)):
            lo, hi = max(0, i - k // 2), min(len(v), i + k // 2 + 1)
            w = v[lo:hi]
            w = w[np.isfinite(w)]
            out[i] = w.mean() if len(w) else np.nan
        return out

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    styles = {0: (":", 1.4, 0.7), 5: ("--", 1.4, 0.8), 10: ("-", 2.0, 1.0)}
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6), sharex=True)
    for ax, (key, name, ylim) in zip(axes, PANELS):
        for i, (label, log) in enumerate(runs):
            us = sorted(u for u in log if a.xmax is None or u <= a.xmax)
            for nd in a.n_dist:
                x = np.array([u for u in us if nd in log[u]], dtype=float)
                y = np.array([log[u][nd].get(key, np.nan) for u in us if nd in log[u]])
                ls, lw, al = styles.get(nd, ("-", 1.6, 1.0))
                ax.plot(x, _smooth(y), ls, color=colors[i % len(colors)],
                        linewidth=lw, alpha=al,
                        label=f"{label} (d={nd})" if key == "found_rate" else None)
        ax.set_title(name)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.18, linewidth=0.7)
        ax.set_xlabel("update")
    axes[0].legend(loc="lower right", fontsize=7.5, framealpha=0.92)
    if a.title:
        fig.suptitle(a.title)
    fig.tight_layout()
    fig.savefig(a.out, dpi=200)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
