"""The explore-first eval series: exploit criterion and explore retention per arm.

Reads `train_navigate` logs (the `[navigate_uN] task=` / `expl=` / `nav=` /
`samples=` lines) for one or more arms and answers the two questions the
plan asks (docs/EXPLORE_FIRST_PLAN.md §3):

  exploit learned   first eval with held-out revisit success >= 0.95 at
                    <= 20 steps AND cos_aq_post >= 0.80 (the task line's
                    level; `task1r_k4_h128` crossed it at u500), reported as
                    an update and as trajectories (`samples=` line)
  explore kept      held-out swept coverage at each eval as a DELTA against
                    the run's own u0 row (a fork scores its parent before
                    its first step), plus first-visit search competence

The criterion is read at d=0, as the task line quotes it; d=10 is carried
beside it. A slide counts as forgetting, not only a cliff, so the whole
series is printed, never just the end.

    python -m analysis.explore_first.series \\
        --run E0=$CLS_LOGS/nav_p2_<job>.out --run E3=... \\
        --out_prefix $CLS_RESULTS/explore_first/wave1

writes `<out_prefix>_series.md` (one table per arm, plus the summary) and
`<out_prefix>_series.png`.
"""
from __future__ import annotations

import argparse
import ast
import math
import os
import re

CRITERION = {"revisit_found": 0.95, "revisit_steps_first": 20.0, "cos_aq_post": 0.80}
COVERAGE_KEY = "swept_coverage"

_LINE = re.compile(r"\[navigate_u(\d+)\] (task|expl|nav|samples)=(\{.*?\})\n")


def parse_log(path: str) -> dict[int, dict]:
    """{update: {"task": {dist: {...}}, "expl": ..., "nav": ..., "samples": {...}}}.

    Each kind is taken from its LAST occurrence at an update, so a resumed
    segment that re-evaluates its start does not double an entry. `nan`
    becomes None.
    """
    txt = open(path, errors="replace").read()
    out: dict[int, dict] = {}
    for m in _LINE.finditer(txt):
        u = int(m.group(1))
        body = re.sub(r"\bnan\b", "None", m.group(3))
        try:
            val = ast.literal_eval(body)
        except Exception:
            continue
        out.setdefault(u, {})[m.group(2)] = val
    return dict(sorted(out.items()))


def _get(row: dict, kind: str, dist: int, key: str):
    d = row.get(kind, {})
    d = d.get(dist, d.get(str(dist), {})) if isinstance(d, dict) else {}
    v = d.get(key) if isinstance(d, dict) else None
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else v


def meets_criterion(row: dict, dist: int = 0) -> bool:
    f = _get(row, "task", dist, "revisit_found")
    s = _get(row, "task", dist, "revisit_steps_first")
    c = _get(row, "task", dist, "cos_aq_post")
    if f is None or s is None or c is None:
        return False
    return (f >= CRITERION["revisit_found"] and s <= CRITERION["revisit_steps_first"]
            and c >= CRITERION["cos_aq_post"])


def first_criterion(series: dict[int, dict], dist: int = 0):
    """(update, trajectories) of the FIRST eval that meets the criterion, or None.

    First, not last: the question is how soon exploit arrives, and a later
    dip is reported by the series itself.
    """
    for u, row in series.items():
        if meets_criterion(row, dist):
            traj = (row.get("samples") or {}).get("episodes")
            return u, traj
    return None


def retention(series: dict[int, dict], dist: int = 0, key: str = COVERAGE_KEY):
    """[(update, coverage, delta_vs_u0, frac_vs_u0)] for every eval with the key."""
    base = _get(series.get(0, {}), "expl", dist, key)
    out = []
    for u, row in series.items():
        v = _get(row, "expl", dist, key)
        if v is None:
            continue
        if base in (None, 0):
            out.append((u, v, None, None))
        else:
            out.append((u, v, v - base, v / base))
    return out


def worst_retention(series: dict[int, dict], dist: int = 0, key: str = COVERAGE_KEY):
    """The minimum fraction of the u0 coverage over the series (None without u0)."""
    fr = [f for _, _, _, f in retention(series, dist, key) if f is not None]
    return min(fr) if fr else None


def _f(v, nd=2):
    return "—" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def arm_table(label: str, series: dict[int, dict]) -> str:
    lines = [f"**{label}**", "",
             "| u | traj | found d0 | first d0 | revisit d0 sr@steps | cos_post d0 | "
             "revisit d10 sr@steps | cos_post d10 | swept d0 (Δ u0) | swept d10 (Δ u0) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    base0 = _get(series.get(0, {}), "expl", 0, COVERAGE_KEY)
    base10 = _get(series.get(0, {}), "expl", 10, COVERAGE_KEY)
    for u, row in series.items():
        traj = (row.get("samples") or {}).get("episodes")
        c0 = _get(row, "expl", 0, COVERAGE_KEY)
        c10 = _get(row, "expl", 10, COVERAGE_KEY)
        d0 = "" if c0 is None or base0 is None else f" ({c0 - base0:+.3f})"
        d10 = "" if c10 is None or base10 is None else f" ({c10 - base10:+.3f})"
        mark = " ✓" if meets_criterion(row, 0) else ""
        lines.append(
            f"| {u}{mark} | {_f(traj)} | {_f(_get(row, 'task', 0, 'found_rate'))} | "
            f"{_f(_get(row, 'task', 0, 'steps_first'), 0)} | "
            f"{_f(_get(row, 'task', 0, 'revisit_found'))}@{_f(_get(row, 'task', 0, 'revisit_steps_first'), 1)} | "
            f"{_f(_get(row, 'task', 0, 'cos_aq_post'))} | "
            f"{_f(_get(row, 'task', 10, 'revisit_found'))}@{_f(_get(row, 'task', 10, 'revisit_steps_first'), 1)} | "
            f"{_f(_get(row, 'task', 10, 'cos_aq_post'))} | "
            f"{_f(c0, 3)}{d0} | {_f(c10, 3)}{d10} |")
    return "\n".join(lines)


def summary_table(runs: dict[str, dict[int, dict]]) -> str:
    lines = ["| arm | evals | u_criterion | traj_criterion | last: revisit d0 sr@steps | "
             "last: cos_post d0 | swept d0 u0 → last (min frac) | swept d10 u0 → last (min frac) | found d0 u0 → last |",
             "|---|---|---|---|---|---|---|---|---|"]
    for label, s in runs.items():
        if not s:
            lines.append(f"| {label} | 0 | | | | | | | |")
            continue
        last_u = max(s)
        last = s[last_u]
        crit = first_criterion(s, 0)
        cu, ct = (crit if crit else (None, None))

        def span(dist):
            b = _get(s.get(0, {}), "expl", dist, COVERAGE_KEY)
            l = _get(last, "expl", dist, COVERAGE_KEY)
            w = worst_retention(s, dist)
            return f"{_f(b, 3)} → {_f(l, 3)}" + (f" ({w:.2f})" if w is not None else "")

        fb = _get(s.get(0, {}), "task", 0, "found_rate")
        fl = _get(last, "task", 0, "found_rate")
        lines.append(
            f"| {label} | {len(s)} (last u{last_u}) | {_f(cu)} | {_f(ct)} | "
            f"{_f(_get(last, 'task', 0, 'revisit_found'))}@{_f(_get(last, 'task', 0, 'revisit_steps_first'), 1)} | "
            f"{_f(_get(last, 'task', 0, 'cos_aq_post'))} | {span(0)} | {span(10)} | "
            f"{_f(fb)} → {_f(fl)} |")
    return "\n".join(lines)


def render(runs: dict[str, dict[int, dict]], out_png: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax_cov, ax_rev, ax_cos = axes
    for label, s in runs.items():
        us = [u for u in s if _get(s[u], "expl", 0, COVERAGE_KEY) is not None]
        if us:
            ax_cov.plot(us, [_get(s[u], "expl", 0, COVERAGE_KEY) for u in us],
                        marker="o", ms=3, label=label)
        ut = [u for u in s if _get(s[u], "task", 0, "revisit_steps_first") is not None]
        if ut:
            ax_rev.plot(ut, [_get(s[u], "task", 0, "revisit_steps_first") for u in ut],
                        marker="o", ms=3, label=label)
            ax_cos.plot(ut, [_get(s[u], "task", 0, "cos_aq_post") for u in ut],
                        marker="o", ms=3, label=label)
    ax_cov.set_title("held-out swept coverage, d=0 (explore kept?)")
    ax_cov.set_xlabel("update"); ax_cov.set_ylim(0, None)
    ax_rev.set_title("revisit steps to first touch, d=0 (exploit learned?)")
    ax_rev.axhline(CRITERION["revisit_steps_first"], ls="--", lw=0.8, color="k")
    ax_rev.set_xlabel("update"); ax_rev.set_ylim(0, None)
    ax_cos.set_title("cos(action, q) after the store, d=0")
    ax_cos.axhline(CRITERION["cos_aq_post"], ls="--", lw=0.8, color="k")
    ax_cos.set_xlabel("update"); ax_cos.set_ylim(-0.2, 1.0)
    ax_cov.legend(fontsize=8, frameon=False)
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True,
                   help="LABEL=path/to/nav_p2_<job>.out (repeatable)")
    p.add_argument("--out_prefix", required=True)
    p.add_argument("--no-png", action="store_true")
    a = p.parse_args()
    runs = {}
    for spec in a.run:
        label, path = spec.split("=", 1)
        runs[label] = parse_log(path) if os.path.exists(path) else {}
    os.makedirs(os.path.dirname(a.out_prefix) or ".", exist_ok=True)
    md = ["## Summary", "", summary_table(runs), ""]
    for label, s in runs.items():
        md += [arm_table(label, s), ""]
    text = "\n".join(md)
    with open(a.out_prefix + "_series.md", "w") as f:
        f.write(text)
    print(text)
    if not a.no_png:
        render(runs, a.out_prefix + "_series.png")
        print(f"wrote {a.out_prefix}_series.png")


if __name__ == "__main__":
    main()
