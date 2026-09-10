"""Success, path optimality and swept coverage for one run, over training.

The three headline metrics live in two different eval dicts in the training log
and are usually read apart, which is how a run gets called "converged" on the
strength of the one metric that saturates first. `joint_curve.py` prints two of
them as a table; this plots all three, on one axis, because they are all in
[0, 1] and the point is their *relative* timing:

    success_rate      nav eval, per distractor level.
    path optimality   (mean_start_dist - goal_radius) / mean_steps. 1.0 is a
                      straight run at one cell per step; it is the probe's
                      `path_efficiency` computed as a ratio of means rather
                      than a mean of ratios, because the log records only the
                      means. Validated against the exact per-episode version at
                      six checkpoints: they agree within 0.02 from u400 on, and
                      the approximation is CONSERVATIVE before that (0.39 vs
                      0.48 at u200), so the early curve understates rather than
                      flatters.
    swept_coverage    explore eval; the union of goal_radius discs along the
                      path, i.e. P(the goal was findable).

TWO THINGS THE LOG CANNOT GIVE, both handled rather than ignored:

`mean_start_dist` is not recorded -- `evaluation/metrics.py` emits only
success_rate, mean_speed, mean_steps and the trial counts. It is a property of
the fixed validation env set and so is constant across evals of one run, which
is why a single measured constant is legitimate; pass it with --start_dist.
Defaults are the values `behavior_probe` measures on the phase-2 held-out set.

`mean_steps` is a mean over SUCCESSES ONLY (metrics.py: steps_sum /
total_successes). While success is below 1.0 the successes are the easy trials,
so path optimality on that stretch is computed over a biased subsample and is
not comparable to the converged value. Those updates are shaded and excluded
from any reported trend. This is the same class of error as reading `follow_q`
without `align_true`: a ratio whose denominator is only defined on a subset.

Usage:
    python -m analysis.nav_tri.training_curve \\
        --log /path/nav_p2_<jobid>.out --out_prefix /path/prefix
"""
from __future__ import annotations

import argparse
import ast
import json
import re

import numpy as np

# behavior_probe on the phase-2 held-out set, 6 envs x 32 trials.
DEFAULT_START_DIST = {0: 10.7037, 5: 11.2016, 10: 10.8714}


def parse_log(path: str) -> dict:
    """{update: {"nav": {n_dist: {...}}, "expl": {n_dist: {...}}}}."""
    with open(path, errors="replace") as fh:
        txt = fh.read()
    out: dict[int, dict] = {}
    for kind in ("nav", "expl"):
        for m in re.finditer(r"\[navigate_u(\d+)\] %s=(\{.*?\})\n" % kind, txt):
            u = int(m.group(1))
            out.setdefault(u, {})[kind] = ast.literal_eval(m.group(2))
    return {u: v for u, v in sorted(out.items()) if "nav" in v and "expl" in v}


def path_optimality(mean_steps: float, start_dist: float,
                    goal_radius: float) -> float:
    """(start_dist - goal_radius) / mean_steps, clipped at 0.

    1.0 means the agent covered the whole remaining distance at one cell per
    step, which is the cap when max_action_norm is 1.0. Undefined (nan) when
    there were no successes, which the log reports as mean_steps == 0.
    """
    if not np.isfinite(mean_steps) or mean_steps <= 0:
        return float("nan")
    return max(0.0, (start_dist - goal_radius) / mean_steps)


def series(log: dict, n_dist: int, goal_radius: float,
           start_dist: float | None = None) -> dict:
    sd = (DEFAULT_START_DIST.get(n_dist) if start_dist is None else start_dist)
    if sd is None:
        raise ValueError("no start_dist for n_dist=%d; pass --start_dist"
                         % n_dist)
    us, succ, opt, swept = [], [], [], []
    for u, v in log.items():
        if n_dist not in v["nav"] or n_dist not in v["expl"]:
            continue
        us.append(u)
        succ.append(float(v["nav"][n_dist]["success_rate"]))
        opt.append(path_optimality(float(v["nav"][n_dist]["mean_steps"]),
                                   sd, goal_radius))
        swept.append(float(v["expl"][n_dist]["swept_coverage"]))
    return {"u": np.array(us, float), "success": np.array(succ),
            "optimality": np.array(opt), "swept": np.array(swept),
            "start_dist": sd, "n_dist": n_dist}


def first_reliable(s: dict, thresh: float) -> float | None:
    """First update after which success never again drops below `thresh`.

    Path optimality is only comparable from here on, because before it the
    mean-over-successes is a mean over the easy trials.
    """
    u, v = s["u"], s["success"]
    for i in range(len(v)):
        if (v[i:] >= thresh).all():
            return float(u[i])
    return None


def render(sets: list[dict], out_prefix: str, thresh: float,
           overlay: dict | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    styles = {0: (":", 1.8, 0.55), 5: ("--", 1.8, 0.75), 10: ("-", 2.4, 1.0)}
    colors = {"success": "#1f77b4", "optimality": "#d62728",
              "swept": "#2ca02c"}
    labels = {"success": "success rate", "optimality": "path optimality",
              "swept": "swept coverage"}

    shade_to = None
    for s in sets:
        if s["n_dist"] == max(x["n_dist"] for x in sets):
            shade_to = first_reliable(s, thresh)
    if shade_to is not None and shade_to > sets[0]["u"][0]:
        ax.axvspan(sets[0]["u"][0] - 5, shade_to, color="0.5", alpha=0.13,
                   zorder=0, linewidth=0)
        # Anchored INSIDE the axes to the right of the band. Anchoring it to
        # the band's left edge puts it outside the figure whenever the band is
        # narrow, which is the usual case once a run converges early.
        ax.text(shade_to + 10, 0.90,
                "← success < %.2f in the shaded band: path optimality there is\n"
                "   a mean over the trials that SUCCEEDED, i.e. the easy ones,\n"
                "   so it is not comparable with the converged value" % thresh,
                ha="left", va="top", fontsize=8.5, color="0.35")

    for s in sets:
        ls, lw, al = styles.get(s["n_dist"], ("-", 2.0, 1.0))
        for key in ("success", "optimality", "swept"):
            ax.plot(s["u"], s[key], ls, color=colors[key], linewidth=lw,
                    alpha=al,
                    label=("%s (d=%d)" % (labels[key], s["n_dist"])))

    if overlay:
        ax.plot(overlay["u"], overlay["optimality"], "o", color=colors["optimality"],
                markersize=5.5, markerfacecolor="white", markeredgewidth=1.6,
                zorder=5, label="path optimality — exact, per-episode probe")

    ax.set_xlabel("training update")
    ax.set_ylabel("all three metrics are fractions in [0, 1]")
    ax.set_ylim(0.0, 1.045)
    ax.axhline(1.0, linestyle=":", linewidth=1.0, color="black", alpha=0.6)
    ax.grid(alpha=0.18, linewidth=0.7)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.92, ncol=2)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_training_curve.png", dpi=220)
    fig.savefig(f"{out_prefix}_training_curve.pdf")
    print("Wrote %s_training_curve.{png,pdf}" % out_prefix)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log", required=True)
    p.add_argument("--out_prefix", required=True)
    p.add_argument("--n_dist", type=int, nargs="+", default=[0, 10])
    p.add_argument("--goal_radius", type=float, default=1.0)
    p.add_argument("--start_dist", type=float, default=None,
                   help="Override the measured constant for every level.")
    p.add_argument("--reliable_at", type=float, default=0.99,
                   help="Success floor above which path optimality is treated "
                        "as comparable.")
    p.add_argument("--overlay_json", default=None,
                   help="{'u': [...], 'optimality': [...]} of exact "
                        "per-episode path_efficiency, drawn as markers.")
    a = p.parse_args()

    log = parse_log(a.log)
    if not log:
        raise SystemExit("no [navigate_uN] nav=/expl= pairs found in %s" % a.log)
    sets = [series(log, nd, a.goal_radius, a.start_dist) for nd in a.n_dist]
    print("%d evals, u%d..u%d" % (len(log), min(log), max(log)))
    for s in sets:
        r = first_reliable(s, a.reliable_at)
        print("  d=%-3d start_dist %.4f   success reliable from u%s"
              % (s["n_dist"], s["start_dist"], r))
        print("    %-14s %s" % ("final", " ".join(
            "%s %.3f" % (k, s[k][-1]) for k in ("success", "optimality", "swept"))))
    ov = json.load(open(a.overlay_json)) if a.overlay_json else None
    render(sets, a.out_prefix, a.reliable_at, ov)


if __name__ == "__main__":
    main()
