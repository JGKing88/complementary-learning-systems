"""Performance over training, two quantities on one 0-1 axis, per model.

    python -m analysis.sensory_key.plot_training --dist 10 --out fig.png \
        --model "Idea 1 (grid MLP, d + c)::Idea 1" 24342872 24659510 \
        --model "Agent-HaSH, unit q + |q|, projection storage::Agent-HaSH" 24570179 24659511

Reads each run's in-training task eval (``[navigate_uN] task=`` lines; sampled,
6 held-out arenas x 16 trials, visits = 2) and plots, per model:

- **found on first visit** -- ``found_rate``: fraction of trajectories that find
  the goal within the 200-step search, before it is stored;
- **path optimality** -- optimal steps / steps taken on the *revisit* (goal
  stored, fresh start), as a ratio of means: ``E[opt] / revisit_steps_first``.
  A revisit starts on a uniform random cell other than the goal and counts as
  reached at the first step within ``goal_radius`` (1 cell); the agent moves at
  most ``max_action_norm`` (1 cell) per step, so the optimal step count from a
  start at distance r is ``ceil(max(0, r - radius) / speed)``. ``E[opt]`` is its
  mean over every start cell of the run's own val arenas (their recorded goals,
  ``world.json``). Clipped to 1.

Each seed is smoothed with a rolling mean over ``--smooth`` eval points; the
line is the seed mean, the band the seed min-max.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np

from analysis.sensory_key.compare_runs import series

RUNS = "/orcd/pool/003/jackking/cls_runs"
# Reference categorical palette (dataviz skill, references/palette.md), light
# mode, slots 1-2 -- validated there; text in ink tokens, never series colour.
SERIES = ["#2a78d6", "#eb6834"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def run_dir(job: str) -> str:
    hits = glob.glob(f"{RUNS}/agent_ckpts/*_{job}")
    if len(hits) != 1:
        raise SystemExit(f"job {job}: expected one run dir, found {hits}")
    return hits[0]


def expected_optimal(rd: str) -> float:
    """Mean optimal revisit steps over every start cell of the run's val arenas."""
    world = json.load(open(os.path.join(rd, "world.json")))
    cfg = json.load(open(os.path.join(rd, "run.json"))).get("config", {})
    env = cfg.get("env", {})
    radius = float(env.get("goal_radius", 1.0))
    speed = float(env.get("max_action_norm") or 1.0)
    per_env = []
    for spec in world["split"]["base_val"]:
        S, (gx, gy) = int(spec["size"]), spec["goal"]
        opt = [math.ceil(max(0.0, math.hypot(x - gx, y - gy) - radius) / speed)
               for x in range(S) for y in range(S) if (x, y) != (gx, gy)]
        per_env.append(np.mean(opt))
    return float(np.mean(per_env))


def curves(job: str, dist: int, smooth: int):
    rows = series(f"{RUNS}/logs/nav_p2_{job}.out")
    u = np.array([r[0] for r in rows])
    found = np.array([r[1][dist]["found_rate"] for r in rows], float)
    steps = np.array([r[1][dist]["revisit_steps_first"] for r in rows], float)
    eopt = expected_optimal(run_dir(job))
    opt = np.clip(eopt / steps, 0, 1)
    k = np.ones(smooth) / smooth
    sm = lambda v: np.convolve(np.nan_to_num(v, nan=np.nanmean(v)), k, "valid")
    return u[smooth - 1:], sm(found), sm(opt), eopt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", nargs="+", action="append", required=True,
                   metavar=("LABEL", "JOB"), help="label then one job id per seed")
    p.add_argument("--dist", type=int, default=10)
    p.add_argument("--smooth", type=int, default=5)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2,
                         "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK})
    fig, ax = plt.subplots(figsize=(9.0, 5.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    report, ends = {}, []
    for i, (label, *jobs) in enumerate(a.model):
        # "Long legend label::Short" -- the short form is the direct label.
        label, short = (label.split("::", 1) + [label])[:2]
        col = SERIES[i]
        per = [curves(j, a.dist, a.smooth) for j in jobs]
        n = min(len(c[0]) for c in per)
        u = per[0][0][:n]
        for m, (idx, style, name, tag) in enumerate([
                (2, "-", "path optimality", "optimality"),
                (1, (0, (5, 3)), "found on first visit", "found")]):
            ys = np.stack([c[idx][:n] for c in per])
            mean = ys.mean(0)
            if len(per) > 1:
                ax.fill_between(u, ys.min(0), ys.max(0), color=col, alpha=0.14, lw=0)
            ax.plot(u, mean, color=col, lw=2, ls=style, solid_capstyle="round",
                    label=f"{label} -- {name}")
            ends.append([mean[-1], f"{short}, {tag} {mean[-1]:.2f}", u[-1]])
            report[f"{label} / {name}"] = dict(final=float(mean[-1]),
                                               seeds=[float(y[-1]) for y in ys])
        report[f"{label} / E[optimal steps]"] = [c[3] for c in per]
    # Direct labels at the right end, nudged apart so none collide.
    ends.sort()
    gap, placed = 0.05, []
    for y, text, x in ends:
        y_lab = max(y, placed[-1] + gap) if placed else y
        placed.append(y_lab)
        ax.annotate(text, (x, y), xytext=(x + 60, y_lab), textcoords="data",
                    va="center", fontsize=8.5, color=INK2, annotation_clip=False)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 4000)
    ax.set_xlabel("PPO update")
    ax.set_ylabel(f"fraction ({a.dist} distractors in memory)")
    ax.grid(axis="y", color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    seeds = max(len(m) - 1 for m in a.model)
    ax.set_title("Performance over training: path optimality (revisit) and goal found "
                 "on first visit", fontsize=11, color=INK, loc="left")
    ax.text(0, -0.2, "Solid: path optimality = optimal steps / steps taken on the revisit "
            "(goal stored, fresh start).\nDashed: fraction finding the goal within the "
            "200-step search, before it is stored.\n"
            f"Mean of {seeds} seed(s), band = seed range; rolling mean over {a.smooth} evals "
            "(one every 50 updates); 6 held-out arenas x 16 sampled trials.",
            transform=ax.transAxes, fontsize=8, color=INK2, va="top")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.subplots_adjust(left=0.09, right=0.74, bottom=0.27, top=0.9)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, facecolor=SURFACE)
    fig.savefig(os.path.splitext(a.out)[0] + ".svg", facecolor=SURFACE)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
