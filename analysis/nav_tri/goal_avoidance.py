"""Is a low goal-find rate avoidance of a *memorised* goal, or of walls?

`EXPERIMENTS_TASK_FAITHFUL.md` §9-§10. The empty-memory test runs the explore
eval (nothing in the Hopfield) on a run's own training arenas: a model that
walks to its fixed goal with no memory has the goal in its weights. The
*converse* reading -- a find rate far below chance means the model steers
clear of a goal it knows -- needs one more control, because `wall_penalty`
makes any model avoid the perimeter, and a run whose training goals all sit
on the perimeter would look identical.

This reports, per env, the fraction of trials that touched the goal and the
coverage, tagged by whether the goal is on the perimeter, for BOTH recorded
env sets. If the gap tracks perimeter-ness rather than train-vs-val, the low
own-arena rate is wall avoidance and says nothing about memorisation.

    python -m analysis.nav_tri.goal_avoidance \\
        --run_dir $CLS_CKPTS/navigate_navp2_task3_k2_h128_nos_s42_22989777 \\
        --update 3000 --trials 96 --device cpu
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from hopfield_nav.encoder_io import load_encoder
from hopfield_nav.evaluation.checkpoint_io import (
    cfg_from_checkpoint, eval_world_from_spec, load_agent, world_spec_for)
from hopfield_nav.evaluation.metrics import evaluate_exploration


def _perimeter(goal, size: int) -> bool:
    return goal[0] in (0, size - 1) or goal[1] in (0, size - 1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--update", type=int, required=True)
    p.add_argument("--trials", type=int, default=96)
    p.add_argument("--n_distractors", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction,
                   default=False, help="default sampled, the convention")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    device = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available())
                          else "cpu")
    ck = os.path.join(a.run_dir, f"navigate_u{a.update}.pt")
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    cfg = cfg_from_checkpoint(blob["config"])
    cfg.device = str(device)
    spec = world_spec_for(a.run_dir)
    if spec is None:
        raise SystemExit(f"{a.run_dir} has no world.json")
    encoder, enc_cfg, gain = load_encoder(cfg.encoder_checkpoint, str(device),
                                          cfg.encoder_gain)
    cfg.encoder_gain = gain
    agent = load_agent(cfg, blob["agent_state_dict"], enc_cfg.out_dim, device)

    print(f"run={os.path.basename(a.run_dir)} u{a.update}  "
          f"{'deterministic' if a.deterministic else 'SAMPLED'}  "
          f"trials={a.trials}  n_dist={a.n_distractors}  empty memory")
    print(f"{'set':>6}  {'env':>3}  {'goal':>9}  {'where':>9}  "
          f"{'found':>6}  {'steps':>6}  {'cov':>5}")
    rows = []
    for which in ("train", "base_val"):
        envs, field, offsets = eval_world_from_spec(
            spec, cfg, encoder, str(device), which=which)
        for i, env in enumerate(envs):
            per_trial: list = []
            evaluate_exploration(
                agent, [env], field, [offsets[i]], cfg, device,
                num_trials=a.trials, max_steps=a.max_steps,
                n_distractors_list=[a.n_distractors],
                deterministic=a.deterministic, per_trial=per_trial)
            hits = np.array([r[4] for r in per_trial], dtype=float)
            steps = np.array([r[5] for r in per_trial], dtype=float)
            cells = np.array([r[3] for r in per_trial], dtype=float)
            g = tuple(env.goal_location)
            where = "perimeter" if _perimeter(g, int(env.size)) else "interior"
            found = hits.mean()
            reach = steps[hits > 0].mean() if hits.any() else float("nan")
            cov = cells.mean() / float(env.size * env.size)
            rows.append((which, where, found, cov))
            print(f"{which:>6}  {i:>3}  {str(g):>9}  {where:>9}  "
                  f"{found:>6.3f}  {reach:>6.1f}  {cov:>5.3f}")

    print()
    for key in ("perimeter", "interior"):
        sel = [r for r in rows if r[1] == key]
        if sel:
            print(f"  all {key:>9} goals: found "
                  f"{np.mean([r[2] for r in sel]):.3f}  "
                  f"cov {np.mean([r[3] for r in sel]):.3f}  (n={len(sel)})")
    for key in ("train", "base_val"):
        sel = [r for r in rows if r[0] == key]
        print(f"  all {key:>9} arenas: found "
              f"{np.mean([r[2] for r in sel]):.3f}  "
              f"cov {np.mean([r[3] for r in sel]):.3f}  (n={len(sel)})")


if __name__ == "__main__":
    main()
