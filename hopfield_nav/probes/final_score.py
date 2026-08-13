"""Score one checkpoint on all three target metrics, with enough trials to trust.

Why this exists rather than reading the training log. `do_eval` runs 10 val envs
x 32 trials every 100 updates, and the composite runs swing far beyond binomial
noise between adjacent evals -- C9 reads `success_rate` 0.413 at u500, 0.991 at
u600 and 0.544 at u700. Those are real policy changes, but it means any single
logged eval is a poor estimate of what a checkpoint is worth, and the deliverable
is a *checkpoint*.

So this re-scores a chosen checkpoint on the same val split (`replay_env_seeds`,
which is the trap in "Resuming this work" #1) with a configurable and much larger
trial count, and reports the three metrics the brief names plus `mean_speed`.

`mean_speed` and `mean_steps` are computed exactly as `evaluation/metrics.py`
does -- averaged over SUCCESSES ONLY -- because that is what the target metric
is, and matching it is more useful than fixing it here. See section 3ab: a low
`success_rate` makes `mean_steps` optimistic, so both are always reported
together.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from hopfield import Hopfield
from ..config import TrainConfig
from ..encoder_io import load_encoder
from ..evaluation.batched import (batched_exploration_trials,
                                  batched_navigation_trials)
from ..evaluation.checkpoint_io import cfg_from_checkpoint
from ..policy.agent import NavAgent, compute_input_dim
from ..rollout.distractors import goal_encoding, sample_distractors
from ..training.world_setup import build_field, replay_env_seeds
from ..world.env import make_env
from ..world.world import build_world


def build(args):
    device = torch.device("cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg: TrainConfig = cfg_from_checkpoint(ck["config"])
    cfg.device = "cpu"
    # A checkpoint predating the field was trained cone-pinned-North; taking the
    # current default here is worth a 3.5x error in coverage.
    if "egocentric_heading" not in ck["config"].get("env", {}):
        cfg.env.egocentric_heading = False
    encoder, _e, gain = load_encoder(cfg.encoder_checkpoint, "cpu",
                                     cfg.encoder_gain)
    cfg.encoder_gain = gain
    if cfg.hopfield.beta is None:
        cfg.hopfield.beta = float(gain)

    field = build_field(cfg, encoder)
    np.random.seed(args.seed)
    env_seeds = replay_env_seeds(cfg, "val", args.n_envs, args.seed)
    envs = [make_env(cfg.env, cfg.agent.movement_mode, seed=s)
            for s in env_seeds]
    world = build_world(field, envs, placement="spread", size=cfg.env.size)
    embed_dim = field.encoded_Phi.shape[2]
    agent = NavAgent(cfg.agent, compute_input_dim(
        cfg.agent, embed_dim, cfg.env.observation_size)).to(device)
    agent.load_state_dict(ck["agent_state_dict"])
    agent.eval()
    return cfg, field, world, agent, embed_dim, device, ck.get("update")


def score_nav(args, cfg, field, world, agent, embed_dim, device, n_dist):
    rng = np.random.RandomState(args.seed + 1)
    succ, speed_sum, steps_sum, total = 0, 0.0, 0.0, 0
    for env, offset in zip(world.envs, world.offsets):
        goal = tuple(int(c) for c in env.goal_location)
        hops, starts = [], []
        for _ in range(args.trials):
            hop = Hopfield(embed_dim, beta=cfg.hopfield.beta, device="cpu")
            pats = [goal_encoding(field, offset, goal)]
            pats += sample_distractors(field, offset, env.size, n_dist, rng)
            rng.shuffle(pats)
            for pat in pats:
                hop.input_memory(torch.from_numpy(pat).float())
            hops.append(hop)
            while True:
                s = (int(rng.randint(0, env.size)), int(rng.randint(0, env.size)))
                if s != goal:
                    break
            starts.append(s)
        steps = batched_navigation_trials(
            agent=agent, env=env, env_offset=offset, vectorhash=field,
            hopfields=hops, cfg=cfg, device=device, starts=starts, goal=goal,
            max_steps=args.max_steps, deterministic=True)
        for st, s0 in zip(steps, starts):
            total += 1
            if st > 0:
                succ += 1
                d0 = float(np.hypot(s0[0] - goal[0], s0[1] - goal[1]))
                speed_sum += d0 / st
                steps_sum += st
    return {
        "success_rate": succ / max(total, 1),
        "mean_speed": speed_sum / max(succ, 1),
        "mean_steps": steps_sum / max(succ, 1),
        "total_trials": total, "total_successes": succ,
    }


def score_expl(args, cfg, field, world, agent, embed_dim, device, n_dist):
    rng = np.random.RandomState(args.seed + 2)
    covs = []
    for env, offset in zip(world.envs, world.offsets):
        goal = tuple(int(c) for c in env.goal_location)
        hops, starts = [], []
        for _ in range(args.trials):
            hop = Hopfield(embed_dim, beta=cfg.hopfield.beta, device="cpu")
            # Explore regime: distractors only, never this env's goal.
            for pat in sample_distractors(field, offset, env.size, n_dist, rng):
                hop.input_memory(torch.from_numpy(pat).float())
            hops.append(hop)
            while True:
                s = (int(rng.randint(0, env.size)), int(rng.randint(0, env.size)))
                if s != goal:
                    break
            starts.append(s)
        visited, _f, _s = batched_exploration_trials(
            agent=agent, env=env, env_offset=offset, vectorhash=field,
            hopfields=hops, cfg=cfg, device=device, starts=starts,
            max_steps=args.max_steps, deterministic=True)
        covs.extend(len(v) / float(env.size * env.size) for v in visited)
    return {"mean_coverage": float(np.mean(covs)),
            "coverage_sem": float(np.std(covs) / np.sqrt(max(len(covs), 1)))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n_envs", type=int, default=10)
    p.add_argument("--trials", type=int, default=64,
                   help="per env; do_eval uses 32, so the default here is 2x")
    p.add_argument("--max_steps", type=int, default=400)
    p.add_argument("--n_dist", type=int, nargs="+", default=[0, 10])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_json", default=None)
    args = p.parse_args()

    cfg, field, world, agent, embed_dim, device, upd = build(args)
    out = {"ckpt": args.ckpt, "update": upd, "n_envs": args.n_envs,
           "trials_per_env": args.trials, "max_steps": args.max_steps,
           "by_n_dist": {}}
    print(f"{args.ckpt}  (update {upd})")
    print(f"{args.n_envs} val envs x {args.trials} trials, "
          f"max_steps={args.max_steps}\n")
    print(f"{'n_dist':>7}{'coverage':>11}{'success':>10}{'steps':>9}{'speed':>8}")
    for nd in args.n_dist:
        nav = score_nav(args, cfg, field, world, agent, embed_dim, device, nd)
        exp = score_expl(args, cfg, field, world, agent, embed_dim, device, nd)
        out["by_n_dist"][nd] = {**nav, **exp}
        print(f"{nd:>7}{exp['mean_coverage']:>11.3f}{nav['success_rate']:>10.3f}"
              f"{nav['mean_steps']:>9.1f}{nav['mean_speed']:>8.3f}")
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.output_json}")


if __name__ == "__main__":
    main()
