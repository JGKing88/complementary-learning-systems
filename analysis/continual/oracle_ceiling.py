"""T0.3 — what fraction of eval trials the BFS oracle itself can solve.

Every `reached` number in the continual figures is "did the policy sit on the
goal within `--max_steps` from a seeded random start". That framing has a
ceiling which is a property of the *environment and the step cap*, not of any
agent: if the oracle only reaches on 92 % of trials, then 0.92 is what a
perfect policy scores and the forgetting curves should be read against 0.92,
not against 1.0.

Nobody has measured it. This does, by running `evaluate_nav_one_env`'s exact
protocol -- same vec construction, same `reset_all()` starts, same "pre-step
position is the goal" success test -- with the BFS teacher in place of the
agent. Any gap it reports is the eval's own headroom.

Writes a JSON summary; prints a table.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from hopfield_nav.config import EnvConfig, RNNAgentConfig, RNNBCConfig, RNNTrainConfig
from hopfield_nav.rollout.oracles import (
    bfs_action_batch_continuous, bfs_action_batch_discrete)
from hopfield_nav.training.rnn_setup import rnn_world
from hopfield_nav.world.env import GridEnv, at_goal
from hopfield_nav.world.vec_env import make_vec


def oracle_nav_one_env(
    env: GridEnv,
    n_trials: int,
    max_steps: int,
    movement_mode: str,
    continuous_scale: float = 1.0,
    continuous_normalize: bool = False,
) -> dict[str, float]:
    """`evaluate_nav_one_env`'s protocol, driven by the BFS teacher.

    Deliberately mirrors `hopfield_nav/evaluation/rnn.py` rather than importing
    it: that function takes an agent, and threading a `teacher_force` flag
    through the evaluator would put a training-only concept into the eval path
    for the sake of one measurement.
    """
    vec = make_vec(env, n_trials, movement_mode, continuous_scale,
                   continuous_normalize, reset=False)
    vec.reset_all()

    goal = (int(vec._goal[0]), int(vec._goal[1]))
    success = np.zeros(n_trials, dtype=bool)
    steps_to_goal = np.full(n_trials, np.nan)

    for t in range(max_steps):
        positions = vec.positions()

        # Pre-step at-goal, exactly as the evaluator scores it.
        hit = at_goal(vec) & ~success
        if hit.any():
            steps_to_goal[hit] = t
        success |= at_goal(vec)
        if success.all():
            break

        if movement_mode == "discrete":
            action = bfs_action_batch_discrete(positions, goal, vec.size, vec._rng)
        else:
            action = bfs_action_batch_continuous(positions, goal, vec._rng)
        vec.step_batch(action)

    reached = steps_to_goal[success]
    return {
        "oracle_reached": float(success.mean()),
        "mean_steps_to_goal": float(reached.mean()) if reached.size else float("nan"),
        "max_steps_to_goal": float(reached.max()) if reached.size else float("nan"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--n_envs", type=int, default=5)
    p.add_argument("--size", type=int, default=20)
    p.add_argument("--observation_size", type=int, default=60)
    p.add_argument("--movement_mode", choices=["discrete", "continuous"],
                   default="continuous")
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--goal_radius", type=float, default=0.5)
    p.add_argument("--n_trials", type=int, default=256)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                   help="World seeds; each builds the same envs the sequential "
                        "protocol would build at that seed.")
    args = p.parse_args()

    cfg = RNNTrainConfig(
        env=EnvConfig(size=args.size, observation_size=args.observation_size,
                      movement_mode=args.movement_mode,
                      goal_radius=args.goal_radius),
        agent=RNNAgentConfig(movement_mode=args.movement_mode),
        bc=RNNBCConfig(),
        n_envs=args.n_envs,
        eval_max_steps=args.max_steps,
    )

    rows: list[dict] = []
    for seed in args.seeds:
        cfg.seed = seed
        np.random.seed(seed)
        envs, _, _, _, _ = rnn_world(cfg, np.random.RandomState(seed))
        for j, env in enumerate(envs):
            m = oracle_nav_one_env(
                env, args.n_trials, args.max_steps, args.movement_mode,
                cfg.env.continuous_scale, cfg.env.continuous_normalize,
            )
            rows.append({"seed": seed, "env": j,
                         "goal": list(env.goal_location), **m})
            print(f"  seed={seed} env={j} goal={env.goal_location}  "
                  f"oracle_reached={m['oracle_reached']:.3f}  "
                  f"mean_steps={m['mean_steps_to_goal']:.1f}  "
                  f"max_steps={m['max_steps_to_goal']:.0f}")

    overall = float(np.mean([r["oracle_reached"] for r in rows]))
    worst = min(rows, key=lambda r: r["oracle_reached"])
    print()
    print(f"[T0.3] oracle ceiling over {len(rows)} envs: {overall:.4f}")
    print(f"[T0.3] worst env: seed={worst['seed']} env={worst['env']} "
          f"-> {worst['oracle_reached']:.3f}")
    if overall < 0.99:
        print(f"[T0.3] NOTE: `reached` is capped at ~{overall:.3f}, not 1.0. "
              "Every retention number should be read against that.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "metadata": {
                "measurement": "T0.3_oracle_ceiling",
                "n_envs": args.n_envs, "size": args.size,
                "movement_mode": args.movement_mode,
                "max_steps": args.max_steps, "n_trials": args.n_trials,
                "seeds": args.seeds,
            },
            "overall_oracle_reached": overall,
            "rows": rows,
        }, f, indent=2)
    print(f"[T0.3] wrote {args.out}")


if __name__ == "__main__":
    main()
