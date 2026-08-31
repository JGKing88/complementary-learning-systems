"""Run the in-context, zero-weight-update control and write its result.

Plan section 5.2. Loads a frozen policy, meets it with environments it has
never seen, and measures whether it gets better at each one across episodes
*without any weight change*. See `hopfield_nav/evaluation/incontext.py` for what
the measurement is and how to read it.

Takes two checkpoints on purpose:

    --load_checkpoint    the model pretrained on lifetimes
    --control_checkpoint the same recipe pretrained on episodes

Without the second one a rising curve proves very little: an agent might look
like it improves across episodes for reasons that have nothing to do with
memory -- a drift toward the centre of the arena helps on every episode equally,
and a policy that simply explores well would post a rising curve if episode 1
happens to start further from the goal. The episodic control is trained
identically and differs only in whether the hidden state survived a goal-reach,
so the *difference* between the two curves is the part attributable to carrying
anything at all.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from hopfield_nav.config import (
    EnvConfig, RNNAgentConfig, RNNBCConfig, RNNTrainConfig)
from hopfield_nav.evaluation.incontext import evaluate_in_context
from hopfield_nav.policy.agent_rnn import RNNAgent, compute_rnn_input_dim
from hopfield_nav.training.rnn_setup import restore_arch_from_ckpt, rnn_world


def _load(path: str, cfg: RNNTrainConfig, device) -> RNNAgent:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    restore_arch_from_ckpt(cfg, ckpt)
    agent = RNNAgent(
        cfg.agent,
        compute_rnn_input_dim(cfg.agent, cfg.env.observation_size),
    ).to(device)
    agent.load_state_dict(ckpt["agent_state_dict"])
    agent.eval()
    for p in agent.parameters():
        p.requires_grad_(False)      # the whole point: nothing may be learned
    return agent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--load_checkpoint", required=True,
                   help="Policy pretrained with --carry_across_episodes.")
    p.add_argument("--control_checkpoint", default=None,
                   help="Same recipe pretrained WITHOUT it. Strongly "
                        "recommended: without it a rising curve is not "
                        "attributable to memory.")
    p.add_argument("--n_envs", type=int, default=8,
                   help="Held-out envs. Use a seed the pretraining never saw.")
    p.add_argument("--seed", type=int, default=9001)
    p.add_argument("--size", type=int, default=20)
    p.add_argument("--observation_size", type=int, default=60)
    p.add_argument("--movement_mode", choices=["discrete", "continuous"],
                   default="continuous")
    p.add_argument("--goal_radius", type=float, default=0.5)
    p.add_argument("--n_lifetimes", type=int, default=64)
    p.add_argument("--n_episodes", type=int, default=10)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available()
                          or args.device == "cpu" else "cpu")

    cfg = RNNTrainConfig(
        env=EnvConfig(size=args.size, observation_size=args.observation_size,
                      movement_mode=args.movement_mode,
                      goal_radius=args.goal_radius),
        agent=RNNAgentConfig(movement_mode=args.movement_mode),
        bc=RNNBCConfig(), n_envs=args.n_envs, seed=args.seed,
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    envs, _, _, _, _ = rnn_world(cfg, np.random.RandomState(args.seed))
    print(f"[incontext] {len(envs)} held-out envs at seed {args.seed}")

    arms: dict[str, str] = {"lifetime": args.load_checkpoint}
    if args.control_checkpoint:
        arms["episodic"] = args.control_checkpoint

    results: dict[str, dict] = {}
    for arm, path in arms.items():
        agent = _load(path, cfg, device)
        per_env = []
        for i, env in enumerate(envs):
            r = evaluate_in_context(
                env, agent, n_lifetimes=args.n_lifetimes,
                n_episodes=args.n_episodes, max_steps=args.max_steps,
                device=device, deterministic=True,
                continuous_scale=cfg.env.continuous_scale,
                continuous_normalize=cfg.env.continuous_normalize,
            )
            per_env.append(r)
            print(f"  [{arm}] env {i}: ep1={r['first_episode']:.3f} -> "
                  f"ep{args.n_episodes}={r['last_episode']:.3f}  "
                  f"adaptation={r['adaptation']:+.3f}")
        curve = np.mean([r["success_by_episode"] for r in per_env], axis=0)
        results[arm] = {
            "checkpoint": path,
            "per_env": per_env,
            "mean_curve": [float(v) for v in curve],
            "first_episode": float(curve[0]),
            "last_episode": float(curve[-1]),
            "adaptation": float(curve[-1] - curve[0]),
        }
        print(f"  [{arm}] MEAN curve: "
              + " ".join(f"{v:.3f}" for v in curve))
        print(f"  [{arm}] adaptation: {results[arm]['adaptation']:+.4f}")

    print()
    lt = results["lifetime"]["adaptation"]
    if "episodic" in results:
        ep = results["episodic"]["adaptation"]
        delta = lt - ep
        print(f"[incontext] lifetime {lt:+.4f} vs episodic control {ep:+.4f}"
              f"  ->  attributable to carrying state: {delta:+.4f}")
        if delta > 0.1:
            print("[incontext] The RNN adapts in-context. Forgetting is not the")
            print("            interesting axis for this comparison, and the")
            print("            framing needs to account for it.")
        else:
            print("[incontext] No adaptation beyond the episodic control.")
            print("            Activation memory does not do this job here.")
    else:
        print(f"[incontext] lifetime adaptation {lt:+.4f} "
              "(no control arm -- not attributable)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "metadata": {
                "measurement": "5.2_in_context_zero_update",
                "n_envs": args.n_envs, "seed": args.seed,
                "n_lifetimes": args.n_lifetimes,
                "n_episodes": args.n_episodes, "max_steps": args.max_steps,
                "size": args.size, "movement_mode": args.movement_mode,
            },
            "arms": results,
        }, f, indent=2)
    print(f"[incontext] wrote {args.out}")


if __name__ == "__main__":
    main()
