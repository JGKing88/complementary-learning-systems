"""Evaluate a checkpoint's three eval metrics across a distractor sweep.

Uses the new dedicated-eval-world setup: a VectorHash scaffold built from only
the val envs, with a fresh Hopfield per trial. Matches what training-time eval
does, just with a broader distractor sweep by default.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from hopfield_nav.config import (
    TrainConfig, EnvConfig, VectorHashConfig, HopfieldConfig,
    AgentConfig, PPOConfig,
)
from hopfield_nav.encoder import load_encoder
from hopfield_nav.env import GridEnv, make_env
from hopfield_nav.vectorhash import VectorHash
from hopfield_nav.agent import NavAgent, compute_input_dim
from hopfield_nav.eval import (
    evaluate_navigation, evaluate_goal_discovery, evaluate_exploration,
)


def _coerce_legacy_cfg(cd: dict) -> dict:
    """Map legacy config fields onto the current schema in-place."""
    # val_envs_per_world used to be scoped to training worlds; it's now a
    # single global num_val_envs for the dedicated eval world.
    if "val_envs_per_world" in cd and "num_val_envs" not in cd:
        cd["num_val_envs"] = cd.pop("val_envs_per_world")
    return cd


def make_cfg(cd: dict) -> TrainConfig:
    cd = _coerce_legacy_cfg(dict(cd))
    env = EnvConfig(**cd["env"])
    vh = VectorHashConfig(**cd["vectorhash"])
    hop = HopfieldConfig(**cd["hopfield"])
    ag = AgentConfig(**cd["agent"])
    ppo = PPOConfig(**cd["ppo"])
    cfg = TrainConfig(env=env, vectorhash=vh, hopfield=hop, agent=ag, ppo=ppo)
    for k, v in cd.items():
        if k in {"env", "vectorhash", "hopfield", "agent", "ppo"}:
            continue
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def build_eval_world(cfg: TrainConfig, encoder, device: str):
    """Recreate the eval-world geometry from the cfg (same layout as training)."""
    rng = np.random.RandomState(cfg.seed)
    size = cfg.env.size

    # Advance the rng past the train-env seeds so val-env seeds match what
    # training would have drawn. Training first consumes envs_per_world *
    # num_worlds integers for train-env seeds before val-env seeds.
    for _ in range(cfg.envs_per_world * cfg.num_worlds):
        rng.randint(0, 10_000_000)

    val_envs = [
        make_env(cfg.env, cfg.agent.movement_mode,
                 seed=int(rng.randint(0, 10_000_000)))
        for _ in range(cfg.num_val_envs)
    ]
    vh = VectorHash(cfg.vectorhash, size=size)
    vh.build_scaffold()
    vh.register_envs(val_envs, placement="spread")
    vh.precompute_encoded_phi(encoder, cfg.fwhm_ratio, device=device)
    val_idxs = list(range(cfg.num_val_envs))
    return val_envs, vh, val_idxs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",
                        default="/home/jackking/cls/checkpoints/r2_r2b_warmup/hopfield_nav_update600.pt")
    parser.add_argument("--encoder",
                        default="/home/jackking/cls/encoders/binary_20260409_083227/encoder_final.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num_trials", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--distractors", nargs="+", type=int,
                        default=[0, 1, 3, 5, 10])
    parser.add_argument(
        "--gbook-only", dest="gbook_only", default=None,
        action=argparse.BooleanOptionalAction,
        help="Override checkpoint: gbook-only scaffold. Omit to use ckpt value.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0); np.random.seed(0)

    encoder, enc_cfg, _ = load_encoder(args.encoder, str(device))
    embed_dim = enc_cfg.out_dim

    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = make_cfg(ck["config"])
    if args.gbook_only is not None:
        cfg.vectorhash.gbook_only = bool(args.gbook_only)

    val_envs, vh, val_idxs = build_eval_world(cfg, encoder, str(device))

    input_dim = compute_input_dim(cfg.agent, embed_dim)
    agent = NavAgent(cfg.agent, input_dim).to(device)
    agent.load_state_dict(ck["agent_state_dict"])
    agent.eval()

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Val envs: {len(val_envs)}  trials/bucket: {args.num_trials}  max_steps: {args.max_steps}\n")

    nav_stoch = evaluate_navigation(
        agent, val_envs, vh, val_idxs, cfg, device,
        num_trials=args.num_trials, max_steps=args.max_steps,
        n_distractors_list=args.distractors, deterministic=False,
    )
    nav_det = evaluate_navigation(
        agent, val_envs, vh, val_idxs, cfg, device,
        num_trials=args.num_trials, max_steps=args.max_steps,
        n_distractors_list=args.distractors, deterministic=True,
    )
    disc = evaluate_goal_discovery(
        agent, val_envs, vh, val_idxs, cfg, device,
        num_trials=args.num_trials, max_steps=args.max_steps,
        n_distractors_list=args.distractors,
    )
    expl = evaluate_exploration(
        agent, val_envs, vh, val_idxs, cfg, device,
        num_trials=args.num_trials, max_steps=args.max_steps,
        n_distractors_list=args.distractors,
    )

    print(f"{'n_dist':>7} {'navD':>6} {'navS':>6} {'stEff':>6} {'stRate':>7} {'reach':>6} {'cov':>6} {'findR':>6}")
    print("-" * 60)
    for n in args.distractors:
        d = disc[n]; e = expl[n]
        print(f"{n:>7} "
              f"{nav_det[n]['success_rate']:>6.2f} "
              f"{nav_stoch[n]['success_rate']:>6.2f} "
              f"{d['store_efficiency']:>6.2f} "
              f"{d['store_success_rate']:>7.2f} "
              f"{d['reach_success_rate']:>6.2f} "
              f"{e['mean_coverage']:>6.2f} "
              f"{e['goal_find_rate']:>6.2f}")


if __name__ == "__main__":
    main()
