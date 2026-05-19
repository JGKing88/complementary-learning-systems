"""Evaluate every checkpoint of a training run across the three eval metrics.

Uses the new dedicated-eval-world setup (fresh Hopfield per trial, unified
structure). Supports old checkpoints via legacy-cfg coercion.

Usage:
    python -m hopfield_nav.eval_checkpoints
"""
from __future__ import annotations

import argparse
import os

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
    if "val_envs_per_world" in cd and "num_val_envs" not in cd:
        cd["num_val_envs"] = cd.pop("val_envs_per_world")
    vh = cd.get("vectorhash")
    if isinstance(vh, dict) and "gbook_only" in vh and "static_vectorhash" not in vh:
        vh["static_vectorhash"] = vh.pop("gbook_only")
    return cd


def make_cfg_from_checkpoint(ck_cfg_dict: dict) -> TrainConfig:
    """Reconstruct a TrainConfig from a checkpoint's saved asdict(cfg)."""
    cd = _coerce_legacy_cfg(dict(ck_cfg_dict))
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
    """Match the val-env seeding order that training uses."""
    rng = np.random.RandomState(cfg.seed)
    size = cfg.env.size
    # Train-env seeds are drawn first (per world) in training; skip them here
    # so the val-env seeds match.
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


@torch.no_grad()
def eval_one_checkpoint(
    path: str, encoder, enc_cfg, device: torch.device,
    num_trials: int = 32, max_steps: int = 200,
    static_vectorhash: bool | None = None,
):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = make_cfg_from_checkpoint(ck["config"])
    if static_vectorhash is not None:
        cfg.vectorhash.static_vectorhash = bool(static_vectorhash)
    embed_dim = enc_cfg.out_dim

    torch.manual_seed(0)
    np.random.seed(0)

    val_envs, vh, val_idxs = build_eval_world(cfg, encoder, str(device))
    input_dim = compute_input_dim(cfg.agent, embed_dim, cfg.env.observation_size)
    agent = NavAgent(cfg.agent, input_dim).to(device)
    agent.load_state_dict(ck["agent_state_dict"])
    agent.eval()

    dist_list = [0]
    nav_det = evaluate_navigation(
        agent, val_envs, vh, val_idxs, cfg, device,
        num_trials=num_trials, max_steps=max_steps,
        n_distractors_list=dist_list, deterministic=True,
    )
    nav_stoch = evaluate_navigation(
        agent, val_envs, vh, val_idxs, cfg, device,
        num_trials=num_trials, max_steps=max_steps,
        n_distractors_list=dist_list, deterministic=False,
    )
    disc = evaluate_goal_discovery(
        agent, val_envs, vh, val_idxs, cfg, device,
        num_trials=num_trials, max_steps=max_steps,
        n_distractors_list=dist_list,
    )
    expl = evaluate_exploration(
        agent, val_envs, vh, val_idxs, cfg, device,
        num_trials=num_trials, max_steps=max_steps,
        n_distractors_list=dist_list,
    )

    return {
        "nav_det": nav_det[0]["success_rate"],
        "nav_stoch": nav_stoch[0]["success_rate"],
        "store_eff": disc[0]["store_efficiency"],
        "store_rate": disc[0]["store_success_rate"],
        "reach": disc[0]["reach_success_rate"],
        "coverage": expl[0]["mean_coverage"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default="/home/jackking/cls/encoders/binary_20260409_083227/encoder_final.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num_trials", type=int, default=32)
    parser.add_argument("--runs", nargs="+", default=[
        "r2_r2a_low_ent", "r2_r2b_warmup", "r2_r2c_bc", "r2_r2d_anneal",
    ])
    parser.add_argument("--updates", nargs="+", type=int,
                        default=[200, 400, 600, 800, 1000, 1200])
    parser.add_argument(
        "--static-vectorhash", dest="static_vectorhash", default=None,
        action=argparse.BooleanOptionalAction,
        help="Override checkpoint: static VectorHash scaffold (gbook + sbook only). "
             "Omit to use ckpt value.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    encoder, enc_cfg, _ = load_encoder(args.encoder, str(device))

    ckpt_root = "/home/jackking/cls/checkpoints"
    rows = []
    print(f"{'run':<18} {'up':>5} {'navD':>6} {'navS':>6} {'stEff':>6} {'stRate':>7} {'reach':>6} {'cov':>6}  pass?")
    print("-" * 78)
    for run in args.runs:
        for up in args.updates:
            path = os.path.join(ckpt_root, run, f"hopfield_nav_update{up}.pt")
            if not os.path.exists(path):
                continue
            m = eval_one_checkpoint(
                path, encoder, enc_cfg, device, num_trials=args.num_trials,
                static_vectorhash=args.static_vectorhash,
            )
            passes = m["nav_stoch"] >= 0.7 and m["store_eff"] >= 0.7 and m["coverage"] >= 0.45
            score = (m["nav_stoch"] + m["store_eff"] + m["coverage"]) / 3
            rows.append((run, up, m, score, passes))
            tag = "***" if passes else ""
            print(f"{run:<18} {up:>5} {m['nav_det']:>6.2f} {m['nav_stoch']:>6.2f} "
                  f"{m['store_eff']:>6.2f} {m['store_rate']:>7.2f} {m['reach']:>6.2f} "
                  f"{m['coverage']:>6.2f}  {tag}")

    print()
    print("Best checkpoint per run by score = mean(nav_stoch, store_eff, coverage):")
    by_run: dict[str, list] = {}
    for r in rows:
        by_run.setdefault(r[0], []).append(r)
    for run, rs in by_run.items():
        best = max(rs, key=lambda r: r[3])
        m = best[2]
        print(f"  {run}: update {best[1]}  score={best[3]:.3f}  "
              f"nav_stoch={m['nav_stoch']:.2f} store_eff={m['store_eff']:.2f} "
              f"coverage={m['coverage']:.2f} (passes_gates={best[4]})")


if __name__ == "__main__":
    main()
