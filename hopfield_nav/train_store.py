"""Phase B store-head pretrain on a frozen Phase-A trunk.

Loads a V10 (or compatible) navigation-training checkpoint, freezes everything except
the store head, trains the store head with BCE-on-at-goal labels through
detached features. Goals are active so the agent encounters at-goal
cells and gets positive labels.

Usage:
    python -m hopfield_nav.train_store \
        --load_checkpoint $CLS_RUNS/agent_ckpts/navigate_<run>/navigate_u160.pt \
        --encoder_checkpoint encoders/run_20260422_185816/encoder_best.pt \
        --phase_b_updates 50 --eval_every 5 \
        --use_wandb --wandb_project hopfield-nav-phase-b
"""
from __future__ import annotations

import argparse
import os
from dataclasses import asdict

import numpy as np
import torch

from cls_paths import run_dir, run_name
import run_manifest
from .config import (
    TrainConfig, EnvConfig, VectorHashConfig, HopfieldConfig,
    AgentConfig, PPOConfig, validate_train_config,
)
from .encoder_io import load_encoder, validate_config
from .policy.agent import NavAgent, compute_input_dim
from .rollout.collector import RolloutCollector
from .updates.ppo import ppo_update
from .training.world_setup import (
    do_eval, make_hops, set_phase_freeze, setup_world,
)
from .evaluation.checkpoint_io import cfg_from_checkpoint


def run_store(
    cfg: TrainConfig,
    agent: NavAgent,
    worlds: list[dict],
    eval_world: dict,
    embed_dim: int,
    device: torch.device,
    n_updates: int,
    eval_every: int,
    ckpt_every: int,
    lr: float,
    use_wandb: bool,
) -> None:
    """Phase B: trunk + move + value frozen, store head trains on BCE.

    cfg should already have:
      - env.goals_active = True (needed for positive labels)
      - ppo.store_bc_weight = 1.0
      - ppo.bce_detach_trunk = True
      - ppo.bce_pos_weight_cap = 5 (or whatever cap chosen)
      - ppo.ent_coef = 0  (don't push log_std around)
    """
    print(f"\n=== Phase B: store-head pretrain, {n_updates} updates ===",
          flush=True)

    set_phase_freeze(agent, freeze_move=True, freeze_store=False,
                     freeze_value=True, freeze_rnn=True)
    trainable = [p for p in agent.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable)} "
          f"(store head only)", flush=True)
    optimizer = torch.optim.Adam(trainable, lr=lr)

    # Per-env empty Hopfield pools (store head fires regardless; the
    # rollout still routes attempted stores into the per-env Hopfield).
    pools = {}
    for w_idx, world in enumerate(worlds):
        vh = world["vectorhash"]
        envs = world["envs"]
        pools[w_idx] = make_hops(
            "empty_shared", cfg, vh, envs, embed_dim, device, cfg.batch_envs,
        )

    n_envs = cfg.envs_per_world

    for update in range(1, n_updates + 1):
        rollouts = []
        for w_idx, world in enumerate(worlds):
            vh = world["vectorhash"]
            collector = RolloutCollector(vh, cfg, embed_dim, device)
            for local_idx, env in enumerate(world["envs"]):
                env_offset = vh.env_offsets[world["env_indices"][local_idx]]
                hop = pools[w_idx][local_idx]
                rollout = collector.collect_rollout(
                    env, agent, hop, h_rnn=None, env_offset=env_offset,
                    update_idx=update, aux_scale=1.0, epsilon_now=0.0,
                )
                rollouts.append(rollout)

        agent.train()
        losses = ppo_update(agent, rollouts, cfg.ppo, optimizer, aux_scale=1.0)

        mean_r = sum(r.rewards.sum().item() for r in rollouts) / max(
            sum(r.rewards.numel() for r in rollouts), 1)

        if use_wandb:
            import wandb
            log = {f"train/{k}": v for k, v in losses.items()}
            log["train/mean_reward"] = mean_r
            log["phase_name"] = "Phase B"
            wandb.log(log)

        if update == 1 or update % 5 == 0:
            print(f"  u{update}: mean_r={mean_r:.4f} | "
                  + " ".join(f"{k}={v:.3f}" for k, v in losses.items()),
                  flush=True)

        if eval_world is not None and update % max(eval_every, 1) == 0:
            do_eval(cfg, agent, eval_world, device,
                    f"store_u{update}", use_wandb,
                    max_steps=cfg.steps_per_rollout)

        # Separate cadence -- see the same block in train_navigate for why.
        if update % max(ckpt_every, 1) == 0:
            os.makedirs(cfg.save_dir, exist_ok=True)
            torch.save({
                "agent_state_dict": agent.state_dict(),
                "config": asdict(cfg),
                "update": update,
            }, os.path.join(cfg.save_dir, f"store_u{update}.pt"))
            run_manifest.record_checkpoint(
                cfg.save_dir, f"store_u{update}.pt", update)

    do_eval(cfg, agent, eval_world, device, "after_store", use_wandb,
            max_steps=cfg.steps_per_rollout)


def train_store(args) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load Phase A checkpoint and reconstruct its config.
    print(f"Loading Phase A checkpoint: {args.load_checkpoint}", flush=True)
    ck = torch.load(args.load_checkpoint, map_location=device, weights_only=False)
    cfg = cfg_from_checkpoint(ck["config"])

    # Phase B overrides on top of Phase A's cfg.
    cfg.env.goals_active = True               # need at-goal labels
    cfg.ppo.store_bc_weight = 1.0
    cfg.ppo.bce_detach_trunk = True
    cfg.ppo.bce_pos_weight_cap = args.bce_pos_weight_cap
    cfg.ppo.ent_coef = 0.0
    cfg.encoder_checkpoint = args.encoder_checkpoint
    cfg.seed = args.seed
    cfg.device = args.device
    cfg.use_wandb = args.use_wandb
    cfg.wandb_project = args.wandb_project
    cfg.eval_every = args.eval_every
    cfg.ckpt_every = args.ckpt_every
    if args.steps_per_rollout is not None:
        cfg.steps_per_rollout = args.steps_per_rollout
    # Not inherited from the Phase A checkpoint: that field holds where Phase A
    # wrote, and reusing it would have Phase B overwrite its own parent.
    cfg.save_dir = args.save_dir

    validate_train_config(cfg)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.RandomState(cfg.seed)

    encoder, enc_cfg, encoder_gain = load_encoder(
        cfg.encoder_checkpoint, str(device), cfg.encoder_gain)
    embed_dim = enc_cfg.out_dim
    validate_config(enc_cfg, cfg.vectorhash.lambdas, encoder_gain, cfg.fwhm_ratio)
    cfg.encoder_gain = encoder_gain
    if cfg.hopfield.beta is None:
        cfg.hopfield.beta = float(encoder_gain)

    # Both train and eval worlds use goals_active=True for Phase B.
    worlds = [setup_world(cfg, encoder, embed_dim, rng, role="train")
              for _ in range(cfg.num_worlds)]
    eval_world = setup_world(cfg, encoder, embed_dim, rng, role="eval")

    input_dim = compute_input_dim(cfg.agent, embed_dim, cfg.env.observation_size)
    print(f"Agent input_dim={input_dim}", flush=True)
    agent = NavAgent(cfg.agent, input_dim).to(device)
    agent.load_state_dict(ck["agent_state_dict"])
    print(f"Loaded agent state from {args.load_checkpoint}", flush=True)

    if cfg.use_wandb:
        import wandb
        wandb.init(project=cfg.wandb_project, config={
            **asdict(cfg),
            "phase_b_updates": args.phase_b_updates,
            "phase_b_lr": args.phase_b_lr,
            "load_checkpoint": args.load_checkpoint,
        })

    if cfg.save_dir is None:
        sub = run_name(*((wandb.run.name, wandb.run.id) if cfg.use_wandb else ()))
        cfg.save_dir = str(run_dir("store", sub))
    else:
        sub = os.path.basename(str(cfg.save_dir).rstrip("/"))

    run_manifest.begin(
        cfg.save_dir, kind="store", name=sub, config=asdict(cfg),
        encoder=run_manifest.encoder_identity(
            cfg.encoder_checkpoint, enc_cfg, encoder_gain),
        parent=args.load_checkpoint,
        wandb_run=wandb.run if cfg.use_wandb else None,
    )

    run_store(
        cfg, agent, worlds, eval_world, embed_dim, device,
        n_updates=args.phase_b_updates,
        eval_every=cfg.eval_every,
        ckpt_every=(cfg.ckpt_every if cfg.ckpt_every is not None
                    else cfg.eval_every),
        lr=args.phase_b_lr,
        use_wandb=cfg.use_wandb,
    )

    os.makedirs(cfg.save_dir, exist_ok=True)
    torch.save({
        "agent_state_dict": agent.state_dict(),
        "config": asdict(cfg),
    }, os.path.join(cfg.save_dir, "store_final.pt"))
    run_manifest.record_checkpoint(cfg.save_dir, "store_final.pt")
    run_manifest.finish(cfg.save_dir)
    print(f"\nDone. Saved to {cfg.save_dir}/store_final.pt", flush=True)

    if cfg.use_wandb:
        import wandb
        wandb.finish()


def main():
    p = argparse.ArgumentParser(description="Phase-B-only store-head pretrain")
    p.add_argument("--load_checkpoint", required=True,
                   help="Phase A checkpoint to start from")
    p.add_argument("--encoder_checkpoint", required=True)
    p.add_argument("--phase_b_updates", type=int, default=50)
    p.add_argument("--phase_b_lr", type=float, default=3e-4)
    p.add_argument("--bce_pos_weight_cap", type=float, default=5.0,
                   help="Cap for BCE pos_weight (0 disables cap; raw "
                        "n_neg/n_pos used). V3 used raw ≈19 which drove "
                        "fire@off to 0.31-0.68. Cap at 5.")
    p.add_argument("--steps_per_rollout", type=int, default=None,
                   help="Override checkpoint's steps_per_rollout if needed")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--ckpt_every", type=int, default=None,
                   help="Checkpoint cadence, in updates. Default: follow "
                        "--eval_every. See train_navigate for why they "
                        "are separate.")
    p.add_argument("--save_dir", type=str, default=None,
                   help="Checkpoint directory. Default: "
                        "<CLS_RUNS>/agent_ckpts/store_<wandb run name "
                        "or timestamp>.")
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="hopfield-nav-phase-b")
    args = p.parse_args()

    train_store(args)


if __name__ == "__main__":
    main()
