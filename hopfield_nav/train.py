"""Main training script for Hopfield navigation.

Usage:
    python -m hopfield_nav.train --encoder_checkpoint encoders/run/encoder_final.pt [...]
"""
from __future__ import annotations

import argparse
import os
from dataclasses import asdict

import numpy as np
import torch

from .config import (
    TrainConfig, EnvConfig, VectorHashConfig, HopfieldConfig,
    AgentConfig, PPOConfig,
)
from .encoder import load_encoder, validate_config
from .env import GridEnv
from .vectorhash import VectorHash
from .hopfield import Hopfield
from .agent import NavAgent, compute_input_dim
from .rollout import RolloutCollector
from .ppo import ppo_update
from .eval import evaluate_navigation, evaluate_goal_discovery, evaluate_exploration


# ---------------------------------------------------------------------------
# World setup
# ---------------------------------------------------------------------------

def setup_world(
    world_idx: int,
    cfg: TrainConfig,
    encoder: torch.nn.Module,
    encoder_gain: float,
    embed_dim: int,
    rng: np.random.RandomState,
) -> tuple[list[GridEnv], list[GridEnv], VectorHash, Hopfield | None]:
    """Create envs + vectorhash for one world.

    Returns (train_envs, val_envs, vectorhash, template_hopfield_or_None).
    """
    n_train = cfg.envs_per_world
    n_val = cfg.val_envs_per_world
    size = cfg.env.size

    # Create environments
    train_envs = [
        GridEnv(size=size, speed=cfg.env.speed, observation_size=cfg.env.observation_size,
                seed=int(rng.randint(0, 10_000_000)), time_penalty=cfg.env.time_penalty)
        for _ in range(n_train)
    ]
    val_envs = [
        GridEnv(size=size, speed=cfg.env.speed, observation_size=cfg.env.observation_size,
                seed=int(rng.randint(0, 10_000_000)), time_penalty=cfg.env.time_penalty)
        for _ in range(n_val)
    ]

    # Build vectorhash — register all envs together so they get non-overlapping offsets
    all_envs = train_envs + val_envs
    vh = VectorHash(cfg.vectorhash, size=size)
    vh.build_scaffold()
    vh.register_envs(all_envs)
    vh.precompute_encoded_phi(encoder, cfg.fwhm_ratio, device=cfg.device)

    # Store the global env_offset index for each env so callers don't need to know
    # the train/val split to look up the right offset.
    train_env_indices = list(range(n_train))
    val_env_indices = list(range(n_train, n_train + n_val))

    # Template Hopfield for pre_stored mode
    template_hop = None
    if cfg.hopfield.init_mode == "pre_stored":
        template_hop = Hopfield(embed_dim, beta=cfg.hopfield.beta, device=cfg.device)
        for pattern in vh.get_goal_encodings(all_envs):
            template_hop.input_memory(torch.from_numpy(pattern).float())
        print(f"  world {world_idx}: pre-stored {template_hop.num_memories} goal patterns")

    return train_envs, val_envs, vh, template_hop, train_env_indices, val_env_indices


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig) -> None:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.RandomState(cfg.seed)

    # Load encoder
    print(f"Loading encoder from {cfg.encoder_checkpoint}")
    encoder, enc_cfg, encoder_gain = load_encoder(
        cfg.encoder_checkpoint, str(device), cfg.encoder_gain)
    embed_dim = enc_cfg.out_dim
    validate_config(enc_cfg, cfg.vectorhash.lambdas, encoder_gain, cfg.fwhm_ratio)
    print(f"Encoder: type={enc_cfg.encoder_type}, out_dim={embed_dim}, gain={encoder_gain}")

    # Setup worlds
    worlds = []
    for w in range(cfg.num_worlds):
        print(f"Setting up world {w}")
        train_envs, val_envs, vh, template_hop, train_idxs, val_idxs = setup_world(
            w, cfg, encoder, encoder_gain, embed_dim, rng)
        worlds.append({
            "train_envs": train_envs,
            "val_envs": val_envs,
            "vectorhash": vh,
            "template_hopfield": template_hop,
            "train_env_indices": train_idxs,
            "val_env_indices": val_idxs,
        })

    # Create agent
    input_dim = compute_input_dim(cfg.agent, embed_dim)
    print(f"Agent input_dim={input_dim}, hidden={cfg.agent.hidden_size}")
    agent = NavAgent(cfg.agent, input_dim).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.ppo.lr)

    # Load checkpoint if provided (for curriculum / fine-tuning)
    if cfg.load_checkpoint:
        print(f"Loading checkpoint from {cfg.load_checkpoint}")
        ckpt = torch.load(cfg.load_checkpoint, map_location=device, weights_only=False)
        agent.load_state_dict(ckpt["agent_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"  loaded (from update {ckpt.get('update', '?')})")

    # Wandb
    if cfg.use_wandb:
        import wandb
        wandb.init(project=cfg.wandb_project, config=asdict(cfg))

    # Training loop
    print(f"Starting training: {cfg.n_updates} updates")
    for update in range(1, cfg.n_updates + 1):
        all_losses: dict[str, float] = {}
        total_reward = 0.0
        total_steps = 0

        for world in worlds:
            vh = world["vectorhash"]
            template_hop = world["template_hopfield"]

            # Create per-batch Hopfield instances for this world pass
            B = cfg.batch_envs
            if cfg.hopfield.agent_can_store:
                if template_hop is not None:
                    hops = [template_hop.clone() for _ in range(B)]
                else:
                    hops = [Hopfield(embed_dim, beta=cfg.hopfield.beta, device=str(device))
                            for _ in range(B)]
            else:
                hops = template_hop if template_hop is not None else Hopfield(
                    embed_dim, beta=cfg.hopfield.beta, device=str(device))

            collector = RolloutCollector(vh, cfg, embed_dim, device)

            all_rollouts = []
            for local_idx, env in enumerate(world["train_envs"]):
                global_idx = world["train_env_indices"][local_idx]
                env_offset = vh.env_offsets[global_idx]

                rollout = collector.collect_rollout(
                    env, agent, hops, h_rnn=None, env_offset=env_offset)
                all_rollouts.append(rollout)

                total_reward += rollout.rewards.sum().item()
                total_steps += rollout.rewards.numel()

            # PPO update on combined rollouts
            agent.train()
            for rollout in all_rollouts:
                losses = ppo_update(agent, rollout, cfg.ppo, optimizer)
                for k, v in losses.items():
                    all_losses[k] = all_losses.get(k, 0.0) + v

        # Average losses
        n_rollouts = sum(len(w["train_envs"]) for w in worlds)
        for k in all_losses:
            all_losses[k] /= max(n_rollouts, 1)

        mean_reward = total_reward / max(total_steps, 1)

        if update % 10 == 0 or update == 1:
            loss_str = " | ".join(f"{k}={v:.4f}" for k, v in all_losses.items())
            print(f"update {update:04d} | mean_reward={mean_reward:.4f} | {loss_str}")

        if cfg.use_wandb:
            import wandb
            log_dict = {f"train/{k}": v for k, v in all_losses.items()}
            log_dict["train/mean_reward"] = mean_reward
            wandb.log(log_dict, step=update)

        # Eval
        if cfg.eval_every > 0 and update % cfg.eval_every == 0:
            for w_idx, world in enumerate(worlds):
                nav_metrics = evaluate_navigation(
                    agent, world["val_envs"], world["vectorhash"],
                    world["val_env_indices"], cfg, device,
                )
                print(f"  eval nav w{w_idx}: {nav_metrics}")

                # Goal discovery + exploration on first val env
                if world["val_envs"]:
                    val_env = world["val_envs"][0]
                    val_idx = world["val_env_indices"][0]

                    disc_metrics = evaluate_goal_discovery(
                        agent, val_env, world["vectorhash"],
                        val_idx, cfg, device,
                        n_distractors_list=[0],
                    )
                    print(f"  eval disc w{w_idx}: {disc_metrics}")

                    expl_metrics = evaluate_exploration(
                        agent, val_env, world["vectorhash"],
                        val_idx, cfg, device,
                        n_distractors_list=[0],
                    )
                    print(f"  eval expl w{w_idx}: {expl_metrics}")

                if cfg.use_wandb:
                    import wandb
                    wandb.log({f"eval/w{w_idx}/{k}": v for k, v in nav_metrics.items()}, step=update)
                    if world["val_envs"]:
                        for n_dist, dm in disc_metrics.items():
                            for k, v in dm.items():
                                wandb.log({f"eval/w{w_idx}/disc_{n_dist}/{k}": v}, step=update)
                        for n_dist, em in expl_metrics.items():
                            for k, v in em.items():
                                wandb.log({f"eval/w{w_idx}/expl_{n_dist}/{k}": v}, step=update)

        # Save
        if cfg.save_every > 0 and update % cfg.save_every == 0:
            os.makedirs(cfg.save_dir, exist_ok=True)
            torch.save({
                "agent_state_dict": agent.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": asdict(cfg),
                "update": update,
            }, os.path.join(cfg.save_dir, f"hopfield_nav_update{update}.pt"))

    print("Training complete.")
    if cfg.use_wandb:
        import wandb
        wandb.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Hopfield navigation agent")
    # Encoder
    parser.add_argument("--encoder_checkpoint", type=str, required=True)
    parser.add_argument("--encoder_gain", type=float, default=None)
    parser.add_argument("--fwhm_ratio", type=float, default=0.25)
    # Env
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--observation_size", type=int, default=512)
    parser.add_argument("--time_penalty", type=float, default=0.01)
    parser.add_argument("--movement_mode", type=str, default="discrete",
                        choices=["discrete", "continuous"])
    # VectorHash
    parser.add_argument("--lambdas", type=int, nargs="+", default=[11, 12])
    parser.add_argument("--Np", type=int, default=1600)
    parser.add_argument("--Npos", type=int, default=None,
                        help="Override Npos (default: product of lambdas). Use to limit memory.")
    # Hopfield
    parser.add_argument("--hopfield_beta", type=float, default=2.0)
    parser.add_argument("--hopfield_alpha", type=float, default=1.0)
    parser.add_argument("--hopfield_steps", type=int, default=1)
    parser.add_argument("--hopfield_init", type=str, default="empty",
                        choices=["empty", "pre_stored"])
    parser.add_argument("--agent_can_store", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--store_cost", type=float, default=0.0,
                        help="Reward penalty per store action (metabolic cost)")
    parser.add_argument("--store_bonus", type=float, default=0.0,
                        help="Reward bonus for storing while at goal")
    # Agent
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_rnn_layers", type=int, default=1)
    parser.add_argument("--hopfield_mode", type=str, default="discrete",
                        choices=["discrete", "continuous"])
    parser.add_argument("--input_encoded_state", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--input_hopfield_signal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--input_prev_action", action=argparse.BooleanOptionalAction, default=False)
    # Training
    parser.add_argument("--num_worlds", type=int, default=1)
    parser.add_argument("--envs_per_world", type=int, default=4)
    parser.add_argument("--val_envs_per_world", type=int, default=2)
    parser.add_argument("--batch_envs", type=int, default=16)
    parser.add_argument("--steps_per_rollout", type=int, default=64)
    parser.add_argument("--n_updates", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--explore_steps", type=int, default=None,
                        help="Two-phase rollout: store allowed for first N steps, frozen after")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    # Checkpoint loading
    parser.add_argument("--load_checkpoint", type=str, default=None,
                        help="Path to agent checkpoint to load (for curriculum / fine-tuning)")
    # Wandb
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="hopfield-nav")

    args = parser.parse_args()

    cfg = TrainConfig(
        env=EnvConfig(
            size=args.size, observation_size=args.observation_size,
            time_penalty=args.time_penalty, movement_mode=args.movement_mode,
        ),
        vectorhash=VectorHashConfig(lambdas=args.lambdas, Np=args.Np, Npos=args.Npos),
        hopfield=HopfieldConfig(
            beta=args.hopfield_beta, alpha=args.hopfield_alpha,
            steps=args.hopfield_steps, init_mode=args.hopfield_init,
            agent_can_store=args.agent_can_store,
            store_cost=args.store_cost,
            store_bonus=args.store_bonus,
        ),
        agent=AgentConfig(
            hidden_size=args.hidden_size, num_rnn_layers=args.num_rnn_layers,
            hopfield_mode=args.hopfield_mode, movement_mode=args.movement_mode,
            input_encoded_state=args.input_encoded_state,
            input_hopfield_signal=args.input_hopfield_signal,
            input_prev_action=args.input_prev_action,
        ),
        ppo=PPOConfig(lr=args.lr),
        encoder_checkpoint=args.encoder_checkpoint,
        encoder_gain=args.encoder_gain,
        load_checkpoint=args.load_checkpoint,
        fwhm_ratio=args.fwhm_ratio,
        num_worlds=args.num_worlds,
        envs_per_world=args.envs_per_world,
        val_envs_per_world=args.val_envs_per_world,
        batch_envs=args.batch_envs,
        steps_per_rollout=args.steps_per_rollout,
        explore_steps=args.explore_steps,
        n_updates=args.n_updates,
        eval_every=args.eval_every,
        save_every=args.save_every,
        save_dir=args.save_dir,
        seed=args.seed,
        device=args.device,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
    )
    train(cfg)


if __name__ == "__main__":
    main()
