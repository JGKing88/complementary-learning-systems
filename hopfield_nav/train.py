"""Main training script for Hopfield navigation.

Usage:
    python -m hopfield_nav.train --encoder_checkpoint encoders/run/encoder_final.pt [...]
"""
from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from datetime import datetime

import numpy as np
import torch

from .config import (
    TrainConfig, EnvConfig, VectorHashConfig, HopfieldConfig,
    AgentConfig, PPOConfig, BCConfig, validate_train_config,
)
from .encoder import load_encoder, validate_config
from .env import make_env, warn_if_offcell_stores
from .vectorhash import VectorHash
from .hopfield import Hopfield
from .agent import NavAgent, compute_input_dim
from .rollout import RolloutCollector
from .ppo import ppo_update
from .bc import bc_update
from .eval import (
    evaluate_navigation, evaluate_goal_discovery, evaluate_exploration,
    evaluate_realistic, evaluate_union_coverage,
)


# ---------------------------------------------------------------------------
# World setup
# ---------------------------------------------------------------------------

def setup_train_world(
    world_idx: int,
    cfg: TrainConfig,
    encoder: torch.nn.Module,
    encoder_gain: float,
    embed_dim: int,
    rng: np.random.RandomState,
) -> dict:
    """Create training envs + VectorHash scaffold for one training world.

    Training worlds are fully independent of eval: their scaffold contains only
    their own train envs, and their template Hopfield (if any) preloads only
    those train envs' goals.
    """
    n_train = cfg.envs_per_world
    size = cfg.env.size

    # make_env forwards the whole EnvConfig. Constructing GridEnv by hand here
    # used to drop goals_active, goal_reward and goal_radius, which VecEnv then
    # read off the base env -- so --goal_radius was silently ignored during
    # training while eval honored it (train_phased.setup_world always used
    # make_env). See docs/REFACTOR_ASSESSMENT.md, phase 2.
    train_envs = [
        make_env(cfg.env, cfg.agent.movement_mode, seed=int(rng.randint(0, 10_000_000)))
        for _ in range(n_train)
    ]

    vh = VectorHash(cfg.vectorhash, size=size)
    vh.build_scaffold()
    vh.register_envs(train_envs, placement="spread")
    vh.precompute_encoded_phi(encoder, cfg.fwhm_ratio, device=cfg.device)

    train_env_indices = list(range(n_train))

    template_hop = None
    if cfg.hopfield.init_mode == "pre_stored":
        template_hop = Hopfield(embed_dim, beta=cfg.hopfield.beta, device=cfg.device)
        for pattern in vh.get_goal_encodings(train_envs):
            template_hop.input_memory(torch.from_numpy(pattern).float())
        print(f"  train world {world_idx}: pre-stored {template_hop.num_memories} goal patterns")

    return {
        "train_envs": train_envs,
        "vectorhash": vh,
        "template_hopfield": template_hop,
        "train_env_indices": train_env_indices,
    }


def setup_eval_world(
    cfg: TrainConfig,
    encoder: torch.nn.Module,
    encoder_gain: float,
    embed_dim: int,
    rng: np.random.RandomState,
) -> dict:
    """Build a single dedicated eval world with its own VectorHash scaffold.

    Decoupled from num_worlds — this is built once at startup and reused for
    every eval pass. Contains num_val_envs val envs only (no train envs), so
    distractors sampled from this scaffold never accidentally coincide with
    training-env regions.
    """
    n_val = cfg.num_val_envs
    size = cfg.env.size

    val_envs = [
        make_env(cfg.env, cfg.agent.movement_mode,
                 seed=int(rng.randint(0, 10_000_000)))
        for _ in range(n_val)
    ]

    vh = VectorHash(cfg.vectorhash, size=size)
    vh.build_scaffold()
    vh.register_envs(val_envs, placement="spread")
    vh.precompute_encoded_phi(encoder, cfg.fwhm_ratio, device=cfg.device)

    val_env_indices = list(range(n_val))

    return {
        "val_envs": val_envs,
        "vectorhash": vh,
        "val_env_indices": val_env_indices,
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig) -> None:
    validate_train_config(cfg)
    warn_if_offcell_stores(cfg.env, where="train")
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

    # Resolve auto-defaults tied to the encoder. Both encoder_gain (via
    # load_encoder's gain_override=None branch) and hopfield.beta fall back to
    # the encoder's own gain when not explicitly set, so a single source of
    # truth governs the embedding scale across encoder + Hopfield.
    cfg.encoder_gain = encoder_gain
    if cfg.hopfield.beta is None:
        cfg.hopfield.beta = float(encoder_gain)
        print(f"  hopfield.beta defaulted to encoder_gain={encoder_gain}")

    # Setup training worlds (train envs + per-world VectorHash scaffold)
    worlds = []
    for w in range(cfg.num_worlds):
        print(f"Setting up train world {w}")
        worlds.append(setup_train_world(w, cfg, encoder, encoder_gain, embed_dim, rng))

    # Setup a single dedicated eval world, built once and reused.
    print(f"Setting up eval world ({cfg.num_val_envs} val envs)")
    eval_world = setup_eval_world(cfg, encoder, encoder_gain, embed_dim, rng)

    # Create agent
    input_dim = compute_input_dim(cfg.agent, embed_dim, cfg.env.observation_size)
    print(f"Agent input_dim={input_dim}, hidden={cfg.agent.hidden_size}")
    agent = NavAgent(cfg.agent, input_dim).to(device)
    # LR comes from the active training mode's config block.
    optim_lr = cfg.bc.lr if cfg.training_mode == "bc" else cfg.ppo.lr
    optimizer = torch.optim.Adam(agent.parameters(), lr=optim_lr)

    # Load checkpoint if provided (for curriculum / fine-tuning)
    if cfg.load_checkpoint:
        print(f"Loading checkpoint from {cfg.load_checkpoint}")
        ckpt = torch.load(cfg.load_checkpoint, map_location=device, weights_only=False)
        agent.load_state_dict(ckpt["agent_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            # Override per-group lr to whatever the CLI args specified —
            # otherwise Adam's saved state pins the lr from the source run,
            # making CLI --bc_lr / --ppo lr a silent no-op when resuming.
            for g in optimizer.param_groups:
                g["lr"] = optim_lr
        print(f"  loaded (from update {ckpt.get('update', '?')})")

    if cfg.use_wandb:
        import wandb
        wandb.init(project=cfg.wandb_project, config=asdict(cfg))

    if cfg.save_dir is None:
        if cfg.use_wandb:
            sub = wandb.run.name or wandb.run.id
        else:
            sub = datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg.save_dir = os.path.join("checkpoint", sub)

    # Distractor RNG for training: deterministic seeding lets the same distractor
    # patterns recur across updates, but with per-update offset so the agent can't
    # memorize a fixed distractor set.
    dist_rng = np.random.RandomState(cfg.seed + 7919)

    # Training loop
    print(f"Starting training: {cfg.n_updates} updates")
    for update in range(1, cfg.n_updates + 1):
        all_losses: dict[str, float] = {}
        total_reward = 0.0
        total_goal_steps = 0     # per-step at-goal indicator summed across rollouts
        total_goal_trajs = 0     # number of (env, batch) trajectories that hit goal at least once
        total_trajs = 0          # total (env, batch) trajectories this update
        total_steps = 0

        # Linear anneal of auxiliary signals (store_bonus, store_bc_weight)
        if cfg.hopfield.aux_anneal_updates > 0:
            aux_scale = max(0.0, 1.0 - (update - 1) / cfg.hopfield.aux_anneal_updates)
        else:
            aux_scale = 1.0

        # Epsilon exploration schedule: constant, or linearly decayed to 0.
        if cfg.hopfield.epsilon_anneal_updates > 0:
            eps_scale = max(0.0, 1.0 - (update - 1) / cfg.hopfield.epsilon_anneal_updates)
        else:
            eps_scale = 1.0
        epsilon_now = cfg.hopfield.epsilon_explore * eps_scale

        # Phase A: collect all rollouts across worlds + envs with the agent
        # FROZEN. Each (world, env) rollout gets a FRESH Hopfield list so
        # memories don't bleed across envs.
        B = cfg.batch_envs
        # Optionally re-sample env_offsets so this update's rollout buffer
        # covers different scaffold patches than the previous one. PPO-correct
        # because the buffer is collected by a single (current) policy. Only
        # safe under static_vectorhash (Wsp/Wps would otherwise need rebuild).
        # Also resamples each env's internal goal so the (env, goal) pair set
        # is fresh every update — without this the 20 train envs share the
        # same 20 goals for the entire run.
        if cfg.hopfield.refresh_envs_each_update:
            for world in worlds:
                vh = world["vectorhash"]
                if not vh.cfg.static_vectorhash:
                    raise RuntimeError(
                        "refresh_envs_each_update requires --static-vectorhash"
                    )
                for env in world["train_envs"]:
                    env.reset_goal()
                vh.register_envs(world["train_envs"], placement="random")
        all_rollouts = []
        for world in worlds:
            vh = world["vectorhash"]
            template_hop = world["template_hopfield"]
            collector = RolloutCollector(vh, cfg, embed_dim, device)

            for local_idx, env in enumerate(world["train_envs"]):
                global_idx = world["train_env_indices"][local_idx]
                env_offset = vh.env_offsets[global_idx]

                # Fresh Hopfield instances per env: no cross-env contamination.
                if cfg.hopfield.agent_can_store:
                    if template_hop is not None:
                        hops = [template_hop.clone() for _ in range(B)]
                    else:
                        hops = [Hopfield(embed_dim, beta=cfg.hopfield.beta, device=str(device))
                                for _ in range(B)]
                else:
                    hops = template_hop if template_hop is not None else Hopfield(
                        embed_dim, beta=cfg.hopfield.beta, device=str(device))

                # Optionally preload N distractor patterns into each per-env Hopfield
                # so training distribution matches eval (where `evaluate_goal_discovery`
                # pre-populates distractors at n_distractors > 0). Only meaningful
                # when we have a per-env hops list (agent_can_store=True).
                use_variable = cfg.hopfield.n_train_distractors_max > 0
                use_fixed = cfg.hopfield.n_train_distractors > 0
                if (use_variable or use_fixed) and isinstance(hops, list):
                    Npos = vh.Npos
                    cx, cy = env_offset
                    env_size = cfg.env.size
                    lo = cfg.hopfield.n_train_distractors_min
                    hi = cfg.hopfield.n_train_distractors_max
                    for b in range(B):
                        if use_variable:
                            n_d = int(dist_rng.randint(lo, hi + 1))
                        else:
                            n_d = cfg.hopfield.n_train_distractors
                        placed = 0
                        while placed < n_d:
                            gx = dist_rng.randint(0, Npos)
                            gy = dist_rng.randint(0, Npos)
                            if cx <= gx < cx + env_size and cy <= gy < cy + env_size:
                                continue
                            pat = vh.encoded_Phi[gx, gy].copy()
                            hops[b].input_memory(torch.from_numpy(pat).float())
                            placed += 1

                rollout = collector.collect_rollout(
                    env, agent, hops, h_rnn=None, env_offset=env_offset,
                    update_idx=update, aux_scale=aux_scale,
                    epsilon_now=epsilon_now,
                )
                all_rollouts.append(rollout)

                total_reward += rollout.rewards.sum().item()
                total_steps += rollout.rewards.numel()
                # goal_reached: (B, T) 1.0 indicator per (env, step). Hits-per-step
                # gives the at-goal density; per-trajectory gives the fraction of
                # 200-step trials that ever reached the goal (the random-walk
                # baseline for an 8x8 grid is ≈0.66 in 200 steps).
                gr = rollout.goal_reached
                total_goal_steps += int(gr.sum().item())
                total_goal_trajs += int((gr.sum(dim=1) > 0).sum().item())
                total_trajs += int(gr.shape[0])

        # Phase B: single pooled update over the full buffer of rollouts.
        # Route to the active training mode's update function.
        agent.train()
        if cfg.training_mode == "bc":
            all_losses = bc_update(
                agent, all_rollouts, cfg.bc, cfg.agent.movement_mode, optimizer,
            )
        else:
            all_losses = ppo_update(
                agent, all_rollouts, cfg.ppo, optimizer, aux_scale=aux_scale,
            )

        mean_reward = total_reward / max(total_steps, 1)
        goal_step_rate = total_goal_steps / max(total_steps, 1)
        goal_traj_rate = total_goal_trajs / max(total_trajs, 1)

        if update % 10 == 0 or update == 1:
            loss_str = " | ".join(f"{k}={v:.4f}" for k, v in all_losses.items())
            print(
                f"update {update:04d} | mean_reward={mean_reward:.4f} | "
                f"goal_traj_rate={goal_traj_rate:.3f} | "
                f"goal_step_rate={goal_step_rate:.4f} | {loss_str}"
            )

        if cfg.use_wandb:
            import wandb
            log_dict = {f"train/{k}": v for k, v in all_losses.items()}
            log_dict["train/mean_reward"] = mean_reward
            log_dict["train/goal_traj_rate"] = goal_traj_rate
            log_dict["train/goal_step_rate"] = goal_step_rate
            wandb.log(log_dict, step=update)

        # Eval — single dedicated eval world, unified structure across all three
        # evals. Each trial gets a fresh Hopfield (no cross-val contamination).
        if cfg.eval_every > 0 and update % cfg.eval_every == 0 and eval_world["val_envs"]:
            val_envs = eval_world["val_envs"]
            val_vh = eval_world["vectorhash"]
            val_idxs = eval_world["val_env_indices"]
            dist_list = cfg.val_n_distractors_list
            n_trials = cfg.n_val_trials

            # Eval rollout budgets scale with grid: ~5×size² steps so the
            # exploration eval has room to plausibly reach cov=0.5 on bigger
            # grids (random-walk hitting time ~ N²).
            eval_max = max(200, 5 * cfg.env.size * cfg.env.size)
            nav_det = evaluate_navigation(
                agent, val_envs, val_vh, val_idxs, cfg, device,
                num_trials=n_trials, max_steps=eval_max,
                n_distractors_list=dist_list, deterministic=True,
            )
            nav_stoch = evaluate_navigation(
                agent, val_envs, val_vh, val_idxs, cfg, device,
                num_trials=n_trials, max_steps=eval_max,
                n_distractors_list=dist_list, deterministic=False,
            )
            disc = evaluate_goal_discovery(
                agent, val_envs, val_vh, val_idxs, cfg, device,
                num_trials=n_trials, max_steps=eval_max,
                n_distractors_list=dist_list,
            )
            expl = evaluate_exploration(
                agent, val_envs, val_vh, val_idxs, cfg, device,
                num_trials=n_trials, max_steps=eval_max,
                n_distractors_list=dist_list,
            )
            union = evaluate_union_coverage(
                agent, val_envs, val_vh, val_idxs, cfg, device,
                num_trials=10, max_steps=eval_max // 2,
                n_distractors_list=dist_list,
            )

            print(f"  eval nav_det: {nav_det}")
            print(f"  eval nav_stoch: {nav_stoch}")
            print(f"  eval disc: {disc}")
            print(f"  eval expl: {expl}")
            print(f"  eval union: {union}")

            if cfg.use_wandb:
                import wandb
                log = {}
                for n_dist in dist_list:
                    for k, v in nav_det[n_dist].items():
                        log[f"eval/nav_det_{n_dist}/{k}"] = v
                    for k, v in nav_stoch[n_dist].items():
                        log[f"eval/nav_stoch_{n_dist}/{k}"] = v
                    for k, v in disc[n_dist].items():
                        log[f"eval/disc_{n_dist}/{k}"] = v
                    for k, v in expl[n_dist].items():
                        log[f"eval/expl_{n_dist}/{k}"] = v
                    for k, v in union[n_dist].items():
                        log[f"eval/union_{n_dist}/{k}"] = v
                wandb.log(log, step=update)

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

    # Realistic end-of-training eval: one persistent Hopfield accumulating
    # memories across envs visited sequentially; retest prior envs after each
    # new visit with storing disabled to measure interference.
    if cfg.realistic_steps_per_env > 0 and eval_world["val_envs"]:
        print(f"Running realistic eval ({cfg.realistic_steps_per_env} steps/env, "
              f"{len(eval_world['val_envs'])} envs)")
        realistic = evaluate_realistic(
            agent, eval_world["val_envs"], eval_world["vectorhash"],
            eval_world["val_env_indices"], cfg, device,
            steps_per_env=cfg.realistic_steps_per_env,
            seed=cfg.seed + 1000,
            deterministic=True,
        )
        print(f"  realistic summary: {realistic['summary']}")

        if cfg.use_wandb:
            import wandb
            log = {}
            for i, m in realistic["primary"].items():
                log[f"realistic/env_{i}/primary/n_reaches"] = m["n_reaches"]
                log[f"realistic/env_{i}/primary/mean_interval"] = m["mean_interval"]
            for (visit_i, retest_j), m in realistic["retest"].items():
                gap = visit_i - retest_j
                log[f"realistic/env_{retest_j}/retest_gap_{gap}/n_reaches"] = m["n_reaches"]
                log[f"realistic/env_{retest_j}/retest_gap_{gap}/mean_interval"] = m["mean_interval"]
            for k, v in realistic["summary"].items():
                log[f"realistic/summary/{k}"] = v
            # Per-env drift curves: reaches-vs-gap tables (plottable in wandb UI)
            for j, curve in realistic["drift"].items():
                if len(curve) < 2:
                    continue
                tbl = wandb.Table(columns=["gap", "n_reaches", "mean_interval"])
                for gap, m in curve:
                    tbl.add_data(gap, m["n_reaches"], m["mean_interval"])
                log[f"realistic/drift/env_{j}"] = tbl
            wandb.log(log, step=cfg.n_updates)

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
    parser.add_argument("--observation_size", type=int, default=60,
                        help="Foveal sensory vector dim (= number of rays evenly spaced "
                             "across the 120° forward cone)")
    parser.add_argument("--time_penalty", type=float, default=0.01)
    parser.add_argument("--movement_mode", type=str, default="discrete",
                        choices=["discrete", "continuous"])
    parser.add_argument("--goal_radius", type=float, default=0.5,
                        help="Euclidean radius around goal that counts as 'at goal'. "
                             "Default 0.5 reproduces snap-equality on integer-snapped "
                             "positions; larger values fuzz the goal region.")
    parser.add_argument("--allow_offcell_store",
                   action=argparse.BooleanOptionalAction, default=False,
                   help="Whether a store fired while at goal may write a cell other than the goal's. Only reachable at goal_radius > 0.5, where at_goal tests the float position but embeddings are read at the snapped cell. Default False: the goal cell's embedding is stored instead, so the pattern written is the one navigation will later recall. Pass --allow_offcell_store for the pre-2026-08 behavior.")
    # VectorHash
    parser.add_argument("--lambdas", type=int, nargs="+", default=[11, 12])
    parser.add_argument("--Np", type=int, default=1600)
    parser.add_argument("--Npos", type=int, default=None,
                        help="Override Npos (default: product of lambdas). Use to limit memory.")
    parser.add_argument(
        # --gbook-only / --no-gbook-only are the pre-rename spelling, kept as a
        # deprecated alias so old sbatch scripts and sweep variants still parse.
        # BooleanOptionalAction generates the --no- form for every long option.
        "--static-vectorhash", "--gbook-only",
        dest="static_vectorhash", action=argparse.BooleanOptionalAction,
        default=False,
        help="Build only gbook (+ encoded_Phi); skip pbook, Wgp, Wsp, and scaffold "
             "self-test. Sensory input (when enabled) reads directly from each env's codebook. "
             "(--gbook-only is a deprecated alias.)",
    )
    # Hopfield
    parser.add_argument("--hopfield_beta", type=float, default=None,
                        help="Hopfield softmax temperature. If omitted, defaults to the encoder's gain.")
    parser.add_argument("--hopfield_alpha", type=float, default=1.0)
    parser.add_argument("--hopfield_steps", type=int, default=1)
    parser.add_argument("--hopfield_init", type=str, default="empty",
                        choices=["empty", "pre_stored"])
    parser.add_argument("--agent_can_store", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--store_cost", type=float, default=0.0,
                        help="Reward penalty per store action (metabolic cost)")
    parser.add_argument("--store_bonus", type=float, default=0.0,
                        help="Reward bonus for storing while at goal")
    parser.add_argument("--store_bc_weight", type=float, default=0.0,
                        help="Auxiliary BCE loss weight on store head: BCE(logits, at_goal)")
    parser.add_argument("--auto_store_warmup", type=int, default=0,
                        help="For first N updates, force-store when at goal regardless of agent action")
    parser.add_argument("--auto_nav_warmup", type=int, default=0,
                        help="For first N updates, force-copy the Hopfield-suggested move for any env that has a stored memory (teacher forcing)")
    parser.add_argument("--aux_anneal_updates", type=int, default=0,
                        help="Linearly decay store_bonus + store_bc_weight to 0 over this many updates")
    parser.add_argument("--novelty_reward", type=float, default=0.0,
                        help="+reward for first-visit to snapped cells during explore phase (per-rollout)")
    parser.add_argument("--n_train_distractors", type=int, default=0,
                        help="If >0, preload this many distractor patterns (sampled from outside the env's region) into each per-env training Hopfield at rollout start. Matches eval-time distractor setup.")
    parser.add_argument("--n_train_distractors_min", type=int, default=0,
                        help="If --n_train_distractors_max > 0, distractor count per rollout is sampled uniformly from [min, max]. Overrides --n_train_distractors when max>0.")
    parser.add_argument("--n_train_distractors_max", type=int, default=0,
                        help="See --n_train_distractors_min. Set max>0 to enable variable-count distractors per rollout.")
    parser.add_argument("--wall_penalty", type=float, default=0.0,
                        help="Per-step reward penalty when the agent is at a grid-edge cell. Counters the perimeter-walk basin that novelty alone tends to reward.")
    parser.add_argument("--revisit_penalty", type=float, default=0.0,
                        help="Per-step reward penalty when the agent revisits a cell already visited in this rollout. Anti-perimeter-loop without penalizing first-time wall-touches en route to wall-adjacent goals.")
    parser.add_argument("--epsilon_explore", type=float, default=0.0,
                        help="Per-step probability of replacing the sampled movement action with a uniform-random direction. The action is injected via the agent override path so log_prob is re-scored under the current policy (PPO ratio stays well-defined).")
    parser.add_argument("--epsilon_anneal_updates", type=int, default=0,
                        help="Linearly decay epsilon_explore from full→0 over this many updates (0 = no decay / constant).")
    parser.add_argument("--refresh_envs_each_update", action="store_true",
                        help="Re-sample env_offsets at the start of every PPO update so each rollout buffer covers a different patch of the global scaffold. Reduces seed-lottery on scaffold position. PPO-correct because the buffer comes from a single policy.")
    # Agent
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_rnn_layers", type=int, default=1)
    parser.add_argument("--hopfield_mode", type=str, default="discrete",
                        choices=["discrete", "continuous"])
    parser.add_argument("--input_encoded_state", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--input_hopfield_signal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--input_prev_action", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--input_prev_reward", action=argparse.BooleanOptionalAction, default=False,
                        help="Feed previous step's reward as input channel (RNN history anchor)")
    parser.add_argument("--input_hopfield_raw", action=argparse.BooleanOptionalAction, default=False,
                        help="Continuous-mode only: feed raw unnormalized q instead of hopfield_signal")
    parser.add_argument("--input_goal_in_memory", action=argparse.BooleanOptionalAction, default=False,
                        help="Add 1-bit input: agent has stored at goal during this rollout (Hopfield is trustworthy). Lets policy distinguish explore (bit=0) and nav (bit=1) modes cleanly.")
    parser.add_argument("--input_sensory", action=argparse.BooleanOptionalAction, default=False,
                        help="Feed raw env observation (codebook vector at current position) to the RNN")
    parser.add_argument("--init_log_std", type=float, default=0.0,
                        help="Continuous policy: initial log std (default 0.0 → std=1.0)")
    parser.add_argument("--freeze_log_std", action="store_true",
                        help="Pin movement_log_std at init (no gradient). Forces PPO to shape the policy mean directly — fixes the navD vs navS gap when learnable log_std lets samples 'hide' a poorly-trained mean.")
    parser.add_argument("--ent_coef", type=float, default=0.01,
                        help="Movement policy entropy bonus weight")
    parser.add_argument("--store_ent_coef", type=float, default=0.05,
                        help="Store policy entropy bonus weight")
    # Training
    parser.add_argument("--num_worlds", type=int, default=1)
    parser.add_argument("--envs_per_world", type=int, default=4)
    parser.add_argument("--num_val_envs", type=int, default=2,
                        help="Number of val envs in the dedicated eval world (independent of num_worlds)")
    parser.add_argument("--n_val_trials", type=int, default=32,
                        help="Number of independent trials per (val_env, n_distractors) bucket in every eval")
    parser.add_argument("--val_distractors", type=int, nargs="+", default=[0],
                        help="Distractor counts swept at every eval (e.g. 0 1 3 5 10)")
    parser.add_argument("--realistic_steps_per_env", type=int, default=1000,
                        help="End-of-training realistic eval: steps per env. Set 0 to skip.")
    parser.add_argument("--hopfield_oracle", action="store_true",
                        help="Eval: use oracle hopfield signal (direct-to-goal in local frame) "
                             "when the goal is considered in memory (see eval.agent_step).")
    parser.add_argument("--action_oracle", action="store_true",
                        help="Eval: greedy best move toward goal when goal is in memory "
                             "(see eval.agent_step); store head unchanged.")
    parser.add_argument("--batch_envs", type=int, default=16)
    parser.add_argument("--steps_per_rollout", type=int, default=64)
    parser.add_argument("--n_updates", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    # Training mode: "ppo" (default) or "bc" (DAgger supervised).
    parser.add_argument("--training_mode", type=str, default="ppo",
                        choices=["ppo", "bc"],
                        help="ppo (default) or bc (DAgger-style supervised against oracle)")
    parser.add_argument("--bc_lr", type=float, default=3e-4,
                        help="Learning rate when --training_mode bc (ignored otherwise)")
    parser.add_argument("--bc_store_weight", type=float, default=1.0,
                        help="Weight on BCE(store) relative to CE(move) in BC loss")
    parser.add_argument("--bc_move_ent_coef", type=float, default=0.0,
                        help="Optional entropy bonus on movement logits in BC update")
    parser.add_argument("--bc_supervise_explore", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="BC mode: supervise pre-memory (novelty) nav labels")
    parser.add_argument("--bc_nav_weight", type=float, default=1.0,
                        help="Per-step weight on trust_hop (Hopfield-follow) move labels. >1 fights dilution from abundant novelty labels.")
    parser.add_argument("--bc_n_minibatches", type=int, default=4,
                        help="BC update: minibatches per epoch over the rollout pool")
    parser.add_argument("--bc_epochs", type=int, default=1,
                        help="BC update: gradient epochs per rollout buffer")
    parser.add_argument("--bc_novelty_fallback", type=str, default="random",
                        choices=["random", "stay"],
                        help="BC novelty oracle fallback when every neighbor is visited")
    parser.add_argument("--bc_bce_pos_weight_cap", type=float, default=0.0,
                        help="Cap on store-BCE pos_weight in BC mode. Mirrors "
                             "--bce_pos_weight_cap on the PPO side.")
    parser.add_argument("--explore_steps", type=int, default=None,
                        help="Two-phase rollout: store allowed for first N steps, frozen after")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Checkpoint directory (default: checkpoint/<wandb run name> with --use_wandb, else checkpoint/<YYYYMMDD_HHMMSS>)",
    )
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
            goal_radius=args.goal_radius,
            allow_offcell_store=args.allow_offcell_store,
        ),
        vectorhash=VectorHashConfig(
            lambdas=args.lambdas, Np=args.Np, Npos=args.Npos,
            static_vectorhash=args.static_vectorhash,
        ),
        hopfield=HopfieldConfig(
            beta=args.hopfield_beta, alpha=args.hopfield_alpha,
            steps=args.hopfield_steps, init_mode=args.hopfield_init,
            agent_can_store=args.agent_can_store,
            store_cost=args.store_cost,
            store_bonus=args.store_bonus,
            auto_store_warmup=args.auto_store_warmup,
            auto_nav_warmup=args.auto_nav_warmup,
            aux_anneal_updates=args.aux_anneal_updates,
            novelty_reward=args.novelty_reward,
            wall_penalty=args.wall_penalty,
            revisit_penalty=args.revisit_penalty,
            n_train_distractors=args.n_train_distractors,
            n_train_distractors_min=args.n_train_distractors_min,
            n_train_distractors_max=args.n_train_distractors_max,
            epsilon_explore=args.epsilon_explore,
            epsilon_anneal_updates=args.epsilon_anneal_updates,
            refresh_envs_each_update=args.refresh_envs_each_update,
        ),
        agent=AgentConfig(
            hidden_size=args.hidden_size, num_rnn_layers=args.num_rnn_layers,
            hopfield_mode=args.hopfield_mode, movement_mode=args.movement_mode,
            input_encoded_state=args.input_encoded_state,
            input_hopfield_signal=args.input_hopfield_signal,
            input_prev_action=args.input_prev_action,
            input_prev_reward=args.input_prev_reward,
            input_hopfield_raw=args.input_hopfield_raw,
            input_sensory=args.input_sensory,
            input_goal_in_memory=args.input_goal_in_memory,
            init_log_std=args.init_log_std,
            freeze_log_std=args.freeze_log_std,
        ),
        ppo=PPOConfig(
            lr=args.lr,
            store_bc_weight=args.store_bc_weight,
            ent_coef=args.ent_coef,
            store_ent_coef=args.store_ent_coef,
        ),
        bc=BCConfig(
            lr=args.bc_lr,
            store_weight=args.bc_store_weight,
            move_ent_coef=args.bc_move_ent_coef,
            epochs=args.bc_epochs,
            n_minibatches=args.bc_n_minibatches,
            bce_pos_weight_cap=args.bc_bce_pos_weight_cap,
            supervise_explore=args.bc_supervise_explore,
            novelty_fallback=args.bc_novelty_fallback,
            nav_weight=args.bc_nav_weight,
        ),
        training_mode=args.training_mode,
        encoder_checkpoint=args.encoder_checkpoint,
        encoder_gain=args.encoder_gain,
        load_checkpoint=args.load_checkpoint,
        fwhm_ratio=args.fwhm_ratio,
        num_worlds=args.num_worlds,
        envs_per_world=args.envs_per_world,
        num_val_envs=args.num_val_envs,
        n_val_trials=args.n_val_trials,
        val_n_distractors_list=args.val_distractors,
        realistic_steps_per_env=args.realistic_steps_per_env,
        hopfield_oracle=args.hopfield_oracle,
        action_oracle=args.action_oracle,
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
