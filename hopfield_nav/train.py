"""Main training script for Hopfield navigation.

Usage:
    python -m hopfield_nav.train --encoder_checkpoint encoders/run/encoder_final.pt [...]
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict

import numpy as np
import torch

from cls_paths import run_dir, run_name
import run_manifest
from .config import (
    TrainConfig, EnvConfig, VectorHashConfig, HopfieldConfig,
    AgentConfig, PPOConfig, BCConfig, validate_train_config,
)
from .encoder_io import load_encoder, validate_config
from .world.env import warn_if_offcell_stores
from .world.scaffold import goal_encodings, place_envs
from hopfield import Hopfield
from .policy.agent import NavAgent, compute_input_dim
from .policy.recurrent import add_recurrent_args
from .rollout.collector import RolloutCollector
from .rollout.distractors import sample_distractors
from .updates.ppo import ppo_update
from .updates.bc import bc_update
from .evaluation.metrics import (
    evaluate_navigation, evaluate_goal_discovery, evaluate_exploration,
    evaluate_realistic,
)
from .evaluation.checkpoint_io import cfg_from_checkpoint
from .training.cfg_args import explicit_dests, overlay_typed
from .training.refresh import Cadence
from .training.world_setup import build_field, setup_run_world


# ---------------------------------------------------------------------------
# World setup
# ---------------------------------------------------------------------------

def build_templates(cfg: TrainConfig, worlds, embed_dim: int) -> list:
    """One pre-stored goal Hopfield per world, or ``None`` per world.

    Rebuilt after **every** refresh tick, not only at startup: the template
    holds the goal *encodings*, which are a function of both the env-local goal
    cell and the env's offset. A place, goal, wall or size tick changes one of
    those, so a template carried over would have the agent begin each rollout
    already remembering a goal that has moved -- a false memory, indistinguish-
    able from a real one and silent in every metric.
    """
    out = []
    for w_idx, world in enumerate(worlds):
        if cfg.hopfield.init_mode != "pre_stored":
            out.append(None)
            continue
        hop = Hopfield(embed_dim, beta=cfg.hopfield.beta, device=cfg.device)
        for pattern in goal_encodings(world.field, world.envs, world.offsets):
            hop.input_memory(torch.from_numpy(pattern).float())
        print(f"  train world {w_idx}: pre-stored {hop.num_memories} "
              "goal patterns")
        out.append(hop)
    return out


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig) -> None:
    validate_train_config(cfg)
    warn_if_offcell_stores(cfg.env, where="train")
    # Before the encoder loads and the scaffold builds: a refresh cadence with
    # no generator to draw from, or the two refresh mechanisms asked for at
    # once, is worth knowing now rather than twenty minutes in.
    cadence = Cadence.from_config(cfg)
    if cadence and cfg.hopfield.refresh_envs_each_update:
        raise SystemExit(
            "  ERROR: --refresh_envs_each_update and the per-trait refresh "
            "flags do the same job, and the old one undoes what the new one "
            "guarantees. It re-places train envs with placement='random' over "
            "the whole scaffold, so a refreshed env can land on the validation "
            "region, and it never touches split.train -- so world.json would "
            "describe envs the run had stopped using. Use --refresh_place / "
            "--refresh_goal, which draw from the declared train domain clear "
            "of the fixed val set, and record every draw.")
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

    # One scaffold field, shared by every train world and the eval world. It
    # is a pure function of (lambdas, Npos, fwhm_ratio, encoder), so the
    # per-world copies this used to build were bit-identical.
    print("Building scaffold field")
    field = build_field(cfg, encoder)

    # The world, its record and the refresher, through the same helper
    # `train_navigate` and `train_store` use -- so all three agree about what a
    # run writes down and none of them can record a world it is not training on.
    encoder_ident = run_manifest.encoder_identity(
        cfg.encoder_checkpoint, enc_cfg, encoder_gain)
    rw = setup_run_world(cfg, encoder, embed_dim, rng, field,
                         cadence=cadence, n_updates=cfg.n_updates,
                         encoder_ident=encoder_ident, where="train",
                         parent_ckpt=cfg.load_checkpoint)
    worlds, eval_world = rw.worlds, rw.eval_world
    templates = build_templates(cfg, worlds, embed_dim)

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
        sub = run_name(*((wandb.run.name, wandb.run.id) if cfg.use_wandb else ()))
        cfg.save_dir = str(run_dir("train", sub))
    else:
        sub = os.path.basename(str(cfg.save_dir).rstrip("/"))

    run_manifest.begin(
        cfg.save_dir, kind="train", name=sub, config=asdict(cfg),
        encoder=encoder_ident,
        parent=cfg.load_checkpoint,
        wandb_run=wandb.run if cfg.use_wandb else None,
    )

    # Written on both paths: a run has to be able to say which envs it used,
    # and the historical path could not (docs/EVAL_SPLITS_DESIGN.md 1.4).
    ckpt_world = rw.record()

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
                if not world.field.cfg.static_vectorhash:
                    raise RuntimeError(
                        "refresh_envs_each_update requires --static-vectorhash"
                    )
                for env in world.envs:
                    env.reset_goal()
                world.offsets = place_envs(
                    len(world.envs), cfg.env.size, world.field.Npos,
                    np.random, placement="random")
        # The per-trait refresh: draws from the declared train domain, clear of
        # the fixed val envs, and records every draw into `split.used`. Mutually
        # exclusive with the block above -- see the check in `train`.
        if rw.refresh(update):
            # Any tick moves a goal or the footprint holding it, so a template
            # built from the old encodings is now a false memory.
            templates = build_templates(cfg, worlds, embed_dim)

        all_rollouts = []
        for w_idx, world in enumerate(worlds):
            vh = world.field
            template_hop = templates[w_idx]
            collector = RolloutCollector(vh, cfg, embed_dim, device)

            for local_idx, env in enumerate(world.envs):
                env_offset = world.offsets[local_idx]

                # Fresh Hopfield instances per env: no cross-env contamination.
                if cfg.hopfield.allow_store:
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
                # when we have a per-env hops list (allow_store=True).
                use_variable = cfg.hopfield.n_train_distractors_max > 0
                use_fixed = cfg.hopfield.n_train_distractors > 0
                if (use_variable or use_fixed) and isinstance(hops, list):
                    env_size = cfg.env.size
                    lo = cfg.hopfield.n_train_distractors_min
                    hi = cfg.hopfield.n_train_distractors_max
                    for b in range(B):
                        if use_variable:
                            n_d = int(dist_rng.randint(lo, hi + 1))
                        else:
                            n_d = cfg.hopfield.n_train_distractors
                        for pat in sample_distractors(
                                vh, env_offset, env_size, n_d, dist_rng):
                            hops[b].input_memory(torch.from_numpy(pat).float())

                rollout = collector.collect_rollout(
                    env, agent, hops, allow_store=cfg.hopfield.allow_store, h_rnn=None, env_offset=env_offset,
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
        if cfg.eval_every > 0 and update % cfg.eval_every == 0 and eval_world.envs:
            val_envs = eval_world.envs
            val_vh = eval_world.field
            val_idxs = eval_world.offsets
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

            print(f"  eval nav_det: {nav_det}")
            print(f"  eval nav_stoch: {nav_stoch}")
            print(f"  eval disc: {disc}")
            print(f"  eval expl: {expl}")

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
                    # `union_coverage` and `union_per_rollout` are keys of
                    # `expl` and are logged by the loop above. A separate
                    # `union` evaluator was absorbed into evaluate_exploration
                    # on 2026-08-06 and this line outlived it, so every
                    # `--use_wandb` run died with NameError at its first eval.
                    for k, v in expl[n_dist].items():
                        log[f"eval/expl_{n_dist}/{k}"] = v
                wandb.log(log, step=update)

        # Save
        if cfg.save_every > 0 and update % cfg.save_every == 0:
            os.makedirs(cfg.save_dir, exist_ok=True)
            # A refreshing run's record grows with it: rewrite before saving so
            # the checkpoint names the file as it stands.
            if rw.refresher is not None:
                ckpt_world = rw.record()
            torch.save({
                "agent_state_dict": agent.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": asdict(cfg),
                "update": update,
                "world_spec": ckpt_world,
            }, os.path.join(cfg.save_dir, f"hopfield_nav_update{update}.pt"))
            run_manifest.record_checkpoint(
                cfg.save_dir, f"hopfield_nav_update{update}.pt", update)

    print("Training complete.")

    # The union of everything training touched, as it ends up. `world.json` is
    # rewritten on the checkpoint cadence, so without this the file would stop
    # at the last saved update rather than at the last refresh tick.
    if rw.refresher is not None:
        rw.record()

    # Realistic end-of-training eval: one persistent Hopfield accumulating
    # memories across envs visited sequentially; retest prior envs after each
    # new visit with storing disabled to measure interference.
    if cfg.realistic_steps_per_env > 0 and eval_world.envs:
        print(f"Running realistic eval ({cfg.realistic_steps_per_env} steps/env, "
              f"{len(eval_world.envs)} envs)")
        realistic = evaluate_realistic(
            agent, eval_world.envs, eval_world.field,
            eval_world.offsets, cfg, device,
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

    run_manifest.finish(cfg.save_dir)

    if cfg.use_wandb:
        import wandb
        wandb.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_args():
    # allow_abbrev=False: `explicit_dests` matches option strings
    # literally, so an abbreviation would reach the Namespace while going
    # unmatched there -- and then lose silently to the inherited value.
    parser = argparse.ArgumentParser(
        description="Train Hopfield navigation agent", allow_abbrev=False)
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
    parser.add_argument("--reset_state_on_teleport",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="Zero the RNN hidden state and prev_reward / prev_action "
                         "when the agent teleports after reaching the goal (C5 of the "
                         "at-goal contract, world/episode.py). Default off since "
                         "2026-08-12: recurrence spans the whole rollout rather than "
                         "restarting at each goal. Applies to training and evaluation "
                         "together -- an answer that differed between them would make "
                         "the two incomparable.")
    parser.add_argument("--allow_offcell_store",
                   action=argparse.BooleanOptionalAction, default=False,
                   help="Whether a store fired while at goal may write a cell other than the goal's. Only reachable at goal_radius > 0.5, where at_goal tests the float position but embeddings are read at the snapped cell. Default False: the goal cell's embedding is stored instead, so the pattern written is the one navigation will later recall. Pass --allow_offcell_store for the pre-2026-08 behavior.")
    parser.add_argument("--wall_resolution", type=int, default=1,
                   help="How many +/-1 wall segments span one grid cell. 1 (default) is one segment per cell, the original coarse barcode. Above 1 a stripe edge can fall inside a cell, which is the only way a ray can report where within a cell it is looking from; at 1 roughly 9-14%% of cells share a bit-identical observation with another cell. 8 drives that to ~0. Changes env identity, so splits and checkpoints are tied to it.")
    parser.add_argument("--egocentric_heading",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Foveal cone turns with the agent: heading is a continuous angle following the direction it actually moved, so a cell looks different depending on how the agent arrived. Sensory input is the only thing heading affects. Pass --no-egocentric_heading to pin every view to North, reproducing pre-2026-08 runs.")
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
    parser.add_argument("--allow_store", action=argparse.BooleanOptionalAction, default=True,
                        help="May the agent's store action write to the Hopfield. "
                             "Named --agent_can_store until 2026-08.")
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
    parser.add_argument("--env_generator", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Draw envs from declared domains (world/generate.py) "
                             "instead of the historical placement path. Off keeps "
                             "today's envs for a given --seed; on fixes the "
                             "offset-reproducibility bug and enforces train/val "
                             "separation. world.json is written either way.")
    parser.add_argument("--place_region", type=str, default="anywhere",
                        help="Where train envs may sit: 'anywhere' or "
                             "'rect:X0,Y0,W,H'. Declaring a rect is what makes "
                             "a place-OOD val set possible later -- its complement.")
    parser.add_argument("--goal_region", type=str, default="any",
                        help="Which env-local cells may hold a goal: 'any', "
                             "'ring:W', 'interior:W' or 'quadrant:Q'.")
    parser.add_argument("--wall_seeds", type=str, default="0,10000000",
                        help="'LO,HI' range training draws wall seeds from.")
    parser.add_argument("--place_margin", type=int, default=None,
                        help="Edge-to-edge train/val clearance in cells. Default "
                             "derives it from the scaffold's own cosine curve.")
    parser.add_argument("--goal_val_frac", type=float, default=0.2,
                        help="Share of goal cells reserved for validation.")
    # Per-trait refresh. All require --env_generator, and all apply to the train
    # set only -- a validation set that moved under the model would make every
    # in-training curve unreadable. These supersede --refresh_envs_each_update,
    # which re-placed randomly over the whole scaffold and recorded nothing.
    parser.add_argument("--refresh_place", type=int, default=None,
                        help="Re-draw train env placements every N updates, from "
                             "--place_region and clear of the fixed val envs by "
                             "the margin. Requires --env_generator.")
    parser.add_argument("--refresh_wall", type=int, default=None,
                        help="Re-draw train wall seeds every N updates, excluding "
                             "every seed the run has already used. Rebuilds the "
                             "envs, so it is the one expensive cadence.")
    parser.add_argument("--refresh_goal", type=int, default=None,
                        help="Re-draw train goals every N updates from the train "
                             "share of --goal_region. Also caps the train goal "
                             "cells at 1 - --goal_val_frac of the region up front.")
    parser.add_argument("--refresh_size", type=int, default=None,
                        help="Re-draw the train env size every N updates. Needs "
                             "more than one declared size; nothing produces that "
                             "yet, so this raises.")
    parser.add_argument("--refresh_envs_each_update", action="store_true",
                        help="Re-sample env_offsets at the start of every PPO update so each rollout buffer covers a different patch of the global scaffold. Reduces seed-lottery on scaffold position. PPO-correct because the buffer comes from a single policy.")
    # Agent
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_rnn_layers", type=int, default=1)
    add_recurrent_args(parser)
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
        help="Checkpoint directory. Default: <CLS_RUNS>/agent_ckpts/<wandb run "
             "name> with --use_wandb, else <CLS_RUNS>/agent_ckpts/<YYYYMMDD_HHMMSS>.",
    )
    # Checkpoint loading
    parser.add_argument("--load_checkpoint", type=str, default=None,
                        help="Path to agent checkpoint to load (for curriculum / fine-tuning)")
    # Wandb
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="hopfield-nav")

    args = parser.parse_args()
    return parser, args


def config_from_args(args) -> TrainConfig:
    """The config a command line asks for, with nothing inherited.

    Pure keyword passing, by design: `overlay_typed` calls this a second time
    with sentinels standing in for untyped flags, and a sentinel has to survive
    the trip to say which config field each flag reaches.
    """
    return TrainConfig(
        env=EnvConfig(
            size=args.size, observation_size=args.observation_size,
            time_penalty=args.time_penalty, movement_mode=args.movement_mode,
            goal_radius=args.goal_radius,
            allow_offcell_store=args.allow_offcell_store,
            egocentric_heading=args.egocentric_heading,
            wall_resolution=args.wall_resolution,
            reset_state_on_teleport=args.reset_state_on_teleport,
        ),
        vectorhash=VectorHashConfig(
            lambdas=args.lambdas, Np=args.Np, Npos=args.Npos,
            static_vectorhash=args.static_vectorhash,
        ),
        hopfield=HopfieldConfig(
            beta=args.hopfield_beta, alpha=args.hopfield_alpha,
            steps=args.hopfield_steps, init_mode=args.hopfield_init,
            allow_store=args.allow_store,
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
            rnn_cell=args.rnn_cell,
            rnn_nonlinearity=args.rnn_nonlinearity,
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
        env_generator=args.env_generator,
        place_region=args.place_region,
        goal_region=args.goal_region,
        wall_seeds=args.wall_seeds,
        place_margin=args.place_margin,
        goal_val_frac=args.goal_val_frac,
        refresh_place=args.refresh_place,
        refresh_wall=args.refresh_wall,
        refresh_goal=args.refresh_goal,
        refresh_size=args.refresh_size,
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


def main():
    parser, args = build_args()
    if args.load_checkpoint:
        # The parent's config is the base, and only the flags actually typed
        # override it. Building from argv alone -- what this did until now --
        # silently substituted train.py's dataclass defaults for the parent's
        # architecture: 8 of 17 agent fields differ, movement_mode among them,
        # which is a different policy head rather than a width mismatch. It
        # failed loudly at load_state_dict, but only after the scaffold built.
        ck = torch.load(args.load_checkpoint, map_location="cpu",
                        weights_only=False)
        cfg = cfg_from_checkpoint(ck["config"])
        overlay_typed(cfg, args, explicit_dests(parser, sys.argv[1:]),
                      config_from_args)
        # Never inherited: that field holds where the parent wrote, and reusing
        # it would have this run overwrite its own parent.
        cfg.save_dir = args.save_dir
        cfg.load_checkpoint = args.load_checkpoint
    else:
        cfg = config_from_args(args)
    train(cfg)


if __name__ == "__main__":
    main()
