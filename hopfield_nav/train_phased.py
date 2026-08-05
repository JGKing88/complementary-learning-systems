"""Phased training orchestrator for Hopfield-nav (see EXPERIMENTS_PHASE2.md).

Four phases run sequentially from a single agent init:

  Phase 1 — store pretrain: BCE(store_logits, at_goal) with trunk detached.
            Move/value heads frozen. Store head learns "fire at goal" from
            pure classification on the observable reward signal.
  Phase 2 — follow pretrain: Hopfield pre-loaded with the goal each episode.
            Store head frozen. auto_nav_warmup teacher-forces correct
            direction for the first N phase-2 updates, then PPO takes over.
  Phase 3 — explore pretrain: empty Hopfield, agent_can_store=False so memory
            stays empty. Move head learns a systematic walk from enriched
            input alone. Store head frozen.
  Phase 4 — compose: all heads unfrozen, small permanent BCE with detached
            trunk as a stability anchor, smaller LR to preserve prior phases.

Usage mirrors train.py; add --phased-config flags on top.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from .config import (
    TrainConfig, EnvConfig, VectorHashConfig, HopfieldConfig,
    AgentConfig, PPOConfig, PhasedConfig, validate_train_config,
)
from .encoder import load_encoder, validate_config
from .env import GridEnv, make_env, warn_if_offcell_stores
from .vectorhash import VectorHash
from .hopfield import Hopfield
from .agent import NavAgent, compute_input_dim
from .rollout import RolloutCollector
from .ppo import ppo_update
from .eval import (
    evaluate_navigation, evaluate_goal_discovery, evaluate_exploration,
    evaluate_union_coverage,
)


# ---------------------------------------------------------------------------
# World setup (reused across phases)
# ---------------------------------------------------------------------------

def setup_world(cfg: TrainConfig, encoder, embed_dim, rng, role: str = "train"):
    """Build envs + VectorHash scaffold. Same for train + eval worlds; role
    just controls count (envs_per_world vs num_val_envs).
    """
    n = cfg.envs_per_world if role == "train" else cfg.num_val_envs
    envs = [
        make_env(cfg.env, cfg.agent.movement_mode,
                 seed=int(rng.randint(0, 10_000_000)))
        for _ in range(n)
    ]
    vh = VectorHash(cfg.vectorhash, size=cfg.env.size)
    vh.build_scaffold()
    vh.register_envs(envs, placement="spread")
    vh.precompute_encoded_phi(encoder, cfg.fwhm_ratio, device=cfg.device)
    return {"envs": envs, "vectorhash": vh, "env_indices": list(range(n))}


def make_hops(
    role: str,
    cfg: TrainConfig,
    vh: VectorHash,
    envs: list[GridEnv],
    embed_dim: int,
    device: torch.device,
    B: int,
):
    """Build a per-env Hopfield setup for one training env.

    role:
      - "pre_stored_shared": one shared Hopfield per env preloaded with the
        goal. No agent writes (agent_can_store=False semantics). Phase 2.
      - "empty_shared": one shared empty Hopfield per env. No agent writes.
        Phase 3.
      - "empty_per_env": B empty Hopfields, agent can write to each. Phases 1
        and 4.
    Returns a list/Hopfield and a flag for whether this is per-env vs shared.
    """
    if role == "pre_stored_shared":
        per_env_templates = []
        for env_idx, env in enumerate(envs):
            hop = Hopfield(embed_dim, beta=cfg.hopfield.beta, device=str(device))
            offset = vh.env_offsets[env_idx]
            gx = min(max(env.goal_location[0] + offset[0], 0), vh.Npos - 1)
            gy = min(max(env.goal_location[1] + offset[1], 0), vh.Npos - 1)
            hop.input_memory(torch.from_numpy(vh.encoded_Phi[gx, gy]).float())
            per_env_templates.append(hop)
        return per_env_templates  # one per env; shared across the B trajectories
    if role == "empty_shared":
        return [Hopfield(embed_dim, beta=cfg.hopfield.beta, device=str(device))
                for _ in envs]
    if role == "empty_per_env":
        # For each env we build a fresh list of B Hopfields each rollout; here
        # we just return a factory.
        def factory():
            return [Hopfield(embed_dim, beta=cfg.hopfield.beta, device=str(device))
                    for _ in range(B)]
        return factory
    raise ValueError(f"unknown role: {role}")


# ---------------------------------------------------------------------------
# Freezing utilities
# ---------------------------------------------------------------------------

def _set_requires_grad(params, flag: bool):
    for p in params:
        p.requires_grad_(flag)


def _move_params(agent: NavAgent) -> list[torch.nn.Parameter]:
    if agent.cfg.movement_mode == "discrete":
        return list(agent.movement_head.parameters())
    return list(agent.movement_mean.parameters()) + [agent.movement_log_std]


def _store_params(agent: NavAgent) -> list[torch.nn.Parameter]:
    return list(agent.store_head.parameters())


def _value_params(agent: NavAgent) -> list[torch.nn.Parameter]:
    return list(agent.value_head.parameters())


def _rnn_params(agent: NavAgent) -> list[torch.nn.Parameter]:
    return list(agent.rnn.parameters())


def set_phase_freeze(agent: NavAgent, freeze_move: bool,
                     freeze_store: bool, freeze_value: bool, freeze_rnn: bool):
    _set_requires_grad(_move_params(agent), not freeze_move)
    _set_requires_grad(_store_params(agent), not freeze_store)
    _set_requires_grad(_value_params(agent), not freeze_value)
    _set_requires_grad(_rnn_params(agent), not freeze_rnn)


# ---------------------------------------------------------------------------
# Rollout collection across all train envs (one "update" worth of data)
# ---------------------------------------------------------------------------

def collect_one_update(
    cfg: TrainConfig,
    worlds: list[dict],
    agent: NavAgent,
    phase_hops,
    embed_dim: int,
    device: torch.device,
    phase_update_idx: int,
) -> list:
    """Collect a RolloutBatch from every (world, env) using the phase's
    hopfield setup. phase_hops is a dict keyed by (world_idx, env_local_idx).
    """
    rollouts = []
    for w_idx, world in enumerate(worlds):
        vh = world["vectorhash"]
        collector = RolloutCollector(vh, cfg, embed_dim, device)
        for local_idx, env in enumerate(world["envs"]):
            env_offset = vh.env_offsets[world["env_indices"][local_idx]]
            hops = phase_hops(w_idx, local_idx)
            rollout = collector.collect_rollout(
                env, agent, hops, h_rnn=None, env_offset=env_offset,
                update_idx=phase_update_idx, aux_scale=1.0,
            )
            rollouts.append(rollout)
    return rollouts


# ---------------------------------------------------------------------------
# Phase 1: store pretrain (BCE only, trunk detached)
# ---------------------------------------------------------------------------

def run_phase1(cfg: TrainConfig, pcfg: PhasedConfig, worlds, agent,
               embed_dim, device, log_prefix: str = "phase1") -> None:
    print(f"\n=== Phase 1: store pretrain ({pcfg.phase1_updates} updates) ===")
    # Freeze everything except store head. Trunk is also frozen — BCE path
    # goes through features.detach() so no gradient reaches the RNN anyway,
    # and keeping RNN frozen keeps features distribution stable for BCE.
    set_phase_freeze(agent, freeze_move=True, freeze_store=False,
                     freeze_value=True, freeze_rnn=True)
    optimizer = torch.optim.Adam(_store_params(agent), lr=pcfg.phase1_lr)

    # Force-store at goal during rollouts so the Hopfield memory distribution
    # at phase-1 rollout time matches phase-4's (agent-written) distribution.
    # phase_update_idx starts at 1; we pass auto_store_warmup > phase1_updates
    # so every phase-1 update sees the force-store behavior.
    saved_auto_store = cfg.hopfield.auto_store_warmup
    saved_auto_nav = cfg.hopfield.auto_nav_warmup
    saved_init_mode = cfg.hopfield.init_mode
    cfg.hopfield.auto_store_warmup = pcfg.phase1_updates + 1
    cfg.hopfield.auto_nav_warmup = 0
    cfg.hopfield.init_mode = "empty"

    def phase_hops(w_idx, local_idx):
        # Per-env Hopfields so auto_store_warmup writes are visible (rollout
        # skips per-env writes when shared_hopfield=True).
        return [Hopfield(embed_dim, beta=cfg.hopfield.beta, device=str(device))
                for _ in range(cfg.batch_envs)]

    for update in range(1, pcfg.phase1_updates + 1):
        rollouts = collect_one_update(cfg, worlds, agent, phase_hops,
                                      embed_dim, device, update)
        # Pool obs + labels across rollouts
        obs = torch.cat([r.obs for r in rollouts], dim=0)       # (N, T, D)
        goal = torch.cat([r.goal_reached for r in rollouts], dim=0)  # (N, T)
        mask = torch.cat([r.explore_mask for r in rollouts], dim=0)

        agent.train()
        # Re-forward to get features (RolloutBatch does not store features).
        _, _, _, _, features = agent(obs, return_features=True)
        store_logits = agent.store_logits_from(features.detach())

        masked_goal = goal * mask
        n_pos = masked_goal.sum().clamp_min(1.0)
        n_neg = (mask - masked_goal).sum().clamp_min(1.0)
        pos_weight = n_neg / n_pos
        bce = F.binary_cross_entropy_with_logits(
            store_logits, goal, reduction="none", pos_weight=pos_weight,
        )
        mask_sum = mask.sum().clamp_min(1.0)
        loss = (bce * mask).sum() / mask_sum * pcfg.phase1_bce_weight

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Diagnostic: store-head accuracy at goal vs off-goal.
        with torch.no_grad():
            probs = torch.sigmoid(store_logits)
            acc_at_goal = ((probs > 0.5).float() * masked_goal).sum() / n_pos
            fire_off_goal = (
                (probs > 0.5).float() * (mask - masked_goal)
            ).sum() / n_neg
        if cfg.use_wandb:
            import wandb
            wandb.log({
                f"{log_prefix}/bce": loss.item(),
                f"{log_prefix}/acc_at_goal": acc_at_goal.item(),
                f"{log_prefix}/fire_off_goal": fire_off_goal.item(),
                "phase_name": "Phase 1 (store pretrain)",
            })
        if update == 1 or update % 5 == 0:
            print(f"  {log_prefix} u{update}: bce={loss.item():.4f} "
                  f"acc@goal={acc_at_goal.item():.2f} "
                  f"fire@off={fire_off_goal.item():.3f}", flush=True)

    cfg.hopfield.auto_store_warmup = saved_auto_store
    cfg.hopfield.auto_nav_warmup = saved_auto_nav
    cfg.hopfield.init_mode = saved_init_mode


# ---------------------------------------------------------------------------
# Phases 2/3/4: PPO with phase-specific freezes + Hopfield setup
# ---------------------------------------------------------------------------

def run_ppo_phase(
    cfg: TrainConfig, pcfg: PhasedConfig, worlds, agent, embed_dim, device,
    phase_name: str,
    n_updates: int,
    hopfield_role: str,
    freeze_move: bool,
    freeze_store: bool,
    freeze_value: bool,
    freeze_rnn: bool,
    lr: float,
    bce_weight: float,
    bce_detach_trunk: bool,
    auto_nav_warmup: int,
    hopfield_init_mode: str,
    eval_fn=None,
) -> None:
    """Generic PPO loop used by phases 2, 3, 4."""
    print(f"\n=== {phase_name} ({n_updates} updates, hopfield={hopfield_role}) ===")
    set_phase_freeze(agent, freeze_move=freeze_move, freeze_store=freeze_store,
                     freeze_value=freeze_value, freeze_rnn=freeze_rnn)
    # Only include grad-enabled params in the optimizer so Adam state for
    # frozen params doesn't accumulate noise from numerical drift.
    trainable = [p for p in agent.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)

    # Save and override cfg flags that the rollout + PPO read.
    saved = {
        "auto_store_warmup": cfg.hopfield.auto_store_warmup,
        "auto_nav_warmup": cfg.hopfield.auto_nav_warmup,
        "init_mode": cfg.hopfield.init_mode,
        "store_bc_weight": cfg.ppo.store_bc_weight,
        "bce_detach_trunk": cfg.ppo.bce_detach_trunk,
    }
    cfg.hopfield.auto_store_warmup = 0
    cfg.hopfield.auto_nav_warmup = auto_nav_warmup
    cfg.hopfield.init_mode = hopfield_init_mode
    cfg.ppo.store_bc_weight = bce_weight
    cfg.ppo.bce_detach_trunk = bce_detach_trunk

    # Pre-build hopfields for "shared" roles (constant across updates).
    shared_per_env = None
    if hopfield_role in ("pre_stored_shared", "empty_shared"):
        shared_per_env = {}
        for w_idx, world in enumerate(worlds):
            shared_per_env[w_idx] = make_hops(
                hopfield_role, cfg, world["vectorhash"], world["envs"],
                embed_dim, device, cfg.batch_envs,
            )

    def phase_hops(w_idx, local_idx):
        if shared_per_env is not None:
            return shared_per_env[w_idx][local_idx]  # single Hopfield, shared across B
        # empty_per_env: fresh list of B per rollout
        return [Hopfield(embed_dim, beta=cfg.hopfield.beta, device=str(device))
                for _ in range(cfg.batch_envs)]

    for update in range(1, n_updates + 1):
        rollouts = collect_one_update(cfg, worlds, agent, phase_hops,
                                      embed_dim, device, update)
        agent.train()
        losses = ppo_update(agent, rollouts, cfg.ppo, optimizer, aux_scale=1.0)

        mean_r = sum(r.rewards.sum().item() for r in rollouts) / max(
            sum(r.rewards.numel() for r in rollouts), 1)

        if cfg.use_wandb:
            import wandb
            log = {f"train/{k}": v for k, v in losses.items()}
            log["train/mean_reward"] = mean_r
            log["phase_name"] = phase_name
            wandb.log(log)

        if update == 1 or update % 10 == 0:
            loss_str = " ".join(f"{k}={v:.3f}" for k, v in losses.items())
            print(f"  {phase_name} u{update}: mean_r={mean_r:.4f} | {loss_str}",
                  flush=True)

        if eval_fn is not None and update % max(cfg.eval_every, 1) == 0:
            eval_fn(update)

    for k, v in saved.items():
        if k in ("auto_store_warmup", "auto_nav_warmup", "init_mode"):
            setattr(cfg.hopfield, k, v)
        else:
            setattr(cfg.ppo, k, v)


# ---------------------------------------------------------------------------
# Eval wrapper used at phase boundaries
# ---------------------------------------------------------------------------

def do_eval(cfg, agent, eval_world, device, update_tag: str,
            use_wandb: bool, max_steps: int = 200) -> None:
    val_envs = eval_world["envs"]
    val_vh = eval_world["vectorhash"]
    val_idxs = eval_world["env_indices"]
    dist = cfg.val_n_distractors_list
    nt = cfg.n_val_trials

    nav = evaluate_navigation(agent, val_envs, val_vh, val_idxs, cfg, device,
                              num_trials=nt, max_steps=max_steps,
                              n_distractors_list=dist, deterministic=True)
    disc = evaluate_goal_discovery(agent, val_envs, val_vh, val_idxs, cfg,
                                   device, num_trials=nt, max_steps=max_steps,
                                   n_distractors_list=dist)
    expl = evaluate_exploration(agent, val_envs, val_vh, val_idxs, cfg, device,
                                num_trials=nt, max_steps=max_steps,
                                n_distractors_list=dist)
    print(f"  [{update_tag}] nav={nav}")
    print(f"  [{update_tag}] disc={disc}")
    print(f"  [{update_tag}] expl={expl}")
    union_cov = None
    union_trials = getattr(cfg, "union_cov_trials", 0)
    if union_trials > 0:
        union_cov = evaluate_union_coverage(
            agent, val_envs, val_vh, val_idxs, cfg, device,
            num_trials=union_trials, max_steps=max_steps,
            n_distractors_list=dist, deterministic=True,
        )
        print(f"  [{update_tag}] union_cov={union_cov}")
    if use_wandb:
        import wandb
        log = {}
        for n_d in dist:
            for k, v in nav[n_d].items(): log[f"eval/nav_{n_d}/{k}"] = v
            for k, v in disc[n_d].items(): log[f"eval/disc_{n_d}/{k}"] = v
            for k, v in expl[n_d].items(): log[f"eval/expl_{n_d}/{k}"] = v
            if union_cov is not None:
                for k, v in union_cov[n_d].items(): log[f"eval/union_cov_{n_d}/{k}"] = v
        log["phase_tag"] = update_tag
        wandb.log(log)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train_phased(cfg: TrainConfig, pcfg: PhasedConfig) -> None:
    validate_train_config(cfg)
    warn_if_offcell_stores(cfg.env, where="train_phased")
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
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

    worlds = [setup_world(cfg, encoder, embed_dim, rng, role="train")
              for _ in range(cfg.num_worlds)]
    eval_world = setup_world(cfg, encoder, embed_dim, rng, role="eval")

    input_dim = compute_input_dim(cfg.agent, embed_dim, cfg.env.observation_size)
    print(f"Agent input_dim={input_dim} (enrichment: "
          f"prev_reward={cfg.agent.input_prev_reward}, "
          f"prev_action={cfg.agent.input_prev_action}, "
          f"hopfield_raw={cfg.agent.input_hopfield_raw}, "
          f"sensory={cfg.agent.input_sensory}@obs={cfg.env.observation_size})")
    agent = NavAgent(cfg.agent, input_dim).to(device)

    if cfg.use_wandb:
        import wandb
        full_cfg = {**asdict(cfg), "phased": asdict(pcfg)}
        wandb.init(project=cfg.wandb_project, config=full_cfg)

    if cfg.save_dir is None:
        sub = (wandb.run.name or wandb.run.id) if cfg.use_wandb else \
              datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg.save_dir = os.path.join("checkpoint", f"phased_{sub}")

    # ----- Phase 1 -----
    run_phase1(cfg, pcfg, worlds, agent, embed_dim, device)
    do_eval(cfg, agent, eval_world, device, "after_phase1", cfg.use_wandb)

    # ----- Phase 2 -----
    run_ppo_phase(
        cfg, pcfg, worlds, agent, embed_dim, device,
        phase_name="Phase 2 (follow pretrain)",
        n_updates=pcfg.phase2_updates,
        hopfield_role="pre_stored_shared",
        freeze_move=False, freeze_store=pcfg.phase2_freeze_store,
        freeze_value=False, freeze_rnn=False,
        lr=cfg.ppo.lr,
        bce_weight=0.0, bce_detach_trunk=False,
        auto_nav_warmup=pcfg.phase2_auto_nav_warmup,
        hopfield_init_mode="pre_stored",
        eval_fn=lambda u: do_eval(cfg, agent, eval_world, device,
                                  f"phase2_u{u}", cfg.use_wandb),
    )
    do_eval(cfg, agent, eval_world, device, "after_phase2", cfg.use_wandb)

    # ----- Phase 3 -----
    # Warm-start: override continuous policy std now that follow is trained.
    if cfg.agent.movement_mode == "continuous":
        with torch.no_grad():
            agent.movement_log_std.fill_(pcfg.phase3_init_log_std)
    run_ppo_phase(
        cfg, pcfg, worlds, agent, embed_dim, device,
        phase_name="Phase 3 (explore pretrain)",
        n_updates=pcfg.phase3_updates,
        hopfield_role="empty_shared",
        freeze_move=False, freeze_store=pcfg.phase3_freeze_store,
        freeze_value=False, freeze_rnn=False,
        lr=cfg.ppo.lr,
        bce_weight=0.0, bce_detach_trunk=False,
        auto_nav_warmup=0,
        hopfield_init_mode="empty",
        eval_fn=lambda u: do_eval(cfg, agent, eval_world, device,
                                  f"phase3_u{u}", cfg.use_wandb),
    )
    do_eval(cfg, agent, eval_world, device, "after_phase3", cfg.use_wandb)

    # ----- Phase 4 -----
    run_ppo_phase(
        cfg, pcfg, worlds, agent, embed_dim, device,
        phase_name="Phase 4 (compose)",
        n_updates=pcfg.phase4_updates,
        hopfield_role="empty_per_env",
        freeze_move=False, freeze_store=False,
        freeze_value=False, freeze_rnn=False,
        lr=pcfg.phase4_lr,
        bce_weight=pcfg.phase4_bce_weight,
        bce_detach_trunk=pcfg.phase4_bce_detach_trunk,
        auto_nav_warmup=0,
        hopfield_init_mode="empty",
        eval_fn=lambda u: do_eval(cfg, agent, eval_world, device,
                                  f"phase4_u{u}", cfg.use_wandb),
    )
    do_eval(cfg, agent, eval_world, device, "after_phase4", cfg.use_wandb)

    os.makedirs(cfg.save_dir, exist_ok=True)
    torch.save({
        "agent_state_dict": agent.state_dict(),
        "config": asdict(cfg),
        "phased_config": asdict(pcfg),
    }, os.path.join(cfg.save_dir, "phased_final.pt"))
    print(f"\nDone. Saved to {cfg.save_dir}/phased_final.pt")

    if cfg.use_wandb:
        import wandb
        wandb.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Phased Hopfield-nav training")
    # Reuse the big knobs from train.py; only the most relevant ones exposed
    # to keep this sketch small. Extend as needed.
    p.add_argument("--encoder_checkpoint", required=True)
    p.add_argument("--size", type=int, default=8)
    p.add_argument("--observation_size", type=int, default=12,
                   help="Phase 2 default: 12 rays across 120° cone.")
    p.add_argument("--movement_mode", default="continuous",
                   choices=["discrete", "continuous"])
    p.add_argument("--goal_radius", type=float, default=0.5,
                   help="Euclidean radius around goal that counts as 'at goal'. "
                        "Default 0.5 reproduces snap-equality on integer-snapped "
                        "positions; larger values fuzz the goal region.")
    p.add_argument("--allow_offcell_store",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Whether a store fired while at goal may write a cell other than the goal's. Only reachable at goal_radius > 0.5, where at_goal tests the float position but embeddings are read at the snapped cell. True (default) preserves the behavior of every run through 2026-08; --no-allow_offcell_store stores the goal cell's embedding instead.")
    p.add_argument("--hopfield_mode", default="continuous",
                   choices=["discrete", "continuous"])
    # Enrichment flags (phase 2 defaults: all ON)
    p.add_argument("--input_prev_reward", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--input_prev_action", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--input_hopfield_raw", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--input_sensory", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--input_encoded_state", action=argparse.BooleanOptionalAction,
                   default=False)
    p.add_argument("--input_hopfield_signal", action=argparse.BooleanOptionalAction,
                   default=True)
    # Phase durations
    p.add_argument("--phase1_updates", type=int, default=20)
    p.add_argument("--phase2_updates", type=int, default=100)
    p.add_argument("--phase3_updates", type=int, default=200)
    p.add_argument("--phase4_updates", type=int, default=300)
    p.add_argument("--phase2_auto_nav_warmup", type=int, default=30)
    p.add_argument("--phase3_init_log_std", type=float, default=-0.5)
    p.add_argument("--phase4_lr", type=float, default=1e-4)
    p.add_argument("--phase4_bce_weight", type=float, default=0.1)
    # Shared
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch_envs", type=int, default=16)
    p.add_argument("--steps_per_rollout", type=int, default=64)
    p.add_argument("--num_worlds", type=int, default=1)
    p.add_argument("--envs_per_world", type=int, default=4)
    p.add_argument("--num_val_envs", type=int, default=2)
    p.add_argument("--n_val_trials", type=int, default=32)
    p.add_argument("--val_distractors", type=int, nargs="+", default=[0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="hopfield-nav-phased")
    # Encoder / vectorhash
    p.add_argument("--lambdas", type=int, nargs="+", default=[11, 12, 13])
    p.add_argument("--Np", type=int, default=1600)
    p.add_argument("--static-vectorhash", dest="static_vectorhash",
                   action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fwhm_ratio", type=float, default=0.25)
    p.add_argument("--encoder_gain", type=float, default=None)

    args = p.parse_args()

    cfg = TrainConfig(
        env=EnvConfig(size=args.size, observation_size=args.observation_size,
                      movement_mode=args.movement_mode,
                      goal_radius=args.goal_radius,
                      allow_offcell_store=args.allow_offcell_store),
        vectorhash=VectorHashConfig(lambdas=args.lambdas, Np=args.Np,
                                    static_vectorhash=args.static_vectorhash),
        hopfield=HopfieldConfig(),
        agent=AgentConfig(
            hopfield_mode=args.hopfield_mode, movement_mode=args.movement_mode,
            input_encoded_state=args.input_encoded_state,
            input_hopfield_signal=args.input_hopfield_signal,
            input_prev_action=args.input_prev_action,
            input_prev_reward=args.input_prev_reward,
            input_hopfield_raw=args.input_hopfield_raw,
            input_sensory=args.input_sensory,
        ),
        ppo=PPOConfig(lr=args.lr),
        encoder_checkpoint=args.encoder_checkpoint,
        encoder_gain=args.encoder_gain,
        fwhm_ratio=args.fwhm_ratio,
        num_worlds=args.num_worlds, envs_per_world=args.envs_per_world,
        num_val_envs=args.num_val_envs, n_val_trials=args.n_val_trials,
        val_n_distractors_list=args.val_distractors,
        batch_envs=args.batch_envs, steps_per_rollout=args.steps_per_rollout,
        eval_every=args.eval_every, save_dir=args.save_dir,
        seed=args.seed, device=args.device,
        use_wandb=args.use_wandb, wandb_project=args.wandb_project,
    )
    pcfg = PhasedConfig(
        phase1_updates=args.phase1_updates,
        phase2_updates=args.phase2_updates,
        phase3_updates=args.phase3_updates,
        phase4_updates=args.phase4_updates,
        phase2_auto_nav_warmup=args.phase2_auto_nav_warmup,
        phase3_init_log_std=args.phase3_init_log_std,
        phase4_lr=args.phase4_lr,
        phase4_bce_weight=args.phase4_bce_weight,
    )
    train_phased(cfg, pcfg)


if __name__ == "__main__":
    main()
