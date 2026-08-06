"""Phase-A-only experiment harness — fast iteration on the explore/follow
balance without paying for Phase B+C.

Skips Phase B (store pretrain) and Phase C (compose). After Phase A finishes,
runs a single after_phase_a eval and exits. Total runtime ~1-2h per seed
instead of 5-7h. Designed for sweeping novelty/interleave configurations to
find a setting that consistently produces coverage ≥ 0.55.

Adds two knobs vs train_phased_v3.run_phase_a:
  - ``warmup_explore_only_updates``: if >0, run that many updates of
    100%-empty rollouts with novelty before starting interleaved Phase A.
  - ``novelty_anneal``: linearly anneal phase_a_novelty_reward from full
    to 0 over the entire Phase A budget (warmup + interleaved).
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
    AgentConfig, PPOConfig, PhasedConfigV3, validate_train_config,
)
from .encoder_io import load_encoder, validate_config
from .world.env import warn_if_offcell_stores
from .world.scaffold import VectorHash
from hopfield import Hopfield
from .policy.agent import NavAgent, compute_input_dim
from .rollout.collector import RolloutCollector
from .rollout.distractors import sample_distractors
from .updates.ppo import ppo_update
from .training.world_setup import (
    do_eval, make_hops, set_phase_freeze, setup_world,
)


def _compute_epsilon(update: int, total: int, base: float, anneal: int) -> float:
    """Linear anneal of ε from `base` → 0 over `anneal` updates (0 = constant).

    `update` is 1-indexed. Returns 0.0 when base==0.
    """
    if base <= 0:
        return 0.0
    if anneal <= 0:
        return base
    scale = max(0.0, 1.0 - (update - 1) / float(anneal))
    return base * scale


def run_phase_a_sweep(
    cfg: TrainConfig,
    pcfg: PhasedConfigV3,
    worlds,
    agent,
    embed_dim,
    device,
    use_wandb: bool,
    eval_world,
    eval_every: int,
    warmup_explore_only_updates: int,
    novelty_anneal: bool,
    interleave_empty_fraction: float,
    randomize_goal_per_rollout: bool = False,
    epsilon_explore: float = 0.0,
    epsilon_anneal_updates: int = 0,
    interleave_empty_target: float | None = None,
    interleave_anneal_updates: int = 0,
    explore_goals_off: bool = False,
    n_train_distractors_min: int = 0,
    n_train_distractors_max: int = 0,
    n_train_emp_distractors_min: int = 0,
    n_train_emp_distractors_max: int = 0,
    n_train_distractors_max_end: int | None = None,
    n_train_emp_distractors_max_end: int | None = None,
    distractor_curriculum_updates: int = 0,
    log_std_anneal_start_update: int = 0,
    log_std_anneal_end_update: int = 0,
    log_std_anneal_target: float | None = None,
    dist_rng: np.random.RandomState | None = None,
) -> None:
    """Phase A with optional pure-explore warmup and novelty annealing.

    interleave_empty_fraction: 0.5 = half envs empty, half pre_stored. 1.0 =
    all empty (pure explore). 0.0 = all pre_stored (pure follow).

    epsilon_explore: per-step probability of replacing the sampled action
    with a uniform-random direction. Linearly annealed to 0 over
    `epsilon_anneal_updates` updates (0 = constant). Applied only to
    empty-regime envs to keep follow gradient clean.

    interleave_empty_target / interleave_anneal_updates: optional curriculum
    on interleave fraction. If `interleave_empty_target` is set, the
    fraction linearly anneals from `interleave_empty_fraction` (initial)
    to `interleave_empty_target` (final) over `interleave_anneal_updates`
    updates, then holds.
    """
    n_updates_total = warmup_explore_only_updates + pcfg.phase_a_updates
    print(f"\n=== Phase A sweep: warmup={warmup_explore_only_updates}, "
          f"interleave_empty={interleave_empty_fraction}, "
          f"novelty={pcfg.phase_a_novelty_reward}, "
          f"anneal={novelty_anneal}, total {n_updates_total} updates ===",
          flush=True)

    set_phase_freeze(agent, freeze_move=False, freeze_store=True,
                     freeze_value=False, freeze_rnn=False)
    trainable = [p for p in agent.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=pcfg.phase_a_lr)

    pre_pools = {}
    emp_pools = {}
    for w_idx, world in enumerate(worlds):
        vh = world["vectorhash"]
        envs = world["envs"]
        pre_pools[w_idx] = make_hops(
            "pre_stored_shared", cfg, vh, envs, embed_dim, device, cfg.batch_envs,
        )
        emp_pools[w_idx] = make_hops(
            "empty_shared", cfg, vh, envs, embed_dim, device, cfg.batch_envs,
        )

    use_distractors = n_train_distractors_max > 0
    use_emp_distractors = n_train_emp_distractors_max > 0
    if use_distractors or use_emp_distractors:
        if dist_rng is None:
            dist_rng = np.random.RandomState(cfg.seed + 7919)
        if use_distractors:
            print(f"Phase A train distractors per pre_stored rollout: "
                  f"~U[{n_train_distractors_min}, {n_train_distractors_max}]",
                  flush=True)
        if use_emp_distractors:
            print(f"Phase A train distractors per empty rollout: "
                  f"~U[{n_train_emp_distractors_min}, {n_train_emp_distractors_max}] "
                  f"(no goal pattern)", flush=True)

    n_envs = cfg.envs_per_world
    # Interleave fraction may anneal over training. Initial value used for
    # update 1; final value reached at u=interleave_anneal_updates.
    interleave_start = interleave_empty_fraction
    interleave_end = (interleave_empty_target if interleave_empty_target is not None
                      else interleave_empty_fraction)

    saved = {
        "auto_nav_warmup": cfg.hopfield.auto_nav_warmup,
        "auto_store_warmup": cfg.hopfield.auto_store_warmup,
        "store_bc_weight": cfg.ppo.store_bc_weight,
        "bce_detach_trunk": cfg.ppo.bce_detach_trunk,
        "novelty_reward": cfg.hopfield.novelty_reward,
    }
    cfg.hopfield.auto_nav_warmup = 0
    cfg.hopfield.auto_store_warmup = 0
    cfg.ppo.store_bc_weight = 0.0
    cfg.ppo.bce_detach_trunk = False

    base_novelty = pcfg.phase_a_novelty_reward

    log_std_anneal_active = (
        log_std_anneal_target is not None
        and log_std_anneal_end_update > log_std_anneal_start_update
    )
    log_std_init_val = float(agent.movement_log_std.detach().mean().item()) \
        if hasattr(agent, "movement_log_std") else None

    for update in range(1, n_updates_total + 1):
        # log_std anneal: programmatically interpolate the parameter from its
        # init value to the target value across [start, end] update window.
        if log_std_anneal_active and log_std_init_val is not None:
            if update >= log_std_anneal_end_update:
                t_ls = 1.0
            elif update <= log_std_anneal_start_update:
                t_ls = 0.0
            else:
                t_ls = (update - log_std_anneal_start_update) / float(
                    log_std_anneal_end_update - log_std_anneal_start_update)
            new_log_std = log_std_init_val + t_ls * (
                log_std_anneal_target - log_std_init_val)
            with torch.no_grad():
                agent.movement_log_std.data.fill_(new_log_std)

        # Anneal novelty if requested.
        if novelty_anneal:
            scale = max(0.0, 1.0 - (update - 1) / max(n_updates_total, 1))
            current_novelty = base_novelty * scale
        else:
            current_novelty = base_novelty

        # Distractor curriculum: linearly ramp max counts from start to end
        # over the first `distractor_curriculum_updates`.
        if distractor_curriculum_updates > 0:
            t = min(1.0, max(0.0, (update - 1) / float(distractor_curriculum_updates)))
        else:
            t = 1.0
        if n_train_distractors_max_end is not None:
            cur_distractors_max = int(round(
                n_train_distractors_max + t * (n_train_distractors_max_end - n_train_distractors_max)
            ))
        else:
            cur_distractors_max = n_train_distractors_max
        if n_train_emp_distractors_max_end is not None:
            cur_emp_distractors_max = int(round(
                n_train_emp_distractors_max
                + t * (n_train_emp_distractors_max_end - n_train_emp_distractors_max)
            ))
        else:
            cur_emp_distractors_max = n_train_emp_distractors_max

        # Compute current interleave fraction (curriculum if anneal>0).
        if interleave_anneal_updates > 0:
            t = min(1.0, max(0.0, (update - 1) / float(interleave_anneal_updates)))
            current_emp_frac = interleave_start + t * (interleave_end - interleave_start)
        else:
            current_emp_frac = interleave_start
        n_emp_interleave = int(round(n_envs * current_emp_frac))
        n_pre_interleave = n_envs - n_emp_interleave

        # First `warmup_explore_only_updates` are 100% empty (pure explore).
        in_warmup = update <= warmup_explore_only_updates
        if in_warmup:
            n_pre_now, n_emp_now = 0, n_envs
        else:
            n_pre_now, n_emp_now = n_pre_interleave, n_emp_interleave

        rollouts = []
        for w_idx, world in enumerate(worlds):
            vh = world["vectorhash"]
            collector = RolloutCollector(vh, cfg, embed_dim, device)
            for local_idx, env in enumerate(world["envs"]):
                env_offset = vh.env_offsets[world["env_indices"][local_idx]]
                # Order: first n_pre_now envs are pre_stored, rest are empty.
                if local_idx < n_pre_now:
                    if use_distractors:
                        # Fresh Hopfield: goal pattern + N distractor patterns,
                        # N ~ Uniform[min, max] resampled per rollout.
                        hop = Hopfield(embed_dim, beta=cfg.hopfield.beta,
                                       device=str(device))
                        gx = min(max(env.goal_location[0] + env_offset[0], 0),
                                 vh.Npos - 1)
                        gy = min(max(env.goal_location[1] + env_offset[1], 0),
                                 vh.Npos - 1)
                        # Collect goal + distractor patterns, then shuffle
                        # the storage order so the goal isn't always first.
                        patterns = [vh.encoded_Phi[gx, gy]]
                        n_dist = int(dist_rng.randint(
                            n_train_distractors_min,
                            cur_distractors_max + 1,
                        ))
                        if n_dist > 0:
                            patterns.extend(sample_distractors(
                                vh, env_offset, cfg.env.size, n_dist, dist_rng))
                        dist_rng.shuffle(patterns)
                        for pat in patterns:
                            hop.input_memory(torch.from_numpy(pat).float())
                    else:
                        hop = pre_pools[w_idx][local_idx]
                    cfg.hopfield.novelty_reward = 0.0
                    # Follow regime: goals on so agent gets +1 + teleport.
                    env.goals_active = True
                else:
                    if use_emp_distractors:
                        # Fresh Hopfield with N distractor patterns (no goal).
                        # Trains explore policy to ignore non-goal recall
                        # signals, so eval-time distractors don't trigger
                        # spurious follow behavior in explore mode.
                        hop = Hopfield(embed_dim, beta=cfg.hopfield.beta,
                                       device=str(device))
                        n_emp_dist = int(dist_rng.randint(
                            n_train_emp_distractors_min,
                            cur_emp_distractors_max + 1,
                        ))
                        if n_emp_dist > 0:
                            for pat in sample_distractors(
                                    vh, env_offset, cfg.env.size,
                                    n_emp_dist, dist_rng):
                                hop.input_memory(torch.from_numpy(pat).float())
                    else:
                        hop = emp_pools[w_idx][local_idx]
                    cfg.hopfield.novelty_reward = current_novelty
                    # Explore regime: optionally turn goals off so the agent
                    # is rewarded purely for novelty/coverage, not goal-find.
                    env.goals_active = not explore_goals_off
                # Goal randomization: pick a fresh goal each rollout to break
                # the memorization shortcut (each env has a fixed sensory
                # codebook → without this, agent learns "in env X go to
                # position Y" via sensory fingerprint, capping coverage).
                # Only sensible in pure-explore phases where there's no
                # pre-stored Hopfield (which would be invalidated by goal
                # change). Skip for pre_stored envs.
                if randomize_goal_per_rollout and local_idx >= n_pre_now:
                    env.reset_goal()
                # Epsilon-greedy injection only on empty-regime envs.
                # Pre_stored envs need clean follow signal; ε would corrupt it.
                if local_idx >= n_pre_now:
                    eps_now = _compute_epsilon(
                        update, n_updates_total, epsilon_explore,
                        epsilon_anneal_updates,
                    )
                else:
                    eps_now = 0.0
                # Pre_stored regime always starts with goal in Hopfield;
                # empty regime never has goal. Caller declares this so
                # the input_goal_in_memory bit (if enabled) reflects the
                # true memory contents at training time, matching eval.
                rollout = collector.collect_rollout(
                    env, agent, hop, h_rnn=None, env_offset=env_offset,
                    update_idx=update, aux_scale=1.0, epsilon_now=eps_now,
                    goal_in_memory_init=(local_idx < n_pre_now),
                )
                rollouts.append(rollout)
        cfg.hopfield.novelty_reward = 0.0

        agent.train()
        losses = ppo_update(agent, rollouts, cfg.ppo, optimizer, aux_scale=1.0)

        mean_r = sum(r.rewards.sum().item() for r in rollouts) / max(
            sum(r.rewards.numel() for r in rollouts), 1)
        if n_pre_now > 0:
            pre_rs = rollouts[:n_pre_now * len(worlds)]
            emp_rs = rollouts[n_pre_now * len(worlds):]
        else:
            pre_rs, emp_rs = [], rollouts
        def _mr(rs):
            if not rs:
                return 0.0
            tot = sum(r.rewards.sum().item() for r in rs)
            n = sum(r.rewards.numel() for r in rs)
            return tot / max(n, 1)

        if use_wandb:
            import wandb
            log = {f"train/{k}": v for k, v in losses.items()}
            log["train/mean_reward"] = mean_r
            log["train/mean_reward_pre"] = _mr(pre_rs)
            log["train/mean_reward_emp"] = _mr(emp_rs)
            log["train/current_novelty"] = current_novelty
            log["train/current_epsilon"] = _compute_epsilon(
                update, n_updates_total, epsilon_explore,
                epsilon_anneal_updates,
            )
            log["train/current_emp_frac"] = current_emp_frac
            log["train/in_warmup"] = float(in_warmup)
            log["phase_name"] = "Phase A sweep"
            wandb.log(log)

        if update == 1 or update % 10 == 0:
            log_std_mean = float(agent.movement_log_std.exp().mean().item())
            print(f"  u{update}{'(warmup)' if in_warmup else ''}: "
                  f"mean_r={mean_r:.4f} (pre={_mr(pre_rs):.4f}, "
                  f"emp={_mr(emp_rs):.4f}) nov={current_novelty:.3f} "
                  f"std={log_std_mean:.3f} | "
                  + " ".join(f"{k}={v:.3f}" for k, v in losses.items()),
                  flush=True)

        if eval_world is not None and update % max(eval_every, 1) == 0:
            do_eval(cfg, agent, eval_world, device,
                    f"phase_a_u{update}", use_wandb,
                    max_steps=cfg.steps_per_rollout)
            # Save per-eval checkpoint so we can pick a best one later.
            os.makedirs(cfg.save_dir, exist_ok=True)
            torch.save({
                "agent_state_dict": agent.state_dict(),
                "config": asdict(cfg),
                "update": update,
            }, os.path.join(cfg.save_dir, f"phase_a_u{update}.pt"))

    do_eval(cfg, agent, eval_world, device, "after_phase_a", use_wandb,
            max_steps=cfg.steps_per_rollout)

    for k, v in saved.items():
        if k in ("auto_nav_warmup", "auto_store_warmup", "novelty_reward"):
            setattr(cfg.hopfield, k, v)
        else:
            setattr(cfg.ppo, k, v)


def train_phase_a_only(
    cfg: TrainConfig, pcfg: PhasedConfigV3,
    warmup_explore_only_updates: int,
    novelty_anneal: bool,
    interleave_empty_fraction: float,
    randomize_goal_per_rollout: bool = False,
    epsilon_explore: float = 0.0,
    epsilon_anneal_updates: int = 0,
    interleave_empty_target: float | None = None,
    interleave_anneal_updates: int = 0,
    load_checkpoint: str | None = None,
    explore_goals_off: bool = False,
    n_train_distractors_min: int = 0,
    n_train_distractors_max: int = 0,
    n_train_emp_distractors_min: int = 0,
    n_train_emp_distractors_max: int = 0,
    n_train_distractors_max_end: int | None = None,
    n_train_emp_distractors_max_end: int | None = None,
    distractor_curriculum_updates: int = 0,
    log_std_anneal_start_update: int = 0,
    log_std_anneal_end_update: int = 0,
    log_std_anneal_target: float | None = None,
) -> None:
    validate_train_config(cfg)
    warn_if_offcell_stores(cfg.env, where="train_phase_a_only")
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
    # Eval always uses goals_active=True so disc/nav metrics work, even when
    # training envs have goals_active=False (V7 pure-explore Phase A).
    saved_goals_active = cfg.env.goals_active
    cfg.env.goals_active = True
    eval_world = setup_world(cfg, encoder, embed_dim, rng, role="eval")
    cfg.env.goals_active = saved_goals_active

    input_dim = compute_input_dim(cfg.agent, embed_dim, cfg.env.observation_size)
    print(f"Agent input_dim={input_dim} init_log_std={cfg.agent.init_log_std}",
          flush=True)
    agent = NavAgent(cfg.agent, input_dim).to(device)

    if load_checkpoint is not None:
        ck = torch.load(load_checkpoint, map_location=device, weights_only=False)
        agent.load_state_dict(ck["agent_state_dict"])
        print(f"Loaded agent state from {load_checkpoint}", flush=True)

    if cfg.use_wandb:
        import wandb
        full_cfg = {
            **asdict(cfg),
            "phased_v3": asdict(pcfg),
            "warmup_explore_only_updates": warmup_explore_only_updates,
            "novelty_anneal": novelty_anneal,
            "interleave_empty_fraction": interleave_empty_fraction,
            "epsilon_explore": epsilon_explore,
            "epsilon_anneal_updates": epsilon_anneal_updates,
        }
        wandb.init(project=cfg.wandb_project, config=full_cfg)

    if cfg.save_dir is None:
        sub = (wandb.run.name or wandb.run.id) if cfg.use_wandb else \
              datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg.save_dir = os.path.join("checkpoint", f"phase_a_only_{sub}")

    run_phase_a_sweep(
        cfg, pcfg, worlds, agent, embed_dim, device,
        cfg.use_wandb, eval_world, cfg.eval_every,
        warmup_explore_only_updates, novelty_anneal,
        interleave_empty_fraction, randomize_goal_per_rollout,
        epsilon_explore=epsilon_explore,
        epsilon_anneal_updates=epsilon_anneal_updates,
        interleave_empty_target=interleave_empty_target,
        interleave_anneal_updates=interleave_anneal_updates,
        explore_goals_off=explore_goals_off,
        n_train_distractors_min=n_train_distractors_min,
        n_train_distractors_max=n_train_distractors_max,
        n_train_emp_distractors_min=n_train_emp_distractors_min,
        n_train_emp_distractors_max=n_train_emp_distractors_max,
        n_train_distractors_max_end=n_train_distractors_max_end,
        n_train_emp_distractors_max_end=n_train_emp_distractors_max_end,
        distractor_curriculum_updates=distractor_curriculum_updates,
        log_std_anneal_start_update=log_std_anneal_start_update,
        log_std_anneal_end_update=log_std_anneal_end_update,
        log_std_anneal_target=log_std_anneal_target,
        dist_rng=rng,
    )

    os.makedirs(cfg.save_dir, exist_ok=True)
    torch.save({
        "agent_state_dict": agent.state_dict(),
        "config": asdict(cfg),
    }, os.path.join(cfg.save_dir, "phase_a_only_final.pt"))
    print(f"\nDone. Saved to {cfg.save_dir}/phase_a_only_final.pt", flush=True)

    if cfg.use_wandb:
        import wandb
        wandb.finish()


def main():
    p = argparse.ArgumentParser(description="Phase-A-only sweep harness")
    p.add_argument("--encoder_checkpoint", required=True)
    p.add_argument("--size", type=int, default=8)
    p.add_argument("--observation_size", type=int, default=12)
    p.add_argument("--movement_mode", default="continuous")
    p.add_argument("--hopfield_mode", default="continuous")
    p.add_argument("--input_prev_reward", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--input_prev_action", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--input_hopfield_raw", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--input_hopfield_multistep", type=int, nargs="*", default=[],
                   help="If non-empty, project recall at these Hopfield iteration counts and pass each as 2-D extra input. Continuous mode only. e.g. --input_hopfield_multistep 1 2 3")
    p.add_argument("--input_sensory", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--input_encoded_state", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--input_hopfield_signal", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--input_goal_in_memory", action=argparse.BooleanOptionalAction, default=False,
                   help="Add 1-bit input: 'goal pattern is in Hopfield'. "
                        "Pre_stored rollouts: bit=True from t=0. Empty: "
                        "bit=False (or flips when agent stores at goal). "
                        "Eval matches: nav-eval bit=True, explore-eval bit "
                        "starts False. Bypasses regime-cue learning.")
    p.add_argument("--init_log_std", type=float, default=-0.5)
    p.add_argument("--freeze_log_std", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Hold movement_log_std at init_log_std with no "
                        "gradient. Pins policy variance so PPO loss directly "
                        "pressures the policy mean.")
    # Phase A core
    p.add_argument("--phase_a_updates", type=int, default=600,
                   help="Number of interleaved updates AFTER any warmup")
    p.add_argument("--phase_a_lr", type=float, default=3e-4)
    p.add_argument("--phase_a_novelty_reward", type=float, default=0.1)
    # New sweep knobs
    p.add_argument("--warmup_explore_only_updates", type=int, default=0,
                   # argparse formats help text with `help % params`, so a bare
                   # `%` must be escaped or --help raises TypeError.
                   help="Pure-explore (100%% empty) prefix; 0 to disable")
    p.add_argument("--novelty_anneal", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--interleave_empty_fraction", type=float, default=0.5,
                   help="Fraction of envs in interleaved phase that run empty")
    p.add_argument("--randomize_goal_per_rollout",
                   action=argparse.BooleanOptionalAction, default=False,
                   help="Re-pick goal each rollout to break memorization "
                        "shortcut. Only applies to empty-regime envs.")
    p.add_argument("--epsilon_explore", type=float, default=0.0,
                   help="Per-step probability of replacing the policy's "
                        "movement with a uniform-random direction (empty-"
                        "regime envs only). 0.0 disables.")
    p.add_argument("--epsilon_anneal_updates", type=int, default=0,
                   help="Linearly anneal epsilon_explore to 0 over this "
                        "many updates. 0 = constant.")
    p.add_argument("--goals_active", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="When False, training envs emit no +1 goal reward "
                        "and never teleport on goal-reach (pure-explore "
                        "Phase A). Eval envs always use goals_active=True.")
    p.add_argument("--move_ent_coef", type=float, default=None,
                   help="Override PPOConfig.ent_coef (entropy bonus on "
                        "movement policy). V7 default 0.0 disables the "
                        "log_std inflation that collapsed V6.")
    p.add_argument("--revisit_penalty", type=float, default=0.0,
                   help="Per-step reward penalty when agent occupies an "
                        "already-visited cell. Densifies the coverage "
                        "gradient: novelty alone goes silent on revisits, "
                        "this keeps signal alive.")
    p.add_argument("--wall_penalty", type=float, default=0.0,
                   help="Per-step reward penalty when agent occupies a "
                        "grid-edge cell. Counters perimeter-walk basins.")
    p.add_argument("--persistence_bonus", type=float, default=0.0,
                   help="Per-step bonus = bonus × cos(action_t, "
                        "action_{t-1}). Encourages straight-line movement "
                        "in explore phase. Stateless alternative to "
                        "revisit_penalty.")
    p.add_argument("--novelty_scale_remaining", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Scale novelty reward by total_cells/n_remaining "
                        "so rare late-game cells pay more.")
    p.add_argument("--novelty_scale_cap", type=float, default=10.0,
                   help="Cap on the remaining-scale multiplier.")
    # Rollout/training
    p.add_argument("--batch_envs", type=int, default=16)
    p.add_argument("--steps_per_rollout", type=int, default=400)
    p.add_argument("--num_worlds", type=int, default=1)
    p.add_argument("--envs_per_world", type=int, default=20)
    p.add_argument("--num_val_envs", type=int, default=10)
    p.add_argument("--n_val_trials", type=int, default=32)
    p.add_argument("--val_distractors", type=int, nargs="+", default=[0])
    p.add_argument("--union_cov_trials", type=int, default=0,
                   help="DEPRECATED and ignored since 2026-08-06. Union "
                        "coverage is now computed by evaluate_exploration "
                        "over its own rollouts, so the union is taken over "
                        "--n_val_trials. Passing a nonzero value warns.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="hopfield-nav-phase-a-sweep")
    p.add_argument("--lambdas", type=int, nargs="+", default=[11, 12, 13])
    p.add_argument("--Np", type=int, default=400)
    p.add_argument("--static-vectorhash", dest="static_vectorhash",
                   action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fwhm_ratio", type=float, default=0.25)
    p.add_argument("--encoder_gain", type=float, default=None)
    p.add_argument("--load_checkpoint", type=str, default=None,
                   help="Phase A/B checkpoint to load before training")
    p.add_argument("--interleave_empty_target", type=float, default=None,
                   help="If set, anneal interleave_empty_fraction → this "
                        "value linearly over interleave_anneal_updates.")
    p.add_argument("--interleave_anneal_updates", type=int, default=0,
                   help="Updates over which to anneal interleave fraction "
                        "from initial → target. 0 = no anneal.")
    p.add_argument("--explore_goals_off", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="If True, disable goals_active for empty-regime envs "
                        "(no goal +1 reward, no teleport). Pre-stored envs "
                        "keep goals_active=True. Forces explore mode to be "
                        "rewarded purely by novelty / revisit / time.")
    p.add_argument("--n_train_distractors_min", type=int, default=0,
                   help="Min distractor patterns added to pre_stored "
                        "Hopfield per rollout. Distractors come from grid "
                        "cells outside this env's region.")
    p.add_argument("--n_train_distractors_max", type=int, default=0,
                   help="Max distractor patterns per pre_stored rollout. "
                        "Per rollout, N ~ Uniform[min, max] is sampled. "
                        "0 disables (legacy: persistent goal-only Hopfield).")
    p.add_argument("--n_train_emp_distractors_min", type=int, default=0,
                   help="Min distractors in empty-regime Hopfield per "
                        "rollout. No goal pattern; explore policy learns "
                        "to ignore non-goal recall signals.")
    p.add_argument("--n_train_emp_distractors_max", type=int, default=0,
                   help="Max distractors in empty-regime Hopfield per "
                        "rollout. 0 disables (legacy: empty Hopfield).")
    p.add_argument("--n_train_distractors_max_end", type=int, default=None,
                   help="Curriculum target for n_train_distractors_max. "
                        "If set, max ramps linearly from start to this over "
                        "--distractor_curriculum_updates. None = no curriculum.")
    p.add_argument("--n_train_emp_distractors_max_end", type=int, default=None,
                   help="Curriculum target for n_train_emp_distractors_max.")
    p.add_argument("--distractor_curriculum_updates", type=int, default=0,
                   help="Number of updates over which to anneal distractor "
                        "max counts from start to end. 0 = no curriculum.")
    p.add_argument("--hidden_size", type=int, default=128,
                   help="RNN hidden dim. V18 baseline used 128.")
    p.add_argument("--num_rnn_layers", type=int, default=1,
                   help="Number of RNN layers. V18 baseline used 1.")
    p.add_argument("--goal_reward", type=float, default=1.0,
                   help="Reward at goal cell when goals_active. Bumping >1 "
                        "strengthens follow PPO updates vs explore reward.")
    p.add_argument("--goal_radius", type=float, default=0.5,
                   help="Euclidean radius around goal that counts as 'at goal'. "
                        "Default 0.5 reproduces snap-equality on integer-snapped "
                        "positions; larger values fuzz the goal region.")
    p.add_argument("--allow_offcell_store",
                   action=argparse.BooleanOptionalAction, default=False,
                   help="Whether a store fired while at goal may write a cell other than the goal's. Only reachable at goal_radius > 0.5, where at_goal tests the float position but embeddings are read at the snapped cell. Default False: the goal cell's embedding is stored instead, so the pattern written is the one navigation will later recall. Pass --allow_offcell_store for the pre-2026-08 behavior.")
    p.add_argument("--time_penalty", type=float, default=None,
                   help="Override EnvConfig.time_penalty (default 0.01). "
                        "Higher values directly punish step count, "
                        "pressuring the policy toward shorter trajectories.")
    p.add_argument("--continuous_normalize", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="If True, env unit-normalizes the action vector before "
                        "applying so step magnitude is fixed at continuous_scale "
                        "(default 1.0). Only takes effect in continuous movement_mode.")
    p.add_argument("--max_action_norm", type=float, default=None,
                   help="Soft cap on the L2 magnitude of move actions in the env. "
                        "Action with ‖a‖ > max_action_norm gets scaled down "
                        "(direction preserved). Only honored when "
                        "continuous_normalize is False. Targets the late-training "
                        "near-goal overshoot from policy mean magnitude growth.")
    p.add_argument("--min_action_norm", type=float, default=None,
                   help="Soft floor on the L2 magnitude of move actions in the env. "
                        "Action with 0 < ‖a‖ < min_action_norm gets scaled UP "
                        "(direction preserved). Only honored when "
                        "continuous_normalize is False. Forces a minimum step "
                        "magnitude so small policy means don't waste steps.")
    p.add_argument("--log_std_anneal_start_update", type=int, default=0,
                   help="Update at which to start ramping movement_log_std toward "
                        "--log_std_anneal_target. 0 disables the anneal.")
    p.add_argument("--log_std_anneal_end_update", type=int, default=0,
                   help="Update at which the anneal completes (log_std reaches "
                        "target). Must be > start. Requires --no-freeze_log_std "
                        "for PPO to subsequently move it; or, with --freeze_log_std, "
                        "the value is set programmatically and stays frozen at "
                        "the new value.")
    p.add_argument("--log_std_anneal_target", type=float, default=None,
                   help="Target log_std at end of anneal. e.g. -1.4 → σ≈0.247.")
    p.add_argument("--ppo_clip_coef", type=float, default=None,
                   help="Override PPOConfig.clip_coef (default 0.2). Lower "
                        "values (0.1-0.15) limit policy update size, helping "
                        "stability when goal_reward inflates value targets.")

    args = p.parse_args()

    env_kwargs = dict(size=args.size, observation_size=args.observation_size,
                      movement_mode=args.movement_mode,
                      goals_active=args.goals_active,
                      goal_reward=args.goal_reward,
                      goal_radius=args.goal_radius,
                      allow_offcell_store=args.allow_offcell_store)
    if args.time_penalty is not None:
        env_kwargs["time_penalty"] = args.time_penalty
    if args.continuous_normalize is not None:
        env_kwargs["continuous_normalize"] = args.continuous_normalize
    if args.max_action_norm is not None:
        env_kwargs["max_action_norm"] = args.max_action_norm
    if args.min_action_norm is not None:
        env_kwargs["min_action_norm"] = args.min_action_norm
    if args.union_cov_trials:
        print("  WARNING: --union_cov_trials is ignored since 2026-08-06. "
              "Union coverage is computed by evaluate_exploration over its own "
              "rollouts, so it is taken over --n_val_trials "
              f"({args.n_val_trials}), not over {args.union_cov_trials}.",
              flush=True)

    cfg = TrainConfig(
        env=EnvConfig(**env_kwargs),
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
            input_hopfield_multistep=args.input_hopfield_multistep,
            input_sensory=args.input_sensory,
            input_goal_in_memory=args.input_goal_in_memory,
            init_log_std=args.init_log_std,
            freeze_log_std=args.freeze_log_std,
            hidden_size=args.hidden_size,
            num_rnn_layers=args.num_rnn_layers,
        ),
        ppo=PPOConfig(),
        encoder_checkpoint=args.encoder_checkpoint,
        encoder_gain=args.encoder_gain,
        fwhm_ratio=args.fwhm_ratio,
        num_worlds=args.num_worlds, envs_per_world=args.envs_per_world,
        num_val_envs=args.num_val_envs, n_val_trials=args.n_val_trials,
        val_n_distractors_list=args.val_distractors,
        union_cov_trials=args.union_cov_trials,   # deprecated; see below
        batch_envs=args.batch_envs, steps_per_rollout=args.steps_per_rollout,
        eval_every=args.eval_every, save_dir=args.save_dir,
        seed=args.seed, device=args.device,
        use_wandb=args.use_wandb, wandb_project=args.wandb_project,
    )
    if args.move_ent_coef is not None:
        cfg.ppo.ent_coef = args.move_ent_coef
    if args.ppo_clip_coef is not None:
        cfg.ppo.clip_coef = args.ppo_clip_coef
    cfg.hopfield.revisit_penalty = args.revisit_penalty
    cfg.hopfield.wall_penalty = args.wall_penalty
    cfg.hopfield.persistence_bonus = args.persistence_bonus
    cfg.hopfield.novelty_scale_remaining = args.novelty_scale_remaining
    cfg.hopfield.novelty_scale_cap = args.novelty_scale_cap
    pcfg = PhasedConfigV3(
        phase_a_updates=args.phase_a_updates,
        phase_a_lr=args.phase_a_lr,
        phase_a_novelty_reward=args.phase_a_novelty_reward,
    )
    train_phase_a_only(
        cfg, pcfg,
        warmup_explore_only_updates=args.warmup_explore_only_updates,
        novelty_anneal=args.novelty_anneal,
        interleave_empty_fraction=args.interleave_empty_fraction,
        randomize_goal_per_rollout=args.randomize_goal_per_rollout,
        epsilon_explore=args.epsilon_explore,
        epsilon_anneal_updates=args.epsilon_anneal_updates,
        interleave_empty_target=args.interleave_empty_target,
        interleave_anneal_updates=args.interleave_anneal_updates,
        load_checkpoint=args.load_checkpoint,
        explore_goals_off=args.explore_goals_off,
        n_train_distractors_min=args.n_train_distractors_min,
        n_train_distractors_max=args.n_train_distractors_max,
        n_train_emp_distractors_min=args.n_train_emp_distractors_min,
        n_train_emp_distractors_max=args.n_train_emp_distractors_max,
        n_train_distractors_max_end=args.n_train_distractors_max_end,
        n_train_emp_distractors_max_end=args.n_train_emp_distractors_max_end,
        distractor_curriculum_updates=args.distractor_curriculum_updates,
        log_std_anneal_start_update=args.log_std_anneal_start_update,
        log_std_anneal_end_update=args.log_std_anneal_end_update,
        log_std_anneal_target=args.log_std_anneal_target,
    )


if __name__ == "__main__":
    main()
