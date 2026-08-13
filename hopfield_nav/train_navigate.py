"""Navigation training: PPO over a schedule of explore and exploit stages.

Explore and exploit are two *regimes* an env can be in, not two halves of a
rollout. An exploit env starts with the goal already in its Hopfield and follows
the recall signal; an explore env starts without it and is paid for coverage.
Every update collects rollouts from both and pools them into a single PPO step,
so what a schedule controls is the *fraction of envs in the explore regime* on
each update:

    --schedule "explore:200 ; interleave:800,empty_frac=1.0->0.5,anneal=50 ; exploit:100"

Until 2026-08 that fraction was implicit in four coupled flags -- a
100%-explore warmup prefix plus one monotone anneal -- which could not express
a third segment at all, let alone a stage-specific learning rate or novelty.
Those flags are gone; `hopfield_nav/training/stages.py` holds the grammar,
`training/explore.py` and `training/exploit.py` hold one regime each, and what
is left here is the composition: walk the stages, build each update's rollout
mix, and run the shared machinery around it.

The store head never trains here -- that is `train_store`'s job.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import asdict

import numpy as np
import torch

from cls_paths import run_dir, run_name
import run_manifest
from .config import TrainConfig, validate_train_config
from .encoder_io import load_encoder, validate_config
from .world.env import warn_if_offcell_stores
from .policy.agent import NavAgent, compute_input_dim
from .policy.recurrent import add_recurrent_args
from .rollout.collector import RolloutCollector
from .updates.ppo import advantage_scale_by_group, ppo_update
from .evaluation.checkpoint_io import cfg_from_checkpoint
from .training.cfg_args import settle_encoder
from .training.explore import ExploreRegime
from .training.exploit import ExploitRegime
from .training.refresh import Cadence, Refresher
from .training.stages import (
    Knobs, ScheduleError, Stage, format_schedule, parse_schedule, resolve,
    stage_at, total_updates,
)
from .training.world_setup import (
    build_field, do_eval, set_phase_freeze, setup_run_world,
)


def _compute_epsilon(update: int, base: float, anneal: int) -> float:
    """Linear anneal of ε from `base` → 0 over `anneal` updates (0 = constant).

    `update` is 1-indexed and global: ε is a property of how far the *run* has
    got, not of the current stage. A stage that wants its own value says
    `eps=` and bypasses this entirely.
    """
    if base <= 0:
        return 0.0
    if anneal <= 0:
        return base
    scale = max(0.0, 1.0 - (update - 1) / float(anneal))
    return base * scale


def run_navigate(
    cfg: TrainConfig,
    stages: list[Stage],
    worlds,
    agent,
    embed_dim,
    device,
    use_wandb: bool,
    eval_world,
    eval_every: int,
    ckpt_every: int,
    dist_rng: np.random.RandomState | None = None,
    ckpt_world: dict | None = None,
    refresher: Refresher | None = None,
    record_world=None,
) -> None:
    """Walk the schedule, one pooled PPO update per step.

    Everything the schedule does not control is read off `cfg`: the run-wide
    novelty / ε / distractor counts and the three global anneals (novelty,
    ε, movement_log_std) that are keyed off the global update counter rather
    than a stage-local one.

    `refresher`, when given, re-draws some of each train env's traits on its own
    cadence; `record_world` rewrites `world.json` so the growing union of values
    training has used stays on disk. Both are None for a run whose envs are
    fixed, which is every run before 2026-08.
    """
    n_updates_total = total_updates(stages)
    print(f"\n=== navigate: {format_schedule(stages)} "
          f"({n_updates_total} updates) ===", flush=True)

    set_phase_freeze(agent, freeze_move=False, freeze_store=cfg.freeze_store,
                     freeze_value=False, freeze_rnn=False)
    trainable = [p for p in agent.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=cfg.ppo.lr)
    # One optimizer for the whole run: stages are segments of a single training
    # trajectory, so Adam's moments carry across a boundary. A stage-level `lr`
    # retunes the existing group instead of building a new optimizer.
    current_lr = cfg.ppo.lr

    # Distractors are a property of the run, not of an update -- see
    # ExploitRegime for why the count reaching 0 must not change the code path.
    use_distractors = (
        cfg.n_train_distractors_max > 0
        or (cfg.n_train_distractors_max_end or 0) > 0
        or any(s.dist_max for s in stages if s.dist_max is not None)
    )
    use_emp_distractors = (
        cfg.n_train_emp_distractors_max > 0
        or (cfg.n_train_emp_distractors_max_end or 0) > 0
        or any(s.emp_dist_max for s in stages if s.emp_dist_max is not None)
    )
    if use_distractors or use_emp_distractors:
        if dist_rng is None:
            dist_rng = np.random.RandomState(cfg.seed + 7919)
        if use_distractors:
            print(f"Exploit-regime distractors per rollout: "
                  f"~U[{cfg.n_train_distractors_min}, "
                  f"{cfg.n_train_distractors_max}]", flush=True)
        if use_emp_distractors:
            print(f"Explore-regime distractors per rollout: "
                  f"~U[{cfg.n_train_emp_distractors_min}, "
                  f"{cfg.n_train_emp_distractors_max}] (no goal pattern)",
                  flush=True)

    exploit_regime = ExploitRegime(cfg, embed_dim, device, dist_rng,
                                   use_distractors=use_distractors)
    if (not cfg.freeze_store
            and not ExploitRegime.allows_store
            and not ExploreRegime.allows_store
            and cfg.hopfield.store_cost == 0
            and cfg.hopfield.store_bonus == 0
            and cfg.ppo.store_bc_weight == 0):
        print(
            "  WARNING: --no-freeze_store, but neither regime allows stores and "
            "there is no store_cost, store_bonus or store_bc_weight. The store "
            "head would take policy-gradient updates from an action with no "
            "consequence -- noise into the store logits and, through the shared "
            "trunk, into movement. Give the action a consequence or leave the "
            "head frozen.", flush=True)

    explore_regime = ExploreRegime(cfg, embed_dim, device, dist_rng,
                                   goals_off=cfg.explore_goals_off,
                                   use_distractors=use_emp_distractors,
                                   ends_on_goal=cfg.explore_ends_on_goal)

    if refresher is not None:
        print(f"Env refresh: {refresher.cadence.describe()} (train envs only; "
              f"the validation set is drawn once and held)", flush=True)

    n_envs = cfg.envs_per_world

    # Snapshot the config *before* the temporary overrides below. Checkpoints
    # are written mid-loop, at a moment when `novelty_reward` has been zeroed
    # for the eval that may follow and `store_bc_weight` is held at 0 for the
    # duration -- so `asdict(cfg)` taken there records the loop's scratch
    # values as if they were the run's settings. That was invisible while a
    # checkpoint's config was only ever read for its architecture; it stops
    # being invisible now that --load_checkpoint inherits the whole recipe.
    ckpt_config = asdict(cfg)

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

    base_novelty = cfg.hopfield.novelty_reward

    log_std_anneal_active = (
        cfg.log_std_anneal_target is not None
        and cfg.log_std_anneal_end_update > cfg.log_std_anneal_start_update
    )
    log_std_init_val = float(agent.movement_log_std.detach().mean().item()) \
        if hasattr(agent, "movement_log_std") else None

    # None follows the rollout length, which is what every run did before
    # `eval_max_steps` existed.
    eval_max_steps = (cfg.eval_max_steps if cfg.eval_max_steps is not None
                      else cfg.steps_per_rollout)

    # Wall-clock per update, excluding eval. Sizing a run from checkpoint
    # mtimes conflates the two and gets the answer wrong by the eval's share.
    t_update_mark = time.time()
    n_updates_timed = 0

    # Whether the store action writes / learns / pays is decided by three
    # unrelated mechanisms in three files; say so once, from the first spec
    # actually built rather than from a flag that could drift from it.

    for update in range(1, n_updates_total + 1):
        stage, local_update = stage_at(stages, update)

        # Before the rollouts, so this update trains on the envs it records.
        # Every value drawn here is folded into `split.used` by the refresher
        # itself -- see training/refresh.py for why that cannot be a second,
        # forgettable call.
        refreshed = () if refresher is None else refresher.maybe_refresh(update)

        # log_std anneal: programmatically interpolate the parameter from its
        # init value to the target value across [start, end] update window.
        if log_std_anneal_active and log_std_init_val is not None:
            if update >= cfg.log_std_anneal_end_update:
                t_ls = 1.0
            elif update <= cfg.log_std_anneal_start_update:
                t_ls = 0.0
            else:
                t_ls = (update - cfg.log_std_anneal_start_update) / float(
                    cfg.log_std_anneal_end_update - cfg.log_std_anneal_start_update)
            new_log_std = log_std_init_val + t_ls * (
                cfg.log_std_anneal_target - log_std_init_val)
            with torch.no_grad():
                agent.movement_log_std.data.fill_(new_log_std)

        # Anneal novelty if requested.
        if cfg.novelty_anneal:
            scale = max(0.0, 1.0 - (update - 1) / max(n_updates_total, 1))
            current_novelty = base_novelty * scale
        else:
            current_novelty = base_novelty

        # Distractor curriculum: linearly ramp max counts from start to end
        # over the first `distractor_curriculum_updates`.
        if cfg.distractor_curriculum_updates > 0:
            t = min(1.0, max(0.0,
                             (update - 1) / float(cfg.distractor_curriculum_updates)))
        else:
            t = 1.0
        if cfg.n_train_distractors_max_end is not None:
            cur_distractors_max = int(round(
                cfg.n_train_distractors_max
                + t * (cfg.n_train_distractors_max_end - cfg.n_train_distractors_max)
            ))
        else:
            cur_distractors_max = cfg.n_train_distractors_max
        if cfg.n_train_emp_distractors_max_end is not None:
            cur_emp_distractors_max = int(round(
                cfg.n_train_emp_distractors_max
                + t * (cfg.n_train_emp_distractors_max_end
                       - cfg.n_train_emp_distractors_max)
            ))
        else:
            cur_emp_distractors_max = cfg.n_train_emp_distractors_max

        # The run-wide values for this update, then the stage's overrides.
        knobs = resolve(stage, local_update, Knobs(
            lr=cfg.ppo.lr,
            empty_frac=0.0,             # always replaced by resolve()
            novelty=current_novelty,
            eps=_compute_epsilon(update, cfg.epsilon_explore,
                                 cfg.epsilon_anneal_updates),
            dist_min=cfg.n_train_distractors_min,
            dist_max=cur_distractors_max,
            emp_dist_min=cfg.n_train_emp_distractors_min,
            emp_dist_max=cur_emp_distractors_max,
        ))

        if knobs.lr != current_lr:
            for group in optimizer.param_groups:
                group["lr"] = knobs.lr
            current_lr = knobs.lr

        n_emp_now = int(round(n_envs * knobs.empty_frac))
        n_pre_now = n_envs - n_emp_now

        rollouts = []
        # One label per rollout, in the order they are appended. The reward
        # split below slices on `n_pre_now` instead, which only agrees with
        # this when `num_worlds == 1`; this list is what the advantage-scale
        # diagnostic uses, and it is right either way.
        regime_labels: list[str] = []
        for w_idx, world in enumerate(worlds):
            vh = world.field
            collector = RolloutCollector(vh, cfg, embed_dim, device)
            for local_idx, env in enumerate(world.envs):
                env_offset = world.offsets[local_idx]
                # Order: the first n_pre_now envs are exploit, the rest explore.
                # The reward split logged below slices on the same boundary.
                is_pre = local_idx < n_pre_now
                regime = exploit_regime if is_pre else explore_regime
                regime_labels.append("pre" if is_pre else "emp")
                spec = regime.spec(w_idx, world, local_idx, env, env_offset, knobs)
                # The collector reads novelty off cfg and the goal reward off
                # the env, so the regime's choice has to be written into both.
                cfg.hopfield.novelty_reward = spec.novelty_reward
                env.goals_active = spec.goals_active
                rollout = collector.collect_rollout(
                    env, agent, spec.hop, allow_store=spec.allow_store,
                    h_rnn=None, env_offset=env_offset,
                    update_idx=update, aux_scale=1.0, epsilon_now=spec.epsilon,
                    goal_in_memory_init=spec.goal_in_memory_init,
                    ends_on_goal=spec.ends_on_goal,
                )
                rollouts.append(rollout)
        cfg.hopfield.novelty_reward = 0.0

        # Only meaningful when an update actually mixes regimes: with one
        # regime the pooled divisor IS that regime's own, and every share is 1
        # by construction. Costs one extra GAE pass over the pool.
        adv_scale = (
            advantage_scale_by_group(rollouts, regime_labels,
                                     cfg.ppo.gamma, cfg.ppo.gae_lambda)
            if 0 < n_pre_now < n_envs else {}
        )

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
            log["train/current_novelty"] = knobs.novelty
            log["train/current_epsilon"] = knobs.eps
            log["train/current_emp_frac"] = knobs.empty_frac
            log["train/current_lr"] = knobs.lr
            log["train/stage_kind"] = stage.kind
            log["train/stage_local_update"] = local_update
            for k, v in adv_scale.items():
                log[f"train/adv_{k}"] = v
            if refresher is not None:
                for trait in refresher.counts:
                    log[f"train/refresh_{trait}"] = int(trait in refreshed)
            log["phase_name"] = "navigate"
            wandb.log(log)

        n_updates_timed += 1
        if update == 1 or update % 10 == 0:
            log_std_mean = float(agent.movement_log_std.exp().mean().item())
            s_per_update = (time.time() - t_update_mark) / max(n_updates_timed, 1)
            print(f"  u{update}({stage.kind}): "
                  f"mean_r={mean_r:.4f} (pre={_mr(pre_rs):.4f}, "
                  f"emp={_mr(emp_rs):.4f}) nov={knobs.novelty:.3f} "
                  f"emp_frac={knobs.empty_frac:.3f} std={log_std_mean:.3f} "
                  f"s/u={s_per_update:.1f} | "
                  + " ".join(f"{k}={v:.3f}" for k, v in losses.items())
                  + (f" | adv_share pre={adv_scale['pre_share']:.2f} "
                     f"emp={adv_scale['emp_share']:.2f}" if adv_scale else "")
                  + (f" | refresh={','.join(refreshed)}" if refreshed else ""),
                  flush=True)
            t_update_mark, n_updates_timed = time.time(), 0

        if eval_world is not None and update % max(eval_every, 1) == 0:
            do_eval(cfg, agent, eval_world, device,
                    f"navigate_u{update}", use_wandb,
                    max_steps=eval_max_steps)
            # Eval time is reported by do_eval and must not be charged to the
            # updates that follow it.
            t_update_mark, n_updates_timed = time.time(), 0

        # Checkpointing on its own cadence. It used to sit inside the eval
        # branch above, which coupled two things with opposite costs: an eval
        # is expensive and wants to be rare, a checkpoint is cheap and wants to
        # be frequent. So `--eval_every 20` on a 300-update run left 15
        # checkpoints, and `analysis.trajectories` -- whose rows are one
        # checkpoint each -- had nothing to draw. `--ckpt_every` defaults to
        # `--eval_every`, so an existing command line is unchanged.
        if update % max(ckpt_every, 1) == 0:
            # Rewrite `world.json` first, so the summary this checkpoint carries
            # names the file as it now stands. The `used` union grows with every
            # refresh tick, and it is what a later `make_val_set` excludes
            # against -- a record written only at startup would let a held-out
            # val env be placed exactly where training later moved.
            if record_world is not None:
                ckpt_world = record_world()
            os.makedirs(cfg.save_dir, exist_ok=True)
            torch.save({
                "agent_state_dict": agent.state_dict(),
                "config": ckpt_config,
                "world_spec": ckpt_world,
                "update": update,
            }, os.path.join(cfg.save_dir, f"navigate_u{update}.pt"))
            run_manifest.record_checkpoint(
                cfg.save_dir, f"navigate_u{update}.pt", update)

    do_eval(cfg, agent, eval_world, device, "after_navigate", use_wandb,
            max_steps=eval_max_steps)

    for k, v in saved.items():
        if k in ("auto_nav_warmup", "auto_store_warmup", "novelty_reward"):
            setattr(cfg.hopfield, k, v)
        else:
            setattr(cfg.ppo, k, v)


def train_navigate(
    cfg: TrainConfig,
    stages: list[Stage],
    load_checkpoint: str | None = None,
) -> None:
    validate_train_config(cfg)
    warn_if_offcell_stores(cfg.env, where="train_navigate")
    # Before the encoder loads and the 12 GB scaffold builds: a refresh cadence
    # without --env_generator has nowhere to draw from, and finding that out
    # twenty minutes in is not the same as finding it out now.
    cadence = Cadence.from_config(cfg)
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

    # One scaffold field for the whole run. It is a pure function of
    # (lambdas, Npos, fwhm_ratio, encoder), so the per-world copies this used
    # to build were bit-identical -- and 12 GB each at Npos=1716.
    field = build_field(cfg, encoder)
    # The world, its record, and the refresher, in one place -- shared with
    # `train` and `train_store` so the three cannot drift about what a run
    # writes down. The preflight runs inside, and a cadence that would run the
    # domains dry raises here, before the manifest exists or a single update.
    encoder_ident = run_manifest.encoder_identity(
        cfg.encoder_checkpoint, enc_cfg, encoder_gain)
    rw = setup_run_world(cfg, encoder, embed_dim, rng, field,
                         cadence=cadence, n_updates=total_updates(stages),
                         encoder_ident=encoder_ident, where="train_navigate",
                         parent_ckpt=load_checkpoint)
    worlds, eval_world, split = rw.worlds, rw.eval_world, rw.split

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
        wandb.init(project=cfg.wandb_project, config=asdict(cfg))

    if cfg.save_dir is None:
        sub = run_name(*((wandb.run.name, wandb.run.id) if cfg.use_wandb else ()))
        cfg.save_dir = str(run_dir("navigate", sub))
    else:
        sub = os.path.basename(str(cfg.save_dir).rstrip("/"))

    run_manifest.begin(
        cfg.save_dir, kind="navigate", name=sub, config=asdict(cfg),
        encoder=encoder_ident,
        parent=load_checkpoint,
        wandb_run=wandb.run if cfg.use_wandb else None,
    )

    # Written on both paths: a run has to be able to say which envs it used,
    # and the historical path could not (see docs/EVAL_SPLITS_DESIGN.md 1.4).
    refresher = rw.refresher
    record_world = rw.record
    ckpt_world = record_world()

    run_navigate(
        cfg, stages, worlds, agent, embed_dim, device,
        cfg.use_wandb, eval_world, cfg.eval_every,
        cfg.ckpt_every if cfg.ckpt_every is not None else cfg.eval_every,
        dist_rng=rng, ckpt_world=ckpt_world, refresher=refresher,
        # A run whose envs never move needs no rewrite: the startup record
        # already describes them for the whole run.
        record_world=(record_world if refresher is not None else None),
    )

    # `run_navigate` rewrote the record on its own cadence, into its own local;
    # take a final one so navigate_final.pt names the file as it ends up, with
    # the complete union of everything training touched.
    if refresher is not None:
        ckpt_world = record_world()

    os.makedirs(cfg.save_dir, exist_ok=True)
    torch.save({
        "agent_state_dict": agent.state_dict(),
        "config": asdict(cfg),
        "world_spec": ckpt_world,
    }, os.path.join(cfg.save_dir, "navigate_final.pt"))
    run_manifest.record_checkpoint(cfg.save_dir, "navigate_final.pt")
    run_manifest.finish(cfg.save_dir)
    print(f"\nDone. Saved to {cfg.save_dir}/navigate_final.pt", flush=True)

    if cfg.use_wandb:
        import wandb
        wandb.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# Which TrainConfig field each flag writes, as a dotted path. One table rather
# than a hand-written constructor because the same mapping now serves two
# callers: a fresh run applies every entry, a run resuming from
# --load_checkpoint applies only the entries the caller actually typed.
# `movement_mode` has two homes and gets a tuple.
CFG_FIELDS: dict[str, tuple[str, ...]] = {
    # env
    "size": ("env.size",),
    "observation_size": ("env.observation_size",),
    "movement_mode": ("env.movement_mode", "agent.movement_mode"),
    "goal_reward": ("env.goal_reward",),
    "goal_radius": ("env.goal_radius",),
    "allow_offcell_store": ("env.allow_offcell_store",),
    "egocentric_heading": ("env.egocentric_heading",),
    "reset_state_on_teleport": ("env.reset_state_on_teleport",),
    "explore_ends_on_goal": ("explore_ends_on_goal",),
    "wall_resolution": ("env.wall_resolution",),
    "time_penalty": ("env.time_penalty",),
    "continuous_normalize": ("env.continuous_normalize",),
    "max_action_norm": ("env.max_action_norm",),
    "min_action_norm": ("env.min_action_norm",),
    # scaffold
    "lambdas": ("vectorhash.lambdas",),
    "Np": ("vectorhash.Np",),
    "static_vectorhash": ("vectorhash.static_vectorhash",),
    # agent
    "hopfield_mode": ("agent.hopfield_mode",),
    "input_encoded_state": ("agent.input_encoded_state",),
    "input_hopfield_signal": ("agent.input_hopfield_signal",),
    "input_prev_action": ("agent.input_prev_action",),
    "input_prev_reward": ("agent.input_prev_reward",),
    "input_hopfield_raw": ("agent.input_hopfield_raw",),
    "input_hopfield_multistep": ("agent.input_hopfield_multistep",),
    "input_sensory": ("agent.input_sensory",),
    "input_goal_in_memory": ("agent.input_goal_in_memory",),
    "init_log_std": ("agent.init_log_std",),
    "freeze_log_std": ("agent.freeze_log_std",),
    "hidden_size": ("agent.hidden_size",),
    "num_rnn_layers": ("agent.num_rnn_layers",),
    "rnn_cell": ("agent.rnn_cell",),
    "rnn_nonlinearity": ("agent.rnn_nonlinearity",),
    # ppo
    "lr": ("ppo.lr",),
    "move_ent_coef": ("ppo.ent_coef",),
    "ppo_clip_coef": ("ppo.clip_coef",),
    # reward shaping
    "novelty_reward": ("hopfield.novelty_reward",),
    "revisit_penalty": ("hopfield.revisit_penalty",),
    "wall_penalty": ("hopfield.wall_penalty",),
    "persistence_bonus": ("hopfield.persistence_bonus",),
    "novelty_scale_remaining": ("hopfield.novelty_scale_remaining",),
    "novelty_scale_cap": ("hopfield.novelty_scale_cap",),
    # run structure
    "encoder_checkpoint": ("encoder_checkpoint",),
    "encoder_gain": ("encoder_gain",),
    "fwhm_ratio": ("fwhm_ratio",),
    "num_worlds": ("num_worlds",),
    "envs_per_world": ("envs_per_world",),
    "num_val_envs": ("num_val_envs",),
    "n_val_trials": ("n_val_trials",),
    "val_distractors": ("val_n_distractors_list",),
    "union_cov_trials": ("union_cov_trials",),
    "batch_envs": ("batch_envs",),
    "steps_per_rollout": ("steps_per_rollout",),
    "eval_every": ("eval_every",),
    "eval_scope": ("eval_scope",),
    "freeze_store": ("freeze_store",),
    "eval_max_steps": ("eval_max_steps",),
    "ckpt_every": ("ckpt_every",),
    "save_dir": ("save_dir",),
    "seed": ("seed",),
    "device": ("device",),
    "use_wandb": ("use_wandb",),
    "wandb_project": ("wandb_project",),
    # env generator
    "env_generator": ("env_generator",),
    "place_region": ("place_region",),
    "goal_region": ("goal_region",),
    "wall_seeds": ("wall_seeds",),
    "place_margin": ("place_margin",),
    "goal_val_frac": ("goal_val_frac",),
    "refresh_place": ("refresh_place",),
    "refresh_wall": ("refresh_wall",),
    "refresh_goal": ("refresh_goal",),
    "refresh_size": ("refresh_size",),
    # schedule
    "schedule": ("schedule",),
    "novelty_anneal": ("novelty_anneal",),
    "epsilon_explore": ("epsilon_explore",),
    "epsilon_anneal_updates": ("epsilon_anneal_updates",),
    "explore_goals_off": ("explore_goals_off",),
    "n_train_distractors_min": ("n_train_distractors_min",),
    "n_train_distractors_max": ("n_train_distractors_max",),
    "n_train_emp_distractors_min": ("n_train_emp_distractors_min",),
    "n_train_emp_distractors_max": ("n_train_emp_distractors_max",),
    "n_train_distractors_max_end": ("n_train_distractors_max_end",),
    "n_train_emp_distractors_max_end": ("n_train_emp_distractors_max_end",),
    "distractor_curriculum_updates": ("distractor_curriculum_updates",),
    "log_std_anneal_start_update": ("log_std_anneal_start_update",),
    "log_std_anneal_end_update": ("log_std_anneal_end_update",),
    "log_std_anneal_target": ("log_std_anneal_target",),
}


def _explicit_dests(parser: argparse.ArgumentParser, argv: list[str]) -> set[str]:
    """The dests of the flags the caller actually typed.

    `--load_checkpoint` makes the checkpoint's config the base, so "not
    mentioned" has to mean "inherit". A parsed Namespace cannot say that -- it
    holds a value either way -- so the information survives only in argv.

    `--flag=value` is split on '='. `BooleanOptionalAction`'s `--no-flag` needs
    no special case: argparse lists it in the same action's `option_strings`.
    The parser is built with `allow_abbrev=False` so a shortened flag cannot
    reach the Namespace while going unmatched here.
    """
    typed = {tok.split("=", 1)[0] for tok in argv if tok.startswith("-")}
    return {action.dest for action in parser._actions
            if any(opt in typed for opt in action.option_strings)}


def _set_path(cfg: TrainConfig, path: str, value) -> None:
    obj = cfg
    *parents, leaf = path.split(".")
    for part in parents:
        obj = getattr(obj, part)
    setattr(obj, leaf, value)


def apply_args(cfg: TrainConfig, args: argparse.Namespace, dests) -> None:
    """Write the named flags onto `cfg`.

    A `None` value means "the flag has no opinion" -- every such flag defaults
    to None precisely so that not passing it leaves the dataclass default (or,
    when resuming, the parent's value) in place.
    """
    for dest in dests:
        paths = CFG_FIELDS.get(dest)
        if paths is None:
            continue
        value = getattr(args, dest, None)
        if value is None:
            continue
        for path in paths:
            _set_path(cfg, path, value)


def build_parser() -> argparse.ArgumentParser:
    # allow_abbrev=False: _explicit_dests matches option strings literally, and
    # an abbreviation would parse fine while silently failing to override an
    # inherited config value.
    p = argparse.ArgumentParser(
        description="Navigation training over an explore/exploit schedule",
        allow_abbrev=False)
    p.add_argument("--encoder_checkpoint", default=None,
                   help="Required for a fresh run. Optional with "
                        "--load_checkpoint, which inherits the parent's -- pass "
                        "it only to deliberately swap encoders, which warns.")
    p.add_argument("--size", type=int, default=8)
    p.add_argument("--observation_size", type=int, default=12)
    p.add_argument("--explore_ends_on_goal",
                   action=argparse.BooleanOptionalAction, default=None,
                   help="An explore rollout ends when the agent reaches "
                        "the goal, instead of teleporting and continuing. "
                        "Per trajectory: the other B-1 keep running. On by "
                        "default since 2026-08-12. Vacuous under "
                        "--explore_goals_off, where there is no goal event "
                        "to end on. Note it truncates novelty accrual -- an "
                        "agent that finds the goal early collects fewer "
                        "coverage steps.")
    p.add_argument("--reset_state_on_teleport",
                   action=argparse.BooleanOptionalAction, default=None,
                   help="Zero the RNN hidden state and prev_reward / prev_action "
                    "when the agent teleports after reaching the goal (C5 of the "
                    "at-goal contract, world/episode.py). Default off since "
                    "2026-08-12: recurrence spans the whole rollout rather than "
                    "restarting at each goal. Applies to training and evaluation "
                    "together -- an answer that differed between them would make "
                    "the two incomparable.")
    p.add_argument("--egocentric_heading", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Foveal cone turns with the agent: heading is a "
                        "continuous angle following the direction it actually "
                        "moved, so a cell looks different depending on how the "
                        "agent arrived. --no-egocentric_heading pins every view "
                        "to North, reproducing pre-2026-08 runs.")
    p.add_argument("--wall_resolution", type=int, default=1,
                   help="How many +/-1 wall segments span one grid cell. 1 (default) is one segment per cell, the original coarse barcode. Above 1 a stripe edge can fall inside a cell, which is the only way a ray can report where within a cell it is looking from; at 1 roughly 9-14%% of cells share a bit-identical observation with another cell. 8 drives that to ~0. Changes env identity, so splits and checkpoints are tied to it.")
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
                        "Exploit rollouts: bit=True from t=0. Explore: "
                        "bit=False (or flips when agent stores at goal). "
                        "Eval matches: nav-eval bit=True, explore-eval bit "
                        "starts False. Bypasses regime-cue learning.")
    p.add_argument("--init_log_std", type=float, default=-0.5)
    p.add_argument("--freeze_log_std", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Hold movement_log_std at init_log_std with no "
                        "gradient. Pins policy variance so PPO loss directly "
                        "pressures the policy mean.")
    # The schedule
    p.add_argument("--schedule", type=str, default=None,
                   help="Stages, separated by ';'. Each is '<kind>:<updates>' "
                        "with optional ',key=value' overrides. Kinds: explore "
                        "(all envs start without the goal in memory), exploit "
                        "(all envs start with it), interleave (a mix). Keys: "
                        "lr, empty_frac (a number, or 'start->end' to anneal "
                        "across the stage), anneal (updates from the stage's "
                        "start to reach the end fraction), novelty, eps, "
                        "dist_min, dist_max, emp_dist_min, emp_dist_max. "
                        "A stage value overrides the run-wide flag outright. "
                        "Example: 'explore:200 ; "
                        "interleave:800,empty_frac=1.0->0.5,anneal=50 ; "
                        "exploit:100,lr=1e-4'. Required, unless "
                        "--load_checkpoint supplies one.")
    p.add_argument("--lr", type=float, default=3e-4,
                   help="Adam learning rate. A stage may override it with "
                        "'lr='; the optimizer is retuned in place, so Adam's "
                        "moments survive the boundary.")
    p.add_argument("--novelty_reward", type=float, default=0.0,
                   help="Reward per first-visit cell, explore regime only.")
    p.add_argument("--novelty_anneal", action=argparse.BooleanOptionalAction, default=False,
                   help="Linearly scale --novelty_reward to 0 across the whole "
                        "schedule. A stage's 'novelty=' ignores this.")
    p.add_argument("--epsilon_explore", type=float, default=0.0,
                   help="Per-step probability of replacing the policy's "
                        "movement with a uniform-random direction (explore-"
                        "regime envs only). 0.0 disables.")
    p.add_argument("--epsilon_anneal_updates", type=int, default=0,
                   help="Linearly anneal epsilon_explore to 0 over this "
                        "many updates. 0 = constant.")
    # No --goals_active here. Every rollout assigns `env.goals_active` from its
    # regime -- exploit always True, explore `not explore_goals_off` -- so a
    # run-wide value was overwritten before the first step and the flag did
    # nothing at all. Use --explore_goals_off, which is the knob that survives.
    # `EnvConfig.goals_active` stays: `train` and `train_phased` still set it.
    p.add_argument("--move_ent_coef", type=float, default=None,
                   help="Override PPOConfig.ent_coef (entropy bonus on "
                        "movement policy).")
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
    p.add_argument("--env_generator", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Draw envs from declared domains (world/generate.py) "
                        "instead of the historical placement path. Off keeps "
                        "today's envs for a given --seed; on fixes the "
                        "offset-reproducibility bug and enforces train/val "
                        "separation. world.json is written either way.")
    p.add_argument("--place_region", type=str, default="anywhere",
                   help="Where train envs may sit: 'anywhere' or "
                        "'rect:X0,Y0,W,H'. Declaring a rect is what makes a "
                        "place-OOD val set possible later -- its complement.")
    p.add_argument("--goal_region", type=str, default="any",
                   help="Which env-local cells may hold a goal: 'any', "
                        "'ring:W', 'interior:W' or 'quadrant:Q'. Declaring a "
                        "region is what makes a goal-OOD val set possible.")
    p.add_argument("--wall_seeds", type=str, default="0,10000000",
                   help="'LO,HI' range training draws wall seeds from.")
    p.add_argument("--place_margin", type=int, default=None,
                   help="Edge-to-edge train/val clearance in cells. Default "
                        "derives it from the scaffold's own cosine-vs-distance "
                        "curve (~80 at lambdas=11,12,13 fwhm=0.25).")
    p.add_argument("--goal_val_frac", type=float, default=0.2,
                   help="Share of goal cells reserved for validation.")
    # Per-trait refresh. All four require --env_generator; all four apply to the
    # train set only, because a validation set that moved under the model would
    # make every in-training curve unreadable.
    p.add_argument("--refresh_place", type=int, default=None,
                   help="Re-draw train env placements every N updates, from "
                        "--place_region and clear of the fixed val envs by the "
                        "margin. Omit to place once and hold. Requires "
                        "--env_generator.")
    p.add_argument("--refresh_wall", type=int, default=None,
                   help="Re-draw train wall seeds every N updates, excluding "
                        "every seed the run has already used. Rebuilds the "
                        "envs, so it is the one expensive cadence.")
    p.add_argument("--refresh_goal", type=int, default=None,
                   help="Re-draw train goals every N updates from the train "
                        "share of --goal_region. Replaces "
                        "--randomize_goal_per_rollout, which drew uniformly "
                        "over the arena and so could land on a cell reserved "
                        "for validation. Setting this also caps the train goal "
                        "cells at 1 - --goal_val_frac of the region up front.")
    p.add_argument("--refresh_size", type=int, default=None,
                   help="Re-draw the train env size every N updates. Needs "
                        "more than one declared size, which nothing produces "
                        "yet (Phase 6), so this currently errors at startup.")
    p.add_argument("--freeze_store", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Pin the store head. True (default) also drops its "
                        "entire objective from the PPO loss, so a frozen head "
                        "cannot steer the shared trunk either. Pass "
                        "--no-freeze_store to train it -- but give the store "
                        "action a consequence first (see --allow_store paths), "
                        "or it learns from pure noise.")
    p.add_argument("--eval_scope", type=str, default="all",
                   choices=("all", "expl", "nav_expl"),
                   help="Which evaluators an in-training eval runs. 'all' is "
                        "nav + goal-discovery + exploration. 'expl' is "
                        "exploration only, for pure-explore schedules where "
                        "the other two are undefined. 'nav_expl' drops only "
                        "goal-discovery: it is the sole unbatched evaluator "
                        "(B=1 per trial), so at 10 envs x 32 trials x 3 "
                        "distractor levels x 400 steps it costs ~10 min "
                        "against ~50 s for the other two together -- and it "
                        "scores the store head, which this trainer freezes.")
    p.add_argument("--eval_max_steps", type=int, default=None,
                   help="Step budget for in-training evals. Default: follow "
                        "--steps_per_rollout, which is what this did "
                        "unconditionally before. Pin it when rollout length is "
                        "the variable under test, so mean_coverage stays the "
                        "same measurement across variants.")
    p.add_argument("--ckpt_every", type=int, default=None,
                   help="Checkpoint cadence, in updates. Default: follow "
                        "--eval_every, which is what this did unconditionally "
                        "before. Set it lower to keep a dense checkpoint "
                        "series (what analysis.trajectories draws rows from) "
                        "without paying for an eval at each one.")
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
                   help="Checkpoint to start from. Its config becomes the base "
                        "for this run -- every setting is inherited except the "
                        "flags you pass explicitly, so a child reproduces its "
                        "parent's recipe without re-listing it. --save_dir is "
                        "never inherited.")
    p.add_argument("--explore_goals_off", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="If True, disable goals_active for explore-regime envs "
                        "(no goal reward, no teleport). Exploit envs keep "
                        "goals_active=True. Forces explore mode to be "
                        "rewarded purely by novelty / revisit / time.")
    p.add_argument("--n_train_distractors_min", type=int, default=0,
                   help="Min distractor patterns added to an exploit-regime "
                        "Hopfield per rollout. Distractors come from grid "
                        "cells outside this env's region.")
    p.add_argument("--n_train_distractors_max", type=int, default=0,
                   help="Max distractor patterns per exploit rollout. "
                        "Per rollout, N ~ Uniform[min, max] is sampled. "
                        "0 disables (legacy: persistent goal-only Hopfield).")
    p.add_argument("--n_train_emp_distractors_min", type=int, default=0,
                   help="Min distractors in an explore-regime Hopfield per "
                        "rollout. No goal pattern; explore policy learns "
                        "to ignore non-goal recall signals.")
    p.add_argument("--n_train_emp_distractors_max", type=int, default=0,
                   help="Max distractors per explore rollout. 0 disables.")
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
                   help="RNN hidden dim.")
    p.add_argument("--num_rnn_layers", type=int, default=1,
                   help="Number of RNN layers.")
    add_recurrent_args(p)
    p.add_argument("--goal_reward", type=float, default=1.0,
                   help="Reward at goal cell when goals_active. Bumping >1 "
                        "strengthens exploit PPO updates vs explore reward.")
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
                        "--log_std_anneal_target. 0 disables the anneal. Counted "
                        "over the whole schedule, not within a stage.")
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
    return p


def main():
    p = build_parser()
    args = p.parse_args()
    explicit = _explicit_dests(p, sys.argv[1:])

    if args.union_cov_trials:
        print("  WARNING: --union_cov_trials is ignored since 2026-08-06. "
              "Union coverage is computed by evaluate_exploration over its own "
              "rollouts, so it is taken over --n_val_trials "
              f"({args.n_val_trials}), not over {args.union_cov_trials}.",
              flush=True)

    parent_cfg = None
    if args.load_checkpoint is not None:
        ck = torch.load(args.load_checkpoint, map_location="cpu",
                        weights_only=False)
        parent_cfg = ck["config"]
        cfg = cfg_from_checkpoint(parent_cfg)
        apply_args(cfg, args, explicit)
        # Never inherited: that field holds where the parent wrote, and reusing
        # it would have this run overwrite its own parent.
        cfg.save_dir = args.save_dir
    else:
        cfg = TrainConfig()
        apply_args(cfg, args, set(CFG_FIELDS))
    settle_encoder(cfg, parent_cfg, p.error)

    if cfg.schedule is None:
        p.error("--schedule is required. It is a list of stages separated by "
                "';', e.g. --schedule 'explore:200 ; "
                "interleave:800,empty_frac=1.0->0.5,anneal=50'. Pure "
                "exploration is --schedule 'explore:600'."
                + ("" if args.load_checkpoint is None else
                   " (--load_checkpoint carried no schedule either: it "
                   "predates the field.)"))
    try:
        stages = parse_schedule(cfg.schedule)
    except ScheduleError as exc:
        p.error(str(exc))
    # Store the canonical form, so run.json says what ran rather than however
    # it happened to be typed.
    cfg.schedule = format_schedule(stages)
    cfg.n_updates = total_updates(stages)

    train_navigate(cfg, stages, load_checkpoint=args.load_checkpoint)


if __name__ == "__main__":
    main()
