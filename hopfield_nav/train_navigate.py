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
from .updates.ppo import ppo_update
from .evaluation.checkpoint_io import cfg_from_checkpoint
from .training.cfg_args import settle_encoder
from .training.explore import ExploreRegime
from .training.exploit import ExploitRegime
from .training.refresh import Cadence, Refresher
from .training import resume as resume_io
from .training.stages import (
    Knobs, ScheduleError, Stage, format_schedule, parse_schedule, resolve,
    stage_at, total_updates,
)
from .policy.action_head import action_bounds_from
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


def _compute_log_kappa_max(update: int, base: float, end, anneal: int) -> float:
    """Linear ramp of the kappa CEILING from `base` -> `end` over `anneal`.

    Same shape as `_compute_epsilon`, and for the same reason: it is a property
    of how far the run has got. `end is None` or `anneal <= 0` means constant,
    which is every run before 2026-09-01.
    """
    if end is None or anneal <= 0:
        return base
    t = min(1.0, max(0.0, (update - 1) / float(anneal)))
    return base + t * (float(end) - base)


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
    start_update: int = 0,
    resume_state: dict | None = None,
    parent_ckpt: str | None = None,
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

    `start_update` is the last update a resumed run completed, so the loop picks
    up at the next one. It is not merely where the counter starts: five separate
    schedules below are keyed off the global update number, and every one of
    them would otherwise re-run its opening from a mid-run policy.
    """
    n_updates_total = total_updates(stages)
    print(f"\n=== navigate: {format_schedule(stages)} "
          f"({n_updates_total} updates"
          + (f", resuming at u{start_update + 1}" if start_update else "")
          + ") ===", flush=True)

    set_phase_freeze(agent, freeze_move=False, freeze_store=cfg.freeze_store,
                     freeze_value=False, freeze_rnn=False)
    trainable = [p for p in agent.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=cfg.ppo.lr)
    # One optimizer for the whole run: stages are segments of a single training
    # trajectory, so Adam's moments carry across a boundary. A stage-level `lr`
    # retunes the existing group instead of building a new optimizer.
    current_lr = cfg.ppo.lr

    # After the freeze, because the freeze is what decides which parameters Adam
    # owns and therefore what shape its state has to be. The RNG is *not*
    # restored here -- see just above the loop for why it has to be last.
    if resume_state is not None:
        resume_io.restore_optimizer(
            optimizer, resume_state["optimizer_state_dict"],
            source=resume_state["_path"])
        print(f"Restored optimizer moments from {resume_state['_path']}",
              flush=True)

    # Captured once: the resume points written below have to name the wandb run
    # they belong to, so a continuation can reattach to it rather than opening
    # a second one.
    wandb_id = None
    if use_wandb:
        import wandb
        wandb_id = getattr(wandb.run, "id", None)

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
    # `getattr(..., None) is not None`, not `hasattr`: under
    # state_dependent_std the attribute exists and holds None, so hasattr is
    # True and the .detach() below dies.
    log_std_init_val = (
        float(agent.movement_log_std.detach().mean().item())
        if getattr(agent, "movement_log_std", None) is not None else None)

    # A knob that is accepted, echoed and then silently discarded is the most
    # expensive kind of bug this project has: `--freeze_log_std` did nothing on
    # this trainer for the whole v35 lineage, and `--epsilon_explore 0.3` on an
    # exploit-only schedule burned a run producing numbers bit-identical to its
    # control. Epsilon reaches the ROLLOUT only through the explore regime --
    # `exploit.py` hard-zeros it, because with the goal already in memory a
    # random action is a wasted step rather than exploration.
    if cfg.epsilon_explore > 0 and not any(
            s.kind in ("explore", "interleave") for s in stages):
        print(f"  WARNING: --epsilon_explore {cfg.epsilon_explore} is INERT for "
              f"this schedule. Epsilon applies to explore rollouts only, and "
              f"this run has none ({', '.join(s.kind for s in stages)}). The "
              f"value will be ignored -- do not read it as an active knob.",
              flush=True)

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

    # Last thing before the first update, and deliberately so. The saved state
    # is the stream as it stood at the *end* of an update, so every draw between
    # here and the loop is one the original run made before its first update,
    # not after its last -- restoring any earlier lets the setup below consume
    # from the restored stream and puts the continuation a few draws out of step
    # forever. Measured: restoring above the ExploreRegime constructor diverged
    # the weights at 8e-3 by the second continued update.
    if resume_state is not None:
        resume_io.restore_rng(resume_state.get("rng"))
        if dist_rng is not None and resume_state.get("dist_rng") is not None:
            dist_rng.set_state(resume_state["dist_rng"])
        resume_io.restore_env_rng(resume_state.get("env_rng"), worlds, eval_world)
        print(f"Restored RNG streams (global, distractor, per-env); "
              f"continuing at u{start_update + 1}", flush=True)

    for update in range(start_update + 1, n_updates_total + 1):
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
                # No global parameter to anneal under a state-dependent head;
                # the schedule would have to become a bias offset, which is a
                # separate design question. Left unapplied rather than
                # half-applied.
                if getattr(agent, "movement_log_std", None) is not None:
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

        # The kappa ceiling for this update. Assigned onto the head rather
        # than passed, because it is read at forward time and nothing else
        # needs to know about it.
        if getattr(cfg.agent, "log_kappa_max_end", None) is not None:
            _lkm = _compute_log_kappa_max(
                update, cfg.agent.log_kappa_max,
                cfg.agent.log_kappa_max_end,
                cfg.agent.log_kappa_anneal_updates)
            _head = getattr(agent, "polar_head", None)
            if _head is not None:
                _head.log_kappa_max = _lkm

        if knobs.lr != current_lr:
            for group in optimizer.param_groups:
                group["lr"] = knobs.lr
            current_lr = knobs.lr

        n_emp_now = int(round(n_envs * knobs.empty_frac))
        n_pre_now = n_envs - n_emp_now

        # WHICH envs are exploit, as opposed to how many. Positionally, the
        # count alone decides: the first `n_pre_now` are exploit every update,
        # so at a fixed `empty_frac` a given env is in the same regime for the
        # whole run. That is a memorization channel -- the policy can learn
        # "this env's walls mean the recall signal is trustworthy" and gate on
        # env identity instead of on the signal, which is exactly the skill the
        # interleaved schedule exists to teach, and which does not transfer to
        # a held-out env.
        #
        # `shuffle` re-draws the assignment every update from the run's own RNG,
        # so an env is exploit on some updates and explore on others and its
        # identity carries no information about its regime. `index` is the
        # historical behaviour and stays the default, because every run before
        # 2026-08-14 was trained under it.
        if cfg.regime_assignment == "shuffle":
            is_pre = np.zeros(n_envs, dtype=bool)
            is_pre[np.random.permutation(n_envs)[:n_pre_now]] = True
        else:
            is_pre = np.arange(n_envs) < n_pre_now

        rollouts = []
        pre_flags: list[bool] = []
        for w_idx, world in enumerate(worlds):
            vh = world.field
            collector = RolloutCollector(vh, cfg, embed_dim, device)
            for local_idx, env in enumerate(world.envs):
                env_offset = world.offsets[local_idx]
                regime = (exploit_regime if is_pre[local_idx]
                          else explore_regime)
                pre_flags.append(bool(is_pre[local_idx]))
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

        agent.train()
        losses = ppo_update(agent, rollouts, cfg.ppo, optimizer, aux_scale=1.0)

        mean_r = sum(r.rewards.sum().item() for r in rollouts) / max(
            sum(r.rewards.numel() for r in rollouts), 1)
        # Split by the flag recorded per rollout, not by a slice. The slice was
        # only ever correct for `index` assignment AND num_worlds == 1 -- the
        # rollout list is world-major, so `rollouts[:n_pre * n_worlds]` mixed
        # regimes as soon as there was more than one world.
        pre_rs = [r for r, pre in zip(rollouts, pre_flags) if pre]
        emp_rs = [r for r, pre in zip(rollouts, pre_flags) if not pre]
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
            if refresher is not None:
                for trait in refresher.counts:
                    log[f"train/refresh_{trait}"] = int(trait in refreshed)
            log["phase_name"] = "navigate"
            wandb.log(log)

        n_updates_timed += 1
        if update == 1 or update % 10 == 0:
            # Under a state-dependent head there is no single sigma to print;
            # the per-update `sigma` in the PPO stats is the realized one, so
            # fall back to it rather than inventing a number here.
            log_std_mean = (
                float(agent.movement_log_std.exp().mean().item())
                if getattr(agent, "movement_log_std", None) is not None
                else float(losses.get("sigma", float("nan"))))
            s_per_update = (time.time() - t_update_mark) / max(n_updates_timed, 1)
            print(f"  u{update}({stage.kind}): "
                  f"mean_r={mean_r:.4f} (pre={_mr(pre_rs):.4f}, "
                  f"emp={_mr(emp_rs):.4f}) nov={knobs.novelty:.3f} "
                  f"emp_frac={knobs.empty_frac:.3f} std={log_std_mean:.3f} "
                  f"s/u={s_per_update:.1f} | "
                  + " ".join(f"{k}={v:.3f}" for k, v in losses.items())
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
            # The rolling resume point, beside the fork point rather than inside
            # it -- see training/resume.py for why the two are separate files.
            # `parent` rides along so a continuation of a continuation still
            # draws its world against the original ancestor's exclusion union.
            resume_io.save(cfg.save_dir, kind="navigate", agent=agent,
                           optimizer=optimizer, update=update,
                           config=ckpt_config, world_spec=ckpt_world,
                           extra={"parent": parent_ckpt,
                                  "wandb_id": wandb_id,
                                  # Its own stream, advanced once per distractor
                                  # draw, so it is not covered by the global
                                  # numpy state and has to be carried too.
                                  "dist_rng": (dist_rng.get_state()
                                               if dist_rng is not None else None),
                                  "env_rng": resume_io.env_rng_states(
                                      worlds, eval_world)})

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
    resume_ck: dict | None = None,
) -> None:
    """Run the schedule. `load_checkpoint` forks a parent; `resume_ck` continues.

    Exactly one of the two, or neither. See `training/resume.py` for why they
    are different operations rather than two spellings of one.
    """
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
    print(f"encoder gain {encoder_gain:g} (code sharpness)   "
          f"hopfield beta {cfg.hopfield.beta:g} (recall sharpness)")

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
    # A continuation is the same run, so it inherits the *original* run's
    # parent -- not None, or `generate_split` would draw without the ancestor's
    # exclusion union and hand the second segment a different world than the
    # first trained on.
    start_update = int(resume_ck["update"]) if resume_ck is not None else 0
    parent_ckpt = (resume_ck.get("parent") if resume_ck is not None
                   else load_checkpoint)

    rw = setup_run_world(cfg, encoder, embed_dim, rng, field,
                         cadence=cadence, n_updates=total_updates(stages),
                         encoder_ident=encoder_ident, where="train_navigate",
                         parent_ckpt=parent_ckpt)
    worlds, eval_world, split = rw.worlds, rw.eval_world, rw.split

    # The worlds above are the run's envs as of update 1. Walk the refresher's
    # ticks forward to where the interruption happened, or the second segment
    # trains on envs the first one had already moved on from.
    if resume_ck is not None and rw.refresher is not None:
        ticks = rw.refresher.fast_forward(start_update)
        print(f"Refresher fast-forwarded through u{start_update}: {ticks} tick(s) "
              f"replayed, used union now "
              f"{rw.refresher.report()['n_used']}", flush=True)

    input_dim = compute_input_dim(cfg.agent, embed_dim, cfg.env.observation_size)
    print(f"Agent input_dim={input_dim} init_log_std={cfg.agent.init_log_std}",
          flush=True)
    agent = NavAgent(cfg.agent, input_dim,
                     action_bounds=action_bounds_from(cfg.env)).to(device)

    if load_checkpoint is not None:
        ck = torch.load(load_checkpoint, map_location=device, weights_only=False)
        agent.load_state_dict(ck["agent_state_dict"])
        print(f"Loaded agent state from {load_checkpoint}", flush=True)
    elif resume_ck is not None:
        agent.load_state_dict(resume_ck["agent_state_dict"])
        print(f"Continuing {resume_ck['_path']} from u{start_update}", flush=True)

    if cfg.use_wandb:
        import wandb
        # Same run, same wandb run: a continuation that opened a second one
        # would split a single training curve across two charts and restate its
        # x-axis from 0.
        wandb_id = resume_ck.get("wandb_id") if resume_ck is not None else None
        wandb.init(project=cfg.wandb_project, config=asdict(cfg),
                   **({"id": wandb_id, "resume": "allow"} if wandb_id else {}))

    if cfg.save_dir is None:
        sub = run_name(*((wandb.run.name, wandb.run.id) if cfg.use_wandb else ()))
        cfg.save_dir = str(run_dir("navigate", sub))
    else:
        sub = os.path.basename(str(cfg.save_dir).rstrip("/"))

    if resume_ck is not None:
        # Not `begin`: that would drop the checkpoint list this run is still
        # adding to and restate `created` as the moment it was interrupted.
        run_manifest.resume(
            cfg.save_dir, update=start_update, config=asdict(cfg),
            wandb_run=wandb.run if cfg.use_wandb else None)
    else:
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
        start_update=start_update, resume_state=resume_ck,
        parent_ckpt=parent_ckpt,
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
    "input_prev_displacement": ("agent.input_prev_displacement",),
    "action_squash": ("agent.action_squash",),
    "state_dependent_std": ("agent.state_dependent_std",),
    "log_std_min": ("agent.log_std_min",),
    "log_std_max": ("agent.log_std_max",),
    "action_polar": ("agent.action_polar",),
    "init_log_kappa": ("agent.init_log_kappa",),
    "log_kappa_min": ("agent.log_kappa_min",),
    "log_kappa_max": ("agent.log_kappa_max",),
    "log_kappa_max_end": ("agent.log_kappa_max_end",),
    "log_kappa_anneal_updates": ("agent.log_kappa_anneal_updates",),
    "init_speed_mu": ("agent.init_speed_mu",),
    "init_speed_nu": ("agent.init_speed_nu",),
    "speed_nu_min": ("agent.speed_nu_min",),
    "speed_nu_max": ("agent.speed_nu_max",),
    "speed_mu_eps": ("agent.speed_mu_eps",),
    "dir_soft": ("agent.dir_soft",),
    "freeze_speed": ("agent.freeze_speed",),
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
    "persistence_realized": ("hopfield.persistence_realized",),
    "novelty_scale_remaining": ("hopfield.novelty_scale_remaining",),
    "novelty_scale_cap": ("hopfield.novelty_scale_cap",),
    # run structure
    "encoder_checkpoint": ("encoder_checkpoint",),
    "encoder_gain": ("encoder_gain",),
    "hopfield_beta": ("hopfield.beta",),
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
    "regime_assignment": ("regime_assignment",),
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
    p.add_argument("--action_squash", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Radial tanh on the policy MEAN, mapping ||mu|| into "
                        "[min_action_norm, max_action_norm]. Squashing the mean "
                        "rather than the sample keeps the distribution Gaussian, "
                        "so no log-prob Jacobian is needed. Fixes the unbounded "
                        "||mu|| drift a hard env clamp permits.")
    p.add_argument("--state_dependent_std", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Per-state log_std head instead of one global parameter. "
                        "Initialized to reproduce the global-sigma policy exactly.")
    p.add_argument("--log_std_min", type=float, default=-2.5)
    p.add_argument("--log_std_max", type=float, default=0.5)
    p.add_argument("--action_polar", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Heading x speed as SEPARATE distributions -- "
                        "VonMises(theta, kappa) on the allocentric heading and "
                        "a Beta on speed over [min,max]_action_norm -- instead "
                        "of one isotropic Cartesian Gaussian. Decouples "
                        "directional exploration from speed, which sigma/||mu|| "
                        "cannot. Under this flag --state_dependent_std and "
                        "--freeze_log_std govern kappa and nu; the speed mean "
                        "stays learnable.")
    p.add_argument("--init_log_kappa", type=float, default=1.85,
                   help="kappa=6.34, matching the Cartesian init sigma=exp(-0.7) "
                        "at mid-speed 1.25 (~23.8 deg of directional noise).")
    p.add_argument("--log_kappa_min", type=float, default=-1.0)
    p.add_argument("--log_kappa_max_end", type=float, default=None,
                   help="Ramp log_kappa_max to this over "
                        "--log_kappa_anneal_updates. The cap is a "
                        "training-time device (kappa does not affect a "
                        "deterministic action, P2 §20.1): on early for "
                        "exploit's policy-space exploration, off late so the "
                        "mean policy is optimized nearer deployment (§24).")
    p.add_argument("--log_kappa_anneal_updates", type=int, default=0,
                   help="Updates over which to ramp log_kappa_max -> "
                        "log_kappa_max_end. 0 = constant.")
    p.add_argument("--log_kappa_max", type=float, default=5.0,
                   help="[-1, 5] -> circular sd from 106 deg down to 4.7 deg.")
    p.add_argument("--init_speed_mu", type=float, default=0.5,
                   help="NORMALIZED mean speed in (0,1); 0.5 -> 1.25 cells, "
                        "the measured billiard-coverage peak.")
    p.add_argument("--init_speed_nu", type=float, default=3.0,
                   help="Beta concentration; 3.0 -> speed sd 0.375.")
    p.add_argument("--speed_nu_min", type=float, default=2.0,
                   help="Floor forbidding a U-shaped speed density for every "
                        "mu (a U-shape needs nu < min(1/mu, 1/(1-mu)) <= 2). "
                        "One constant, so nu stays a single freezable scalar.")
    p.add_argument("--speed_nu_max", type=float, default=200.0)
    p.add_argument("--speed_mu_eps", type=float, default=0.05)
    p.add_argument("--dir_soft", type=float, default=0.05,
                   help="Softens the direction head's magnitude, which is a "
                        "gauge freedom (atan2 is scale-invariant) whose decay "
                        "would send the heading gradient to infinity. A short "
                        "direction vector becomes a LOW concentration instead. "
                        "Watch the dir_norm column: sustained values near this "
                        "mean the heading is being held near-uniform.")
    p.add_argument("--freeze_speed", type=float, default=None,
                   help="Hold speed constant at this many GRID CELLS and drop "
                        "the speed factor entirely (not a degenerate limit: "
                        "its log-prob and entropy slots are exactly zero). "
                        "All exploration becomes directional. Requires "
                        "--action_polar; inexpressible under the Cartesian "
                        "head at any parameter setting.")
    p.add_argument("--input_prev_displacement",
                   action=argparse.BooleanOptionalAction, default=False,
                   help="Feed the REALIZED displacement of the previous step "
                        "as a separate 2-D channel. Not redundant with "
                        "--input_prev_action: the norm clamp and the arena "
                        "clip both make the executed move differ from the "
                        "commanded one, and the regime cues that compare a "
                        "change in q against distance travelled need the "
                        "executed one. Continuous movement only.")
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
    p.add_argument("--persistence_realized", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Score the persistence bonus on the REALIZED "
                        "displacement rather than the commanded action. "
                        "Default off, which is what every run up to P20 "
                        "trained under. On the commanded action a "
                        "wall-pinned agent collects the full bonus for not "
                        "moving (P2 doc §18.7-18.8).")
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
                   choices=("all", "navexpl", "expl"),
                   help="Which evaluators an in-training eval runs. 'all' is "
                        "nav + goal-discovery + exploration. 'navexpl' drops "
                        "goal discovery, which is the only evaluator that "
                        "measures the store head -- a head this trainer never "
                        "trains -- and the only unbatched one, so it costs "
                        "~73 s against ~5 s for the other two together. "
                        "'expl' is exploration only, for pure-explore "
                        "schedules where the other two are undefined.")
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
    p.add_argument("--encoder_gain", type=float, default=None,
                   help="Sharpness of the encoder's output nonlinearity: the "
                        "code is normalize(tanh(gain * z)), so raising it "
                        "makes the embedding more BINARY without changing its "
                        "magnitude. Overriding it now also applies to the "
                        "model, which it previously did not -- see "
                        "encoder_io.load_encoder.")
    p.add_argument("--hopfield_beta", type=float, default=None,
                   help="Sharpness of the Hopfield recall: the update is "
                        "tanh(beta * W x). Defaults to the encoder's gain, "
                        "which is why the two were never separable before. "
                        "At the scale W x actually takes here (~1e-4) the "
                        "default leaves tanh in its linear region, so recall "
                        "is a weighted blend rather than an attractor -- see "
                        "EXPERIMENTS_NAV_P2 section 14. Raising it is how you "
                        "get a saturating, genuinely Hopfield-like recall.")
    p.add_argument("--load_checkpoint", type=str, default=None,
                   help="FORK a new run from this checkpoint's weights. Its "
                        "config becomes the base -- every setting is inherited "
                        "except the flags you pass explicitly, so a child "
                        "reproduces its parent's recipe without re-listing it. "
                        "--save_dir is never inherited. Adam's moments are NOT "
                        "inherited: a fork is expected to be retuned, and "
                        "moments from a different objective are stale. To pick "
                        "an interrupted run back up instead, use "
                        "--continue_from.")
    p.add_argument("--continue_from", type=str, default=None,
                   help="CONTINUE this run's own trajectory from a "
                        f"{resume_io.RESUME_FILE} written by an earlier "
                        "segment (pass the file or its run directory). Unlike "
                        "--load_checkpoint this is the same run: optimizer "
                        "moments, RNG streams, the global update counter and "
                        "the env refresher's tick history all come back, and "
                        "output continues into the same --save_dir and wandb "
                        "run. Its config is taken from the resume point, so no "
                        "other flag may be given except --device and a "
                        "--schedule that lengthens the original.")
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
    p.add_argument("--regime_assignment", choices=("index", "shuffle"),
                   default=None,
                   help="Which envs take the exploit regime, given how many do."
                        " 'index' (default) takes the first n_pre in order, so"
                        " at a fixed empty_frac an env keeps its regime for the"
                        " whole run and the policy can gate on env identity"
                        " instead of on the recall signal -- a shortcut that"
                        " does not transfer to a held-out env. 'shuffle'"
                        " re-draws the assignment every update.")
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

    if args.continue_from is not None and args.load_checkpoint is not None:
        p.error("--continue_from and --load_checkpoint are different "
                "operations and cannot be combined. --load_checkpoint forks a "
                "new run from a parent's weights; --continue_from picks this "
                "run's own trajectory back up. Pass one.")

    parent_cfg = None
    resume_ck = None
    if args.continue_from is not None:
        # The config comes from the resume point in full, so nothing here reads
        # `args` except the handful of flags below that describe *this segment*
        # rather than the run. `reject_overrides` is what stops a typed
        # --goal_reward from being quietly discarded.
        resume_io.reject_overrides(
            explicit, allowed=_CONTINUE_ALLOWED, flag="--continue_from")
        resume_ck = resume_io.load(args.continue_from, "cpu", kind="navigate")
        resume_ck["_path"] = args.continue_from
        cfg = cfg_from_checkpoint(resume_ck["config"])
        if "device" in explicit:
            cfg.device = args.device
        if "schedule" in explicit:
            # Checked against the original below: lengthening is allowed,
            # rewriting an update that already ran is not.
            cfg.schedule = args.schedule
    elif args.load_checkpoint is not None:
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

    if resume_ck is not None:
        _check_schedule_extends(resume_ck, stages, cfg, p.error)

    train_navigate(cfg, stages, load_checkpoint=args.load_checkpoint,
                   resume_ck=resume_ck)


# The flags that describe *this segment* rather than the run, and so may be
# typed alongside --continue_from. Everything else comes from the resume point:
# a continuation that silently accepted --goal_reward would be the same class of
# bug as an optimizer dropped without a word.
_CONTINUE_ALLOWED = frozenset({"continue_from", "device", "schedule"})


# Sentinel defaults for the schedule comparison below. `resolve` overwrites a
# field only where the stage sets one, so a knob that reads back as the sentinel
# is inherited from cfg -- identical on both sides, since a continuation's cfg
# comes from the resume point.
_PROBE = Knobs(lr=-1.0, empty_frac=-1.0, novelty=-1.0, eps=-1.0,
               dist_min=-1, dist_max=-1, emp_dist_min=-1, emp_dist_max=-1)


def _check_schedule_extends(resume_ck: dict, stages: list[Stage],
                            cfg: TrainConfig, fail) -> None:
    """A continuation may lengthen its schedule; it may not rewrite its past.

    Lengthening is the reason to retype `--schedule` at all: a run that hit the
    wall at u320 of 500 is continued by asking for the same 500, and one whose
    schedule turned out too short is continued by asking for more. What must not
    change is any update the first segment already ran, because those updates
    are in the weights already and a schedule disagreeing with them describes a
    run that never happened.

    Compared by *resolved knobs* rather than by stage identity, because the two
    come apart in both directions. `interleave:2` -> `interleave:4` at a flat
    empty_frac changes no past update, and rejecting it would refuse the most
    ordinary continuation there is. But `empty_frac=1.0->0.5` with no explicit
    `anneal=` spans the stage, so the same lengthening rescales the anneal and
    every past update silently moves -- which comparing the stage's *length*
    would catch and comparing its *kind* would not.

    `novelty_anneal` is checked separately: it is keyed off the run total rather
    than off any stage, so lengthening the schedule moves it even when every
    stage resolves identically.
    """
    start_update = int(resume_ck["update"])
    old_schedule = resume_ck["config"].get("schedule")
    if not old_schedule:
        return
    old_stages = parse_schedule(old_schedule)
    new_total, old_total = total_updates(stages), total_updates(old_stages)

    if new_total < start_update:
        fail(f"--continue_from: the resume point is at u{start_update}, but "
             f"--schedule '{format_schedule(stages)}' runs only {new_total} "
             "updates. A continuation cannot be shorter than what has already "
             "run.")

    if cfg.novelty_anneal and new_total != old_total and start_update > 0:
        fail(f"--continue_from: --novelty_anneal scales novelty by "
             f"1-(u-1)/n_updates, so changing the run total from {old_total} to "
             f"{new_total} changes the novelty every one of the {start_update} "
             "updates already run was trained with. Continue at the original "
             "length, or fork with --load_checkpoint.")

    for u in range(1, start_update + 1):
        old_stage, old_local = stage_at(old_stages, u)
        new_stage, new_local = stage_at(stages, u)
        if old_stage.kind != new_stage.kind:
            fail(f"--continue_from: --schedule '{format_schedule(stages)}' puts "
                 f"u{u} in a '{new_stage.kind}' stage, but the run it continues "
                 f"ran it as '{old_stage.kind}' (was '{old_schedule}').")
        if resolve(old_stage, old_local, _PROBE) != resolve(new_stage, new_local,
                                                            _PROBE):
            fail(f"--continue_from: --schedule '{format_schedule(stages)}' "
                 f"changes the knobs u{u} already ran with (was "
                 f"'{old_schedule}'):\n"
                 f"    was {resolve(old_stage, old_local, _PROBE)}\n"
                 f"    now {resolve(new_stage, new_local, _PROBE)}\n"
                 "  A continuation may add updates to the end of a schedule but "
                 "cannot change one that has already run. To train the same "
                 "weights under a different schedule, fork with "
                 "--load_checkpoint.")


if __name__ == "__main__":
    main()
