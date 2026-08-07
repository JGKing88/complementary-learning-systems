"""World construction, head freezing and the phase-boundary eval.

`train_phase_a_only` and `train_phase_b_only` need all four of these, and until
this module they imported them from `train_phased` -- one entry point reaching
into another for its internals, which is why the three could not be reasoned
about separately and why moving any of them meant checking two other CLIs.

None of it is phase-specific: `setup_world` builds envs plus a scaffold,
`make_hops` builds the per-env Hopfield arrangement a phase asks for by name,
`set_phase_freeze` toggles `requires_grad` on the four parameter groups, and
`do_eval` runs the standard evaluator set and logs it. The *choice* of role,
freeze mask and schedule stays in each entry point, because that choice is what
distinguishes them.
"""
from __future__ import annotations

import time

import torch

from ..policy.agent import NavAgent
from ..config import TrainConfig
from ..world.env import make_env
from ..evaluation.metrics import (
    evaluate_exploration, evaluate_goal_discovery, evaluate_navigation,
)
from hopfield import Hopfield
from ..world.scaffold import VectorHash, goal_encodings
from ..world.world import World, build_world


# ---------------------------------------------------------------------------
# World setup (reused across phases)
# ---------------------------------------------------------------------------

def build_field(cfg: TrainConfig, encoder) -> VectorHash:
    """The scaffold field, built once and shared by every world.

    It is a pure function of ``(lambdas, Npos, fwhm_ratio, encoder)``, so the
    per-world copies this used to make were bit-identical -- and 12 GB each at
    ``Npos=1716, out_dim=1024``.
    """
    field = VectorHash(cfg.vectorhash)
    field.build_scaffold()
    field.precompute_encoded_phi(encoder, cfg.fwhm_ratio, device=cfg.device)
    return field


def setup_world(cfg: TrainConfig, encoder, embed_dim, rng, role: str = "train",
                field: VectorHash | None = None) -> World:
    """Build envs and place them in a scaffold. Same for train + eval worlds;
    role just controls count (envs_per_world vs num_val_envs).

    ``field=None`` builds one, which is what a single-world caller wants. Pass a
    field to share it -- the train worlds and the eval world should always share
    one.
    """
    n = cfg.envs_per_world if role == "train" else cfg.num_val_envs
    envs = [
        make_env(cfg.env, cfg.agent.movement_mode,
                 seed=int(rng.randint(0, 10_000_000)))
        for _ in range(n)
    ]
    if field is None:
        field = build_field(cfg, encoder)
    return build_world(field, envs, placement="spread", size=cfg.env.size)


def make_hops(
    role: str,
    cfg: TrainConfig,
    world: World,
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
    envs = world.envs
    if role == "pre_stored_shared":
        per_env_templates = []
        for pattern in goal_encodings(world.field, envs, world.offsets):
            hop = Hopfield(embed_dim, beta=cfg.hopfield.beta, device=str(device))
            hop.input_memory(torch.from_numpy(pattern).float())
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

def set_requires_grad(params, flag: bool):
    for p in params:
        p.requires_grad_(flag)


def move_params(agent: NavAgent) -> list[torch.nn.Parameter]:
    if agent.cfg.movement_mode == "discrete":
        return list(agent.movement_head.parameters())
    return list(agent.movement_mean.parameters()) + [agent.movement_log_std]


def store_params(agent: NavAgent) -> list[torch.nn.Parameter]:
    return list(agent.store_head.parameters())


def value_params(agent: NavAgent) -> list[torch.nn.Parameter]:
    return list(agent.value_head.parameters())


def rnn_params(agent: NavAgent) -> list[torch.nn.Parameter]:
    return list(agent.rnn.parameters())


def set_phase_freeze(agent: NavAgent, freeze_move: bool,
                     freeze_store: bool, freeze_value: bool, freeze_rnn: bool):
    set_requires_grad(move_params(agent), not freeze_move)
    # `movement_log_std` is a movement parameter, so unfreezing the movement
    # head re-enables its gradient and silently undoes `--freeze_log_std` --
    # which every caller does, because no phase freezes movement. The flag was
    # therefore a no-op on train_navigate: gentle-terrain-124's lineage ran
    # with a *learnable* log_std despite asking for a frozen one, visible as
    # std drifting 0.166 -> 0.294 over 250 updates. The agent's own config is
    # the authority; a phase mask must not overrule it.
    if getattr(agent.cfg, "freeze_log_std", False) \
            and hasattr(agent, "movement_log_std"):
        agent.movement_log_std.requires_grad = False
    set_requires_grad(store_params(agent), not freeze_store)
    set_requires_grad(value_params(agent), not freeze_value)
    set_requires_grad(rnn_params(agent), not freeze_rnn)


# ---------------------------------------------------------------------------
# Eval wrapper used at phase boundaries
# ---------------------------------------------------------------------------

def do_eval(cfg, agent, eval_world: World, device, update_tag: str,
            use_wandb: bool, max_steps: int = 200) -> None:
    val_envs = eval_world.envs
    val_vh = eval_world.field
    val_offsets = eval_world.offsets
    dist = cfg.val_n_distractors_list
    nt = cfg.n_val_trials

    # "expl" skips the two evaluators a pure-explore run cannot be scored on.
    # They stay in the wandb log as empty dicts rather than as stale values, so
    # a scope switch mid-project cannot be mistaken for a collapse in nav.
    expl_only = getattr(cfg, "eval_scope", "all") == "expl"

    t0 = time.time()
    nav = {} if expl_only else evaluate_navigation(
        agent, val_envs, val_vh, val_offsets, cfg, device,
        num_trials=nt, max_steps=max_steps,
        n_distractors_list=dist, deterministic=True)
    disc = {} if expl_only else evaluate_goal_discovery(
        agent, val_envs, val_vh, val_offsets, cfg, device,
        num_trials=nt, max_steps=max_steps, n_distractors_list=dist)
    expl = evaluate_exploration(agent, val_envs, val_vh, val_offsets, cfg, device,
                                num_trials=nt, max_steps=max_steps,
                                n_distractors_list=dist)
    eval_s = time.time() - t0
    if not expl_only:
        print(f"  [{update_tag}] nav={nav}")
        print(f"  [{update_tag}] disc={disc}")
    print(f"  [{update_tag}] expl={expl}")
    # Sizing a run needs the eval's own cost, not just the per-update total it
    # is folded into -- see docs/EXPERIMENTS_SCHEDULE_REPRO.md on how badly a
    # run can be mis-sized when that number has to be inferred after the fact.
    print(f"  [{update_tag}] eval_seconds={eval_s:.1f} scope="
          f"{'expl' if expl_only else 'all'}", flush=True)
    if use_wandb:
        import wandb
        log = {"eval/eval_seconds": eval_s}
        for n_d in dist:
            for k, v in nav.get(n_d, {}).items(): log[f"eval/nav_{n_d}/{k}"] = v
            for k, v in disc.get(n_d, {}).items(): log[f"eval/disc_{n_d}/{k}"] = v
            # union_coverage / redundancy now arrive inside expl.
            for k, v in expl[n_d].items(): log[f"eval/expl_{n_d}/{k}"] = v
        log["phase_tag"] = update_tag
        wandb.log(log)


__all__ = [
    "build_field", "do_eval", "make_hops", "move_params", "rnn_params",
    "set_phase_freeze", "set_requires_grad", "setup_world", "store_params",
    "value_params",
]
