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

import torch

from ..policy.agent import NavAgent
from ..config import TrainConfig
from ..world.env import GridEnv, make_env
from ..evaluation.metrics import (
    evaluate_exploration, evaluate_goal_discovery, evaluate_navigation,
)
from hopfield import Hopfield
from ..world.scaffold import VectorHash


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
    set_requires_grad(store_params(agent), not freeze_store)
    set_requires_grad(value_params(agent), not freeze_value)
    set_requires_grad(rnn_params(agent), not freeze_rnn)


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
    if use_wandb:
        import wandb
        log = {}
        for n_d in dist:
            for k, v in nav[n_d].items(): log[f"eval/nav_{n_d}/{k}"] = v
            for k, v in disc[n_d].items(): log[f"eval/disc_{n_d}/{k}"] = v
            # union_coverage / redundancy now arrive inside expl.
            for k, v in expl[n_d].items(): log[f"eval/expl_{n_d}/{k}"] = v
        log["phase_tag"] = update_tag
        wandb.log(log)


__all__ = [
    "do_eval", "make_hops", "move_params", "rnn_params", "set_phase_freeze",
    "set_requires_grad", "setup_world", "store_params", "value_params",
]
