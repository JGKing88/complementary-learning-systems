"""World and mode setup for the goal-conditioned pair experiment (plan §3).

Shared by the trainer (`train_goal_pairs`) and the pre-flight
(`scripts/goal_nav_preflight.py`), so the two build the same envs from the
same config and neither imports the other -- a CLI is a program, and rule 5
of the layering keeps programs unimported.
"""
from __future__ import annotations

import numpy as np

from ..config import RNNAgentConfig, RNNTrainConfig
from ..evaluation.goal_pairs import EnvTensors, aggregate_tables, evaluate_pairs
from ..utils import smooth_gbook
from ..world import generate as gen
from .rnn_setup import rnn_world

MODES = ("xy", "grid", "regular")


def agent_cfg_for_mode(mode: str, movement_mode: str, **kw) -> RNNAgentConfig:
    """The three input modes, each a complete channel configuration (plan §2.2).

        xy       [xy(p), xy(g)]            the coordinate ceiling
        grid     [gbook(p), gbook(g)]      the grid code, no sensory
        regular  [omni(p), omni(g)]        the ray-cast, heading-free
    """
    base = dict(movement_mode=movement_mode, input_prev_action=False,
                input_prev_reward=False, **kw)
    if mode == "xy":
        return RNNAgentConfig(input_sensory=False, input_xy_state=True,
                              goal_channel="abs", **base)
    if mode == "grid":
        return RNNAgentConfig(input_sensory=False, input_grid_state=True,
                              input_goal_grid_state=True, **base)
    if mode == "regular":
        return RNNAgentConfig(input_sensory=True, sensory_mode="omni",
                              goal_sensory="omni", **base)
    raise ValueError(f"unknown mode {mode!r}; one of {MODES}")


class EnvSet:
    """Envs + their precomputed tensors, under one name, for the table."""

    def __init__(self, name: str, envs, offsets, sgb) -> None:
        self.name = name
        self.envs = envs
        self.offsets = offsets
        self.tensors = [EnvTensors.build(e, o, sgb) for e, o in zip(envs, offsets)]

    def __len__(self) -> int:
        return len(self.envs)


def build_env_sets(cfg: RNNTrainConfig, rng, *, n_same: int, keep_field: bool = False):
    """Train, held-out (`base_val`) and `same` env sets, plus split, field, sgb.

    `same` is a fixed subset of the ACTUAL training envs -- same wall, same
    offset -- not a `make_val_set(same)` draw, which re-pairs walls and
    offsets. The env-side probe wants the envs themselves.
    """
    envs, offsets, split, vh, kind = rnn_world(cfg, rng)
    if kind != "declared":
        raise SystemExit("train_goal_pairs needs --env_generator: the holdouts "
                         "are defined by the declared split")
    sgb = smooth_gbook(vh.gbook, vh.lambdas, cfg.fwhm_ratio)
    train = EnvSet("train", envs, offsets, sgb)
    val_envs = gen.build_envs(split.base_val, cfg.env, "discrete")
    heldout = EnvSet("heldout", val_envs, [s.offset for s in split.base_val], sgb)
    k = min(n_same, len(envs))
    same = EnvSet("same", envs[:k], offsets[:k], sgb)
    # `sgb` is 434 x 1716 x 1716 float32 (~5 GB) and every cell this run will
    # ever read is now in the EnvTensors. Drop the field and the smoothed
    # book so the process does not hold 10 GB it never touches again; the
    # pre-flight rebuilds them itself for the checks that need the scaffold.
    if not keep_field:
        vh.gbook = None
        sgb = None
    return train, heldout, same, split, vh, sgb


def eval_all(model, sets, acfg, cells, movement_mode, device, *, n_per_quadrant, seed):
    """Aggregate quadrant table per env set."""
    model.eval()
    out = {}
    for es in sets:
        tabs = []
        for i, t in enumerate(es.tensors):
            tabs.append(evaluate_pairs(
                model, t, acfg, cells, movement_mode=movement_mode, device=device,
                env_set=es.name, n_per_quadrant=n_per_quadrant,
                rng=np.random.RandomState(seed * 1000 + i)))
        out[es.name] = aggregate_tables(tabs)
    model.train()
    return out


def jsonable(tables: dict) -> dict:
    return {es: {f"{s}x{g}": {k: (list(v) if isinstance(v, tuple) else v)
                               for k, v in row.items()}
                 for (s, g), row in agg.items()}
            for es, agg in tables.items()}
