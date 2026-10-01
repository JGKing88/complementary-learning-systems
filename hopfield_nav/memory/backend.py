"""The ``memory_backend`` switch: what the old code calls when it is not Hopfield.

``AgentConfig.memory_backend`` is ``"hopfield"`` (default; every existing path
unchanged) or ``"sensory_kv"`` (idea 1, docs/GRID_MLP_NAV_PLAN.md §1). The call
sites -- the rollout collector, ``TaskRegime``, ``evaluate_task`` -- branch on
``is_kv(cfg)`` and call the functions here; every evaluator that has not been
taught the new backend refuses it through ``require_hopfield``.

Stage 1 supports the task-faithful training path only (``task`` schedule stages,
oracle store). ``validate`` states the rest of the contract up front.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .readout import GridMemoryReadout
from .sensory_kv import SensoryKVMemory, omni_key

HOPFIELD = "hopfield"
SENSORY_KV = "sensory_kv"
BACKENDS = (HOPFIELD, SENSORY_KV)

# How many foreign arenas feed the distractor keys (``ForeignKeyPool``).
FOREIGN_POOL_SIZE = 256


def is_kv(cfg) -> bool:
    backend = getattr(cfg.agent, "memory_backend", HOPFIELD)
    if backend not in BACKENDS:
        raise ValueError(f"memory_backend must be one of {BACKENDS}, got {backend!r}")
    return backend == SENSORY_KV


def require_hopfield(cfg, where: str) -> None:
    """For evaluators not yet taught the sensory_kv backend: fail loudly."""
    if is_kv(cfg):
        raise NotImplementedError(
            f"{where} does not support memory_backend='sensory_kv' yet "
            "(GRID_MLP_NAV_PLAN stage 1 covers the task training path and "
            "evaluate_task only)")


def validate(cfg) -> None:
    """Refuse configurations the sensory_kv backend does not implement."""
    if not is_kv(cfg):
        return
    a, problems = cfg.agent, []
    if a.movement_mode != "continuous" or a.hopfield_mode != "continuous":
        problems.append("continuous movement and hopfield_mode")
    if not a.input_hopfield_signal:
        problems.append("input_hopfield_signal (it carries d)")
    if a.input_hopfield_multistep:
        problems.append("no input_hopfield_multistep (a second Hopfield readout)")
    if getattr(a, "input_chart_frac", False):
        problems.append("no input_chart_frac (a Hopfield-recall statistic)")
    if a.input_goal_in_memory:
        problems.append("no input_goal_in_memory")
    if getattr(cfg.hopfield, "storage_rule", "hebb") != "hebb":
        problems.append("no --hopfield_storage_rule (there is no Hopfield)")
    if getattr(a, "input_q_magnitude", False):
        problems.append("no input_q_magnitude (c on memory_conf is its analog)")
    if not a.grid_mlp_checkpoint:
        problems.append("--grid_mlp_checkpoint")
    if getattr(cfg, "training_mode", "ppo") != "ppo":
        problems.append("training_mode ppo (the BC teachers read Hopfield q)")
    if cfg.hopfield.auto_nav_warmup or cfg.hopfield.auto_store_warmup:
        problems.append("no auto_nav_warmup / auto_store_warmup")
    from ..world.env import FOVEAL_HALF_ANGLE_DEG, PANORAMA_BIN_DEG
    n_full = int(round(2 * FOVEAL_HALF_ANGLE_DEG / PANORAMA_BIN_DEG))
    if cfg.env.observation_size != n_full:
        # The key is the four-heading view binned by direction; only when the
        # ray spacing equals the panorama's bin width does it cover every bin.
        # At 12 rays it covers 48 of 180 and most reads come back exactly 0.
        problems.append(f"observation_size {n_full} (one ray per "
                        f"{PANORAMA_BIN_DEG:g}-degree panorama slice)")
    if not getattr(cfg.env, "distal_amp", 0.0):
        problems.append("--distal_amp > 0 (without the panorama no sensory key "
                        "matches away from the goal; plan §2)")
    if problems:
        raise ValueError("memory_backend='sensory_kv' needs: " + "; ".join(problems))


@lru_cache(maxsize=4)
def _readout(ckpt: str, lambdas: tuple, fwhm: float, npos: int, scale: bool):
    from types import SimpleNamespace
    cfg = SimpleNamespace(
        agent=SimpleNamespace(grid_mlp_checkpoint=ckpt, scale_q_by_c=scale),
        vectorhash=SimpleNamespace(lambdas=list(lambdas), Npos=npos),
        fwhm_ratio=fwhm)
    return GridMemoryReadout.from_config(cfg)


def readout_for(cfg) -> GridMemoryReadout:
    """One readout per (checkpoint, scaffold, flag), shared by every caller."""
    vh = cfg.vectorhash
    npos = vh.Npos or int(np.prod(vh.lambdas))
    return _readout(cfg.agent.grid_mlp_checkpoint, tuple(vh.lambdas),
                    float(cfg.fwhm_ratio), int(npos), bool(cfg.agent.scale_q_by_c))


class ForeignKeyPool:
    """Goal keys of arenas that are not this one: the distractor keys.

    The Hopfield distractors are encodings of scaffold cells outside the arena.
    The sensory analogue needs a *key* too -- what another arena's goal looks
    like -- so this builds ``FOREIGN_POOL_SIZE`` arenas with the run's env
    config (panorama included) from their own seed stream and keys each at its
    goal. A distractor is one pool key paired with the grid code of a random
    outside-arena cell.
    """

    def __init__(self, env_cfg, movement_mode: str, seed: int,
                 size: int = FOREIGN_POOL_SIZE) -> None:
        from ..world.env import make_env
        rng = np.random.RandomState([int(seed) % (2 ** 32), 0xF0E1])
        seeds = rng.randint(0, 2 ** 31 - 1, size=size)
        self.keys = np.stack([omni_key(e, e._goal) for e in
                              (make_env(env_cfg, movement_mode, seed=int(s))
                               for s in seeds)])


@lru_cache(maxsize=4)
def _pool(env_cfg_items: tuple, movement_mode: str, seed: int) -> ForeignKeyPool:
    from ..config import EnvConfig
    return ForeignKeyPool(EnvConfig(**dict(env_cfg_items)), movement_mode, seed)


def foreign_pool_for(cfg, env_size: int | None = None) -> ForeignKeyPool:
    import dataclasses
    env_cfg = cfg.env if env_size is None else dataclasses.replace(cfg.env, size=int(env_size))
    items = tuple(sorted(dataclasses.asdict(env_cfg).items()))
    return _pool(items, cfg.agent.movement_mode, int(cfg.seed))


def sample_offarena_cells(Npos: int, offset, env_size: int, n: int,
                          rng: np.random.RandomState) -> np.ndarray:
    """``(n, 2)`` global cells outside the arena -- ``sample_distractors``'s
    rejection loop, returning positions instead of encodings."""
    cx, cy = offset
    out = []
    while len(out) < n:
        gx, gy = rng.randint(0, Npos), rng.randint(0, Npos)
        if cx <= gx < cx + env_size and cy <= gy < cy + env_size:
            continue
        out.append((gx, gy))
    return np.array(out, dtype=np.int64).reshape(n, 2)


def new_kv_memory(cfg, env_offset, env_size: int, n_distractors: int,
                  rng: np.random.RandomState) -> SensoryKVMemory:
    """A fresh store holding ``n_distractors`` foreign goals."""
    ro = readout_for(cfg)
    mem = SensoryKVMemory(value_dim=ro.grid_code.Ng)
    if n_distractors > 0:
        pool = foreign_pool_for(cfg, env_size)
        cells = sample_offarena_cells(ro.grid_code.Npos, env_offset, env_size,
                                      n_distractors, rng)
        vals = ro.grid_code.at(cells[:, 0], cells[:, 1])
        for i in range(n_distractors):
            mem.write(pool.keys[rng.randint(len(pool.keys))], vals[i])
    return mem


__all__ = [
    "BACKENDS", "ForeignKeyPool", "HOPFIELD", "SENSORY_KV", "foreign_pool_for",
    "is_kv", "new_kv_memory", "readout_for", "require_hopfield",
    "sample_offarena_cells", "validate",
]
