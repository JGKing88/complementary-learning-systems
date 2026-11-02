"""The ``memory_backend`` switch: what the old code calls when it is not Hopfield.

``AgentConfig.memory_backend`` is ``"hopfield"`` (default; every existing path
unchanged), ``"sensory_kv"`` (idea 1, docs/GRID_MLP_NAV_PLAN.md §1) or
``"none"`` (idea 3, §7: no memory module; the GRU's hidden state is kept
wherever a memory would be). The call sites -- the rollout collector,
``TaskRegime``, ``evaluate_task`` -- branch on ``is_kv(cfg)`` / ``is_none(cfg)``
and call the functions here; every evaluator that has not been taught the new
backends refuses them through ``require_hopfield``.

Stage 1 supports the task-faithful training path only (``task`` schedule stages,
oracle store). ``validate`` states the rest of the contract up front.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .readout import GridMemoryReadout
from .sensory_kv import KVEntry, SensoryKVMemory, omni_key

HOPFIELD = "hopfield"
SENSORY_KV = "sensory_kv"
NONE = "none"
BACKENDS = (HOPFIELD, SENSORY_KV, NONE)

# How many foreign arenas feed the distractor keys (``ForeignKeyPool``).
FOREIGN_POOL_SIZE = 256


def _backend(cfg) -> str:
    backend = getattr(cfg.agent, "memory_backend", HOPFIELD)
    if backend not in BACKENDS:
        raise ValueError(f"memory_backend must be one of {BACKENDS}, got {backend!r}")
    return backend


def is_kv(cfg) -> bool:
    return _backend(cfg) == SENSORY_KV


def is_none(cfg) -> bool:
    """Idea 3: no memory module. Wherever a memory would persist (across the
    visits of a task sequence, across the continual protocol), the policy's
    hidden state persists instead."""
    return _backend(cfg) == NONE


def require_hopfield(cfg, where: str) -> None:
    """For evaluators not yet taught the other backends: fail loudly."""
    backend = _backend(cfg)
    if backend != HOPFIELD:
        raise NotImplementedError(
            f"{where} does not support memory_backend={backend!r} yet "
            "(GRID_MLP_NAV_PLAN covers the task training path and the "
            "evaluators taught it)")


def validate_none(cfg) -> None:
    """Refuse configurations the no-memory backend (idea 3) does not implement."""
    a, problems = cfg.agent, []
    if a.input_hopfield_signal:
        problems.append("--no-input_hopfield_signal (there is no memory readout)")
    if a.input_hopfield_multistep:
        problems.append("no input_hopfield_multistep")
    if getattr(a, "input_chart_frac", False):
        problems.append("no input_chart_frac")
    if a.input_goal_in_memory:
        problems.append("no input_goal_in_memory")
    if getattr(a, "input_q_magnitude", False):
        problems.append("no input_q_magnitude")
    if getattr(cfg, "training_mode", "ppo") != "ppo":
        problems.append("training_mode ppo (the BC teachers read Hopfield q)")
    if getattr(cfg, "hopfield_oracle", False):
        problems.append("no hopfield_oracle")
    if cfg.hopfield.auto_nav_warmup:
        problems.append("no auto_nav_warmup (it follows a memory readout)")
    if a.movement_mode != "continuous":
        problems.append("continuous movement")
    if getattr(cfg.env, "reset_state_on_teleport", False):
        # The state is the memory; PPO also replays each sequence from its
        # start without resets.
        problems.append("no reset_state_on_teleport")
    if problems:
        raise ValueError("memory_backend='none' needs: " + "; ".join(problems))


def validate(cfg) -> None:
    """Refuse configurations the sensory_kv and none backends do not implement."""
    if is_none(cfg):
        validate_none(cfg)
        return
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
    if getattr(cfg, "hopfield_oracle", False):
        problems.append("no hopfield_oracle (an encoded_Phi oracle)")
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


@lru_cache(maxsize=4)
def _grid_code(lambdas: tuple, fwhm: float, npos: int):
    from .grid_code import SmoothedGridCode
    return SmoothedGridCode(list(lambdas), fwhm, npos)


def grid_code_for(cfg):
    """The smoothed grid code on the run's scaffold, shared by every caller
    (the ``grid_code`` input channel; the same code idea 1's MLP reads)."""
    vh = cfg.vectorhash
    npos = vh.Npos or int(np.prod(vh.lambdas))
    return _grid_code(tuple(int(l) for l in vh.lambdas), float(cfg.fwhm_ratio), int(npos))


def grid_code_dim(cfg) -> int:
    return sum(int(l) ** 2 for l in cfg.vectorhash.lambdas)


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


def kv_entry_at(cfg, env, cell, env_offset) -> KVEntry:
    """What a store at ``cell`` writes: its four-heading key, its grid code."""
    ro = readout_for(cfg)
    return KVEntry(omni_key(env, cell), ro.goal_value(cell, env_offset))


def kv_distractor_entries(cfg, env_offset, env_size: int, n: int,
                          rng: np.random.RandomState) -> list[KVEntry]:
    """``n`` foreign goals for the evaluators, drawn so ``rng`` advances
    exactly as ``rollout.distractors.sample_distractors`` advances it.

    The evaluators draw distractors and then trial starts from one stream; a
    different consumption here would hand the sensory_kv model different
    starts from the Hopfield model's. Cells come from the same rejection loop
    (``sample_offarena_cells``); which foreign arena's key each cell gets is
    drawn from a second stream seeded by the cells, so it never touches
    ``rng``.
    """
    ro = readout_for(cfg)
    cells = sample_offarena_cells(ro.grid_code.Npos, env_offset, env_size, n, rng)
    if n == 0:
        return []
    keys = foreign_pool_for(cfg, env_size).keys
    side = np.random.RandomState(
        int((cells[:, 0] * 7919 + cells[:, 1] * 104729).sum() + n) % (2 ** 32))
    vals = ro.grid_code.at(cells[:, 0], cells[:, 1])
    return [KVEntry(keys[side.randint(len(keys))], vals[i]) for i in range(n)]


def kv_memory_with(cfg, entries) -> SensoryKVMemory:
    """A fresh store holding ``entries`` in order."""
    mem = SensoryKVMemory(value_dim=readout_for(cfg).grid_code.Ng)
    for e in entries:
        mem.input_memory(e)
    return mem


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
    "BACKENDS", "ForeignKeyPool", "HOPFIELD", "KVEntry", "NONE", "SENSORY_KV",
    "foreign_pool_for", "grid_code_dim", "grid_code_for", "is_kv", "is_none", "kv_distractor_entries", "kv_entry_at",
    "kv_memory_with", "new_kv_memory", "readout_for", "require_hopfield",
    "sample_offarena_cells", "validate", "validate_none",
]
