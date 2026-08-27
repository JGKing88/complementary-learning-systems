"""Shared setup: the encoder, the worlds, the memory, and the cell bank.

Everything here is what Tests A-D have in common, and the parts of it that are
easy to get subtly wrong are the ones with comments.

Three deviations from a naive reading of the spec, each deliberate:

``EnvSpec`` is sampled directly; no ``GridEnv`` is ever built.
    An env reaches this harness as ``(wall_seed, size, offset, goal)`` and
    nothing here reads a wall code -- there is no sensory channel in any of
    these probes, and Test D's clipping is the size box, not the barcode.
    Building 2500 ``GridEnv`` objects would raycast a four-heading sensory
    codebook per cell for nothing. ``wall_seed`` is still drawn and recorded so
    a world can be replayed into real envs later.

``generate_split`` is not used.
    It exists to hold traits out between a train and a val set, and this
    harness has no training and no holdout -- every env is a test env. What is
    reused is the part that carries meaning: ``EnvSpec`` itself and
    ``place_envs``, so placement is the same ``spread`` lattice a real world
    gets.

The storage shuffle is done once, not averaged over.
    ``W`` is a sum of rank-1 terms with the diagonal zeroed, so pattern order
    changes the result only through float accumulation order. Shuffling once
    per (world, K) keeps the spec's intent -- no privileged first pattern --
    without pretending an average over orders would measure anything.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch

from encoder_training.config import EncoderModelConfig
from hopfield import Hopfield
from hopfield_nav.world.scaffold import place_envs
from hopfield_nav.world.spec import EnvSpec

from .encode import Field

MEMORY_MODES = ("multi_env_goals", "goal+distractors", "same_env_goals")

# Outcome categories for Test A retrieval, in severity order. The ordering is
# load-bearing: the report colours the first three as one sequential ramp on
# `retr_dist` and the last two as their own hues.
OUTCOMES = ("exact", "near", "far_same_env", "other_env", "alias")
NEAR_CELLS = 2.0            # retr_dist <= this and not exact -> "near"


# ---------------------------------------------------------------------------
# Encoder loading
# ---------------------------------------------------------------------------

def _dig(ckpt: dict, key: str):
    """Find ``key`` wherever this checkpoint generation happened to put it."""
    for holder in ("train_config", "model_config", "config"):
        sub = ckpt.get(holder)
        if isinstance(sub, dict):
            if key in sub:
                return sub[key]
            inner = sub.get("model_params")
            if isinstance(inner, dict) and key in inner:
                return inner[key]
    return ckpt.get(key)


def load_probe_encoder(
    path: str,
    *,
    device: str = "cpu",
    fwhm_override: float | None = None,
    fwhm_fallback: float | None = None,
) -> tuple[torch.nn.Module, EncoderModelConfig, float, float, dict]:
    """Load an encoder and *inherit* its ``gain`` and ``fwhm_ratio``.

    Neither is ever a CLI default here. ``gain`` already resolves from the
    checkpoint in production (``encoder_io.load_encoder``). ``fwhm_ratio`` does
    not: it lives in ``ckpt["train_config"]``, which ``EncoderModelConfig``
    filters out, so ``train_navigate`` supplies it from an argparse default of
    0.25 -- and ``encoder_io.validate_config`` accepts a ``fwhm_ratio``
    argument and checks only ``lambdas``, so a mismatch raises nothing. An
    encoder evaluated at the wrong smoothing width is not the encoder that was
    trained, so this raises instead.

    Returns ``(encoder, model_config, gain, fwhm_ratio, header)``.
    """
    from hopfield_nav.encoder_io import load_encoder

    encoder, cfg, gain = load_encoder(path, device)

    stored = _dig(torch.load(path, map_location="cpu", weights_only=False),
                  "fwhm_ratio")
    fwhm = stored
    # `fwhm_fallback` fills in only where the checkpoint carries nothing, so a
    # sweep can pass one value for a batch that includes `untrained_mlp.pt`
    # without silently masking the real fwhm of every other checkpoint in it.
    # `fwhm_override` forces, and is the rarer, louder option.
    if fwhm_override is not None:
        fwhm = float(fwhm_override)
    elif stored is None and fwhm_fallback is not None:
        fwhm = float(fwhm_fallback)
    elif fwhm is None:
        raise ValueError(
            f"{path} carries no fwhm_ratio (checked train_config, "
            f"model_config, config and top level). It is a property of the "
            f"encoder, and evaluating at the wrong smoothing width silently "
            f"produces embeddings the encoder was never fitted to. Pass "
            f"--fwhm_fallback to state one; it is recorded in the result "
            f"header as an override."
        )

    header = encoder_header(
        path, cfg, gain, float(fwhm),
        fwhm_was_overridden=(fwhm_override is not None
                             or (stored is None and fwhm_fallback is not None)))
    return encoder, cfg, float(gain), float(fwhm), header


def encoder_header(
    path: str,
    cfg: EncoderModelConfig,
    gain: float,
    fwhm_ratio: float,
    *,
    fwhm_was_overridden: bool = False,
) -> dict:
    """Provenance block copied onto every result file.

    ``unique_radius`` is *copied* out of the checkpoint when it is there, never
    recomputed: it is ``encoder_training``'s metric and this package does not
    re-derive another module's numbers. Checkpoints without it leave it null.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict")) or {}
    n_params = int(sum(v.numel() for v in state.values()
                       if hasattr(v, "numel")))

    ur = ckpt.get("unique_radius")
    ur_summary = None
    if isinstance(ur, dict):
        keep = ("r_min", "r_median", "r_max", "n_refs", "headline",
                "headline_trim", "alias_ceiling_max", "alias_ceiling_mean",
                "cos_floor_mean", "r_at_cos0.5_median")
        ur_summary = {k: _jsonable(ur[k]) for k in keep if k in ur}

    return {
        "path": str(path),
        "name": Path(path).parent.name + "/" + Path(path).name,
        "gain": float(gain),
        "fwhm_ratio": float(fwhm_ratio),
        "fwhm_was_overridden": bool(fwhm_was_overridden),
        "lambdas": list(cfg.lambdas),
        "out_dim": int(cfg.out_dim),
        "hidden_dim": int(getattr(cfg, "hidden_dim", 0)),
        "num_hidden_layers": int(getattr(cfg, "num_hidden_layers", 0)),
        "encoder_type": str(cfg.encoder_type),
        "output_nonlinearity": str(cfg.output_nonlinearity),
        "n_params": n_params,
        "epoch": _jsonable(ckpt.get("epoch")),
        "val_nav_acc": _jsonable(ckpt.get("val_nav_acc")),
        "unique_radius": ur_summary,
    }


def _jsonable(v):
    if v is None or isinstance(v, (str, bool)):
        return v
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, (int,)):
        return int(v)
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_jsonable(x) for x in np.asarray(v).tolist()]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return str(v)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ProbeConfig:
    """Everything the four tests read. Defaults are the spec's Sec 9 table."""

    # scaffold / world
    Npos: int = 1716
    env_size: int = 20
    n_worlds: int = 50
    n_envs_per_world: int = 50        # pinned at max(k_values); see Sec 2.3
    spread_jitter: float = 0.4
    seed: int = 0

    # memory
    memory_mode: str = "multi_env_goals"
    k_values: tuple[int, ...] = (1, 2, 3, 5, 10, 20, 50)
    # How many of a world's envs are scored against its W. Capping this is what
    # keeps the K axis a *load* axis: scoring all K would make the measured
    # population grow with K, so a K-to-K comparison would confound "more
    # memories" with "a different, larger set of envs". At or above this value
    # the population is identical at every K and sample size comes from worlds.
    n_score_envs: int = 5

    # hopfield
    steps: tuple[int, ...] = (1, 2, 3, 5, 10, 15)
    alpha: float = 1.0
    use_tanh: bool = True
    zero_diag: bool = True
    hopfield_scale: float | None = None      # None -> 1/D, the production value
    beta_override: float | None = None       # None -> encoder gain

    # retrieval
    n_alias: int = 20_000

    # continuous sampling (Test C)
    n_cont_samples: int = 200_000
    n_cont_annulus: int = 50_000
    annulus_radius: float = 3.0
    subcell_bins: int = 8

    # flow (Test D)
    flow_max_steps_factor: int = 4
    continuous_scale: float = 1.0

    # Raw per-cell maps are kept only for this many (world, env) pairs. The
    # full set is 3e9 numbers at the defaults, and every figure but the raw
    # example panels reads a pooled accumulator instead. One raw example is
    # what catches a harness bug; keeping every example is what makes the
    # output unusable.
    n_map_worlds: int = 1
    n_map_envs: int = 2

    # How many recall steps the real-space trajectory probe decodes. 0 disables
    # it. Costs one bank retrieval per step, so it runs on n_map_worlds only.
    trajectory_steps: int = 15

    # execution
    device: str = "cpu"
    chunk: int = 4096
    cos_chunk: int = 2048            # cue rows per retrieval matmul

    def validate(self) -> None:
        if self.memory_mode not in MEMORY_MODES:
            raise ValueError(
                f"memory_mode={self.memory_mode!r} not in {MEMORY_MODES}")
        if max(self.k_values) > self.n_envs_per_world:
            raise ValueError(
                f"max(k_values)={max(self.k_values)} exceeds "
                f"n_envs_per_world={self.n_envs_per_world}. Sec 2.3 pins the "
                f"world size at the largest K and stores only the first K, so "
                f"placement is identical across the sweep -- raising K without "
                f"raising the world would confound load with packing.")
        if self.env_size * 2 > self.Npos:
            raise ValueError("env_size too large for this Npos")

    def to_json(self) -> dict:
        d = asdict(self)
        for k in ("k_values", "steps"):
            d[k] = list(d[k])
        return d


# ---------------------------------------------------------------------------
# Worlds
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class World:
    """``n_envs_per_world`` envs placed in one scaffold, plus its own RNG seed.

    A world is the sampling unit for ``W``; the *env* stays the reporting unit.
    Under ``multi_env_goals`` one world at load ``K`` yields ``K`` scored envs
    against a single ``W``, which is what makes ``K=50`` affordable.
    """

    index: int
    seed: int
    specs: tuple[EnvSpec, ...]

    def offsets(self) -> np.ndarray:
        return np.array([s.offset for s in self.specs], dtype=np.int64)

    def goals(self) -> np.ndarray:
        return np.array([s.goal for s in self.specs], dtype=np.int64)

    def to_json(self) -> dict:
        return {"index": self.index, "seed": self.seed,
                "specs": [s.to_json() for s in self.specs]}


def sample_worlds(cfg: ProbeConfig) -> list[World]:
    """Draw ``cfg.n_worlds`` worlds, each ``n_envs_per_world`` placed envs."""
    worlds = []
    for w in range(cfg.n_worlds):
        seed = int(cfg.seed) * 1_000_003 + w
        rng = np.random.RandomState(seed)
        offsets = place_envs(
            cfg.n_envs_per_world, cfg.env_size, cfg.Npos, rng,
            placement="spread", spread_jitter=cfg.spread_jitter,
        )
        goals = rng.randint(0, cfg.env_size, size=(cfg.n_envs_per_world, 2))
        wall_seeds = rng.randint(0, 2 ** 31 - 1, size=cfg.n_envs_per_world)
        specs = tuple(
            EnvSpec(wall_seed=int(wall_seeds[i]), size=int(cfg.env_size),
                    offset=(int(offsets[i][0]), int(offsets[i][1])),
                    goal=(int(goals[i][0]), int(goals[i][1])))
            for i in range(cfg.n_envs_per_world)
        )
        worlds.append(World(index=w, seed=seed, specs=specs))
    return worlds


def scored_envs(cfg: "ProbeConfig", k: int) -> list[int]:
    """Which envs of a world are measured at load ``k``.

    ``multi_env_goals`` scores a fixed prefix, capped by ``n_score_envs`` --
    see the field's comment. The other modes define a single test env by
    construction, so they score one.
    """
    if cfg.memory_mode != "multi_env_goals":
        return [0]
    return list(range(min(k, cfg.n_score_envs)))


def env_offset_distances(world: World, k: int) -> np.ndarray:
    """Pairwise scaffold-offset distance among the first ``k`` envs: ``(k, k)``.

    The report orders the confusion matrix by this rather than by env index --
    aliasing shows as mass on the near-offset band, interference as a uniform
    field, and index order would scramble both into noise.
    """
    off = world.offsets()[:k].astype(np.float64)
    d = off[:, None, :] - off[None, :, :]
    return np.sqrt((d ** 2).sum(-1))


# ---------------------------------------------------------------------------
# Cell bank
# ---------------------------------------------------------------------------

@dataclass
class CellBank:
    """Every cell retrieval is allowed to return, encoded once per world.

    Rows ``[0, k*size^2)`` are env-major -- env ``e``, local ``(x, y)`` at
    ``e*size^2 + x*size + y`` -- and the tail is ``n_alias`` uniformly drawn
    scaffold cells outside every footprint.

    Deliberately not all ``Npos^2`` cells: at 1716 that is 2.94M rows, a
    ``(size^2, 2.94M)`` cosine per env, and it would dominate the suite to
    estimate a tail that a uniform sample already estimates. The alias rows are
    what catch the ``alias_ceiling`` outliers the header reports.
    """

    Z: np.ndarray                    # (M, D) unit rows
    k: int
    size: int
    n_alias: int

    @property
    def n_env_rows(self) -> int:
        return self.k * self.size * self.size

    def decode(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bank row -> ``(env_idx, x, y)``; ``env_idx = -1`` for alias rows."""
        idx = np.asarray(idx)
        s2 = self.size * self.size
        env = np.where(idx < self.n_env_rows, idx // s2, -1)
        within = np.where(idx < self.n_env_rows, idx % s2, 0)
        return env, within // self.size, within % self.size


def local_cells(size: int) -> np.ndarray:
    """All local cells, in the bank's row order: ``(size^2, 2)``."""
    xs, ys = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    return np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.int64)


def build_cell_bank(
    field: Field, world: World, k: int, cfg: ProbeConfig,
    rng: np.random.RandomState,
) -> CellBank:
    size = cfg.env_size
    cells = local_cells(size)

    gx_parts, gy_parts = [], []
    for e in range(k):
        ox, oy = world.specs[e].offset
        gx_parts.append(cells[:, 0] + ox)
        gy_parts.append(cells[:, 1] + oy)

    if cfg.n_alias > 0:
        ax, ay = _sample_alias_cells(world, k, cfg, rng)
        gx_parts.append(ax)
        gy_parts.append(ay)

    gx = np.concatenate(gx_parts)
    gy = np.concatenate(gy_parts)
    return CellBank(Z=field.encode(gx, gy), k=k, size=size,
                    n_alias=cfg.n_alias)


def _sample_alias_cells(
    world: World, k: int, cfg: ProbeConfig, rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform scaffold cells outside every one of the ``k`` footprints.

    Rejection sampled in blocks rather than one at a time -- the footprints
    cover ``k*size^2 / Npos^2`` of the scaffold, well under 1% at every
    configuration here, so a block is almost never short.
    """
    size = cfg.env_size
    off = world.offsets()[:k]
    out_x, out_y = [], []
    have = 0
    while have < cfg.n_alias:
        n = int((cfg.n_alias - have) * 1.1) + 64
        gx = rng.randint(0, cfg.Npos, size=n)
        gy = rng.randint(0, cfg.Npos, size=n)
        inside = np.zeros(n, dtype=bool)
        for ox, oy in off:
            inside |= ((gx >= ox) & (gx < ox + size)
                       & (gy >= oy) & (gy < oy + size))
        keep = ~inside
        out_x.append(gx[keep])
        out_y.append(gy[keep])
        have += int(keep.sum())
    return (np.concatenate(out_x)[:cfg.n_alias],
            np.concatenate(out_y)[:cfg.n_alias])


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

@dataclass
class Memory:
    """One Hopfield, the patterns in it, and what each pattern *is*.

    ``owner[i]`` is the env index pattern ``i`` belongs to (``-1`` for a
    distractor drawn from open scaffold), so a retrieval that lands on a stored
    pattern can be named.
    """

    hopfield: Hopfield
    Z: np.ndarray                     # (K, D) as stored (post-normalisation)
    owner: np.ndarray                 # (K,) int
    diag_frac: float


def build_memory(
    field: Field, world: World, k: int, cfg: ProbeConfig,
    rng: np.random.RandomState, *, test_env: int = 0,
) -> Memory:
    """Fill a Hopfield according to ``cfg.memory_mode``."""
    if cfg.memory_mode == "multi_env_goals":
        goals = world.goals()[:k]
        offs = world.offsets()[:k]
        Z = field.encode(goals[:, 0] + offs[:, 0], goals[:, 1] + offs[:, 1])
        owner = np.arange(k)
    elif cfg.memory_mode == "goal+distractors":
        spec = world.specs[test_env]
        gz = field.encoded_state(np.array([spec.goal]), spec.offset)
        if k > 1:
            dx, dy = _distractor_cells(world, test_env, k - 1, cfg, rng)
            Z = np.concatenate([gz, field.encode(dx, dy)], axis=0)
        else:
            Z = gz
        owner = np.full(k, -1)
        owner[0] = test_env
    elif cfg.memory_mode == "same_env_goals":
        spec = world.specs[test_env]
        cells = local_cells(cfg.env_size)
        pick = rng.choice(len(cells), size=k, replace=False)
        # The test env's own goal must be one of them, and first.
        goal_row = spec.goal[0] * cfg.env_size + spec.goal[1]
        pick = np.concatenate([[goal_row], pick[pick != goal_row]])[:k]
        Z = field.encoded_state(cells[pick], spec.offset)
        owner = np.full(k, test_env)
    else:                                             # pragma: no cover
        raise ValueError(cfg.memory_mode)

    return _store(Z, owner, field, cfg, rng)


def _distractor_cells(
    world: World, test_env: int, n: int, cfg: ProbeConfig,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray]:
    """``rollout.distractors.sample_distractors``, vectorised.

    Same distribution -- uniform scaffold cells outside *this* env's footprint
    -- but drawn in blocks, so the RNG stream is not the production one. That
    matters only for bit-replay of a training run, which this harness never
    does.
    """
    size = cfg.env_size
    cx, cy = world.specs[test_env].offset
    out_x, out_y = [], []
    have = 0
    while have < n:
        m = int((n - have) * 1.1) + 16
        gx = rng.randint(0, cfg.Npos, size=m)
        gy = rng.randint(0, cfg.Npos, size=m)
        keep = ~((gx >= cx) & (gx < cx + size) & (gy >= cy) & (gy < cy + size))
        out_x.append(gx[keep])
        out_y.append(gy[keep])
        have += int(keep.sum())
    return np.concatenate(out_x)[:n], np.concatenate(out_y)[:n]


def _store(
    Z: np.ndarray, owner: np.ndarray, field: Field, cfg: ProbeConfig,
    rng: np.random.RandomState,
) -> Memory:
    """Store patterns in a real ``Hopfield``, in shuffled order."""
    k, dim = Z.shape
    order = rng.permutation(k)

    beta = cfg.beta_override if cfg.beta_override is not None else field.gain
    hop = Hopfield(dim, beta=float(beta), zero_diag=cfg.zero_diag,
                   scale=cfg.hopfield_scale, device=cfg.device)

    Zt = torch.from_numpy(np.ascontiguousarray(Z)).float().to(cfg.device)
    for i in order:
        hop.input_memory(Zt[int(i)])

    # diag_frac is measured on the un-zeroed outer-product sum, because the
    # question it answers is how much signal `zero_diag` throws away -- which
    # is invisible once it has been thrown away.
    Zn = Z / np.linalg.norm(Z, axis=1, keepdims=True).clip(1e-12)
    diag = (Zn ** 2).sum(axis=0)
    full_sq = float((Zn @ Zn.T).__pow__(2).sum())     # ||sum_k z z^T||_F^2
    diag_frac = float(np.sqrt((diag ** 2).sum() / max(full_sq, 1e-30)))

    stored = Zn.astype(np.float32)
    return Memory(hopfield=hop, Z=stored, owner=np.asarray(owner),
                  diag_frac=diag_frac)


def recall_trajectory(
    mem: Memory, cues: np.ndarray, steps: tuple[int, ...], cfg: ProbeConfig,
) -> dict[int, np.ndarray]:
    """``Hopfield.recall_batch_trajectory`` on numpy in, numpy out.

    One call yields every requested step count, which is what the production
    ``multistep_q`` does -- and, like it, the basis is computed once outside
    and does not move with the recall.
    """
    X = torch.from_numpy(np.ascontiguousarray(cues)).float().to(cfg.device)
    traj = mem.hopfield.recall_batch_trajectory(
        X, list(steps), alpha=cfg.alpha, use_tanh=cfg.use_tanh,
    )
    return {s: v.cpu().numpy().astype(np.float32) for s, v in traj.items()}


def tanh_argument(mem: Memory, cues: np.ndarray, cfg: ProbeConfig) -> np.ndarray:
    """``beta * (W x)`` over real cues -- the evidence for or against Sec 1.3.

    Sec 1.3 argues the ``tanh`` is numerically inert because ``||W x|| ~ K/D``.
    That is a claim about a *product*, and ``beta`` differs 27x between the
    encoders under test, so it is measured rather than assumed.
    """
    X = torch.from_numpy(np.ascontiguousarray(cues)).float().to(cfg.device)
    with torch.no_grad():
        h = X @ mem.hopfield.W.T
        return (mem.hopfield.beta * h).cpu().numpy().ravel()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=1))


__all__ = [
    "CellBank", "Memory", "MEMORY_MODES", "OUTCOMES", "ProbeConfig", "World",
    "build_cell_bank", "build_memory", "encoder_header", "env_offset_distances",
    "load_probe_encoder", "local_cells", "recall_trajectory", "sample_worlds",
    "scored_envs", "tanh_argument", "write_json",
]
