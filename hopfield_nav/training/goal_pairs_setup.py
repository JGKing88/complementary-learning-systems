"""World and mode setup for the goal-conditioned pair experiment (plan §3).

Shared by the trainer (`train_goal_pairs`) and the pre-flight
(`scripts/goal_nav_preflight.py`), so the two build the same envs from the
same config and neither imports the other -- a CLI is a program, and rule 5
of the layering keeps programs unimported.
"""
from __future__ import annotations

from dataclasses import replace
from typing import NamedTuple

import numpy as np

from gridcode.lattice import gbook_at
from ..config import RNNAgentConfig, RNNTrainConfig
from ..evaluation.goal_pairs import EnvTensors, aggregate_tables, evaluate_pairs
from ..utils import smooth_gbook
from ..world import generate as gen
from .rnn_setup import rnn_world

MODES = ("xy", "grid", "regular")

# Experiment B's arms (plan sec 4.2): the factorial that makes a B win
# attributable. Lives here so the eval CLI can rebuild an arm's agent
# without importing the training CLI.
ARMS = {
    "full": dict(rnn_cell="gru", input_prev_action=True),
    "rec":  dict(rnn_cell="gru", input_prev_action=False),
    "dist": dict(rnn_cell="mlp", input_prev_action=False),
}


def agent_cfg_for_mode(mode: str, movement_mode: str, **kw) -> RNNAgentConfig:
    """The three input modes, each a complete channel configuration (plan §2.2).

        xy       [xy(p), xy(g)]            the coordinate ceiling
        grid     [gbook(p), gbook(g)]      the grid code, no sensory
        regular  [omni(p), omni(g)]        the ray-cast, heading-free
    """
    # A has no prev_action (memoryless); B's `full` arm turns it on. Neither
    # ever has prev_reward (plan sec 2.2: arrival is the only reward event).
    base = {"movement_mode": movement_mode, "input_prev_action": False,
            "input_prev_reward": False, **kw}
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
    """Envs + their precomputed tensors, under one name, for the table.

    ``lambdas`` and ``fwhm_ratio`` are what ``lattice_gbook`` / ``with_lattice``
    need to re-synthesise an env's grid code on another lattice (plan sec
    4B.2); the tensors built here are the scaffold's own lattice, (0, 1).
    """

    def __init__(self, name: str, envs, offsets, sgb, *, lambdas=None,
                 fwhm_ratio: float | None = None, tensors=None,
                 theta: float = 0.0, scale: float = 1.0) -> None:
        self.name = name
        self.envs = envs
        self.offsets = offsets
        self.lambdas = None if lambdas is None else [int(l) for l in lambdas]
        self.fwhm_ratio = fwhm_ratio
        self.theta, self.scale = float(theta), float(scale)
        if tensors is None:
            tensors = [EnvTensors.build(e, o, sgb) for e, o in zip(envs, offsets)]
        self.tensors = tensors

    def __len__(self) -> int:
        return len(self.envs)

    def lattice_gbook(self, k: int, theta: float, scale: float = 1.0,
                      shift=(0.0, 0.0)) -> np.ndarray:
        """Env ``k``'s ``(S * S, Ng)`` code under lattice ``(theta, scale, shift)``, from its offset."""
        if self.lambdas is None or self.fwhm_ratio is None:
            raise ValueError("EnvSet was built without lambdas/fwhm_ratio; cannot re-synthesise")
        S = self.envs[k].size
        cells = np.array([(x, y) for x in range(S) for y in range(S)], dtype=np.float64)
        ox, oy = self.offsets[k]
        return gbook_at(cells + np.array([ox, oy], dtype=np.float64), self.lambdas,
                        self.fwhm_ratio, theta, scale, shift)

    def with_lattice(self, theta: float, scale: float = 1.0, name: str | None = None) -> "EnvSet":
        """A copy of this set with every env's grid code on lattice ``(theta, scale)``."""
        tensors = [replace(t, gbook=self.lattice_gbook(k, theta, scale), theta=float(theta),
                           scale=float(scale)) for k, t in enumerate(self.tensors)]
        return EnvSet(name or f"{self.name}@{np.degrees(theta):.0f}", self.envs, self.offsets, None,
                      lambdas=self.lambdas, fwhm_ratio=self.fwhm_ratio, tensors=tensors,
                      theta=theta, scale=scale)


class Lattice(NamedTuple):
    theta: float
    scale: float
    shift: tuple[float, float]


class LatticeSampler:
    """Per-lifetime lattice draws for the lattice-randomised runs (plan sec 4B.2).

    theta ~ U[0, 2 pi) minus the held-out band ``|theta| < holdout_deg``, so
    the standard lattice (theta = 0) and its neighbourhood are never trained
    on; scale ~ U[lo, hi] minus ``[0.95, 1.05]`` unless the range is the
    point ``(1, 1)``; with probability ``mix_standard_frac`` the draw is the
    standard lattice (0, 1) instead (B2-mix, sec 4B.8).

    ``translate`` adds a lattice translation uniform over the combined period
    (``prod(lambdas)``, 1716 for 11 12 13) to every draw, the standard ones
    included. Without it the absolute phases of a training env -- whose
    scaffold offset is fixed and can be memorised -- pin theta through the
    weights, which is the route B2 exists to close (seen on `dist`,
    2026-09-13). With it the absolute phases are uniform for every theta and
    only phase differences, and the trajectory, carry information.
    """

    def __init__(self, rng, *, holdout_deg: float = 15.0, scale_range=(1.0, 1.0),
                 mix_standard_frac: float = 0.0, translate: bool = False,
                 period: float = 1716.0) -> None:
        self.rng = rng
        self.holdout = np.radians(float(holdout_deg))
        self.scale_range = (float(scale_range[0]), float(scale_range[1]))
        self.mix = float(mix_standard_frac)
        self.translate = bool(translate)
        self.period = float(period)
        if not (0.0 <= self.holdout < np.pi):
            raise ValueError("holdout_deg must be in [0, 180)")

    def _shift(self) -> tuple[float, float]:
        if not self.translate:
            return (0.0, 0.0)
        return (float(self.rng.uniform(0, self.period)), float(self.rng.uniform(0, self.period)))

    def draw(self) -> Lattice:
        if self.mix > 0 and self.rng.uniform() < self.mix:
            return Lattice(0.0, 1.0, self._shift())
        theta = self.rng.uniform(self.holdout, 2 * np.pi - self.holdout)
        lo, hi = self.scale_range
        if lo == hi:
            return Lattice(float(theta), lo, self._shift())
        for _ in range(100):
            s = self.rng.uniform(lo, hi)
            if not (0.95 <= s <= 1.05):
                return Lattice(float(theta), float(s), self._shift())
        raise RuntimeError("scale range is inside the held-out band [0.95, 1.05]")

    def in_holdout(self, theta: float) -> bool:
        t = np.mod(float(theta) + np.pi, 2 * np.pi) - np.pi
        return abs(t) < self.holdout


def parse_thetas(spec: str) -> list[float]:
    """``"0,7,45"`` (degrees) -> radians."""
    return [float(np.radians(float(v))) for v in str(spec).split(",") if v.strip() != ""]


def build_env_sets(cfg: RNNTrainConfig, rng, *, n_same: int, keep_field: bool = False,
                   n_ood_place: int = 0):
    """Train, held-out (`base_val`) and `same` env sets, plus split, field, sgb.

    `same` is a fixed subset of the ACTUAL training envs -- same wall, same
    offset -- not a `make_val_set(same)` draw, which re-pairs walls and
    offsets. The env-side probe wants the envs themselves.

    With `n_ood_place > 0` a fourth set, `heldout_out`, is minted at
    `place = ood` -- envs whose footprint lies OUTSIDE the declared place
    region by at least the margin (plan sec 2.4, the corner holdout). It
    only means something when the region is a `Rect`; `Anywhere` has no
    complement and `make_val_set` raises. The split's own `base_val` is
    then `heldout_in`: new walls inside the region, so the phase effect and
    the wall effect are separated.
    """
    envs, offsets, split, vh, kind = rnn_world(cfg, rng)
    if kind != "declared":
        raise SystemExit("train_goal_pairs needs --env_generator: the holdouts "
                         "are defined by the declared split")
    sgb = smooth_gbook(vh.gbook, vh.lambdas, cfg.fwhm_ratio)
    lat = dict(lambdas=vh.lambdas, fwhm_ratio=cfg.fwhm_ratio)
    train = EnvSet("train", envs, offsets, sgb, **lat)
    val_envs = gen.build_envs(split.base_val, cfg.env, "discrete")
    in_name = "heldout_in" if n_ood_place > 0 else "heldout"
    heldout = EnvSet(in_name, val_envs, [s.offset for s in split.base_val], sgb, **lat)
    k = min(n_same, len(envs))
    same = EnvSet("same", envs[:k], offsets[:k], sgb, **lat)
    heldout_out = None
    if n_ood_place > 0:
        specs = gen.make_val_set(
            split, n_ood_place,
            {"place": "ood", "wall": "held_out", "goal": "held_out"},
            seed=int(cfg.seed) + 7919)
        # The gate: every minted box clears the training rect by >= margin,
        # on the torus, on at least one axis. Checked here, at launch, so a
        # run whose "outside" set is not outside never starts.
        region = split.domains.place
        if not hasattr(region, "x0"):
            raise SystemExit("--n_ood_place needs --place_region rect:...; "
                             "'anywhere' has no outside")
        for sp in specs:
            gx = gen.axis_separation(region.x0, region.w, sp.offset[0], sp.size, split.period)
            gy = gen.axis_separation(region.y0, region.h, sp.offset[1], sp.size, split.period)
            if max(gx, gy) < split.margin:
                raise SystemExit(
                    f"ood env at {sp.offset} is only {max(gx, gy)} cells from the "
                    f"training rect (margin {split.margin}); refusing to run")
        out_envs = gen.build_envs(specs, cfg.env, "discrete")
        heldout_out = EnvSet("heldout_out", out_envs, [sp.offset for sp in specs], sgb, **lat)
    # `sgb` is 434 x 1716 x 1716 float32 (~5 GB) and every cell this run will
    # ever read is now in the EnvTensors. Drop the field and the smoothed
    # book so the process does not hold 10 GB it never touches again; the
    # pre-flight rebuilds them itself for the checks that need the scaffold.
    if not keep_field:
        vh.gbook = None
        sgb = None
    if heldout_out is not None:
        return train, heldout, same, split, vh, sgb, heldout_out
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
