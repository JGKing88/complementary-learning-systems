"""Resolved environment specs, and the split a generator produces.

An ``EnvSpec`` is a *fully resolved* environment: the four traits, concrete. It
is what makes a world declarable, checkable and rebuildable -- today a world is
implied by an RNG replay, which is why ``build_eval_world`` recovers a
checkpoint's val wall codes and goals but not their offsets (see
``docs/EVAL_SPLITS_DESIGN.md`` §1.4).

``GeneratedSplit`` bundles the specs with the domains that produced them and the
union of values each trait actually took. Phase 3 serializes it to
``world.json``; both halves are load-bearing:

    domains   let a later eval mint *fresh* envs and derive complements
    used      let it exclude what training actually saw, and make `same` mean
              "any value the trait ever took" once refresh exists
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import numpy as np

from . import domains as dom


@dataclass(frozen=True)
class EnvSpec:
    """One environment, resolved.

    ``wall_seed`` reaches ``GridEnv(seed=...)`` unchanged, so the wall code is
    bit-identical to what that seed has always produced. ``goal`` is applied
    afterwards via ``GridEnv.set_goal``; the constructor's own goal draw becomes
    dead entropy.
    """

    wall_seed: int
    size: int
    offset: tuple[int, int]
    goal: tuple[int, int]

    def to_json(self) -> dict:
        return {"wall_seed": int(self.wall_seed), "size": int(self.size),
                "offset": [int(self.offset[0]), int(self.offset[1])],
                "goal": [int(self.goal[0]), int(self.goal[1])]}

    @staticmethod
    def from_json(d: dict) -> "EnvSpec":
        return EnvSpec(int(d["wall_seed"]), int(d["size"]),
                       (int(d["offset"][0]), int(d["offset"][1])),
                       (int(d["goal"][0]), int(d["goal"][1])))


@dataclass(frozen=True)
class TraitDomains:
    """The declared legal region for each trait.

    ``place`` and ``goal`` have complements and therefore support an OOD level;
    ``wall`` and ``size`` do not -- see ``domains`` module docstring.
    """

    place: dom.PlaceDomain
    wall: dom.SeedRange
    goal: dom.GoalDomain
    size: dom.Sizes

    def to_json(self) -> dict:
        return {"place": self.place.to_json(), "wall": self.wall.to_json(),
                "goal": self.goal.to_json(), "size": self.size.to_json()}

    @staticmethod
    def from_json(d: dict) -> "TraitDomains":
        return TraitDomains(
            place=dom.from_json(d["place"]), wall=dom.from_json(d["wall"]),
            goal=dom.from_json(d["goal"]), size=dom.from_json(d["size"]),
        )


@dataclass
class GeneratedSplit:
    """Train + base-val env sets, the domains behind them, and what was used.

    ``used`` is the union over refresh ticks once Phase 4 exists. In Phase 2 a
    trait is drawn once, so it is just the resolved values -- but the type is
    already a set per trait so refresh has somewhere to accumulate.
    """

    domains: TraitDomains
    train: list[EnvSpec]
    base_val: list[EnvSpec]
    goal_cells_train: frozenset
    goal_cells_val: frozenset
    margin: int
    period: int
    Npos: int
    used: dict = dc_field(default_factory=dict)
    diagnostics: dict = dc_field(default_factory=dict)

    def record_used(self, specs: list[EnvSpec]) -> None:
        """Fold ``specs`` into the per-trait union."""
        self.used.setdefault("place", set()).update(s.offset for s in specs)
        self.used.setdefault("wall", set()).update(s.wall_seed for s in specs)
        self.used.setdefault("goal", set()).update(s.goal for s in specs)
        self.used.setdefault("size", set()).update(s.size for s in specs)

    def to_json(self) -> dict:
        return {
            "domains": self.domains.to_json(),
            "margin": int(self.margin),
            "period": int(self.period),
            "Npos": int(self.Npos),
            "goal_cells_train": sorted([int(c[0]), int(c[1])]
                                       for c in self.goal_cells_train),
            "goal_cells_val": sorted([int(c[0]), int(c[1])]
                                     for c in self.goal_cells_val),
            "train": [s.to_json() for s in self.train],
            "base_val": [s.to_json() for s in self.base_val],
            "used": {
                "place": sorted([int(o[0]), int(o[1])]
                                for o in self.used.get("place", ())),
                "wall": sorted(int(s) for s in self.used.get("wall", ())),
                "goal": sorted([int(c[0]), int(c[1])]
                               for c in self.used.get("goal", ())),
                "size": sorted(int(s) for s in self.used.get("size", ())),
            },
            "diagnostics": _jsonable(self.diagnostics),
        }

    @staticmethod
    def from_json(d: dict) -> "GeneratedSplit":
        split = GeneratedSplit(
            domains=TraitDomains.from_json(d["domains"]),
            train=[EnvSpec.from_json(s) for s in d["train"]],
            base_val=[EnvSpec.from_json(s) for s in d["base_val"]],
            goal_cells_train=frozenset(tuple(c) for c in d["goal_cells_train"]),
            goal_cells_val=frozenset(tuple(c) for c in d["goal_cells_val"]),
            margin=int(d["margin"]), period=int(d["period"]),
            Npos=int(d["Npos"]), diagnostics=d.get("diagnostics", {}),
        )
        u = d.get("used", {})
        split.used = {
            "place": {tuple(o) for o in u.get("place", ())},
            "wall": set(u.get("wall", ())),
            "goal": {tuple(c) for c in u.get("goal", ())},
            "size": set(u.get("size", ())),
        }
        return split


def _jsonable(obj):
    """numpy scalars are not JSON-serializable; diagnostics are full of them."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


__all__ = ["EnvSpec", "GeneratedSplit", "TraitDomains"]
