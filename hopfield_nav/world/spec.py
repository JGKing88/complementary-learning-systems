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

import json
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


@dataclass
class WorldSpec:
    """A run's world, recorded: which scaffold, which envs, how they were chosen.

    This is what makes a world reproducible instead of replayable.
    ``build_eval_world`` recovers a checkpoint's val wall codes and goals by
    replaying the seed stream, but *not* their offsets -- placement drew from
    global ``np.random``, whose state depended on everything built before it. So
    every post-hoc eval has been scoring checkpoints on scaffold patches training
    never used (§1.4, measured deltas up to 10 cells). Writing the resolved specs
    fixes that outright.

    ``generator`` says how the envs were chosen, not how they are stored:

        "declared"  drawn by the Phase-2 generator from declared domains
        "legacy"    drawn by the historical placement path; domains are the
                    permissive defaults and `split.train` records what it drew

    Both are equally reproducible from this file. The flag only tells you whether
    the *domains* mean anything or are just describing an unconstrained draw.
    """

    scaffold: dict
    generator: str
    split: GeneratedSplit
    spec_version: int = 1

    def _payload(self) -> dict:
        return {"spec_version": self.spec_version, "generator": self.generator,
                "scaffold": self.scaffold, "split": self.split.to_json()}

    def spec_hash(self) -> str:
        """sha256 over the canonical payload -- key order cannot change it."""
        import hashlib
        blob = json.dumps(self._payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def to_json(self) -> dict:
        return {**self._payload(), "spec_hash": self.spec_hash()}

    @staticmethod
    def from_json(d: dict) -> "WorldSpec":
        spec = WorldSpec(scaffold=d["scaffold"], generator=d["generator"],
                         split=GeneratedSplit.from_json(d["split"]),
                         spec_version=int(d.get("spec_version", 1)))
        recorded = d.get("spec_hash")
        if recorded is not None and recorded != spec.spec_hash():
            raise ValueError(
                f"world spec hash mismatch: file says {recorded[:12]}..., "
                f"contents hash to {spec.spec_hash()[:12]}.... The file was "
                f"edited by hand or written by a different spec_version.")
        return spec

    def summary(self, path: str | None = None) -> dict:
        """The small block that rides in a checkpoint.

        Domains and the hash, never the resolved lists -- those grow with every
        refresh tick once Phase 4 lands, and a checkpoint is written far more
        often than a world changes.
        """
        return {"spec_version": self.spec_version, "generator": self.generator,
                "spec_hash": self.spec_hash(),
                "domains": self.split.domains.to_json(),
                "world_json": path}

    def write(self, save_dir) -> str:
        import os
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(str(save_dir), WORLD_SPEC_NAME)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.to_json(), f, indent=2, sort_keys=True)
        os.replace(tmp, path)
        return path

    @staticmethod
    def read(path) -> "WorldSpec":
        import os
        p = str(path)
        if os.path.isdir(p):
            p = os.path.join(p, WORLD_SPEC_NAME)
        with open(p) as f:
            return WorldSpec.from_json(json.load(f))


WORLD_SPEC_NAME = "world.json"


def _jsonable(obj):
    """numpy scalars are not JSON-serializable; diagnostics are full of them."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


__all__ = ["WORLD_SPEC_NAME", "EnvSpec", "GeneratedSplit",
           "TraitDomains", "WorldSpec"]
