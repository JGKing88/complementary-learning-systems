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
import warnings
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


def _hash_payload(payload: dict) -> str:
    """sha256 over the canonical JSON -- key order cannot change it."""
    import hashlib
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _boxes_from_json(entries, sizes) -> set:
    """Read ``used["place"]`` from either the box form or the pre-Phase-6 one.

    ``[r, c, size]`` is self-describing. ``[r, c]`` is not, so it is paired with
    the run's recorded size -- unambiguous for every world.json written before
    this, since size refresh has never shipped a mixed-size run. If one somehow
    exists, the largest size is the only safe guess: it over-excludes rather than
    placing a validation env on top of a training one.
    """
    out, seen = set(), sorted(int(s) for s in sizes)
    for e in entries:
        if len(e) == 3:
            out.add(((int(e[0]), int(e[1])), int(e[2])))
            continue
        if not seen:
            raise ValueError(
                "world.json records used places as [r, c] with no used size to "
                "pair them with; the file cannot say how large those envs were.")
        if len(seen) > 1:
            warnings.warn(
                f"world.json records used places as [r, c] but {len(seen)} "
                f"sizes {seen}; assuming the largest ({seen[-1]}), which "
                "over-excludes rather than risking an overlap.", stacklevel=3)
        out.add(((int(e[0]), int(e[1])), seen[-1]))
    return out


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
        self.used.setdefault("place", set()).update(
            (s.offset, int(s.size)) for s in specs)
        self.used.setdefault("wall", set()).update(s.wall_seed for s in specs)
        self.used.setdefault("goal", set()).update(s.goal for s in specs)
        self.used.setdefault("size", set()).update(s.size for s in specs)

    def absorb_used(self, other: dict) -> None:
        """Fold another run's ``used`` union into this one.

        For a run continuing from ``--load_checkpoint``: the parent's envs are
        as much "what training saw" as this run's own, so a later ``held_out``
        validation set has to be disjoint from both. Recording the union *here*
        rather than teaching every evaluator to walk the checkpoint chain means
        ``world.json`` answers the question on its own -- and transitively, since
        the parent's file already absorbed *its* parent's.

        Only the four trait unions merge. ``train``, ``base_val`` and the goal
        cell partition stay this run's own: they say what this run trained and
        evaluated on, which is a different question from what any run ever used.
        """
        for trait in ("place", "wall", "goal", "size"):
            self.used.setdefault(trait, set()).update(other.get(trait, ()))

    def used_boxes(self) -> list[tuple[tuple[int, int], int]]:
        """Every ``(offset, size)`` training ever placed, sorted.

        A place value is not an offset: it is a *box*, and the region it forbids
        depends on its own size as well as the candidate's (``_forbidden_span``
        is asymmetric in the two). Storing bare offsets and re-labelling them at
        the call site with whatever size was convenient is the §6.2 bug. Its
        sharpest form, which is now a test: four size-4 envs at margin 3 on a
        77-wide scaffold leave 421 legal offsets for a size-30 validation env,
        and **zero** if those same four are relabelled at size 30. So the
        pairing lives here, once, and callers that only want coordinates ask
        for ``used_offsets``.
        """
        return sorted(self.used.get("place", ()))

    def used_offsets(self) -> set[tuple[int, int]]:
        """Just the coordinates, for readers with no use for the extent."""
        return {o for o, _ in self.used.get("place", ())}

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
                # [r, c, size] -- see `used_boxes`. Files written before Phase 6
                # carry [r, c]; `from_json` pairs those with the recorded size.
                "place": sorted([int(o[0]), int(o[1]), int(s)]
                                for o, s in self.used.get("place", ())),
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
            "place": _boxes_from_json(u.get("place", ()), u.get("size", ())),
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
    # 2 since Phase 6: ``used.place`` entries gained their size (``[r, c, size]``
    # rather than ``[r, c]``). Version 1 files load unchanged -- see `from_json`.
    spec_version: int = 2

    def _payload(self) -> dict:
        return {"spec_version": self.spec_version, "generator": self.generator,
                "scaffold": self.scaffold, "split": self.split.to_json()}

    def spec_hash(self) -> str:
        """sha256 over the canonical payload -- key order cannot change it."""
        return _hash_payload(self._payload())

    def to_json(self) -> dict:
        return {**self._payload(), "spec_hash": self.spec_hash()}

    @staticmethod
    def from_json(d: dict) -> "WorldSpec":
        # Hash the file's own payload rather than a re-render of the object.
        # The hash answers "was this edited by hand", which is a claim about the
        # bytes -- and re-rendering makes it a claim about the current
        # serializer instead, so *any* format change invalidates every file
        # already on disk. Phase 6 changed one field's shape and would have
        # rejected every world.json ever written, with a message blaming the
        # user for editing it.
        recorded = d.get("spec_hash")
        if recorded is not None:
            actual = _hash_payload({k: v for k, v in d.items()
                                    if k != "spec_hash"})
            if recorded != actual:
                raise ValueError(
                    f"world spec hash mismatch: file says {recorded[:12]}..., "
                    f"contents hash to {actual[:12]}.... The file was edited "
                    f"after it was written.")
        return WorldSpec(scaffold=d["scaffold"], generator=d["generator"],
                         split=GeneratedSplit.from_json(d["split"]),
                         spec_version=int(d.get("spec_version", 1)))

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
        """Atomically write `world.json` into `save_dir`.

        The temp file carries the writing process's pid. A fixed `.tmp` name is
        safe for one writer and quietly broken for several: two processes
        writing into the same directory both create `world.json.tmp`, the first
        `os.replace` consumes it, and the second raises

            FileNotFoundError: '.../world.json.tmp' -> '.../world.json'

        which is exactly how 246 of a 272-run sweep died. The rename stays
        atomic, so a reader still never sees a half-written file; only the
        staging name needed to be unique.
        """
        import os
        import tempfile
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(str(save_dir), WORLD_SPEC_NAME)
        # mkstemp rather than a pid suffix: unique across threads as well as
        # processes, and it creates the file atomically. Same directory, so the
        # rename below stays on one filesystem and stays atomic.
        fd, tmp = tempfile.mkstemp(
            dir=str(save_dir), prefix=WORLD_SPEC_NAME + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self.to_json(), f, indent=2, sort_keys=True)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
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
