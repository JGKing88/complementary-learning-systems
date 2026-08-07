"""Per-trait domains: where an environment's four traits are allowed to land.

An env is four independent things -- wall pattern, env-local goal cell, scaffold
placement, size (see ``docs/EVAL_SPLITS_DESIGN.md`` Part 1). Each gets a
*domain*: a declared, serializable description of the legal values. Training
declares its domains; validation draws from the same domain minus what training
used (``held_out``), or from the complement (``ood``).

Two of the four have a bounded universe and therefore a meaningful complement:

    place   offsets live in [0, Npos)^2, so "outside this Rect" is well-defined
    goal    cells live in [0, size)^2, so "not in this ring" is well-defined

The other two do not. There is no universe of all wall patterns to take a
complement of -- novelty there just means *a seed training never drew*, which is
what ``held_out`` already is. And size is unbounded above, so a size OOD set is
named outright rather than derived. ``complement()`` therefore raises on
``WallDomain`` and ``SizeDomain``, with a message saying which knob to use
instead. That asymmetry is real, not an omission.

Domains are values: immutable, hashable, JSON round-trippable.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Deterministic seeding
# ---------------------------------------------------------------------------

def stable_hash(*parts) -> int:
    """A seed derived from ``parts``, stable across processes and machines.

    **Not** Python's ``hash()``: that is salted per interpreter run
    (``PYTHONHASHSEED``), so a run seeded from it would not reproduce when
    re-launched. blake2b over a canonical string has no such problem.
    """
    payload = "\x1f".join(str(p) for p in parts)
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2 ** 32 - 1)


def trait_rng(run_seed: int, trait: str, tick: int = 0,
              role: str = "train") -> np.random.RandomState:
    """An independent stream for one (role, trait, tick).

    *Derived* per key rather than advanced from a single stream. That is what
    makes "refresh placement only, hold walls and goals" reproducible without
    replaying the other traits: each trait's values depend on its own key alone,
    so changing the refresh cadence of one trait cannot move another.
    """
    return np.random.RandomState(stable_hash(run_seed, role, trait, tick))


# ---------------------------------------------------------------------------
# Place
# ---------------------------------------------------------------------------

class PlaceDomain:
    """Legal top-left offsets for an env of a given size.

    ``Rect`` means *the env footprint is contained in the rect* -- "the area in
    which envs can be placed" -- not "the top-left corner is in the rect". The
    two differ by ``size`` at the far edge and the first reading is the one that
    keeps an env inside the region you drew.
    """

    kind = "place"

    def bounds(self, size: int, Npos: int) -> tuple[int, int, int, int]:
        """Inclusive offset box (x_lo, y_lo, x_hi, y_hi) this domain samples in."""
        raise NotImplementedError

    def candidate(self, rng, size: int, Npos: int) -> tuple[int, int]:
        x_lo, y_lo, x_hi, y_hi = self.bounds(size, Npos)
        if x_hi < x_lo or y_hi < y_lo:
            raise ValueError(
                f"{self!r} has no legal offset for size={size}, Npos={Npos}"
            )
        return (int(rng.randint(x_lo, x_hi + 1)), int(rng.randint(y_lo, y_hi + 1)))

    def contains(self, offset, size: int, Npos: int) -> bool:
        x_lo, y_lo, x_hi, y_hi = self.bounds(size, Npos)
        return x_lo <= offset[0] <= x_hi and y_lo <= offset[1] <= y_hi

    def capacity(self, size: int, margin: int, Npos: int) -> int:
        """Envs of ``size`` placeable on a lattice with ``margin`` clearance."""
        x_lo, y_lo, x_hi, y_hi = self.bounds(size, Npos)
        pitch = size + margin
        nx = (x_hi - x_lo) // pitch + 1 if x_hi >= x_lo else 0
        ny = (y_hi - y_lo) // pitch + 1 if y_hi >= y_lo else 0
        return max(0, nx) * max(0, ny)

    def complement(self, margin: int = 0) -> "PlaceDomain":
        raise NotImplementedError

    def to_json(self) -> dict:
        raise NotImplementedError


@dataclass(frozen=True)
class Anywhere(PlaceDomain):
    """The whole scaffold."""

    def bounds(self, size, Npos):
        return (0, 0, Npos - size, Npos - size)

    def complement(self, margin: int = 0):
        raise ValueError(
            "Anywhere has no complement -- there is nothing outside the whole "
            "scaffold. A model trained with place=Anywhere has no place-OOD set; "
            "declare a Rect at training time if you want to test OOD placement."
        )

    def to_json(self):
        return {"kind": "Anywhere"}


@dataclass(frozen=True)
class Rect(PlaceDomain):
    """Envs whose footprint lies wholly inside [x0, x0+w) x [y0, y0+h)."""

    x0: int
    y0: int
    w: int
    h: int

    def bounds(self, size, Npos):
        return (self.x0, self.y0,
                min(self.x0 + self.w - size, Npos - size),
                min(self.y0 + self.h - size, Npos - size))

    def complement(self, margin: int = 0):
        return OutsideRect(self, int(margin))

    def to_json(self):
        return {"kind": "Rect", "x0": self.x0, "y0": self.y0,
                "w": self.w, "h": self.h}


@dataclass(frozen=True)
class OutsideRect(PlaceDomain):
    """Envs whose footprint clears ``inner`` by at least ``margin`` cells.

    Sampled by rejection over the whole scaffold; ``capacity`` is an estimate
    (whole-scaffold capacity minus the excluded band), which is all the
    preflight check needs.
    """

    inner: Rect
    margin: int = 0

    def bounds(self, size, Npos):
        return (0, 0, Npos - size, Npos - size)

    def contains(self, offset, size, Npos):
        if not super().contains(offset, size, Npos):
            return False
        r, m = self.inner, self.margin
        # Flat (non-toroidal) clearance: the excluded region is a literal
        # rectangle in scaffold coordinates, not a phase class.
        sep_x = max(r.x0 - (offset[0] + size), offset[0] - (r.x0 + r.w))
        sep_y = max(r.y0 - (offset[1] + size), offset[1] - (r.y0 + r.h))
        return max(sep_x, sep_y) >= m

    def capacity(self, size, margin, Npos):
        whole = Anywhere().capacity(size, margin, Npos)
        blocked = Rect(
            max(0, self.inner.x0 - self.margin),
            max(0, self.inner.y0 - self.margin),
            self.inner.w + 2 * self.margin,
            self.inner.h + 2 * self.margin,
        ).capacity(size, margin, Npos)
        return max(0, whole - blocked)

    def complement(self, margin: int = 0):
        return self.inner

    def to_json(self):
        return {"kind": "OutsideRect", "inner": self.inner.to_json(),
                "margin": self.margin}


# ---------------------------------------------------------------------------
# Goal
# ---------------------------------------------------------------------------

class GoalDomain:
    """Legal goal cells in **env-local** coordinates, [0, size)^2.

    Local, not global: "if any training env ever puts its goal at (12, 9), no
    base-val env may use (12, 9)" is a statement about the cell within an
    arena, and the forbidden set is the union over all envs and all refresh
    ticks.
    """

    kind = "goal"

    def cells(self, size: int) -> frozenset[tuple[int, int]]:
        raise NotImplementedError

    def complement(self) -> "GoalDomain":
        raise NotImplementedError

    def to_json(self) -> dict:
        raise NotImplementedError


def _all_cells(size: int) -> frozenset[tuple[int, int]]:
    return frozenset((x, y) for x in range(size) for y in range(size))


@dataclass(frozen=True)
class AnyCells(GoalDomain):
    def cells(self, size):
        return _all_cells(size)

    def complement(self):
        raise ValueError(
            "AnyCells has no complement -- every cell is already legal. A model "
            "trained with goal=AnyCells has no goal-region-OOD set; declare "
            "Ring/Interior/Quadrant at training time to test that axis."
        )

    def to_json(self):
        return {"kind": "AnyCells"}


@dataclass(frozen=True)
class Cells(GoalDomain):
    """An explicit cell set. What both goal branches serialize to (§2.8)."""

    members: frozenset

    def cells(self, size):
        return frozenset(c for c in self.members
                         if 0 <= c[0] < size and 0 <= c[1] < size)

    def complement(self):
        return NotCells(self.members)

    def to_json(self):
        return {"kind": "Cells",
                "members": sorted([int(c[0]), int(c[1])] for c in self.members)}


@dataclass(frozen=True)
class NotCells(GoalDomain):
    members: frozenset

    def cells(self, size):
        return _all_cells(size) - frozenset(self.members)

    def complement(self):
        return Cells(self.members)

    def to_json(self):
        return {"kind": "NotCells",
                "members": sorted([int(c[0]), int(c[1])] for c in self.members)}


@dataclass(frozen=True)
class Ring(GoalDomain):
    """Cells within ``width`` of the arena border."""

    width: int = 1

    def cells(self, size):
        w = self.width
        return frozenset((x, y) for x in range(size) for y in range(size)
                         if x < w or y < w or x >= size - w or y >= size - w)

    def complement(self):
        return Interior(self.width)

    def to_json(self):
        return {"kind": "Ring", "width": self.width}


@dataclass(frozen=True)
class Interior(GoalDomain):
    """Everything ``Ring(width)`` is not."""

    width: int = 1

    def cells(self, size):
        return _all_cells(size) - Ring(self.width).cells(size)

    def complement(self):
        return Ring(self.width)

    def to_json(self):
        return {"kind": "Interior", "width": self.width}


@dataclass(frozen=True)
class Quadrant(GoalDomain):
    """One of the four arena quadrants, indexed 0..3 as (x_hi, y_hi) bits."""

    index: int

    def cells(self, size):
        half = size // 2
        xs = range(half, size) if self.index & 1 else range(half)
        ys = range(half, size) if self.index & 2 else range(half)
        return frozenset((x, y) for x in xs for y in ys)

    def complement(self):
        raise ValueError(
            "Quadrant.complement() needs the arena size to enumerate its cells "
            "-- a quadrant is defined relative to `size // 2`. Use "
            "complement_for(domain, size), which every caller inside the "
            "generator already does."
        )

    def to_json(self):
        return {"kind": "Quadrant", "index": self.index}


def complement_for(domain: GoalDomain, size: int) -> GoalDomain:
    """Size-aware complement: always an explicit ``Cells`` over [0, size)^2.

    ``Ring``/``Interior`` know their own complement without a size, but
    ``Quadrant`` does not -- enumerating it needs the arena width. Going through
    ``Cells`` makes every complement concrete and serializable the same way.
    """
    return Cells(_all_cells(size) - domain.cells(size))


# ---------------------------------------------------------------------------
# Wall and size -- no complement (see module docstring)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SeedRange:
    """Wall-pattern seeds drawn from [lo, hi). Novelty is seed disjointness."""

    kind = "wall"
    lo: int
    hi: int

    def sample(self, rng, n: int, exclude: frozenset = frozenset()) -> list[int]:
        if self.hi - self.lo - len(exclude) < n:
            raise ValueError(
                f"SeedRange({self.lo}, {self.hi}) cannot yield {n} seeds "
                f"outside the {len(exclude)} already used"
            )
        out: list[int] = []
        seen = set(exclude)
        while len(out) < n:
            s = int(rng.randint(self.lo, self.hi))
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    def complement(self):
        raise ValueError(
            "SeedRange has no complement: there is no bounded universe of wall "
            "patterns. Wall novelty is 'a seed training never drew', which is "
            "what level='held_out' already gives you."
        )

    def to_json(self):
        return {"kind": "SeedRange", "lo": self.lo, "hi": self.hi}


@dataclass(frozen=True)
class Sizes:
    """Legal env sizes. A set of one, in every current run."""

    kind = "size"
    values: tuple[int, ...]

    def sample(self, rng, n: int) -> list[int]:
        return [int(self.values[rng.randint(len(self.values))]) for _ in range(n)]

    def complement(self):
        raise ValueError(
            "Sizes has no complement: env size is unbounded above, so there is "
            "no set to take a complement of. Name the OOD size outright -- "
            "make_val_set(..., size=28)."
        )

    def to_json(self):
        return {"kind": "Sizes", "values": list(self.values)}


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------

_REGISTRY = {
    "Anywhere": lambda d: Anywhere(),
    "Rect": lambda d: Rect(d["x0"], d["y0"], d["w"], d["h"]),
    "OutsideRect": lambda d: OutsideRect(from_json(d["inner"]), d["margin"]),
    "AnyCells": lambda d: AnyCells(),
    "Cells": lambda d: Cells(frozenset(tuple(c) for c in d["members"])),
    "NotCells": lambda d: NotCells(frozenset(tuple(c) for c in d["members"])),
    "Ring": lambda d: Ring(d["width"]),
    "Interior": lambda d: Interior(d["width"]),
    "Quadrant": lambda d: Quadrant(d["index"]),
    "SeedRange": lambda d: SeedRange(d["lo"], d["hi"]),
    "Sizes": lambda d: Sizes(tuple(d["values"])),
}


def from_json(d: dict):
    """Rebuild any domain from its ``to_json`` form."""
    kind = d.get("kind")
    if kind not in _REGISTRY:
        raise ValueError(f"unknown domain kind {kind!r}")
    return _REGISTRY[kind](d)


__all__ = [
    "Anywhere", "AnyCells", "Cells", "GoalDomain", "Interior", "NotCells",
    "OutsideRect", "PlaceDomain", "Quadrant", "Rect", "Ring", "SeedRange",
    "Sizes", "complement_for", "from_json", "stable_hash", "trait_rng",
]
