"""Re-drawing some of a train env's traits on a cadence, from the train domain.

An env is four independent traits (``docs/EVAL_SPLITS_DESIGN.md`` Part 1), so
"refresh the environments" is four separate decisions, not one. A run can move
its envs around the scaffold every update while holding wall patterns fixed, or
re-roll goals while everything else stays put. Each trait gets its own cadence
and its own derived RNG stream, so changing one cannot move another.

**Only the train set refreshes.** ``base_val`` is drawn once and held. A
validation set that moved under the model would make every in-training curve
uninterpretable -- an eval that improved could just as easily have drawn an
easier world, and nothing in the number would say which.

The invariant this module exists to protect
-------------------------------------------

``make_val_set`` excludes against ``split.used[trait]`` -- the union of every
value the trait has *ever* taken. A tick that re-draws placements without
folding them into that union silently narrows the exclusion set, and a later
``held_out`` validation env can be placed exactly where training was. Nothing
raises; the split just stops meaning anything.

So drawing and recording are one operation, and the drawn specs are never handed
back: ``_draw`` writes ``split.train`` and calls ``record_used``, and ``_apply``
reads ``split.train``. There is no arrangement of these calls that applies a
refresh nobody recorded.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import TrainConfig
from ..world import domains as dom
from ..world.generate import build_envs, sample_places
from ..world.scaffold import fit_env_assoc
from ..world.spec import EnvSpec, GeneratedSplit
from ..world.world import World

TRAITS = ("place", "wall", "goal", "size")


@dataclass(frozen=True)
class Cadence:
    """How often each trait is re-drawn, in updates. ``None`` means never."""

    place: int | None = None
    wall: int | None = None
    goal: int | None = None
    size: int | None = None

    def __bool__(self) -> bool:
        return any(getattr(self, t) for t in TRAITS)

    def due(self, tick: int) -> tuple[str, ...]:
        """Traits due on update ``tick`` (1-indexed, as the loop counts).

        ``tick % n == 0``, so ``--refresh_place 10`` first fires on update 10 and
        the generated draw stands until then, while ``1`` refreshes from the
        first update onward.
        """
        return tuple(t for t in TRAITS
                     if getattr(self, t) and tick % getattr(self, t) == 0)

    def describe(self) -> str:
        on = [f"{t}/{getattr(self, t)}u" for t in TRAITS if getattr(self, t)]
        return ", ".join(on) if on else "off"

    def to_json(self) -> dict:
        return {t: getattr(self, t) for t in TRAITS}

    @staticmethod
    def from_config(cfg: TrainConfig) -> "Cadence":
        """Read the four flags off a config, rejecting the unusable combinations.

        Refresh has nowhere to draw *from* without declared domains: the legacy
        placement path has no place domain, no goal-cell partition and no seed
        range. That has to be a startup error rather than a flag that quietly
        does nothing.
        """
        cad = Cadence(**{t: getattr(cfg, f"refresh_{t}", None) for t in TRAITS})
        for t in TRAITS:
            n = getattr(cad, t)
            if n is not None and n < 1:
                raise ValueError(
                    f"--refresh_{t} is a cadence in updates and must be >= 1, "
                    f"got {n}. Omit the flag to never refresh {t}.")
        if cad and not cfg.env_generator:
            raise ValueError(
                f"env refresh ({cad.describe()}) needs --env_generator. The "
                "legacy placement path declares no domains, so there is nothing "
                "to re-draw from -- no place region, no wall seed range, and no "
                "train/val goal-cell partition to keep a refreshed goal out of "
                "the validation set.")
        return cad


class Refresher:
    """Owns the cadence and rewrites the train worlds in place.

    Constructed only when something actually refreshes, so a run that does not
    ask for it pays nothing and behaves exactly as before.
    """

    def __init__(self, cadence: Cadence, split: GeneratedSplit,
                 worlds: list[World], env_cfg, movement_mode: str, seed: int):
        self.cadence = cadence
        self.split = split
        self.worlds = worlds
        self.env_cfg = env_cfg
        self.movement_mode = movement_mode
        self.seed = int(seed)
        self.ticks = 0
        self.counts = {t: 0 for t in TRAITS}

        # `split.train` is the flat, in-order truth for every train env across
        # every world; `_apply` slices it back into worlds on that assumption.
        # If the two ever disagree a refresh would write env i's traits onto env
        # j -- silently, since both are legal envs.
        live = [off for w in worlds for off in w.offsets]
        if live != [s.offset for s in split.train]:
            raise ValueError(
                f"the worlds hold {len(live)} envs whose offsets do not match "
                f"the {len(split.train)} recorded train specs, in order. "
                "Refresh slices split.train back into worlds and would pair the "
                "wrong env with the wrong spec.")

        if cadence.size and len(split.domains.size.values) < 2:
            raise ValueError(
                "--refresh_size needs more than one declared env size; the "
                f"size domain is {split.domains.size.values}, taken from "
                "--size, so a refresh could only ever redraw the same value. "
                "Declaring multiple sizes is Phase 6 (size OOD).")

        # Validation is fixed, so its footprints and seeds are computed once.
        self._val_footprints = [(v.offset, v.size) for v in split.base_val]
        self._val_seeds = frozenset(v.wall_seed for v in split.base_val)

    # -- the one public entry ------------------------------------------------

    def maybe_refresh(self, tick: int) -> tuple[str, ...]:
        """Refresh whatever is due at ``tick``. Returns the traits refreshed."""
        due = self.cadence.due(tick)
        if not due:
            return ()
        traits = self._draw(due, tick)
        self._apply(traits)
        self.ticks += 1
        for t in traits:
            self.counts[t] += 1
        return traits

    def report(self) -> dict:
        """What refreshed, how often, and how far the used union has grown."""
        return {
            "cadence": self.cadence.to_json(),
            "ticks": self.ticks,
            "counts": dict(self.counts),
            "n_used": {t: len(self.split.used.get(t, ())) for t in TRAITS},
        }

    # -- draw and record, one operation --------------------------------------

    def _draw(self, traits, tick: int) -> tuple[str, ...]:
        """Re-draw ``traits`` across the whole train set and record the result.

        Returns the traits actually refreshed, which can be wider than asked.
        Nothing is returned to the caller but the *names*: the values go into
        ``split.train`` and ``split.used`` here, which is what makes an
        unrecorded refresh unrepresentable rather than merely discouraged.
        """
        split, cur = self.split, self.split.train
        n = len(cur)
        traits = set(traits)

        sizes = [s.size for s in cur]
        if "size" in traits:
            rng = dom.trait_rng(self.seed, "size", tick)
            sizes = [int(split.domains.size.sample(rng, 1)[0])] * n
            # A new footprint invalidates the packing and the arena at once:
            # offsets were spaced for the old size, and a goal cell can fall
            # outside the new one outright. Neither is left to a separate
            # cadence, because a run with only --refresh_size would otherwise
            # produce envs whose goals are out of bounds.
            traits |= {"place", "goal"}
        size = sizes[0]

        offsets = [s.offset for s in cur]
        if "place" in traits:
            cap = split.domains.place.capacity(size, split.margin, split.Npos)
            if cap < n + len(split.base_val):
                raise ValueError(
                    f"place domain {split.domains.place!r} holds ~{cap} envs of "
                    f"size {size} at margin {split.margin}, but a refresh needs "
                    f"{n} train envs clear of {len(split.base_val)} fixed val "
                    "envs. At the initial size this bound passed, so the size "
                    "refresh is what broke it.")
            rng = dom.trait_rng(self.seed, "place", tick)
            # Excluding the fixed val set at margin is what keeps the train/val
            # separation a property of every draw, not just the first one.
            offsets = sample_places(
                split.domains.place, rng, n, size=size, Npos=split.Npos,
                period=split.period, exclude=self._val_footprints,
                margin=split.margin, self_margin=split.margin)

        seeds = [s.wall_seed for s in cur]
        if "wall" in traits:
            rng = dom.trait_rng(self.seed, "wall", tick)
            # Excluded from every seed training has already drawn, so each tick
            # is genuinely new wall patterns rather than a reshuffle, and from
            # the val seeds, which must stay disjoint for the whole run.
            seeds = split.domains.wall.sample(
                rng, n, exclude=frozenset(split.used["wall"]) | self._val_seeds)

        goals = [s.goal for s in cur]
        if "goal" in traits:
            rng = dom.trait_rng(self.seed, "goal", tick)
            # From the *partition*, never from the domain minus what was used:
            # a fresh cell each tick would consume the whole grid in a handful
            # of updates and leave validation nothing. `generate_split` caps the
            # train share up front when refresh_goal is on, and this draws
            # inside that cap forever.
            pool = sorted(c for c in split.goal_cells_train
                          if c[0] < size and c[1] < size)
            if not pool:
                raise ValueError(
                    f"no train goal cell fits an env of size {size}: the "
                    f"partition holds {len(split.goal_cells_train)} cells, none "
                    "inside the refreshed arena.")
            goals = [pool[i] for i in
                     rng.choice(len(pool), n, replace=len(pool) < n)]

        split.train = [EnvSpec(int(seeds[i]), int(sizes[i]),
                               (int(offsets[i][0]), int(offsets[i][1])),
                               (int(goals[i][0]), int(goals[i][1])))
                       for i in range(n)]
        split.record_used(split.train)
        return tuple(t for t in TRAITS if t in traits)

    # -- apply the recorded draw ---------------------------------------------

    def _apply(self, traits: tuple[str, ...]) -> None:
        """Push ``split.train`` -- the recorded draw, and nothing else -- onto
        the worlds.

        Takes no specs by design. The only thing appliable is what ``_draw``
        wrote, so there is no path that moves an env without the move being in
        the union ``make_val_set`` excludes against.
        """
        rebuild = bool({"wall", "size"} & set(traits))
        i = 0
        for world in self.worlds:
            chunk = self.split.train[i:i + len(world.envs)]
            i += len(chunk)
            if rebuild:
                # A wall code is fixed at construction from (seed, size), so a
                # new one means a new env. `build_envs` applies the goal too.
                world.envs = build_envs(chunk, self.env_cfg, self.movement_mode)
            elif "goal" in traits:
                for env, spec in zip(world.envs, chunk):
                    env.set_goal(spec.goal)
            if "place" in traits:
                world.offsets = [s.offset for s in chunk]
            # Wsp/Wps are fitted to (observation, global cell) pairs, so a new
            # wall code or a moved footprint invalidates them. `assoc` is None
            # under static_vectorhash -- every current run -- and this whole
            # branch costs nothing there.
            if world.assoc is not None and (rebuild or "place" in traits):
                world.assoc = fit_env_assoc(world.field, world.envs,
                                            world.offsets)


__all__ = ["Cadence", "Refresher", "TRAITS"]
