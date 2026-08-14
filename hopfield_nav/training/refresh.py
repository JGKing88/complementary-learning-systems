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

import numpy as np

from ..config import TrainConfig
from ..world import domains as dom
from ..world import generate
from ..world.generate import build_envs, sample_places
from ..world.scaffold import fit_env_assoc
from ..world.spec import EnvSpec, GeneratedSplit
from ..world.world import World

TRAITS = ("place", "wall", "goal", "size")

# Draw orders the preflight tries before believing a val set of the requested
# size is reachable. Greedy packing is order-dependent and `make_val_set`
# shuffles with the eval's own seed, so one order is not evidence.
_PREFLIGHT_ORDERS = 5


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
                 worlds: list[World] | None, env_cfg, movement_mode: str,
                 seed: int):
        self.cadence = cadence
        self.split = split
        self.worlds = worlds
        self.env_cfg = env_cfg
        self.movement_mode = movement_mode
        self.seed = int(seed)
        self.ticks = 0
        self.counts = {t: 0 for t in TRAITS}

        # ``worlds=None`` is *draw only*: the values are drawn and recorded, and
        # nothing is built. That is what makes the preflight cheap and, more to
        # the point, exact -- it runs `_draw`, the same function the run will,
        # rather than a reimplementation that could drift from it. Skipping
        # `_apply` also skips the env rebuild, which is the whole cost of a wall
        # tick.
        if worlds is not None:
            # `split.train` is the flat, in-order truth for every train env
            # across every world; `_apply` slices it back into worlds on that
            # assumption. If the two ever disagree a refresh would write env i's
            # traits onto env j -- silently, since both are legal envs.
            live = [off for w in worlds for off in w.offsets]
            if live != [s.offset for s in split.train]:
                raise ValueError(
                    f"the worlds hold {len(live)} envs whose offsets do not "
                    f"match the {len(split.train)} recorded train specs, in "
                    "order. Refresh slices split.train back into worlds and "
                    "would pair the wrong env with the wrong spec.")

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

    def fast_forward(self, through_tick: int) -> int:
        """Replay every tick up to ``through_tick``, then apply the result once.

        A resume rebuilds its worlds from ``(seed, config)``, which reproduces
        the run's *first* update rather than its Nth. Without this the second
        segment of a resumed run would train on the envs the first segment
        started with, silently undoing every refresh that happened in between --
        the same class of bug as the frozen goals v22 fixed, and just as quiet.

        The ticks cannot be skipped to the last one. The wall draw excludes
        ``split.used``, so tick N's seeds depend on every tick before it. They
        can be *replayed* exactly: each draw is a pure function of
        ``(seed, trait, tick)`` and the accumulated union, and nothing training
        does enters -- the same property `preflight` relies on to be exact
        rather than an estimate.

        Only the draws are replayed. ``_apply`` rebuilds envs and refits
        ``assoc``, and every intermediate world it would build is overwritten by
        the next tick, so it runs once at the end over the union of the traits
        that came due. That union is what makes one apply equivalent to N: a run
        whose last tick was goals-only still needs the rebuild its earlier wall
        tick asked for. Costs about a second for a 300-update run.
        """
        if through_tick <= 0:
            return 0
        seen: set[str] = set()
        for tick in range(1, through_tick + 1):
            due = self.cadence.due(tick)
            if not due:
                continue
            traits = self._draw(due, tick)
            self.ticks += 1
            for t in traits:
                self.counts[t] += 1
            seen |= set(traits)
        if seen:
            self._apply(tuple(t for t in TRAITS if t in seen))
        return self.ticks

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
        if self.worlds is None:              # draw-only; see __init__
            return
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


def preflight(split: GeneratedSplit, cadence: Cadence, n_updates: int,
              env_cfg, movement_mode: str, seed: int, *,
              n_val_envs: int) -> dict:
    """What a post-hoc held-out validation set will still be able to ask for.

    Refreshed values are a pure function of ``(seed, tick)`` and the declared
    domains -- nothing training does enters -- so this does not *estimate* the
    end-state union, it replays the exact ticks. Draw-only, so it costs about a
    second for a 300-update run and builds no environments.

    Two outcomes, and they are not the same kind of thing. A *shrinking eval
    ceiling* is recorded and the run proceeds -- a run that only ever evaluates
    on ``--split recorded`` is fine with a tight union, and that is not the
    trainer's call to veto. A domain that *runs dry* is different: the run will
    raise partway through, hours in, at a tick fixed before it started. The
    caller raises on that one, because a warning is only useful to someone
    watching stdout at second two.

    **The ceiling is a ceiling, not a predicted failure.** An earlier reading of the
    Phase 4 measurements had `place='held_out'` becoming infeasible after ~20-30
    ticks; that was the rejection sampler failing to find the survivors, not
    their absence (see the correction in ``docs/ENV_GENERATOR_STATUS.md``). What
    actually happens is that the largest held-out place set a run can support
    falls from ~187 envs to ~10 and then plateaus -- the last few offsets are
    never consumed, because a tick draws from the ~2.9M positions clear of
    ``base_val`` and hits one of them with probability ~1e-5.

    That ceiling is decided at the start of the run and only visible at the end
    of it, which is the reason to measure it here.
    """
    import copy

    sim = copy.deepcopy(split)
    runner = Refresher(cadence, sim, None, env_cfg, movement_mode, seed)
    dies_at, dies_of = None, None
    for tick in range(1, int(n_updates) + 1):
        try:
            runner.maybe_refresh(tick)
        except Exception as exc:
            # A domain that runs dry mid-run is a different failure from a
            # shrinking eval ceiling: it kills the training run outright, hours
            # in, at a tick that was decided before it started. Report the tick
            # rather than letting the exception escape a diagnostic.
            dies_at, dies_of = tick, f"{type(exc).__name__}: {exc}"
            break

    size = sim.train[0].size if sim.train else int(env_cfg.size)
    # The boxes as recorded -- each with the size it was placed at, not this
    # one relabelled onto all of them (§6.2).
    used_place = sim.used_boxes()
    # The same two calls `make_val_set` makes -- `legal_offsets` then
    # `greedy_pack` -- so the two cannot disagree about which offsets are legal
    # or about the margin rule.
    cand = generate.legal_offsets(used_place, sim.domains.place, size=size,
                                  Npos=sim.Npos, period=sim.period,
                                  margin=sim.margin)

    # They *can* disagree about the count, because greedy packing depends on the
    # order it sees candidates in and `make_val_set` shuffles with the eval's own
    # seed. Measured: an estimate from one fixed order over-promises -- at the
    # reported ceiling, make_val_set succeeded for one val seed and failed for
    # two others. So take the worst of several orders, and cap the search at what
    # was actually asked for: the question is "can a post-hoc eval get
    # `n_val_envs`", not "what is the largest packing that exists", and the cap
    # keeps the work bounded when the legal set still runs to ~10^6 offsets.
    got = []
    for k in range(_PREFLIGHT_ORDERS):
        rng = np.random.RandomState(dom.stable_hash(seed, "preflight", k))
        order = [cand[i] for i in rng.permutation(len(cand))]
        got.append(len(generate.greedy_pack(
            order, size=size, period=sim.period, self_margin=sim.margin,
            limit=int(n_val_envs))))
    available = min(got) if got else 0

    wall_left = (sim.domains.wall.hi - sim.domains.wall.lo
                 - len(sim.used["wall"]))
    report = {
        "n_updates": int(n_updates),
        "ticks": runner.ticks,
        "n_val_envs": int(n_val_envs),
        "used_at_end": {t: len(sim.used.get(t, ())) for t in TRAITS},
        "place_legal_offsets": len(cand),
        # Worst over `_PREFLIGHT_ORDERS` draw orders, capped at n_val_envs.
        "place_val_envs_available": int(available),
        "wall_seeds_left": int(wall_left),
        "goal_cells_held_out": len(sim.goal_cells_val),
        "refresh_dies_at_update": dies_at,
        "refresh_dies_of": dies_of,
        "ok": (dies_at is None and available >= n_val_envs
               and wall_left >= n_val_envs and len(sim.goal_cells_val) > 0),
    }
    return report


def format_preflight(report: dict) -> str:
    """One line if it is fine, and the specific shortfall if it is not."""
    n = report["n_val_envs"]
    head = (f"held-out eval headroom after {report['ticks']} refresh ticks: "
            f"place {report['place_val_envs_available']}/{n} envs "
            f"({report['place_legal_offsets']} legal offsets), "
            f"wall {report['wall_seeds_left']} seeds, "
            f"goal {report['goal_cells_held_out']} cells")
    if report["refresh_dies_at_update"] is not None:
        return (
            f"this run cannot finish: refresh runs out of values at update "
            f"{report['refresh_dies_at_update']} of {report['n_updates']} and "
            f"raises\n    {report['refresh_dies_of']}\n"
            f"  Nothing about the draw depends on training, so it will happen "
            f"exactly there. Widen the domain that ran dry, or lower that "
            f"trait's cadence.")
    if report["ok"]:
        return f"  preflight: {head} — enough for --num_val_envs {n}."
    short = []
    if report["place_val_envs_available"] < n:
        short.append(
            f"place supports {report['place_val_envs_available']} val envs, "
            f"not {n} — {report['used_at_end']['place']} offsets will be in use "
            f"by the end, and a held-out env has to clear every one of them")
    if report["wall_seeds_left"] < n:
        short.append(f"only {report['wall_seeds_left']} wall seeds unused; "
                     "widen --wall_seeds")
    if not report["goal_cells_held_out"]:
        short.append("no goal cells reserved; raise --goal_val_frac")
    return (f"  WARNING: {head}.\n"
            + "".join(f"    - {s}\n" for s in short)
            + "    Training and its own base_val are unaffected — this only "
              "limits a post-hoc --split with held_out. Recorded in world.json; "
              "the run continues.")


__all__ = ["Cadence", "Refresher", "TRAITS", "format_preflight", "preflight"]
