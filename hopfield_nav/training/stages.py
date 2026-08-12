"""The training schedule: what mix of explore and exploit each update runs.

Explore and exploit are not consecutive halves of a rollout. They are two
*regimes* an env can be in, differing only in what its Hopfield holds at t=0 --
the goal pattern (exploit, "follow") or nothing (explore) -- and every update
collects rollouts from both and pools them into one PPO step. So a schedule does
not say "run explore, then run exploit"; it says, for each update, *what
fraction of the envs are in the empty regime*. 1.0 is pure explore, 0.0 is pure
exploit, in between is interleaved.

Before this module that fraction was implicit in four coupled flags -- a
100%-explore warmup prefix plus a single monotone anneal from an initial to a
target fraction -- which between them could not express "200 pure explore, then
800 interleaved, then 100 pure exploit". A schedule is a list of stages:

    explore:200,novelty=0.3 ; interleave:800,empty_frac=1.0->0.5,anneal=50 ; exploit:100,lr=1e-4

Nothing here imports torch. The schedule is a description of a run, and keeping
it a plain value means the grammar and the arithmetic are testable without
building a world.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

# empty-regime fraction implied by a stage kind. `interleave` has none: it is
# the kind that exists precisely to be given one.
KIND_FRACTIONS: dict[str, float | None] = {
    "explore": 1.0,
    "exploit": 0.0,
    "interleave": None,
}

INTERLEAVE_DEFAULT_FRACTION = 0.5

# Per-stage override keys, in the order format_schedule emits them.
STAGE_KEYS = (
    "lr", "empty_frac", "anneal", "novelty", "eps",
    "dist_min", "dist_max", "emp_dist_min", "emp_dist_max",
)


class ScheduleError(ValueError):
    """A schedule string that cannot be parsed, or parses to something absurd."""


@dataclass
class Stage:
    """One contiguous run of updates at a fixed or annealing empty-fraction.

    `empty_frac_start` / `empty_frac_end` are always resolved by the time a
    Stage exists -- `explore` pins both to 1.0, `exploit` to 0.0 -- so nothing
    downstream has to re-derive a fraction from the kind name.

    Every other field is None when the stage does not override it, because
    "unset" and "set to the same value the run defaults to" have to stay
    distinguishable: the first inherits later changes to the default, the
    second does not.
    """
    kind: str
    updates: int
    empty_frac_start: float
    empty_frac_end: float
    anneal: int | None = None       # updates from stage start to reach _end
    lr: float | None = None
    novelty: float | None = None
    eps: float | None = None
    dist_min: int | None = None
    dist_max: int | None = None
    emp_dist_min: int | None = None
    emp_dist_max: int | None = None


@dataclass
class Knobs:
    """The values one update actually runs with.

    Built from the run-wide defaults (after any global anneal) and then
    overwritten by whatever the current stage overrides. See `resolve`.
    """
    lr: float
    empty_frac: float
    novelty: float
    eps: float
    dist_min: int
    dist_max: int
    emp_dist_min: int
    emp_dist_max: int


@dataclass
class RolloutSpec:
    """What one env's rollout needs, as decided by its regime.

    The collector reads novelty off `cfg.hopfield` and the goal reward off the
    env, so somebody has to mutate global state before each rollout. Returning
    this instead lets the regime modules stay pure and puts all the mutation in
    one place in the composer.
    """
    hop: Any                    # a hopfield.Hopfield
    allow_store: bool           # may the agent's store action write to `hop`?
    novelty_reward: float
    goals_active: bool
    reset_goal: bool
    epsilon: float
    goal_in_memory_init: bool


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_fraction(raw: str, where: str) -> tuple[float, float]:
    """`0.5` -> (0.5, 0.5); `1.0->0.5` -> (1.0, 0.5)."""
    parts = [p.strip() for p in raw.split("->")]
    if len(parts) > 2:
        raise ScheduleError(
            f"{where}: empty_frac takes at most one '->', got {raw!r}")
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        raise ScheduleError(
            f"{where}: empty_frac must be a number or 'start->end', "
            f"got {raw!r}") from None
    for v in vals:
        if not 0.0 <= v <= 1.0:
            raise ScheduleError(
                f"{where}: empty_frac must be within [0, 1], got {v}")
    return (vals[0], vals[-1])


def _parse_stage(text: str) -> Stage:
    fields = [f.strip() for f in text.split(",")]
    head = fields[0]
    if ":" not in head:
        raise ScheduleError(
            f"stage {text!r}: expected '<kind>:<updates>', found no ':'. "
            f"Kinds are {', '.join(sorted(KIND_FRACTIONS))}.")
    kind, _, raw_updates = head.partition(":")
    kind = kind.strip()
    if kind not in KIND_FRACTIONS:
        raise ScheduleError(
            f"stage {text!r}: unknown kind {kind!r}; "
            f"expected one of {', '.join(sorted(KIND_FRACTIONS))}")
    try:
        updates = int(raw_updates.strip())
    except ValueError:
        raise ScheduleError(
            f"stage {text!r}: update count must be an integer, "
            f"got {raw_updates.strip()!r}") from None
    if updates < 1:
        raise ScheduleError(
            f"stage {text!r}: update count must be at least 1, got {updates}")

    kwargs: dict[str, str] = {}
    for field in fields[1:]:
        if not field:
            continue
        if "=" not in field:
            raise ScheduleError(
                f"stage {text!r}: expected 'key=value', got {field!r}. "
                f"Keys are {', '.join(STAGE_KEYS)}.")
        key, _, value = field.partition("=")
        key, value = key.strip(), value.strip()
        if key not in STAGE_KEYS:
            raise ScheduleError(
                f"stage {text!r}: unknown key {key!r}; "
                f"expected one of {', '.join(STAGE_KEYS)}")
        if key in kwargs:
            raise ScheduleError(f"stage {text!r}: {key} given twice")
        kwargs[key] = value

    implied = KIND_FRACTIONS[kind]
    if "empty_frac" in kwargs:
        if implied is not None:
            raise ScheduleError(
                f"stage {text!r}: {kind!r} already means empty_frac={implied}; "
                f"use 'interleave' to set a fraction explicitly")
        start, end = _parse_fraction(kwargs.pop("empty_frac"), f"stage {text!r}")
    elif implied is not None:
        start = end = implied
    else:
        start = end = INTERLEAVE_DEFAULT_FRACTION

    anneal = None
    if "anneal" in kwargs:
        raw = kwargs.pop("anneal")
        try:
            anneal = int(raw)
        except ValueError:
            raise ScheduleError(
                f"stage {text!r}: anneal must be an integer, got {raw!r}"
            ) from None
        if start == end:
            raise ScheduleError(
                f"stage {text!r}: anneal is meaningless without an "
                f"'empty_frac=start->end' to anneal along")
        if anneal < 1:
            raise ScheduleError(
                f"stage {text!r}: anneal must be at least 1, got {anneal}")
        if anneal > updates:
            raise ScheduleError(
                f"stage {text!r}: anneal={anneal} exceeds the stage's "
                f"{updates} updates, so the ramp would never finish. Lengthen "
                f"the stage or shorten the anneal.")

    def _num(key, cast):
        if key not in kwargs:
            return None
        raw = kwargs[key]
        try:
            return cast(raw)
        except ValueError:
            raise ScheduleError(
                f"stage {text!r}: {key} must be "
                f"{'an integer' if cast is int else 'a number'}, got {raw!r}"
            ) from None

    stage = Stage(
        kind=kind, updates=updates,
        empty_frac_start=start, empty_frac_end=end, anneal=anneal,
        lr=_num("lr", float),
        novelty=_num("novelty", float),
        eps=_num("eps", float),
        dist_min=_num("dist_min", int),
        dist_max=_num("dist_max", int),
        emp_dist_min=_num("emp_dist_min", int),
        emp_dist_max=_num("emp_dist_max", int),
    )
    for key in ("lr", "novelty", "eps"):
        v = getattr(stage, key)
        if v is not None and v < 0:
            raise ScheduleError(f"stage {text!r}: {key} must be >= 0, got {v}")
    for key in ("dist_min", "dist_max", "emp_dist_min", "emp_dist_max"):
        v = getattr(stage, key)
        if v is not None and v < 0:
            raise ScheduleError(f"stage {text!r}: {key} must be >= 0, got {v}")
    for lo, hi in (("dist_min", "dist_max"), ("emp_dist_min", "emp_dist_max")):
        a, b = getattr(stage, lo), getattr(stage, hi)
        if a is not None and b is not None and a > b:
            raise ScheduleError(f"stage {text!r}: {lo}={a} exceeds {hi}={b}")
    return stage


def parse_schedule(text: str) -> list[Stage]:
    """`"explore:200 ; interleave:800,empty_frac=1.0->0.5"` -> [Stage, Stage].

    Stages are separated by ';', a stage is `<kind>:<updates>` followed by
    optional `,key=value` overrides. Whitespace anywhere is ignored.
    """
    stages = [_parse_stage(chunk.strip())
              for chunk in text.split(";") if chunk.strip()]
    if not stages:
        raise ScheduleError(
            f"schedule {text!r} declares no stages; expected something like "
            f"'explore:200 ; interleave:800,empty_frac=1.0->0.5'")
    return stages


def format_schedule(stages: list[Stage]) -> str:
    """Canonical form, round-tripping through `parse_schedule`."""
    out = []
    for s in stages:
        parts = [f"{s.kind}:{s.updates}"]
        if s.kind == "interleave" or s.empty_frac_start != s.empty_frac_end:
            frac = (f"{s.empty_frac_start:g}" if s.empty_frac_start == s.empty_frac_end
                    else f"{s.empty_frac_start:g}->{s.empty_frac_end:g}")
            parts.append(f"empty_frac={frac}")
        if s.anneal is not None:
            parts.append(f"anneal={s.anneal}")
        for key in ("lr", "novelty", "eps",
                    "dist_min", "dist_max", "emp_dist_min", "emp_dist_max"):
            v = getattr(s, key)
            if v is not None:
                parts.append(f"{key}={v:g}")
        out.append(",".join(parts))
    return " ; ".join(out)


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------

def total_updates(stages: list[Stage]) -> int:
    return sum(s.updates for s in stages)


def stage_at(stages: list[Stage], update: int) -> tuple[Stage, int]:
    """Global 1-indexed update -> (its stage, its 1-indexed position in it)."""
    if update < 1:
        raise IndexError(f"update is 1-indexed, got {update}")
    consumed = 0
    for stage in stages:
        if update <= consumed + stage.updates:
            return stage, update - consumed
        consumed += stage.updates
    raise IndexError(
        f"update {update} is past the schedule's {consumed} updates")


def empty_fraction_at(stage: Stage, local_update: int) -> float:
    """Fraction of envs in the empty (explore) regime, at a stage-local update.

    The anneal clock is stage-local: `anneal=50` means "50 updates into *this
    stage*", not into the run. That differs from the pre-schedule code, whose
    single anneal was keyed off the global counter and so was already partway
    through by the time a warmup prefix ended. Stage-local is the only reading
    that survives a stage appearing anywhere in a schedule.
    """
    start, end = stage.empty_frac_start, stage.empty_frac_end
    if start == end:
        return start
    span = stage.anneal if stage.anneal is not None else stage.updates
    t = min(1.0, max(0.0, (local_update - 1) / float(span)))
    return start + t * (end - start)


def resolve(stage: Stage, local_update: int, defaults: Knobs) -> Knobs:
    """The run-wide defaults for this update, with the stage's overrides on top.

    Overrides are absolute, not multiplicative: `explore:200,novelty=0.3` means
    novelty is exactly 0.3 for those updates even when `--novelty_anneal` has
    scaled the default down. One rule, so a schedule can be read without also
    reading the global flags.
    """
    knobs = replace(defaults, empty_frac=empty_fraction_at(stage, local_update))
    for field in ("lr", "novelty", "eps",
                  "dist_min", "dist_max", "emp_dist_min", "emp_dist_max"):
        v = getattr(stage, field)
        if v is not None:
            setattr(knobs, field, v)
    return knobs


__all__ = [
    "INTERLEAVE_DEFAULT_FRACTION", "KIND_FRACTIONS", "Knobs", "RolloutSpec",
    "STAGE_KEYS", "ScheduleError", "Stage", "empty_fraction_at",
    "format_schedule", "parse_schedule", "resolve", "stage_at",
    "total_updates",
]
