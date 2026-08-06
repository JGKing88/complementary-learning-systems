"""The training schedule: grammar, lookup, and the arithmetic it replaced.

Until 2026-08 the explore/exploit mix over a run was implicit in four coupled
`train_navigate` flags -- a 100%-explore warmup prefix, an initial fraction, a
target fraction, and an anneal length. `--schedule` replaces them with an
explicit stage list. What could silently regress is the arithmetic, so these
drive it directly rather than running a trainer; the smoke suite covers the
trainer end to end.

The one deliberate behavior change is pinned here too: the old anneal clock was
*global*, so with a warmup prefix it was already partway through by the time the
first interleaved update ran. The stage-local clock is the only reading that
survives a stage appearing anywhere in a schedule.
"""
from __future__ import annotations

import argparse

import pytest

from hopfield_nav.training.stages import (
    Knobs, ScheduleError, empty_fraction_at, format_schedule, parse_schedule,
    resolve, stage_at, total_updates,
)


DEFAULTS = Knobs(lr=3e-4, empty_frac=0.0, novelty=0.1, eps=0.0,
                 dist_min=0, dist_max=0, emp_dist_min=0, emp_dist_max=0)


# ---------------------------------------------------------------------------
# Grammar
# ---------------------------------------------------------------------------

def test_kinds_pin_their_own_fraction():
    """`explore` and `exploit` are the two poles; `interleave` is the dial."""
    explore, exploit, inter = parse_schedule("explore:5 ; exploit:6 ; interleave:7")
    assert (explore.empty_frac_start, explore.empty_frac_end) == (1.0, 1.0)
    assert (exploit.empty_frac_start, exploit.empty_frac_end) == (0.0, 0.0)
    assert (inter.empty_frac_start, inter.empty_frac_end) == (0.5, 0.5)
    assert [s.updates for s in (explore, exploit, inter)] == [5, 6, 7]


@pytest.mark.parametrize("text", [
    "explore:200;exploit:100",
    "explore:200 ; exploit:100",
    "  explore : 200  ;  exploit : 100  ",
    "explore:200 ; exploit:100 ;",
])
def test_whitespace_and_trailing_separators_are_ignored(text):
    stages = parse_schedule(text)
    assert [(s.kind, s.updates) for s in stages] == [("explore", 200), ("exploit", 100)]


def test_per_stage_overrides_parse():
    (s,) = parse_schedule(
        "interleave:800,empty_frac=1.0->0.5,anneal=50,lr=1e-4,novelty=0.3,"
        "eps=0.4,dist_min=1,dist_max=10,emp_dist_min=2,emp_dist_max=8")
    assert (s.empty_frac_start, s.empty_frac_end, s.anneal) == (1.0, 0.5, 50)
    assert (s.lr, s.novelty, s.eps) == (1e-4, 0.3, 0.4)
    assert (s.dist_min, s.dist_max) == (1, 10)
    assert (s.emp_dist_min, s.emp_dist_max) == (2, 8)


def test_unset_overrides_stay_none():
    """None and 'set to the default value' have to stay distinguishable.

    `resolve` inherits the run-wide value for a None field, so a stage that
    said nothing tracks a later change to the flag while one that said
    `novelty=0.1` does not.
    """
    (s,) = parse_schedule("explore:10")
    assert s.lr is None and s.novelty is None and s.eps is None
    assert s.dist_max is None and s.emp_dist_max is None


@pytest.mark.parametrize("text", [
    "explore:200 ; interleave:800,empty_frac=1->0.5,anneal=50 ; exploit:100,lr=0.0001",
    "interleave:13,empty_frac=1->0.25,anneal=6",
    "explore:50,novelty=0.3,eps=0.4",
    "exploit:8",
])
def test_format_round_trips(text):
    stages = parse_schedule(text)
    assert parse_schedule(format_schedule(stages)) == stages


@pytest.mark.parametrize("text,message", [
    ("explore:200,empty_frac=0.5", "already means empty_frac"),
    ("exploit:200,empty_frac=0.5", "already means empty_frac"),
    ("nonsense:10", "unknown kind"),
    ("explore", "found no ':'"),
    ("explore:abc", "must be an integer"),
    ("explore:0", "at least 1"),
    ("explore:-5", "at least 1"),
    ("interleave:10,empty_frac=1.5", "within [0, 1]"),
    ("interleave:10,empty_frac=a->b", "must be a number"),
    ("interleave:10,empty_frac=1->0.5->0", "at most one '->'"),
    ("explore:10,anneal=5", "meaningless without"),
    ("interleave:10,empty_frac=1->0.5,anneal=20", "exceeds the stage's"),
    ("explore:10,bogus=1", "unknown key"),
    ("explore:10,lr=1,lr=2", "given twice"),
    ("explore:10,lr=-1", "must be >= 0"),
    ("exploit:10,dist_min=5,dist_max=2", "exceeds"),
    ("", "declares no stages"),
    ("  ;  ", "declares no stages"),
])
def test_bad_schedules_say_what_is_wrong(text, message):
    with pytest.raises(ScheduleError) as exc:
        parse_schedule(text)
    assert message in str(exc.value)


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------

def test_stage_at_walks_the_boundaries():
    stages = parse_schedule("explore:3 ; interleave:2 ; exploit:2")
    assert total_updates(stages) == 7
    got = [(stage.kind, local)
           for stage, local in (stage_at(stages, u) for u in range(1, 8))]
    assert got == [
        ("explore", 1), ("explore", 2), ("explore", 3),
        ("interleave", 1), ("interleave", 2),
        ("exploit", 1), ("exploit", 2),
    ]


@pytest.mark.parametrize("update", [0, -1])
def test_stage_at_rejects_non_positive_updates(update):
    with pytest.raises(IndexError, match="1-indexed"):
        stage_at(parse_schedule("explore:3"), update)


def test_stage_at_rejects_past_the_end():
    with pytest.raises(IndexError, match="past the schedule"):
        stage_at(parse_schedule("explore:3"), 4)


def test_constant_fraction_never_moves():
    (s,) = parse_schedule("interleave:10,empty_frac=0.25")
    assert [empty_fraction_at(s, u) for u in range(1, 11)] == [0.25] * 10


def test_anneal_defaults_to_the_whole_stage():
    (s,) = parse_schedule("interleave:5,empty_frac=1.0->0.0")
    got = [empty_fraction_at(s, u) for u in range(1, 6)]
    assert got == pytest.approx([1.0, 0.8, 0.6, 0.4, 0.2])


def test_anneal_shorter_than_the_stage_then_holds():
    (s,) = parse_schedule("interleave:6,empty_frac=1.0->0.5,anneal=2")
    got = [empty_fraction_at(s, u) for u in range(1, 7)]
    assert got == pytest.approx([1.0, 0.75, 0.5, 0.5, 0.5, 0.5])


# ---------------------------------------------------------------------------
# The arithmetic this replaced
# ---------------------------------------------------------------------------

def _old_fraction(update, warmup, start, target, anneal):
    """The pre-2026-08 `train_navigate` computation, verbatim.

    Kept as a local copy for the same reason test_ckpt_cadence.py keeps one:
    the point is to compare against what the code *used to do*, and importing
    the current implementation would make the comparison vacuous.
    """
    end = target if target is not None else start
    if anneal > 0:
        t = min(1.0, max(0.0, (update - 1) / float(anneal)))
        frac = start + t * (end - start)
    else:
        frac = start
    return 1.0 if update <= warmup else frac


def test_a_constant_interleave_after_a_warmup_is_unchanged():
    """`warmup=W, fraction=F` is exactly `explore:W ; interleave:N,empty_frac=F`."""
    stages = parse_schedule("explore:3 ; interleave:10,empty_frac=0.5")
    new = [empty_fraction_at(*stage_at(stages, u)) for u in range(1, 14)]
    old = [_old_fraction(u, warmup=3, start=0.5, target=None, anneal=0)
           for u in range(1, 14)]
    assert new == pytest.approx(old)


def test_an_anneal_with_no_warmup_is_unchanged():
    """With nothing before it, the stage-local clock *is* the global clock."""
    stages = parse_schedule("interleave:13,empty_frac=1.0->0.25,anneal=6")
    new = [empty_fraction_at(*stage_at(stages, u)) for u in range(1, 14)]
    old = [_old_fraction(u, warmup=0, start=1.0, target=0.25, anneal=6)
           for u in range(1, 14)]
    assert new == pytest.approx(old)


def test_a_warmup_before_an_anneal_deliberately_differs():
    """The one behavior change, pinned so it cannot happen again by accident.

    The old anneal clock ran during the warmup, where the fraction it computed
    was discarded -- so the first interleaved update inherited a partly-annealed
    fraction (0.7 of the way from 1.0 to 0.5 here, at t=3/5) instead of starting
    at the top. Stage-local starts where the stage starts.
    """
    stages = parse_schedule("explore:3 ; interleave:10,empty_frac=1.0->0.5,anneal=5")
    new = [empty_fraction_at(*stage_at(stages, u)) for u in range(1, 14)]
    old = [_old_fraction(u, warmup=3, start=1.0, target=0.5, anneal=5)
           for u in range(1, 14)]

    assert new[:3] == pytest.approx(old[:3] )      # warmup: both pinned at 1.0
    assert new[3] == pytest.approx(1.0)            # new: the stage starts at the top
    assert old[3] == pytest.approx(0.7)            # old: already 3/5 annealed
    assert new[4:] != pytest.approx(old[4:])


# ---------------------------------------------------------------------------
# resolve()
# ---------------------------------------------------------------------------

def test_a_stage_without_overrides_inherits_every_default():
    (s,) = parse_schedule("explore:10")
    knobs = resolve(s, 1, DEFAULTS)
    assert knobs.lr == DEFAULTS.lr
    assert knobs.novelty == DEFAULTS.novelty
    assert knobs.eps == DEFAULTS.eps
    assert knobs.empty_frac == 1.0          # from the kind, not from defaults


def test_stage_overrides_are_absolute_not_relative():
    """`novelty=0.3` means 0.3, whatever a global anneal has done to the default.

    The alternative -- scaling the annealed value -- would make a schedule
    unreadable without also reading the flags.
    """
    (s,) = parse_schedule("explore:10,novelty=0.3,lr=1e-4,eps=0.4")
    annealed = Knobs(**{**vars(DEFAULTS), "novelty": 0.007})
    knobs = resolve(s, 1, annealed)
    assert knobs.novelty == 0.3
    assert knobs.lr == 1e-4
    assert knobs.eps == 0.4


def test_resolve_does_not_mutate_the_defaults():
    """The composer reuses one Knobs per update; aliasing would leak forward."""
    (s,) = parse_schedule("explore:10,novelty=0.3")
    before = vars(DEFAULTS).copy()
    resolve(s, 1, DEFAULTS)
    assert vars(DEFAULTS) == before


# ---------------------------------------------------------------------------
# The CLI's explicit-flag scan
# ---------------------------------------------------------------------------

def _parser():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta_long", type=int, default=2)
    p.add_argument("--flag", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--untouched", type=str, default="x")
    return p


@pytest.mark.parametrize("argv,expected", [
    ([], set()),
    (["--alpha", "2.0"], {"alpha"}),
    (["--alpha=2.0"], {"alpha"}),
    (["--flag"], {"flag"}),
    (["--no-flag"], {"flag"}),
    (["--alpha", "2.0", "--beta_long", "3"], {"alpha", "beta_long"}),
])
def test_explicit_dests_sees_exactly_what_was_typed(argv, expected):
    """--load_checkpoint inheritance turns on this distinction.

    A parsed Namespace holds a value whether or not the flag was passed, so
    "inherit the parent's value" is only expressible by looking at argv.
    """
    from hopfield_nav.train_navigate import _explicit_dests
    p = _parser()
    p.parse_args(argv)                       # the argv must actually be valid
    assert _explicit_dests(p, argv) == expected


def test_explicit_dests_never_reports_an_unpassed_flag():
    from hopfield_nav.train_navigate import _explicit_dests
    p = _parser()
    assert "untouched" not in _explicit_dests(p, ["--alpha", "2.0"])


def test_abbreviations_cannot_slip_past_the_scan():
    """allow_abbrev=False is load-bearing, not stylistic.

    With abbreviation on, `--unt x` would parse into the Namespace while
    matching no option string here -- so it would set a value that the resume
    path then silently declined to apply.
    """
    p = _parser()
    with pytest.raises(SystemExit):
        p.parse_args(["--unt", "y"])
