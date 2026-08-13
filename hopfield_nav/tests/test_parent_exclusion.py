"""A continuation inherits what its parent trained on.

The curriculum this exists for is two stages: `train_navigate` learns explore
and exploit across many envs, then `train.py` resumes that checkpoint and trains
the whole task at once on a handful of *different* envs. The question the whole
evaluation then rests on -- "has the model ever seen this env?" -- has two
answers, one per stage, and until 2026-08-12 a validation set minted from the
child's `world.json` only ever knew the second.

Two separate failures, both silent:

  * the child *placed* wherever it liked. Measured on a real chain before this:
    `min_place_gap_vs_parent = -2`, i.e. a stage-2 train env overlapping a
    stage-1 one.
  * the child *recorded* only its own envs, so `--split place=held_out` came
    back "disjoint from training" while reusing stage-1 wall seeds and goal
    cells. On a size-8 arena there are 64 goal cells and 20 stage-1 envs claim
    20 of them, so this was not a tail risk.

`world_parent.json` has existed for the diagnosis since 2026-08-12 -- it reports
the overlap -- but reporting is not excluding.

The no-op half matters as much as the feature: a run with no parent must draw
exactly what it drew before, RNG consumption included, or every existing
checkpoint's world becomes unreproducible.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from hopfield_nav.config import EnvConfig, VectorHashConfig
from hopfield_nav.world import domains as dom
from hopfield_nav.world import generate as gen
from hopfield_nav.world.scaffold import VectorHash
from hopfield_nav.world.spec import EnvSpec, GeneratedSplit, TraitDomains

LAMBDAS = [3, 4, 5]          # Npos = 60
SIZE = 6
OBS = 12
MARGIN = 6


@pytest.fixture(scope="module")
def field():
    vh = VectorHash(VectorHashConfig(lambdas=LAMBDAS, static_vectorhash=True))
    vh.build_scaffold()
    torch.manual_seed(0)
    enc = torch.nn.Linear(vh.Ng, 16)
    enc.eval()
    vh.precompute_encoded_phi(enc, 0.25, device="cpu")
    return vh


@pytest.fixture
def env_cfg():
    return EnvConfig(size=SIZE, observation_size=OBS)


@pytest.fixture
def domains():
    return TraitDomains(place=dom.Anywhere(), wall=dom.SeedRange(0, 100_000),
                        goal=dom.AnyCells(), size=dom.Sizes((SIZE,)))


def _split(field, env_cfg, domains, n_train=3, n_val=2, seed=0, inherited=None):
    return gen.generate_split(field, env_cfg, domains, n_train, n_val,
                              seed=seed, margin=MARGIN, inherited=inherited)


# ---------------------------------------------------------------------------
# The no-op: a parentless run is untouched
# ---------------------------------------------------------------------------

def test_no_parent_draws_exactly_what_it_drew_before(field, env_cfg, domains):
    """`inherited=None` and `inherited={}` must both reduce to the old path.

    Stated on the resolved specs rather than "equivalent up to a mask": the
    place branch takes a *different code path* when there is something to
    exclude (`use_mask`), so the gate has to be on emptiness, not on presence.
    """
    base = _split(field, env_cfg, domains)
    for inh in (None, {}, {"place": set(), "wall": set(), "goal": set()}):
        got = _split(field, env_cfg, domains, inherited=inh)
        assert got.train == base.train, f"train envs moved with inherited={inh!r}"
        assert got.base_val == base.base_val
        assert got.goal_cells_val == base.goal_cells_val
        assert got.used == base.used


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------

def test_a_child_places_clear_of_its_parents_envs(field, env_cfg, domains):
    parent = _split(field, env_cfg, domains, seed=0)
    # A different seed AND a different count, which is the case that bites:
    # `generate_split` is deterministic in its arguments, so two stages of
    # different sizes are two unrelated draws, not a prefix of one.
    child = _split(field, env_cfg, domains, n_train=2, n_val=2, seed=1,
                   inherited=parent.used)
    gaps = [gen.toroidal_gap(c.offset, c.size, o, s, child.period)
            for c in child.train for o, s in parent.used_boxes()]
    assert gaps, "no pairs compared; the fixture cannot show anything"
    assert min(gaps) >= MARGIN, (
        f"a child train env is {min(gaps)} cells from a parent's, below margin "
        f"{MARGIN}")


def test_without_inheriting_the_child_can_land_on_the_parent(field, env_cfg,
                                                             domains):
    """The bug, stated as the difference. If this ever comes back clear, the
    test above is passing for a reason other than the feature."""
    parent = _split(field, env_cfg, domains, seed=0)
    worst = None
    for seed in range(12):
        child = _split(field, env_cfg, domains, n_train=2, n_val=2, seed=seed)
        gaps = [gen.toroidal_gap(c.offset, c.size, o, s, child.period)
                for c in child.train for o, s in parent.used_boxes()]
        worst = min(gaps) if worst is None else min(worst, min(gaps))
    assert worst < MARGIN, (
        "no unheld child draw came within the margin of the parent over 12 "
        "seeds, so this scaffold cannot demonstrate the failure")


# ---------------------------------------------------------------------------
# Wall seeds and goal cells
# ---------------------------------------------------------------------------

def _narrow_walls():
    """A seed range tight enough that a collision is the *expected* outcome.

    At the working range (0, 10_000_000) a child drawing four seeds misses three
    parent seeds ~99.99% of the time, so a test written there passes whether or
    not the exclusion exists -- which is how the first version of this file was
    written, and the mutation survived it. Eight seeds makes an unheld draw
    collide about 93% of the time, and the paired form below removes the
    remaining luck.
    """
    return TraitDomains(place=dom.Anywhere(), wall=dom.SeedRange(0, 8),
                        goal=dom.AnyCells(), size=dom.Sizes((SIZE,)))


def test_the_child_never_reuses_a_parents_wall_seed(field, env_cfg):
    narrow = _narrow_walls()
    parent = _split(field, env_cfg, narrow, seed=0)
    assert parent.used["wall"], "parent drew no wall seeds; test is vacuous"
    collided = 0
    for seed in range(1, 9):
        held = _split(field, env_cfg, narrow, n_train=2, n_val=2, seed=seed,
                      inherited=parent.used)
        seen = ({t.wall_seed for t in held.train}
                | {v.wall_seed for v in held.base_val})
        assert not (seen & parent.used["wall"]), (
            f"seed {seed}: child reused parent wall seed(s) "
            f"{sorted(seen & parent.used['wall'])}")
        # Same draw without inheriting, to show the exclusion is what did it.
        loose = _split(field, env_cfg, narrow, n_train=2, n_val=2, seed=seed)
        if ({t.wall_seed for t in loose.train}
                | {v.wall_seed for v in loose.base_val}) & parent.used["wall"]:
            collided += 1
    assert collided, (
        "no unheld draw reused a parent seed across 8 seeds, so this range "
        "cannot demonstrate the exclusion doing anything")


def test_a_parents_goal_cell_is_not_offered_as_held_out(field, env_cfg, domains):
    """`goal_cells_val` is what a later `held_out` draws from. A cell some
    earlier stage put a goal on is not held out from the model."""
    parent = _split(field, env_cfg, domains, seed=0)
    child = _split(field, env_cfg, domains, n_train=2, n_val=2, seed=1,
                   inherited=parent.used)
    assert parent.used["goal"], "parent used no goal cells; test is vacuous"
    assert not (child.goal_cells_val & parent.used["goal"])


def test_the_child_may_still_train_on_a_cell_the_parent_used(field, env_cfg,
                                                             domains):
    """Inheriting constrains *validation*, not what this run may train on.

    Nothing about the curriculum says stage two must avoid stage one's goal
    cells -- only that neither's may be offered as held out. Over-constraining
    here would shrink the train pool for no reason.
    """
    region = dom.AnyCells().cells(SIZE)
    parent = _split(field, env_cfg, domains, seed=0)
    child = _split(field, env_cfg, domains, n_train=2, n_val=2, seed=1,
                   inherited=parent.used)
    # The partition is exhaustive either way: every cell is train or val.
    assert child.goal_cells_train | child.goal_cells_val == frozenset(region)
    assert parent.used["goal"] <= child.goal_cells_train


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------

def test_used_is_the_union_so_the_record_stands_alone(field, env_cfg, domains):
    """The point of doing this at training time. `make_val_set` reads one
    split; if the union is not in it, every evaluator would have to learn to
    walk the checkpoint chain instead."""
    parent = _split(field, env_cfg, domains, seed=0)
    child = _split(field, env_cfg, domains, n_train=2, n_val=2, seed=1,
                   inherited=parent.used)
    for trait in ("place", "wall", "goal", "size"):
        assert parent.used[trait] <= child.used[trait], (
            f"child.used[{trait!r}] dropped values the parent had used")


def test_train_stays_this_runs_own(field, env_cfg, domains):
    """`used` is "what any stage saw"; `train` is "what this run trains on".
    Collapsing them would have the child build and roll out the parent's envs."""
    parent = _split(field, env_cfg, domains, seed=0)
    child = _split(field, env_cfg, domains, n_train=2, n_val=2, seed=1,
                   inherited=parent.used)
    assert len(child.train) == 2
    assert not (set(child.train) & set(parent.train))


def test_a_minted_val_set_is_disjoint_from_both_stages(field, env_cfg, domains):
    """End to end: the question the whole curriculum rests on.

    `make_val_set` sees only the child's split, which is exactly why the union
    has to be in it.
    """
    parent = _split(field, env_cfg, domains, seed=0)
    child = _split(field, env_cfg, domains, n_train=2, n_val=2, seed=1,
                   inherited=parent.used)
    specs = gen.make_val_set(child, 2, {"place": "held_out", "wall": "held_out",
                                        "goal": "held_out"}, seed=7)
    for stage, split in (("parent", parent), ("child", child)):
        gaps = [gen.toroidal_gap(v.offset, v.size, o, s, child.period)
                for v in specs for o, s in split.used_boxes()]
        assert min(gaps) >= MARGIN, f"val env within margin of {stage} envs"
        assert not ({v.wall_seed for v in specs} & split.used["wall"]), stage
        assert not ({v.goal for v in specs} & split.used["goal"]), stage


def test_world_overlap_reports_this_run_not_the_union(field, env_cfg, domains):
    """The diagnostic has to survive the record it now reads from.

    `child.used` contains the parent's envs by construction, so a
    union-vs-parent comparison reports total overlap unconditionally -- and
    compares boxes against themselves, giving `min_place_gap_vs_parent = -size`.
    Observed on the first real chain run after inheriting shipped: a correctly
    separated pair reported `-4` with `shared_walls=6`, which reads exactly like
    the bug the field exists to catch.
    """
    from hopfield_nav.training.world_setup import world_overlap

    parent = _split(field, env_cfg, domains, seed=0)
    child = _split(field, env_cfg, domains, n_train=2, n_val=2, seed=1,
                   inherited=parent.used)
    ov = world_overlap(parent, child, env_cfg)
    assert ov["min_place_gap_vs_parent"] >= MARGIN, (
        f"reported {ov['min_place_gap_vs_parent']} for a pair that is "
        f"separated -- the diagnostic is reading the absorbed union")
    assert ov["n_wall_seeds_shared"] == 0
    assert ov["n_train_offsets_shared"] == 0


def test_absorb_used_leaves_the_resolved_lists_alone():
    """`train` / `base_val` answer a different question from `used` and must
    not drift into it."""
    a = GeneratedSplit(
        domains=TraitDomains(dom.Anywhere(), dom.SeedRange(0, 10),
                             dom.AnyCells(), dom.Sizes((4,))),
        train=[EnvSpec(1, 4, (0, 0), (0, 0))], base_val=[],
        goal_cells_train=frozenset(), goal_cells_val=frozenset(),
        margin=0, period=12, Npos=12)
    a.record_used(a.train)
    a.absorb_used({"place": {((6, 6), 4)}, "wall": {9}, "goal": {(3, 3)},
                   "size": {4}})
    assert a.train == [EnvSpec(1, 4, (0, 0), (0, 0))]
    assert a.used["wall"] == {1, 9}
    assert a.used["goal"] == {(0, 0), (3, 3)}
    assert a.used_boxes() == [((0, 0), 4), ((6, 6), 4)]


# ---------------------------------------------------------------------------
# verify_split now covers the inherited half
# ---------------------------------------------------------------------------

def test_verify_split_catches_a_val_env_on_an_inherited_footprint(env_cfg):
    """Hand-built, because the generator cannot produce this any more -- which
    is the point. The assertion is what stops a future `make_val_set` change
    from reintroducing it quietly."""
    split = GeneratedSplit(
        domains=TraitDomains(dom.Anywhere(), dom.SeedRange(0, 100_000),
                             dom.AnyCells(), dom.Sizes((SIZE,))),
        train=[EnvSpec(1, SIZE, (0, 0), (0, 0))],
        base_val=[EnvSpec(2, SIZE, (30, 30), (1, 1))],
        goal_cells_train=frozenset({(0, 0)}), goal_cells_val=frozenset({(1, 1)}),
        margin=MARGIN, period=60, Npos=60)
    split.record_used(split.train)
    gen.verify_split(split, env_cfg)              # clean before inheriting
    split.absorb_used({"place": {((30, 30), SIZE)}})
    with pytest.raises(AssertionError, match="training used"):
        gen.verify_split(split, env_cfg)


def test_verify_split_catches_an_inherited_goal_cell(env_cfg):
    split = GeneratedSplit(
        domains=TraitDomains(dom.Anywhere(), dom.SeedRange(0, 100_000),
                             dom.AnyCells(), dom.Sizes((SIZE,))),
        train=[EnvSpec(1, SIZE, (0, 0), (0, 0))],
        base_val=[EnvSpec(2, SIZE, (30, 30), (1, 1))],
        goal_cells_train=frozenset({(0, 0)}), goal_cells_val=frozenset({(1, 1)}),
        margin=MARGIN, period=60, Npos=60)
    split.record_used(split.train)
    split.absorb_used({"goal": {(1, 1)}})
    with pytest.raises(AssertionError, match="goal cell"):
        gen.verify_split(split, env_cfg)
