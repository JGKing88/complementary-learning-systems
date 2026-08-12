"""Asking an evaluation for a particular kind of validation set.

The surface is `--split`, and the properties worth pinning are the ones that
would be invisible if wrong: that `recorded` really is the run's own validation
envs rather than a lookalike, that a minted set's separation is measured rather
than assumed, that the 12 GB scaffold is built once across combinations, and
that the output file keeps the shape nine other readers depend on.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from hopfield_nav.config import EnvConfig, TrainConfig, VectorHashConfig
from hopfield_nav.evaluation import checkpoint_io as cio
from hopfield_nav.world import domains as dom
from hopfield_nav.world import generate as gen
from hopfield_nav.world.spec import TraitDomains, WorldSpec
from hopfield_nav.training.world_setup import build_field

LAMBDAS = [5, 7]        # Npos = 35
SIZE = 4
OBS = 16


def _cfg() -> TrainConfig:
    return TrainConfig(
        env=EnvConfig(size=SIZE, observation_size=OBS),
        vectorhash=VectorHashConfig(lambdas=LAMBDAS, Np=40,
                                    static_vectorhash=True),
        envs_per_world=3, num_worlds=1, num_val_envs=2, device="cpu",
    )


@pytest.fixture(scope="module")
def enc():
    torch.manual_seed(0)
    e = torch.nn.Linear(int(np.sum(np.square(LAMBDAS))), 8)
    e.eval()
    return e


@pytest.fixture(scope="module")
def field(enc):
    return build_field(_cfg(), enc)


@pytest.fixture
def spec(field):
    domains = TraitDomains(place=dom.Anywhere(), wall=dom.SeedRange(0, 100_000),
                           goal=dom.Ring(1), size=dom.Sizes((SIZE,)))
    split = gen.generate_split(field, EnvConfig(size=SIZE, observation_size=OBS),
                               domains, 3, 2, seed=0, margin=4)
    return WorldSpec(scaffold={"Npos": field.Npos}, generator="declared",
                     split=split)


# ---------------------------------------------------------------------------
# The grammar
# ---------------------------------------------------------------------------

def test_recorded_is_its_own_level_not_a_synonym_for_held_out():
    """The two answer different questions and must stay separable.

    `recorded` replays the envs a checkpoint was actually scored against;
    all-`held_out` mints a fresh set that is *also* disjoint from training. One
    asks how the run did on its own validation set, the other whether that
    survives a second draw from the same rule. Collapsing them would make a
    whole class of result unaskable.
    """
    assert gen.parse_levels("recorded") is None
    assert gen.parse_levels(None) is None
    assert gen.parse_levels("place=held_out,wall=held_out,goal=held_out") == {
        "place": "held_out", "wall": "held_out", "goal": "held_out"}
    assert gen.levels_key(None) == "recorded"


def test_unnamed_traits_default_to_held_out():
    """So `--split place=same` reads as the memorization probe it looks like."""
    assert gen.parse_levels("place=same") == {
        "place": "same", "wall": "held_out", "goal": "held_out"}


@pytest.mark.parametrize("text, match", [
    ("place", "trait=level"),
    ("size=held_out", "no bounded universe"),
    ("colour=same", "unknown trait"),
    ("place=novel", "unknown level"),
])
def test_the_grammar_rejects_what_it_cannot_mean(text, match):
    with pytest.raises(ValueError, match=match):
        gen.parse_levels(text)


def test_levels_key_is_stable_across_orderings():
    a = gen.parse_levels("goal=ood,place=same")
    b = gen.parse_levels("place=same,goal=ood")
    assert gen.levels_key(a) == gen.levels_key(b) == "place=same,wall=held_out,goal=ood"


# ---------------------------------------------------------------------------
# Resolving env sets
# ---------------------------------------------------------------------------

def test_recorded_returns_the_runs_own_validation_envs(spec):
    """Offsets included -- which is the whole reason world.json exists."""
    got = cio.eval_specs(spec, None, n_envs=2, seed=0)
    assert got == spec.split.base_val


def test_a_minted_held_out_set_is_not_the_recorded_one(spec):
    minted = cio.eval_specs(spec, gen.parse_levels("place=held_out"),
                            n_envs=2, seed=7)
    assert minted != spec.split.base_val
    assert not {m.wall_seed for m in minted} & {b.wall_seed
                                                for b in spec.split.base_val}


def test_the_scaffold_is_built_once_across_combinations(spec, enc):
    """Four splits, one field. At Npos=1716 `encoded_Phi` is 12 GB, so a rebuild
    per combination would cost more than every evaluator put together."""
    from hopfield_nav.eval_all import resolve_env_sets
    splits = [None, gen.parse_levels("place=same"),
              gen.parse_levels("place=held_out"), gen.parse_levels("goal=ood")]
    sets = resolve_env_sets(_cfg(), enc, torch.device("cpu"), spec, splits,
                            ckpt_path="unused", val_seed=3, n_val_envs=2)
    assert len(sets) == 4
    assert len({id(s["field"]) for s in sets}) == 1
    assert [s["key"] for s in sets] == [
        "recorded", "place=same,wall=held_out,goal=held_out",
        "place=held_out,wall=held_out,goal=held_out",
        "place=held_out,wall=held_out,goal=ood"]


def test_a_split_without_a_record_fails_rather_than_evaluating_something_else(
        spec, enc):
    """Minting needs the declared domains and the union training used; an RNG
    replay recovers neither, so there is nothing to hold out *from*."""
    from hopfield_nav.eval_all import resolve_env_sets
    with pytest.raises(SystemExit, match="needs a world.json"):
        resolve_env_sets(_cfg(), enc, torch.device("cpu"), None,
                         [gen.parse_levels("place=held_out")],
                         ckpt_path="/nowhere/x.pt", val_seed=0, n_val_envs=2)


# ---------------------------------------------------------------------------
# What a minted set asserts about itself (5.2)
# ---------------------------------------------------------------------------

def test_the_report_measures_separation_rather_than_assuming_it(spec):
    """Once a CLI mints envs on request, their separation *is* the claim the
    evaluation makes, so it lands in the results file as a number."""
    levels = gen.parse_levels("place=held_out")
    minted = cio.eval_specs(spec, levels, n_envs=2, seed=5)
    env_cfg = EnvConfig(size=SIZE, observation_size=OBS)
    rep = gen.val_set_report(spec.split, minted, env_cfg, levels)
    assert rep["levels"] == gen.levels_key(levels)
    assert rep["min_place_gap_vs_train"] >= spec.split.margin
    assert rep["n_wall_seeds_shared"] == 0
    assert rep["n_goal_cells_shared"] == 0


def test_the_same_level_is_reported_as_overlapping_because_it_is(spec):
    """`same` reuses training's values on purpose, so the report shows a
    negative gap and shared seeds -- and must not treat that as a violation."""
    levels = gen.parse_levels("place=same,wall=same,goal=same")
    minted = cio.eval_specs(spec, levels, n_envs=2, seed=5)
    rep = gen.val_set_report(spec.split, minted,
                             EnvConfig(size=SIZE, observation_size=OBS), levels)
    assert rep["n_wall_seeds_shared"] == len(minted)
    assert rep["min_place_gap_vs_train"] < spec.split.margin


def test_a_violated_held_out_claim_raises_rather_than_being_reported(spec):
    """A number in a report is not a strong enough place for a generator bug.

    Constructed by handing the reporter a set that reuses a training wall seed
    while claiming held_out -- what a broken exclusion would produce.
    """
    import dataclasses
    bad = [dataclasses.replace(spec.split.base_val[0],
                               wall_seed=spec.split.train[0].wall_seed)]
    with pytest.raises(AssertionError, match="reused .* training wall seeds"):
        gen.val_set_report(spec.split, bad,
                           EnvConfig(size=SIZE, observation_size=OBS),
                           gen.parse_levels("wall=held_out"))


# ---------------------------------------------------------------------------
# Every driver, one answer
# ---------------------------------------------------------------------------

def test_every_eval_driver_can_be_asked_for_a_split():
    """`--split place=ood` has to mean the same thing in all five.

    Each driver used to build its own eval world, so a level would have meant
    whatever that driver's copy of the logic did. They now share
    `eval_env_set`, and this reads the source to keep it that way -- a driver
    that grows a private path would still run, and would quietly answer a
    different question.
    """
    import inspect
    import hopfield_nav.eval_all as eval_all
    from analysis import trajectories
    from analysis.continual import agenthash
    from analysis.phase_decoding import exp1, exp2, rollout

    # The four single-set drivers go through the shared helper.
    for mod in (trajectories, agenthash, rollout):
        src = inspect.getsource(mod)
        assert "eval_world_for_split(" in src, (
            f"{mod.__name__} no longer resolves its env set through the shared "
            "helper, so --split may mean something different there")

    # ...and eval_all through the same underlying call, for its N-combination
    # table.
    assert "eval_env_set(" in inspect.getsource(eval_all)

    # Every CLI that owns a RolloutEngine passes the flag through; a driver that
    # forgets it silently decodes the recorded set while reporting a level.
    for mod in (exp1, exp2):
        src = inspect.getsource(mod)
        assert '"--split"' in src and "split=args.split" in src, (
            f"{mod.__name__} parses no --split, or does not forward it")


def test_the_shared_helper_and_eval_all_agree_on_one_split(spec, enc):
    """Same envs whichever entry point asked, so a trajectory figure and a
    metrics table are describing the same arenas."""
    from hopfield_nav.eval_all import resolve_env_sets

    levels = gen.parse_levels("place=held_out,goal=ood")
    via_table = resolve_env_sets(_cfg(), enc, torch.device("cpu"), spec,
                                 [levels], ckpt_path="unused", val_seed=11,
                                 n_val_envs=2)[0]
    via_single = cio.eval_env_set(_cfg(), enc, "cpu", ckpt_path="unused",
                                  levels=levels, val_seed=11, n_envs=2,
                                  spec=spec)
    assert via_table["key"] == via_single["key"]
    assert via_table["offsets"] == via_single["offsets"]
    assert ([e.seed for e in via_table["envs"]]
            == [e.seed for e in via_single["envs"]])
