"""Reconciling a typed command line with a config inherited from a checkpoint.

The failure this guards against is silent in the only way that matters: a flag
the caller did not type reads as the *dataclass default* rather than as the
parent's value, so a resumed run trains a differently-configured agent than the
one it loaded. `train.py` did exactly that until 2026-08-12 -- 8 of 17 agent
fields differ between a `train_navigate` checkpoint and `train.py`'s defaults,
`movement_mode` among them, which is a different policy head rather than a width
mismatch.
"""
from __future__ import annotations

import argparse

import pytest

from hopfield_nav.config import (AgentConfig, BCConfig, EnvConfig, PPOConfig,
                                 TrainConfig)
from hopfield_nav.evaluation.checkpoint_io import cfg_from_checkpoint
from hopfield_nav.training.cfg_args import explicit_dests, overlay_typed


# ---------------------------------------------------------------------------
# cfg_from_checkpoint reconstructs every nested block
# ---------------------------------------------------------------------------

def test_every_nested_config_comes_back_as_its_dataclass():
    """`bc` used to come back a raw dict, because the reconstruction named five
    blocks by hand and there are six. `cfg.bc.lr` raised on any config read from
    a checkpoint -- invisible while nothing resumed a BC run."""
    import dataclasses

    src = TrainConfig()
    src.bc = BCConfig(lr=1.25e-4, epochs=7)
    got = cfg_from_checkpoint(dataclasses.asdict(src))

    for f in dataclasses.fields(TrainConfig):
        want = getattr(TrainConfig(), f.name)
        if dataclasses.is_dataclass(want):
            assert not isinstance(getattr(got, f.name), dict), (
                f"cfg.{f.name} came back as a dict, not {type(want).__name__}")
            assert isinstance(getattr(got, f.name), type(want))
    # And the values survived, not just the type.
    assert got.bc.lr == 1.25e-4 and got.bc.epochs == 7


def test_a_field_added_after_a_checkpoint_was_written_reads_as_its_default():
    """Older checkpoints are missing keys the dataclass has now. That must read
    as the default rather than failing the load."""
    import dataclasses

    d = dataclasses.asdict(TrainConfig())
    d["agent"].pop("init_log_std")
    d["bc"].pop("epochs")
    got = cfg_from_checkpoint(d)
    assert got.agent.init_log_std == AgentConfig().init_log_std
    assert got.bc.epochs == BCConfig().epochs


# ---------------------------------------------------------------------------
# overlay_typed: untyped inherits, typed overrides
# ---------------------------------------------------------------------------

def _build(args) -> TrainConfig:
    """A stand-in for an entry point's constructor: pure keyword passing."""
    return TrainConfig(
        env=EnvConfig(size=args.size, movement_mode=args.movement_mode),
        agent=AgentConfig(movement_mode=args.movement_mode,
                          hidden_size=args.hidden_size),
        ppo=PPOConfig(lr=args.lr),
        num_val_envs=args.num_val_envs,
    )


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--size", type=int, default=8)
    p.add_argument("--movement_mode", default="discrete")
    p.add_argument("--hidden_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num_val_envs", type=int, default=2)
    return p


def _resolve(parent: TrainConfig, argv: list[str]) -> TrainConfig:
    p = _parser()
    args = p.parse_args(argv)
    overlay_typed(parent, args, explicit_dests(p, argv), _build)
    return parent


def test_an_untyped_flag_keeps_the_parents_value_not_the_default():
    """The whole point. The parent trained continuous; the default is discrete;
    saying nothing must mean continuous."""
    parent = TrainConfig(env=EnvConfig(size=4, movement_mode="continuous"),
                         agent=AgentConfig(movement_mode="continuous"))
    got = _resolve(parent, ["--num_val_envs", "5"])
    assert got.env.movement_mode == "continuous"
    assert got.agent.movement_mode == "continuous"
    assert got.env.size == 4
    assert got.num_val_envs == 5


def test_a_typed_flag_overrides_even_when_it_equals_the_default():
    """"Typed", not "differs from the default". Comparing against defaults is
    the cheap implementation and it is wrong exactly when the caller means to
    pin a value the parent changed."""
    parent = TrainConfig(num_val_envs=99)
    got = _resolve(parent, ["--num_val_envs", "2"])     # 2 IS the default
    assert got.num_val_envs == 2, (
        "a typed flag lost to the parent because its value matched the default")


def test_one_flag_reaching_two_fields_reaches_both():
    """`movement_mode` writes env and agent both. A hand-written table gets this
    wrong by omission; deriving it from the constructor cannot."""
    parent = TrainConfig(env=EnvConfig(movement_mode="continuous"),
                         agent=AgentConfig(movement_mode="continuous"))
    got = _resolve(parent, ["--movement_mode", "discrete"])
    assert got.env.movement_mode == "discrete"
    assert got.agent.movement_mode == "discrete"


def test_nested_untyped_fields_are_left_alone_wholesale():
    """A block the caller never touched keeps every one of the parent's values,
    not just the ones the constructor happens to mention."""
    parent = TrainConfig(ppo=PPOConfig(lr=1e-5, ent_coef=0.123))
    got = _resolve(parent, ["--size", "6"])
    assert got.ppo.lr == 1e-5
    assert got.ppo.ent_coef == 0.123          # never named by `_build` at all
    assert got.env.size == 6


def test_explicit_dests_reads_argv_not_the_namespace():
    p = _parser()
    assert explicit_dests(p, ["--size", "4"]) == {"size"}
    assert explicit_dests(p, ["--size=4", "--lr=1e-3"]) == {"size", "lr"}
    assert explicit_dests(p, []) == set()


def test_a_boolean_optional_no_flag_counts_as_typed():
    """`--no-flag` is the same action as `--flag`, so it must register."""
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--allow_store", action=argparse.BooleanOptionalAction,
                   default=True)
    assert explicit_dests(p, ["--no-allow_store"]) == {"allow_store"}
    assert explicit_dests(p, ["--allow_store"]) == {"allow_store"}
    assert explicit_dests(p, []) == set()


def test_train_py_forbids_abbreviations():
    """`explicit_dests` matches option strings literally, so an abbreviation
    would parse fine and then silently lose to the inherited value. Both parsers
    that feed it must refuse abbreviations rather than rely on nobody typing
    one."""
    from hopfield_nav import train, train_navigate

    for build in (train.build_args, train_navigate.build_parser):
        p = build.__wrapped__ if hasattr(build, "__wrapped__") else build
        # build_args parses argv; construct its parser via the module instead.
        break
    p = train_navigate.build_parser()
    assert p.allow_abbrev is False

    # train.py's parser is built inside build_args, so check it by behaviour:
    # an abbreviation must be rejected outright.
    with pytest.raises(SystemExit):
        import sys
        old = sys.argv
        try:
            sys.argv = ["train", "--encoder_checkpoint", "x", "--siz", "4"]
            train.build_args()
        finally:
            sys.argv = old
