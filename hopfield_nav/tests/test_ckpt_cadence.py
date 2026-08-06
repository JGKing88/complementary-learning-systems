"""Checkpointing runs on its own cadence, not the evaluation one.

Until 2026-08 the per-update `torch.save` in phase A and phase B sat *inside*
the `if update % eval_every == 0` branch, which coupled two things with
opposite costs. An eval is expensive and wants to be rare; a checkpoint is
cheap and wants to be frequent. The result was that making a long run
affordable (`--eval_every 20` on 300 updates) also thinned it to 15
checkpoints, and `analysis.trajectories` -- which draws one figure row per
checkpoint -- had almost nothing to plot.

Two properties are pinned here, and the second is the one that matters for the
101 existing sweep variants: with `--ckpt_every` unset, behavior is *exactly*
what it was. Every one of those command lines has to keep producing the same
checkpoint series.

These drive the cadence arithmetic directly rather than running a trainer: the
smoke suite already runs both trainers end to end, and what could silently
regress here is the schedule, not the plumbing.
"""
from __future__ import annotations

import pytest

from hopfield_nav.config import TrainConfig


def _saves(n_updates: int, every: int) -> list[int]:
    """The updates a `update % max(every, 1) == 0` branch fires on."""
    return [u for u in range(1, n_updates + 1) if u % max(every, 1) == 0]


def _resolve(cfg: TrainConfig) -> int:
    """The expression both trainers use to pick the checkpoint cadence."""
    return cfg.ckpt_every if cfg.ckpt_every is not None else cfg.eval_every


def test_unset_ckpt_every_reproduces_the_old_schedule():
    """The compatibility property. 101 sweep variants depend on it.

    `run_phase_a_sweep_evelina.sh` passes `--eval_every` and nothing else, so
    an unset `--ckpt_every` must give the same checkpoint series the coupled
    branch gave.
    """
    cfg = TrainConfig(eval_every=20)
    assert cfg.ckpt_every is None, "the default must be 'follow eval_every'"
    assert _resolve(cfg) == 20
    assert _saves(300, _resolve(cfg)) == _saves(300, cfg.eval_every)


def test_ckpt_every_decouples_the_two():
    """The point of the change: 60 checkpoints, 15 evals, over 300 updates."""
    cfg = TrainConfig(eval_every=20, ckpt_every=5)
    evals = _saves(300, cfg.eval_every)
    ckpts = _saves(300, _resolve(cfg))
    assert len(evals) == 15
    assert len(ckpts) == 60
    # Every eval still lands on a checkpoint here, so an eval's numbers can
    # still be tied to a saved agent when the cadences are multiples.
    assert set(evals) <= set(ckpts)


def test_ckpt_every_may_be_sparser_than_eval_every():
    """The other direction has to work too: cheap evals, expensive storage.

    Nothing forces ckpt_every to divide eval_every, and a 512 MB-per-checkpoint
    run may well want to evaluate more often than it saves.
    """
    cfg = TrainConfig(eval_every=5, ckpt_every=50)
    assert len(_saves(300, cfg.eval_every)) == 60
    assert len(_saves(300, _resolve(cfg))) == 6


@pytest.mark.parametrize("every", [0, -1])
def test_a_nonpositive_cadence_saves_every_update(every):
    """`max(every, 1)` is what stops a ZeroDivisionError on `--ckpt_every 0`.

    Saving every update is the sane reading of "0": the alternative is a crash
    partway into a training job.
    """
    assert _saves(4, every) == [1, 2, 3, 4]


def test_ckpt_every_is_recorded_in_the_config():
    """It has to reach the checkpoint's config dict, and so the manifest.

    A cadence you cannot read back off a finished run is a cadence you will
    misremember when you come to reproduce the figure.
    """
    from dataclasses import asdict
    cfg = TrainConfig(eval_every=20, ckpt_every=5)
    assert asdict(cfg)["ckpt_every"] == 5


def test_old_checkpoints_load_without_the_field():
    """Additive only: a config dict written before ckpt_every existed still loads.

    ~350 run directories have no such key, and `cfg_from_checkpoint` has to
    keep reading them -- checkpoints are keyed by dataclass field name.
    """
    from dataclasses import asdict
    from hopfield_nav.evaluation.checkpoint_io import cfg_from_checkpoint

    legacy = asdict(TrainConfig(eval_every=20))
    del legacy["ckpt_every"]
    restored = cfg_from_checkpoint(legacy)
    assert restored.ckpt_every is None
    assert _resolve(restored) == 20
