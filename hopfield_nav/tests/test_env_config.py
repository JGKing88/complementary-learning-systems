"""Env-config wiring: the flags that were silently dropped, and off-cell stores.

Two things are pinned here.

1. ``EnvConfig`` fields reach the environments the rollout actually steps.
   ``train.setup_train_world`` used to build its ``GridEnv``s by hand and drop
   ``goals_active`` / ``goal_reward`` / ``goal_radius``; ``VecEnv`` reads those
   off the base env, so ``--goal_radius`` was a no-op during training while
   eval honoured it.

2. The off-cell store case. With ``goal_radius > 0.5`` in continuous mode,
   ``at_goal`` tests the float position while embeddings are read at the
   *snapped* cell, so a store fired "at goal" can write a neighbouring cell's
   embedding as the goal memory. ``EnvConfig.allow_offcell_store`` names that
   behavior and defaults to False, so the goal cell's embedding is stored
   instead. True restores what every run through 2026-08 did.
"""
from __future__ import annotations

import io
import contextlib

import numpy as np
import pytest

from hopfield_nav.config import EnvConfig
from hopfield_nav.world.env import (
    at_goal, make_env, max_offcell_offset, warn_if_offcell_stores,
)
from hopfield_nav.world.vec_env import VecEnv
from hopfield_nav.world.scaffold import VectorHash


# ---------------------------------------------------------------------------
# EnvConfig -> GridEnv -> VecEnv
# ---------------------------------------------------------------------------

GOAL_FIELDS = [
    ("goals_active", False),
    ("goal_reward", 3.5),
    ("goal_radius", 1.5),
]


@pytest.mark.parametrize("field,value", GOAL_FIELDS)
@pytest.mark.parametrize("movement_mode", ["discrete", "continuous"])
def test_goal_fields_reach_env_and_vec(field, value, movement_mode):
    """make_env forwards every goal field, and VecEnv reads them off the base."""
    cfg = EnvConfig(size=6, movement_mode=movement_mode, **{field: value})
    env = make_env(cfg, movement_mode, seed=0)
    assert getattr(env, field) == value
    vec = VecEnv(env, batch_size=4)
    assert getattr(vec, field) == value


def test_goal_radius_changes_at_goal_rate():
    """The behavioral consequence: a bigger radius makes more positions at-goal.

    This is what --goal_radius was failing to do on the train.py path.
    """
    rates = {}
    for radius in (0.5, 1.5):
        cfg = EnvConfig(size=6, goal_radius=radius)
        env = make_env(cfg, "discrete", seed=0)
        vec = VecEnv(env, batch_size=256)
        vec.reset_all()
        rates[radius] = float(np.mean(at_goal(vec)))
    assert rates[0.5] < rates[1.5], rates


# ---------------------------------------------------------------------------
# Off-cell geometry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("radius,expected", [
    (0.5, 0),   # default: every at-goal position snaps to the goal cell
    (0.7, 1),
    (1.0, 1),
    (1.5, 1),
    (2.0, 2),
])
def test_max_offcell_offset(radius, expected):
    assert max_offcell_offset(radius) == expected


@pytest.mark.parametrize("radius", [0.5, 1.0, 1.5, 2.0])
def test_max_offcell_offset_matches_sampling(radius):
    """Sample the at-goal ball, snap, and confirm the predicted bound is tight."""
    rng = np.random.RandomState(0)
    pts = rng.uniform(-radius, radius, size=(200_000, 2))
    pts = pts[(pts ** 2).sum(1) <= radius * radius]
    observed = int(np.abs(np.round(pts)).max())
    assert observed == max_offcell_offset(radius)


@pytest.mark.parametrize("radius,allow,should_warn", [
    (0.5, True, False),    # radius too small for the two policies to differ
    (0.5, False, False),
    (1.0, True, True),     # opted back into writing a neighbour's embedding
    (1.0, False, False),   # suppressed: a note, not a warning
    (2.0, True, True),
])
def test_warning_gating(radius, allow, should_warn):
    cfg = EnvConfig(size=6, goal_radius=radius, allow_offcell_store=allow)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        warn_if_offcell_stores(cfg)
    assert ("WARNING" in buf.getvalue()) is should_warn


def test_offcell_store_is_disallowed_by_default():
    """The default must not let a store write a cell other than the goal's."""
    assert EnvConfig().allow_offcell_store is False


def test_suppression_is_reported_when_the_radius_makes_it_matter():
    """Silent at 0.5 (nothing to substitute); a note above it."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        warn_if_offcell_stores(EnvConfig(size=6, goal_radius=0.5))
    assert buf.getvalue() == ""

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        warn_if_offcell_stores(EnvConfig(size=6, goal_radius=1.0))
    out = buf.getvalue()
    assert "WARNING" not in out
    assert "goal cell's embedding" in out


# ---------------------------------------------------------------------------
# Which pattern a store writes
# ---------------------------------------------------------------------------

def _coordinate_scaffold(Npos: int = 8) -> VectorHash:
    """A VectorHash whose encoded_Phi[x, y] is literally (x, y).

    Lets a test read back which cell a returned pattern came from, without
    building a real scaffold or encoder.
    """
    vh = VectorHash.__new__(VectorHash)
    vh.Npos = Npos
    phi = np.zeros((Npos, Npos, 2), dtype=np.float32)
    for x in range(Npos):
        for y in range(Npos):
            phi[x, y] = (x, y)
    vh.encoded_Phi = phi
    return vh


def test_store_patterns_allowed_is_a_plain_lookup():
    """allow_offcell=True must be byte-identical to get_encoded_state."""
    vh = _coordinate_scaffold()
    positions = np.array([[4, 4], [5, 4], [1, 1]], dtype=np.int32)
    mask = np.array([True, True, False])
    got = vh.get_store_patterns(positions, (0, 0), at_goal_mask=mask,
                                goal=(4, 4), allow_offcell=True)
    assert np.array_equal(got, vh.get_encoded_state(positions, (0, 0)))


def test_store_patterns_suppressed_redirects_only_offcell_at_goal_rows():
    vh = _coordinate_scaffold()
    positions = np.array([
        [4, 4],   # at goal, on the goal cell
        [5, 4],   # at goal, one cell east
        [3, 5],   # at goal, diagonal neighbour
        [1, 1],   # not at goal
    ], dtype=np.int32)
    mask = np.array([True, True, True, False])
    got = vh.get_store_patterns(positions, (0, 0), at_goal_mask=mask,
                                goal=(4, 4), allow_offcell=False)
    assert np.array_equal(got[0], (4, 4))
    assert np.array_equal(got[1], (4, 4))   # substituted
    assert np.array_equal(got[2], (4, 4))   # substituted
    assert np.array_equal(got[3], (1, 1))   # untouched -- not at goal


def test_store_patterns_does_not_mutate_encoded_phi():
    vh = _coordinate_scaffold()
    positions = np.array([[5, 4]], dtype=np.int32)
    before = vh.encoded_Phi.copy()
    vh.get_store_patterns(positions, (0, 0), at_goal_mask=np.array([True]),
                          goal=(4, 4), allow_offcell=False)
    assert np.array_equal(vh.encoded_Phi, before)


def test_store_patterns_honours_env_offset():
    """The goal is in local coords; the offset shifts both lookups equally."""
    vh = _coordinate_scaffold()
    positions = np.array([[1, 0]], dtype=np.int32)
    got = vh.get_store_patterns(positions, (2, 3), at_goal_mask=np.array([True]),
                                goal=(0, 0), allow_offcell=False)
    assert np.array_equal(got[0], (2, 3))   # goal (0,0) + offset (2,3)


def test_store_patterns_without_goal_falls_back_to_current_cell():
    """No goal supplied (e.g. goals_active=False) -> nothing to substitute."""
    vh = _coordinate_scaffold()
    positions = np.array([[5, 4]], dtype=np.int32)
    got = vh.get_store_patterns(positions, (0, 0), at_goal_mask=np.array([True]),
                                goal=None, allow_offcell=False)
    assert np.array_equal(got[0], (5, 4))
