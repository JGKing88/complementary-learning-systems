"""Movement bounds must be identical in the single env and the batched one.

`make_vec` used to take no action-norm arguments at all, so it silently dropped
`min_action_norm` / `max_action_norm`. Training builds `ContinuousVecEnv`
directly and passes both; every eval path comes through `make_vec`. Setting the
bounds therefore clamped training and left every reported number unclamped --
a mismatch with no symptom other than wrong numbers.

These tests pin the two halves of the fix: the bounds reach the batched env,
and the two envs agree about them for the same config.
"""
from __future__ import annotations

import numpy as np
import pytest

from hopfield_nav.config import EnvConfig
from hopfield_nav.world.env import ContinuousGridEnv, make_env
from hopfield_nav.world.vec_env import make_vec


def _env(min_norm=0.5, max_norm=2.0):
    env = ContinuousGridEnv(
        size=20, seed=0, observation_size=12,
        time_penalty=0.05, goal_reward=2.0, goals_active=True,
        scale=1.0, normalize=False,
        min_action_norm=min_norm, max_action_norm=max_norm,
    )
    env._goal = (19, 19)  # far from the starts used below
    return env


def _step_from(vec, start, action):
    """Realized displacement of a single batched step from `start`."""
    vec.set_positions(np.tile(np.asarray(start, dtype=np.int32), (vec.B, 1)))
    vec._pos_f[:] = np.asarray(start, dtype=np.float64)
    before = vec._pos_f.copy()
    vec.step_batch(np.tile(np.asarray(action, dtype=np.float64), (vec.B, 1)))
    return vec._pos_f - before


class TestBoundsReachTheBatchedEnv:

    def test_make_vec_inherits_bounds_from_wrapped_env(self):
        vec = make_vec(_env(), batch=4, movement_mode="continuous", reset=False)
        assert vec.min_action_norm == 0.5
        assert vec.max_action_norm == 2.0

    def test_oversized_action_is_clamped_to_max(self):
        vec = make_vec(_env(), batch=4, movement_mode="continuous", reset=False)
        # Norm 5 along +x, from the middle of the arena so the wall cannot clip.
        disp = _step_from(vec, (10, 10), (5.0, 0.0))
        assert np.allclose(np.linalg.norm(disp, axis=-1), 2.0)

    def test_undersized_action_is_raised_to_min(self):
        vec = make_vec(_env(), batch=4, movement_mode="continuous", reset=False)
        disp = _step_from(vec, (10, 10), (0.01, 0.0))
        assert np.allclose(np.linalg.norm(disp, axis=-1), 0.5)

    def test_direction_is_preserved_by_the_clamp(self):
        vec = make_vec(_env(), batch=4, movement_mode="continuous", reset=False)
        disp = _step_from(vec, (10, 10), (3.0, 4.0))  # norm 5, direction (.6,.8)
        unit = disp / np.linalg.norm(disp, axis=-1, keepdims=True)
        assert np.allclose(unit, np.array([0.6, 0.8]), atol=1e-9)

    def test_explicit_none_overrides_the_inherited_bounds(self):
        vec = make_vec(_env(), batch=4, movement_mode="continuous", reset=False,
                       min_action_norm=None, max_action_norm=None)
        assert vec.min_action_norm is None
        disp = _step_from(vec, (10, 10), (5.0, 0.0))
        assert np.allclose(np.linalg.norm(disp, axis=-1), 5.0)

    def test_unbounded_env_still_yields_unbounded_vec(self):
        vec = make_vec(_env(None, None), batch=4, movement_mode="continuous",
                       reset=False)
        assert vec.min_action_norm is None
        assert vec.max_action_norm is None


class TestTrainEvalAgreement:
    """The property the bug violated, asserted directly on one config."""

    @pytest.mark.parametrize("min_norm,max_norm", [(0.5, 2.0), (None, None),
                                                   (None, 1.0), (0.25, None)])
    def test_make_env_and_make_vec_agree(self, min_norm, max_norm):
        cfg = EnvConfig(size=20, observation_size=12,
                        min_action_norm=min_norm, max_action_norm=max_norm)
        single = make_env(cfg, "continuous", seed=0)
        vec = make_vec(single, batch=2, movement_mode="continuous",
                       continuous_scale=cfg.continuous_scale,
                       continuous_normalize=cfg.continuous_normalize,
                       reset=False)
        assert vec.min_action_norm == single.min_action_norm
        assert vec.max_action_norm == single.max_action_norm

    def test_realized_displacement_matches_between_the_two(self):
        cfg = EnvConfig(size=20, observation_size=12,
                        min_action_norm=0.5, max_action_norm=2.0)
        single = make_env(cfg, "continuous", seed=0)
        single._goal = (19, 19)
        vec = make_vec(single, batch=1, movement_mode="continuous",
                       continuous_scale=cfg.continuous_scale,
                       continuous_normalize=cfg.continuous_normalize,
                       reset=False)
        for action in [(5.0, 0.0), (0.01, 0.0), (3.0, 4.0), (1.0, 1.0)]:
            single._continuous_pos = np.array([10.0, 10.0])
            single._pos = (10, 10)
            single.step(np.asarray(action, dtype=np.float64))
            single_disp = single._continuous_pos - np.array([10.0, 10.0])
            vec_disp = _step_from(vec, (10, 10), action)[0]
            assert np.allclose(single_disp, vec_disp), (
                f"train/eval disagree on action {action}: "
                f"{single_disp} vs {vec_disp}")


class TestRealizedDisplacement:
    """`last_displacement` must report the move, not the command.

    The two channels exist separately because they disagree, and the
    disagreement is the signal: a clamp means the policy asked for more than it
    can take, a clip means a wall is there.
    """

    def test_clamped_step_reports_the_clamped_distance(self):
        vec = make_vec(_env(), batch=2, movement_mode="continuous", reset=False)
        _step_from(vec, (10, 10), (5.0, 0.0))
        assert np.allclose(np.linalg.norm(vec.last_displacement(), axis=-1), 2.0)

    def test_step_into_a_wall_reports_the_truncated_distance(self):
        vec = make_vec(_env(), batch=2, movement_mode="continuous", reset=False)
        # Sitting 0.5 from the +x wall (size 20 -> max coordinate 19.0) and
        # commanding a full-size step: the clip must eat most of it.
        _step_from(vec, (10, 10), (0.0, 0.0))
        vec._pos_f[:] = np.array([18.5, 10.0])
        vec.step_batch(np.tile(np.array([2.0, 0.0]), (vec.B, 1)))
        disp = vec.last_displacement()
        assert np.allclose(disp[:, 0], 0.5), disp
        assert np.allclose(np.linalg.norm(disp, axis=-1), 0.5)

    def test_displacement_disagrees_with_the_commanded_action(self):
        vec = make_vec(_env(), batch=1, movement_mode="continuous", reset=False)
        vec.set_positions(np.array([[18, 10]], dtype=np.int32))
        vec._pos_f[:] = np.array([18.5, 10.0])
        commanded = np.array([[5.0, 0.0]])
        vec.step_batch(commanded)
        realized = vec.last_displacement()
        assert not np.allclose(realized, commanded)
        assert np.linalg.norm(realized) < np.linalg.norm(commanded)

    def test_unmoved_episodes_report_zero(self):
        vec = make_vec(_env(), batch=4, movement_mode="continuous", reset=False)
        vec.set_positions(np.tile(np.array([10, 10], dtype=np.int32), (4, 1)))
        vec._pos_f[:] = np.array([10.0, 10.0])
        vec.step_batch(np.tile(np.array([1.0, 0.0]), (2, 1)),
                       indices=np.array([0, 1]))
        disp = vec.last_displacement()
        assert np.allclose(np.linalg.norm(disp[:2], axis=-1), 1.0)
        assert np.allclose(disp[2:], 0.0)

    def test_single_env_and_vec_env_report_the_same_displacement(self):
        cfg = EnvConfig(size=20, observation_size=12,
                        min_action_norm=0.5, max_action_norm=2.0)
        single = make_env(cfg, "continuous", seed=0)
        single._goal = (19, 19)
        vec = make_vec(single, batch=1, movement_mode="continuous",
                       continuous_scale=cfg.continuous_scale,
                       continuous_normalize=cfg.continuous_normalize,
                       reset=False)
        for action in [(5.0, 0.0), (0.01, 0.0), (3.0, 4.0)]:
            single._continuous_pos = np.array([10.0, 10.0])
            single._pos = (10, 10)
            single.step(np.asarray(action, dtype=np.float64))
            _step_from(vec, (10, 10), action)
            assert np.allclose(single._last_displacement,
                               vec.last_displacement()[0])


class TestChannelSpec:

    def test_prev_displacement_channel_appears_when_enabled(self):
        from hopfield_nav.config import AgentConfig
        from hopfield_nav.policy import channels
        cfg = AgentConfig(movement_mode="continuous", hopfield_mode="continuous",
                          input_prev_action=True, input_prev_displacement=True)
        names = [s.name for s in channels.channel_specs(cfg, embed_dim=8)]
        assert "prev_action" in names
        assert "prev_displacement" in names
        widths = {s.name: s.width for s in channels.channel_specs(cfg, embed_dim=8)}
        assert widths["prev_displacement"] == 2

    def test_absent_by_default(self):
        from hopfield_nav.config import AgentConfig
        from hopfield_nav.policy import channels
        cfg = AgentConfig(movement_mode="continuous", hopfield_mode="continuous")
        names = [s.name for s in channels.channel_specs(cfg, embed_dim=8)]
        assert "prev_displacement" not in names
