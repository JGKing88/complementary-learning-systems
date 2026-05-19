"""Characterization tests for every place the at-goal predicate is evaluated.

Pre-refactor: these tests pin the *current* behavior of the inline
``pos == goal`` expression at each call site by exercising it directly.

Post-refactor: a single ``at_goal(pos, goal)`` helper replaces the inline
expressions. Every test in this file must still pass — any divergence is
a behavior change.

Sites covered (see survey in conversation 2026-05-10):
  Simulators (gated by ``goals_active``):
    env.py:105       GridEnv.reward            single, tuple
    vec_env.py:116   VecEnv.step_batch         per-row, int32
    vec_env.py:298   ContinuousVecEnv.step_batch  per-row, int32
  Pure positional (not gated by goals_active):
    rollout.py:191, 567               (B, 2) int32
    rollout_rnn.py:129                (B, 2) int32
    eval_rnn.py:85                    (B, 2) int32
    eval.py:346,431,446,551,569,757,907,1051   tuple[int,int]
    inspect_sequential.py:56          tuple
    inspect_trajectories.py:62        tuple
    visualize_trajectories.py:169,180,222,277,334  tuple
    probe_alignment.py:107            tuple
    phase_decoding/collect_trajectory.py:100   tuple
    phase_decoding/collect.py:135     tuple
    phase_decoding_v2/rollout.py:220  tuple
    final_plotting/agenthash.py:201   tuple
"""
from __future__ import annotations

import numpy as np
import pytest

from hopfield_nav.env import ContinuousGridEnv, GridEnv
from hopfield_nav.vec_env import ContinuousVecEnv, VecEnv


# ---------------------------------------------------------------------------
# Inline-expression reference implementations.
#
# These mirror exactly what the source does today at each site. After the
# refactor the helper must agree with these for every input.
# ---------------------------------------------------------------------------

def _ref_tuple_eq(pos, goal):
    """Reference for the tuple-comparison sites (env.py, eval.py, tooling).

    The source uses ``env.current_location == goal`` which is a Python
    tuple equality returning a Python bool.
    """
    return pos == goal


def _ref_vec_step_check(pos_row, goal_tuple):
    """Reference for vec_env.py:116 / :298.

    The source casts the int32 row to (int, int) then compares to the
    goal tuple. Returns a Python bool.
    """
    return (int(pos_row[0]), int(pos_row[1])) == goal_tuple


def _ref_batched_check(positions, goal):
    """Reference for rollout.py / rollout_rnn.py / eval_rnn.py.

    The source builds a numpy bool vector via two element-wise compares
    and a logical AND. ``goal`` is converted to ``np.array(...)`` first
    in some sites; we emulate both call patterns here.
    """
    goal_arr = np.asarray(goal)
    return (positions[:, 0] == goal_arr[0]) & (positions[:, 1] == goal_arr[1])


# ---------------------------------------------------------------------------
# Reference-output checkpoints. These pin the inline expressions to specific
# values for crafted inputs. They will be re-asserted against the helper after
# the refactor.
# ---------------------------------------------------------------------------

class TestReferenceBehavior:
    """Pin the current behavior of the inline expressions before any change."""

    def test_tuple_match_returns_python_bool_true(self):
        result = _ref_tuple_eq((3, 4), (3, 4))
        assert result is True
        assert isinstance(result, bool)

    def test_tuple_mismatch_returns_python_bool_false(self):
        result = _ref_tuple_eq((3, 4), (3, 5))
        assert result is False
        assert isinstance(result, bool)

    def test_tuple_partial_match_x_only(self):
        # Same x, different y — not at goal.
        assert _ref_tuple_eq((3, 4), (3, 0)) is False

    def test_tuple_partial_match_y_only(self):
        # Same y, different x — not at goal.
        assert _ref_tuple_eq((3, 4), (0, 4)) is False

    def test_vec_row_match_int32_row(self):
        # vec_env stores _pos as int32; mimic a sliced row.
        row = np.array([5, 7], dtype=np.int32)
        assert _ref_vec_step_check(row, (5, 7)) is True

    def test_vec_row_mismatch_int32_row(self):
        row = np.array([5, 7], dtype=np.int32)
        assert _ref_vec_step_check(row, (5, 8)) is False

    def test_batch_all_match(self):
        positions = np.array([[1, 1], [1, 1], [1, 1]], dtype=np.int32)
        out = _ref_batched_check(positions, (1, 1))
        assert out.dtype == bool
        assert out.shape == (3,)
        assert out.tolist() == [True, True, True]

    def test_batch_no_match(self):
        positions = np.array([[0, 0], [2, 3], [4, 5]], dtype=np.int32)
        out = _ref_batched_check(positions, (1, 1))
        assert out.tolist() == [False, False, False]

    def test_batch_mixed(self):
        positions = np.array([[1, 1], [0, 1], [1, 0], [1, 1]], dtype=np.int32)
        out = _ref_batched_check(positions, (1, 1))
        assert out.tolist() == [True, False, False, True]

    def test_batch_zero_size(self):
        positions = np.zeros((0, 2), dtype=np.int32)
        out = _ref_batched_check(positions, (1, 1))
        assert out.shape == (0,)
        assert out.dtype == bool

    def test_batch_goal_as_array(self):
        # rollout.py wraps goal in np.array(...) first; should be equivalent.
        positions = np.array([[1, 1], [2, 2]], dtype=np.int32)
        goal_tuple = (1, 1)
        goal_arr = np.array(goal_tuple)
        out_tuple = _ref_batched_check(positions, goal_tuple)
        out_arr = _ref_batched_check(positions, goal_arr)
        assert np.array_equal(out_tuple, out_arr)


# ---------------------------------------------------------------------------
# Simulator end-to-end tests. These exercise the env.step and step_batch
# code paths, which is the only place the at-goal check feeds into the
# returned reward / goal_reached array. ``goals_active`` gating must be
# preserved exactly.
# ---------------------------------------------------------------------------

class TestGridEnvReward:
    """env.py:105 — GridEnv.reward gates on goals_active and on _pos == _goal."""

    def _make(self, goals_active=True):
        env = GridEnv(
            size=8, seed=0, observation_size=12,
            time_penalty=0.01, goal_reward=1.0, goals_active=goals_active,
        )
        env._goal = (4, 4)
        return env

    def test_reward_at_goal_with_goals_active(self):
        env = self._make(goals_active=True)
        env._pos = (4, 4)
        assert env.reward() == pytest.approx(1.0)

    def test_reward_off_goal_with_goals_active(self):
        env = self._make(goals_active=True)
        env._pos = (4, 5)
        assert env.reward() == pytest.approx(-0.01)

    def test_reward_at_goal_with_goals_inactive(self):
        env = self._make(goals_active=False)
        env._pos = (4, 4)
        # goals_active=False -> never reaps goal_reward, even on goal cell.
        assert env.reward() == pytest.approx(-0.01)

    def test_reward_off_goal_with_goals_inactive(self):
        env = self._make(goals_active=False)
        env._pos = (4, 5)
        assert env.reward() == pytest.approx(-0.01)

    def test_step_into_goal_returns_goal_reward(self):
        env = self._make(goals_active=True)
        env._pos = (4, 3)
        state = env.step((0, 1))  # move N into goal at (4,4)
        assert state.position == (4, 4)
        assert state.reward == pytest.approx(1.0)

    def test_step_away_from_goal_loses_reward(self):
        env = self._make(goals_active=True)
        env._pos = (4, 4)
        state = env.step((0, 1))  # leave goal — post-step pos = (4,5)
        assert state.position == (4, 5)
        assert state.reward == pytest.approx(-0.01)


class TestContinuousGridEnvReward:
    """env.py:105 via ContinuousGridEnv — same gate, snapped position."""

    def _make(self, goals_active=True):
        env = ContinuousGridEnv(
            size=8, seed=0, observation_size=12,
            time_penalty=0.01, goal_reward=1.0, goals_active=goals_active,
            scale=1.0, normalize=False, max_action_norm=None,
        )
        env._goal = (4, 4)
        return env

    def test_step_to_goal_snapped_position_matches(self):
        env = self._make(goals_active=True)
        env._continuous_pos = np.array([4.0, 3.0])
        env._pos = (4, 3)
        state = env.step(np.array([0.0, 1.0]))  # +y by 1 -> (4,4)
        assert state.position == (4, 4)
        assert state.reward == pytest.approx(1.0)

    def test_snap_near_goal_returns_goal_reward(self):
        env = self._make(goals_active=True)
        # Float pos rounds to (4,4)
        env._continuous_pos = np.array([3.7, 4.1])
        env._pos = env.current_location
        # Take a tiny no-op-ish step that keeps snap at (4,4)
        state = env.step(np.array([0.1, -0.05]))
        assert state.position == (4, 4)
        assert state.reward == pytest.approx(1.0)

    def test_goals_inactive_at_goal_no_reward(self):
        env = self._make(goals_active=False)
        env._continuous_pos = np.array([4.0, 4.0])
        env._pos = (4, 4)
        state = env.step(np.array([0.0, 0.0]))
        assert state.position == (4, 4)
        assert state.reward == pytest.approx(-0.01)


class TestVecEnvStepBatchGoalReached:
    """vec_env.py:116 — discrete batched step_batch.

    Verifies the per-row tuple-comparison-with-goals_active-gate semantics.
    """

    def _make(self, B=4, goals_active=True):
        base = GridEnv(
            size=8, seed=1, observation_size=12,
            time_penalty=0.01, goal_reward=1.0, goals_active=goals_active,
        )
        base._goal = (3, 5)
        vec = VecEnv(base, batch_size=B)
        vec.reset_all()
        return vec

    def test_pre_step_at_goal_flagged(self):
        vec = self._make(B=4)
        # Crafted positions: rows 0 and 2 at goal, others not.
        vec._pos = np.array([[3, 5], [0, 0], [3, 5], [7, 7]], dtype=np.int32)
        # Action 0 = N. Pre-step at-goal rows ignore movement and teleport.
        rewards, goal_reached, _pos = vec.step_batch(np.array([0, 0, 0, 0]))
        assert goal_reached.dtype == bool
        assert goal_reached.tolist() == [True, False, True, False]
        np.testing.assert_allclose(rewards, [1.0, -0.01, 1.0, -0.01])

    def test_no_rows_at_goal(self):
        vec = self._make(B=3)
        vec._pos = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int32)
        _r, goal_reached, _ = vec.step_batch(np.array([0, 0, 0]))
        assert goal_reached.tolist() == [False, False, False]

    def test_goals_inactive_never_flags_at_goal(self):
        vec = self._make(B=2, goals_active=False)
        vec._pos = np.array([[3, 5], [3, 5]], dtype=np.int32)
        _r, goal_reached, _ = vec.step_batch(np.array([0, 0]))
        # goals_active=False -> no row should be flagged as reached.
        assert goal_reached.tolist() == [False, False]

    def test_indices_subset_only_checks_those_rows(self):
        vec = self._make(B=4)
        vec._pos = np.array([[3, 5], [3, 5], [3, 5], [3, 5]], dtype=np.int32)
        idx = np.array([1, 3])
        _r, goal_reached, _ = vec.step_batch(np.array([0, 0]), indices=idx)
        # Only the two requested rows are returned, both at goal.
        assert goal_reached.tolist() == [True, True]


class TestContinuousVecEnvStepBatchGoalReached:
    """vec_env.py:298 — continuous batched step_batch."""

    def _make(self, B=4, goals_active=True):
        base = ContinuousGridEnv(
            size=8, seed=2, observation_size=12,
            time_penalty=0.01, goal_reward=1.0, goals_active=goals_active,
            scale=1.0, normalize=False, max_action_norm=None,
        )
        base._goal = (3, 5)
        vec = ContinuousVecEnv(base, batch_size=B)
        vec.reset_all()
        return vec

    def test_pre_step_at_goal_flagged(self):
        vec = self._make(B=4)
        vec._pos_f = np.array([[3.0, 5.0], [0.0, 0.0], [3.0, 5.0], [7.0, 7.0]])
        vec._update_snapped()
        rewards, goal_reached, _ = vec.step_batch(np.zeros((4, 2)))
        assert goal_reached.tolist() == [True, False, True, False]
        np.testing.assert_allclose(rewards, [1.0, -0.01, 1.0, -0.01])

    def test_goals_inactive_never_flags(self):
        vec = self._make(B=2, goals_active=False)
        vec._pos_f = np.array([[3.0, 5.0], [3.0, 5.0]])
        vec._update_snapped()
        _r, goal_reached, _ = vec.step_batch(np.zeros((2, 2)))
        assert goal_reached.tolist() == [False, False]


# ---------------------------------------------------------------------------
# Site-by-site equivalence tests. Each rebuilds the inline expression on
# crafted inputs, asserting the exact value. Post-refactor we will add a
# parallel helper-based assertion in the same tests.
# ---------------------------------------------------------------------------

class TestRolloutInlineExpr:
    """rollout.py:191, 567 — (B,2) int32 positions vs goal_arr = np.array(goal)."""

    def test_pre_step_expr_returns_bool_array(self):
        positions = np.array([[3, 5], [0, 0], [3, 5], [7, 7]], dtype=np.int32)
        goal_arr = np.array((3, 5))
        at_goal = (positions[:, 0] == goal_arr[0]) & (positions[:, 1] == goal_arr[1])
        assert at_goal.dtype == bool
        assert at_goal.tolist() == [True, False, True, False]

    def test_bootstrap_expr_size_one(self):
        # Edge case: B=1 row.
        positions = np.array([[2, 2]], dtype=np.int32)
        goal_arr = np.array((2, 2))
        at_goal = (positions[:, 0] == goal_arr[0]) & (positions[:, 1] == goal_arr[1])
        assert at_goal.shape == (1,)
        assert at_goal.tolist() == [True]


class TestRolloutRnnInlineExpr:
    """rollout_rnn.py:129 / eval_rnn.py:85 — same shape, goal as tuple."""

    def test_expr_with_goal_tuple(self):
        positions = np.array([[1, 1], [3, 5]], dtype=np.int32)
        goal = (3, 5)
        at_goal = (positions[:, 0] == goal[0]) & (positions[:, 1] == goal[1])
        assert at_goal.tolist() == [False, True]


class TestEvalSingleEnvInlineExpr:
    """eval.py × 8 / inspect / visualize / probe / phase / final_plotting.

    All do ``env.current_location == goal``. Mirror the exact expression
    using a stand-in tuple for current_location.
    """

    @pytest.mark.parametrize("pos,goal,expected", [
        ((0, 0), (0, 0), True),
        ((0, 0), (0, 1), False),
        ((0, 0), (1, 0), False),
        ((4, 7), (4, 7), True),
        ((4, 7), (7, 4), False),
    ])
    def test_tuple_eq_pinned(self, pos, goal, expected):
        result = pos == goal
        assert result is expected
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Helper test scaffold: filled in once the helper is added. Each parametrized
# input is checked against the inline reference. After refactor the helper
# replaces the reference inside the source; here we hold both for parity.
# ---------------------------------------------------------------------------

# Inputs covering every shape/dtype combination seen across the 27 sites.
_HELPER_CASES_SCALAR = [
    ((0, 0), (0, 0), True),
    ((3, 4), (3, 4), True),
    ((3, 4), (3, 5), False),
    ((3, 4), (4, 4), False),
    # int32 row sliced from a (B,2) array
    (np.array([5, 7], dtype=np.int32), (5, 7), True),
    (np.array([5, 7], dtype=np.int32), (5, 8), False),
    # int64 row
    (np.array([5, 7], dtype=np.int64), (5, 7), True),
    # tuple goal vs ndarray goal
    ((3, 4), np.array([3, 4]), True),
    ((3, 4), np.array([3, 5]), False),
]

_HELPER_CASES_BATCH = [
    (
        np.array([[3, 5], [0, 0], [3, 5], [7, 7]], dtype=np.int32),
        (3, 5),
        np.array([True, False, True, False]),
    ),
    (
        np.zeros((0, 2), dtype=np.int32),
        (1, 1),
        np.zeros((0,), dtype=bool),
    ),
    (
        np.array([[1, 1]], dtype=np.int32),
        (1, 1),
        np.array([True]),
    ),
    (
        np.array([[1, 1], [1, 1]], dtype=np.int32),
        np.array([1, 1]),  # goal as ndarray (rollout.py path)
        np.array([True, True]),
    ),
]


class TestHelperParity:
    """Raw-math parity using the private ``_at_goal_l2`` helper.

    Production code goes through ``at_goal(env)`` which auto-resolves to
    the env's actual position. These tests target the underlying L2 math
    directly so we can pin numeric behavior without env scaffolding.
    """

    @pytest.fixture(scope="class")
    def helper(self):
        from hopfield_nav.env import _at_goal_l2
        return _at_goal_l2

    @pytest.mark.parametrize("pos,goal,expected", _HELPER_CASES_SCALAR)
    def test_scalar_inputs(self, helper, pos, goal, expected):
        result = helper(pos, goal)
        assert isinstance(result, bool), (
            f"scalar input must return Python bool, got {type(result)}")
        assert result is expected

    @pytest.mark.parametrize("positions,goal,expected", _HELPER_CASES_BATCH)
    def test_batch_inputs(self, helper, positions, goal, expected):
        result = helper(positions, goal)
        assert isinstance(result, np.ndarray), (
            f"batch input must return ndarray, got {type(result)}")
        assert result.dtype == bool
        assert result.shape == expected.shape
        np.testing.assert_array_equal(result, expected)

    def test_helper_matches_simulator_step_batch(self, helper):
        """End-to-end: helper output equals simulator's goal_reached vector."""
        base = GridEnv(size=8, seed=3, observation_size=12)
        base._goal = (2, 6)
        vec = VecEnv(base, batch_size=5)
        vec.reset_all()
        vec._pos = np.array(
            [[2, 6], [0, 0], [2, 6], [3, 6], [2, 5]], dtype=np.int32)
        # What the helper would say, given the pre-step positions.
        helper_says = helper(vec._pos.copy(), vec._goal)
        # What step_batch actually reports.
        _r, goal_reached, _ = vec.step_batch(np.zeros(5, dtype=int))
        np.testing.assert_array_equal(helper_says, goal_reached)

    def test_helper_matches_continuous_step_batch(self, helper):
        base = ContinuousGridEnv(
            size=8, seed=4, observation_size=12,
            scale=1.0, normalize=False,
        )
        base._goal = (2, 6)
        vec = ContinuousVecEnv(base, batch_size=4)
        vec.reset_all()
        vec._pos_f = np.array(
            [[2.0, 6.0], [0.0, 0.0], [2.0, 6.0], [3.0, 6.0]])
        vec._update_snapped()
        helper_says = helper(vec._pos.copy(), vec._goal)
        _r, goal_reached, _ = vec.step_batch(np.zeros((4, 2)))
        np.testing.assert_array_equal(helper_says, goal_reached)


# ---------------------------------------------------------------------------
# Radius semantics. The helper now accepts a continuous position and a
# Euclidean (L2) radius. Default radius=0.5 must preserve the snap-
# equality behavior on integer positions (verified above) and define a
# circle on continuous positions.
# ---------------------------------------------------------------------------

class TestRadiusSemantics:

    @pytest.fixture(scope="class")
    def helper(self):
        from hopfield_nav.env import _at_goal_l2
        return _at_goal_l2

    # --- Default radius=0.5 on continuous positions ----------------------

    def test_continuous_at_goal_exactly(self, helper):
        # Float pos exactly at goal -> distance 0 <= 0.5
        assert helper((3.0, 4.0), (3, 4)) is True

    def test_continuous_just_inside_radius_axis_aligned(self, helper):
        # 0.5 along +x axis: distance == radius (inclusive boundary)
        assert helper((3.5, 4.0), (3, 4)) is True

    def test_continuous_just_outside_radius_axis_aligned(self, helper):
        # 0.51 along +x axis: distance > radius
        assert helper((3.51, 4.0), (3, 4)) is False

    def test_continuous_diagonal_outside_l2_inside_linf(self, helper):
        # The exact case from the AskUserQuestion preview. With L2 this is
        # NOT at goal even though it WOULD snap to the goal cell.
        # Distance = sqrt(0.4^2 + 0.4^2) = 0.566 > 0.5.
        assert helper((3.4, 4.4), (3, 4)) is False

    def test_continuous_diagonal_just_inside(self, helper):
        # sqrt(0.3^2 + 0.3^2) = 0.424 < 0.5
        assert helper((3.3, 4.3), (3, 4)) is True

    # --- Custom radii ----------------------------------------------------

    def test_custom_radius_zero_only_exact_match(self, helper):
        # radius=0: only exact equality counts.
        assert helper((3, 4), (3, 4), radius=0.0) is True
        assert helper((3.0001, 4.0), (3, 4), radius=0.0) is False

    def test_custom_radius_one_includes_adjacent_cells(self, helper):
        # radius=1 includes axis-adjacent integer cells (distance 1).
        assert helper((4, 4), (3, 4), radius=1.0) is True
        # And diagonal neighbors (distance sqrt(2) ≈ 1.414) excluded.
        assert helper((4, 5), (3, 4), radius=1.0) is False
        # ...but included if radius >= sqrt(2):
        assert helper((4, 5), (3, 4), radius=1.5) is True

    def test_custom_radius_large_always_true_within_grid(self, helper):
        # radius=100 — anything within a small grid is "at goal".
        assert helper((0, 0), (7, 7), radius=100.0) is True

    # --- Batched continuous positions ------------------------------------

    def test_batch_continuous_default_radius(self, helper):
        positions = np.array([
            [3.0, 4.0],   # exact -> True
            [3.5, 4.0],   # boundary -> True
            [3.51, 4.0],  # outside -> False
            [3.4, 4.4],   # snap-yes-but-L2-no -> False
            [3.3, 4.3],   # diag inside -> True
        ])
        out = helper(positions, (3, 4))
        assert out.dtype == bool
        assert out.shape == (5,)
        assert out.tolist() == [True, True, False, False, True]

    def test_batch_continuous_custom_radius(self, helper):
        positions = np.array([
            [3.0, 4.0],
            [4.0, 4.0],   # axial neighbor at distance 1
            [4.0, 5.0],   # diagonal neighbor at distance sqrt(2)
        ])
        # With radius=1 only the first two are included.
        out_r1 = helper(positions, (3, 4), radius=1.0)
        assert out_r1.tolist() == [True, True, False]
        # With radius=sqrt(2) all three are included.
        out_rsq2 = helper(positions, (3, 4), radius=float(np.sqrt(2)))
        assert out_rsq2.tolist() == [True, True, True]

    def test_batch_zero_size_with_radius(self, helper):
        positions = np.zeros((0, 2), dtype=np.float64)
        out = helper(positions, (1, 1), radius=2.0)
        assert out.shape == (0,)
        assert out.dtype == bool

    # --- Backward compat reaffirmed: integer positions unchanged ----------

    def test_default_radius_preserves_integer_match_semantics(self, helper):
        # Any pair of integer positions: dist is 0 (match) or >= 1 (no match).
        # Default radius=0.5 separates these cleanly.
        for x in range(5):
            for y in range(5):
                expected = (x == 2 and y == 3)
                assert helper((x, y), (2, 3)) is expected

    def test_default_radius_preserves_int32_row_semantics(self, helper):
        # int32 sliced rows from VecEnv._pos must continue to work.
        for x, y, expected in [(2, 3, True), (2, 4, False), (3, 3, False)]:
            row = np.array([x, y], dtype=np.int32)
            assert helper(row, (2, 3)) is expected


# ---------------------------------------------------------------------------
# End-to-end cfg plumbing: EnvConfig.goal_radius → make_env → env.goal_radius
# → simulator step methods. Verifies the radius actually changes goal_reached
# semantics in step_batch when configured.
# ---------------------------------------------------------------------------

class TestGoalRadiusPlumbing:

    def test_env_config_default_is_half(self):
        from hopfield_nav.config import EnvConfig
        cfg = EnvConfig()
        assert cfg.goal_radius == 0.5

    def test_make_env_threads_goal_radius(self):
        from hopfield_nav.config import EnvConfig
        from hopfield_nav.env import make_env
        cfg = EnvConfig(size=8, observation_size=12, goal_radius=1.5)
        env = make_env(cfg, "discrete", seed=0)
        assert env.goal_radius == 1.5

    def test_make_env_threads_goal_radius_continuous(self):
        from hopfield_nav.config import EnvConfig
        from hopfield_nav.env import make_env
        cfg = EnvConfig(size=8, observation_size=12, movement_mode="continuous",
                        goal_radius=2.0)
        env = make_env(cfg, "continuous", seed=0)
        assert env.goal_radius == 2.0

    def test_vec_env_inherits_goal_radius(self):
        base = GridEnv(size=8, seed=0, observation_size=12, goal_radius=1.5)
        vec = VecEnv(base, batch_size=3)
        assert vec.goal_radius == 1.5

    def test_continuous_vec_env_inherits_goal_radius(self):
        base = ContinuousGridEnv(
            size=8, seed=0, observation_size=12, goal_radius=1.5,
            scale=1.0, normalize=False,
        )
        vec = ContinuousVecEnv(base, batch_size=3)
        assert vec.goal_radius == 1.5

    def test_grid_env_reward_respects_goal_radius(self):
        # With radius=1, axial neighbors of goal also pay goal_reward.
        env = GridEnv(size=8, seed=0, observation_size=12,
                      time_penalty=0.01, goal_reward=1.0, goal_radius=1.0)
        env._goal = (4, 4)
        env._pos = (4, 5)  # axial neighbor — distance 1, within radius 1
        assert env.reward() == pytest.approx(1.0)
        env._pos = (5, 5)  # diagonal — distance sqrt(2) > 1
        assert env.reward() == pytest.approx(-0.01)

    def test_vec_env_step_batch_respects_goal_radius(self):
        base = GridEnv(size=8, seed=0, observation_size=12,
                       time_penalty=0.01, goal_reward=1.0, goal_radius=1.0)
        base._goal = (3, 5)
        vec = VecEnv(base, batch_size=4)
        vec.reset_all()
        vec._pos = np.array([
            [3, 5],   # exact: distance 0
            [3, 4],   # axial: distance 1, inside radius 1
            [4, 5],   # axial: distance 1, inside radius 1
            [4, 6],   # diagonal: distance sqrt(2) > 1, outside
        ], dtype=np.int32)
        rewards, goal_reached, _ = vec.step_batch(np.zeros(4, dtype=int))
        assert goal_reached.tolist() == [True, True, True, False]
        np.testing.assert_allclose(rewards, [1.0, 1.0, 1.0, -0.01])

    def test_continuous_vec_env_step_batch_respects_goal_radius(self):
        base = ContinuousGridEnv(
            size=8, seed=0, observation_size=12,
            time_penalty=0.01, goal_reward=1.0, goal_radius=1.0,
            scale=1.0, normalize=False,
        )
        base._goal = (3, 5)
        vec = ContinuousVecEnv(base, batch_size=3)
        vec.reset_all()
        # Snapped positions are still int — radius gate applies on snap-distance.
        vec._pos_f = np.array([[3.0, 5.0], [4.0, 5.0], [4.0, 6.0]])
        vec._update_snapped()
        _r, goal_reached, _ = vec.step_batch(np.zeros((3, 2)))
        assert goal_reached.tolist() == [True, True, False]

    def test_default_radius_preserves_step_batch_semantics(self):
        # Sanity: with default radius=0.5, only exact integer matches count
        # (the pre-goal_radius behavior).
        base = GridEnv(size=8, seed=0, observation_size=12)
        base._goal = (3, 5)
        vec = VecEnv(base, batch_size=4)
        vec.reset_all()
        vec._pos = np.array([
            [3, 5], [3, 4], [4, 5], [4, 6],
        ], dtype=np.int32)
        _r, goal_reached, _ = vec.step_batch(np.zeros(4, dtype=int))
        assert goal_reached.tolist() == [True, False, False, False]


# ---------------------------------------------------------------------------
# at_goal(env) / at_goal(vec): env-based dispatch, the production API.
# Verifies the helper auto-resolves to the env's actual position (continuous
# when available) — the snap is never used in continuous mode.
# ---------------------------------------------------------------------------

class TestAtGoalEnvDispatch:

    def test_grid_env_uses_pos(self):
        from hopfield_nav.env import at_goal
        env = GridEnv(size=8, seed=0, observation_size=12)
        env._goal = (3, 4)
        env._pos = (3, 4)
        assert at_goal(env) is True
        env._pos = (4, 4)  # adjacent, distance 1 > 0.5
        assert at_goal(env) is False

    def test_continuous_grid_env_uses_continuous_pos_not_snap(self):
        """Regression: the visualize_trajectories bug.

        Continuous pos (3.45, 4.45) snaps to (3, 4) (= goal) but L2 distance
        to (3, 4) is sqrt(0.405) ≈ 0.636 > 0.5. The OLD behavior (using
        env.current_location, which is the snap) said at_goal=True. The
        FIXED behavior uses the actual continuous position, so at_goal=False.
        """
        from hopfield_nav.env import at_goal
        env = ContinuousGridEnv(
            size=8, seed=0, observation_size=12,
            scale=1.0, normalize=False,
        )
        env._goal = (3, 4)
        # Continuous pos in the snap-square but outside the L2-radius circle.
        env._continuous_pos = np.array([3.45, 4.45])
        env._pos = env.current_location  # snaps to (3, 4) = goal
        assert env.current_location == (3, 4)        # snap matches goal
        assert at_goal(env) is False                  # but L2 says no

    def test_continuous_grid_env_uses_continuous_pos_inside(self):
        from hopfield_nav.env import at_goal
        env = ContinuousGridEnv(
            size=8, seed=0, observation_size=12,
            scale=1.0, normalize=False,
        )
        env._goal = (3, 4)
        # Inside the L2 circle of radius 0.5.
        env._continuous_pos = np.array([3.3, 4.3])
        env._pos = env.current_location
        assert at_goal(env) is True

    def test_continuous_grid_env_axis_aligned_at_radius(self):
        from hopfield_nav.env import at_goal
        env = ContinuousGridEnv(
            size=8, seed=0, observation_size=12,
            scale=1.0, normalize=False, goal_radius=0.5,
        )
        env._goal = (3, 4)
        env._continuous_pos = np.array([3.5, 4.0])  # exactly on circle
        env._pos = env.current_location
        assert at_goal(env) is True

    def test_vec_env_uses_pos(self):
        from hopfield_nav.env import at_goal
        base = GridEnv(size=8, seed=0, observation_size=12)
        base._goal = (3, 4)
        vec = VecEnv(base, batch_size=4)
        vec.reset_all()
        vec._pos = np.array([[3, 4], [3, 5], [4, 4], [0, 0]], dtype=np.int32)
        out = at_goal(vec)
        assert out.dtype == bool
        assert out.tolist() == [True, False, False, False]

    def test_continuous_vec_env_uses_pos_f_not_pos(self):
        """Vec-env analog of the snap-vs-continuous regression."""
        from hopfield_nav.env import at_goal
        base = ContinuousGridEnv(
            size=8, seed=0, observation_size=12,
            scale=1.0, normalize=False,
        )
        base._goal = (3, 4)
        vec = ContinuousVecEnv(base, batch_size=3)
        vec.reset_all()
        # Row 0: continuous (3.45, 4.45) — snap matches, L2 outside.
        # Row 1: continuous (3.3, 4.3) — both match.
        # Row 2: continuous (3.0, 4.0) — exact.
        vec._pos_f = np.array([[3.45, 4.45], [3.3, 4.3], [3.0, 4.0]])
        vec._update_snapped()
        # Snap says rows 0, 2 match the goal (row 1 snaps to (3, 4) too).
        assert (vec._pos[0] == np.array([3, 4])).all()
        # But the helper uses _pos_f, so row 0 is OUT (distance ≈ 0.636).
        out = at_goal(vec)
        assert out.tolist() == [False, True, True]

    def test_continuous_vec_env_step_batch_uses_continuous_position(self):
        """End-to-end: the bug surfaces in step_batch's goal_reached."""
        from hopfield_nav.env import at_goal
        base = ContinuousGridEnv(
            size=8, seed=0, observation_size=12,
            scale=1.0, normalize=False,
        )
        base._goal = (3, 4)
        vec = ContinuousVecEnv(base, batch_size=2)
        vec.reset_all()
        vec._pos_f = np.array([[3.45, 4.45], [3.3, 4.3]])
        vec._update_snapped()
        _r, goal_reached, _ = vec.step_batch(np.zeros((2, 2)))
        # Pre-fix: goal_reached would be [True, True] (snap-based).
        # Post-fix: only row 1 is within L2 radius 0.5.
        assert goal_reached.tolist() == [False, True]

    def test_continuous_grid_env_step_at_corner_of_snap_square(self):
        """End-to-end via env.step → reward → at_goal."""
        env = ContinuousGridEnv(
            size=8, seed=0, observation_size=12,
            scale=1.0, normalize=False,
            time_penalty=0.01, goal_reward=1.0,
        )
        env._goal = (3, 4)
        env._continuous_pos = np.array([3.45, 4.45])
        env._pos = env.current_location  # (3, 4) — snap matches goal
        # Pre-fix: env.reward would return goal_reward (snap matches).
        # Post-fix: at_goal(self) uses _continuous_pos, L2 ≈ 0.636 > 0.5.
        assert env.reward() == pytest.approx(-0.01)

    def test_at_goal_rejects_explicit_goal_or_radius(self):
        from hopfield_nav.env import at_goal
        env = GridEnv(size=8, seed=0, observation_size=12)
        env._goal = (3, 4)
        env._pos = (3, 4)
        # Helper must refuse to silently override env-owned config.
        with pytest.raises(TypeError):
            at_goal(env, goal=(0, 0))
        with pytest.raises(TypeError):
            at_goal(env, radius=2.0)
