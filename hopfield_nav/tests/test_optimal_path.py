"""`optimal_path_to_goal` and the two fields the evaluator records from it.

The suite already recorded `steps_to_goal` and `path_to_goal`, and neither is
comparable between arms on its own. Steps mixes route quality with step
magnitude, because a continuous-mode policy chooses how far it moves each step
and every trained arm here moves well under the teacher's unit vector. Path
isolates distance but is conditioned on success, so an arm that solves only the
near goals is scored on a nearer subpopulation than one that also solves the
far ones -- which is how the arm that forgets almost everything comes out
looking like the fastest navigator in the suite.

The optimum is what removes that, and here it is exact rather than a bound:
the arena has no interior obstacles, so the box is convex and the straight
segment is traversable. These tests pin that exactness, the units decision
(distance, not steps, under continuous movement), and the fact that the
recorded pair is conditioned the way the docstring claims.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from hopfield_nav.config import RNNAgentConfig
from hopfield_nav.evaluation.rnn import (
    evaluate_nav_one_env, optimal_path_to_goal)
from hopfield_nav.policy.agent_rnn import RNNAgent, compute_rnn_input_dim
from hopfield_nav.world.env import GridEnv

OBS = 24
SIZE = 6


def _build(movement_mode: str, **kw):
    cfg = RNNAgentConfig(
        hidden_size=16, num_rnn_layers=1, movement_mode=movement_mode,
        init_log_std=-1.0, freeze_log_std=True, **kw,
    )
    agent = RNNAgent(cfg, compute_rnn_input_dim(cfg, OBS))
    return agent, GridEnv(size=SIZE, observation_size=OBS, seed=3)


# --------------------------------------------------------------------------
# the helper
# --------------------------------------------------------------------------

def test_continuous_optimum_is_euclidean_less_the_goal_ball():
    """`at_goal` is an L2 ball, so the agent stops at its edge, not its centre."""
    starts = np.array([[0.0, 0.0], [3.0, 4.0]])
    out = optimal_path_to_goal(starts, (0, 0), goal_radius=0.5,
                               movement_mode="continuous")
    # Start (3,4) is 5.0 from the goal; it only has to travel 4.5 of that.
    assert out[1] == pytest.approx(4.5)
    # Standing on the goal, there is nothing left to travel -- and in
    # particular the radius must not make this negative.
    assert out[0] == pytest.approx(0.0)


def test_continuous_optimum_never_goes_negative_inside_the_ball():
    """A start inside the goal ball is 0 distance away, not a negative one.

    An unclamped `d - radius` would put a negative number in the denominator of
    every ratio built from this, which is worse than wrong -- it flips the
    sign of the efficiency figure rather than obviously breaking it.
    """
    out = optimal_path_to_goal(np.array([[0.2, 0.0]]), (0, 0),
                               goal_radius=0.5, movement_mode="continuous")
    assert out[0] == 0.0


def test_discrete_optimum_is_manhattan():
    """Four cardinal actions, one cell each: Manhattan is distance *and* steps."""
    out = optimal_path_to_goal(np.array([[1.0, 1.0], [0.0, 3.0]]), (3, 0),
                               goal_radius=0.5, movement_mode="discrete")
    assert out[0] == pytest.approx(3.0)   # |3-1| + |0-1|
    assert out[1] == pytest.approx(6.0)   # |3-0| + |0-3|


def test_discrete_optimum_ignores_the_goal_radius():
    """The radius is a continuous-geometry notion; on the grid it must not
    shorten a Manhattan count, or discrete optima would come out fractional."""
    a = optimal_path_to_goal(np.array([[0.0, 0.0]]), (2, 0),
                             goal_radius=0.5, movement_mode="discrete")
    b = optimal_path_to_goal(np.array([[0.0, 0.0]]), (2, 0),
                             goal_radius=1.5, movement_mode="discrete")
    assert a[0] == b[0] == pytest.approx(2.0)


def test_optimum_is_attainable_not_a_lower_bound():
    """The straight line must stay inside the arena for the optimum to be real.

    This is the property the whole metric rests on: the box is convex and has
    no interior obstacles, so the segment between any two interior points is
    traversable. If interior walls were ever added, this test is the one that
    should start failing.
    """
    env = GridEnv(size=SIZE, observation_size=OBS, seed=3)
    gx, gy = int(env._goal[0]), int(env._goal[1])
    starts = np.array([[float(x), float(y)]
                       for x in range(SIZE) for y in range(SIZE)])
    out = optimal_path_to_goal(starts, (gx, gy), env.goal_radius, "continuous")
    for (sx, sy), d in zip(starts, out):
        for t in np.linspace(0.0, 1.0, 11):
            px = sx + t * (gx - sx)
            py = sy + t * (gy - sy)
            assert 0.0 <= px <= SIZE - 1 and 0.0 <= py <= SIZE - 1, (
                "straight path left the arena; the optimum is not attainable")
        assert d <= math.hypot(gx - sx, gy - sy) + 1e-9


# --------------------------------------------------------------------------
# what the evaluator records
# --------------------------------------------------------------------------

@pytest.mark.parametrize("movement_mode", ["discrete", "continuous"])
def test_evaluator_reports_both_optimum_fields(movement_mode):
    agent, env = _build(movement_mode)
    m = evaluate_nav_one_env(env, agent, n_trials=6, max_steps=8,
                             device=torch.device("cpu"))
    assert "mean_optimal_to_goal" in m and "mean_optimal_all" in m
    # `_all` is over every trial, so it is finite whether or not any succeeded.
    assert math.isfinite(m["mean_optimal_all"])
    assert m["mean_optimal_all"] > 0.0


@pytest.mark.parametrize("movement_mode", ["discrete", "continuous"])
def test_optimum_to_goal_is_nan_exactly_when_nothing_succeeded(movement_mode):
    """It must follow `mean_path_to_goal`'s conditioning, not diverge from it.

    The point of recording it is that the two divide; if one is conditioned on
    success and the other is not, the ratio is silently comparing different
    trials.
    """
    agent, env = _build(movement_mode)
    # max_steps=1 gives the agent no chance to move, so nothing reaches.
    m = evaluate_nav_one_env(env, agent, n_trials=4, max_steps=1,
                             device=torch.device("cpu"))
    assert m["nav_det"] == 0.0
    assert math.isnan(m["mean_path_to_goal"])
    assert math.isnan(m["mean_optimal_to_goal"])
    assert math.isfinite(m["mean_optimal_all"])


def test_optimum_is_measured_from_the_start_not_after_a_teleport():
    """Reaching the goal teleports the agent, so the start must be captured at
    reset. Reading positions later would measure the *next* episode's start.

    The guard is that the recorded optimum stays within the range a genuine
    start in this arena can produce.
    """
    agent, env = _build("continuous")
    m = evaluate_nav_one_env(env, agent, n_trials=8, max_steps=6,
                             device=torch.device("cpu"))
    gx, gy = int(env._goal[0]), int(env._goal[1])
    far = math.hypot(max(gx, SIZE - 1 - gx), max(gy, SIZE - 1 - gy))
    assert 0.0 < m["mean_optimal_all"] <= far


def test_history_schema_carries_the_optimum():
    """The field is useless if it stops at the evaluator and never reaches the
    recorded history, which is what every later analysis actually reads."""
    from analysis.continual.baseline import _to_emit_metrics
    row = _to_emit_metrics({
        "nav_det": 1.0, "mean_steps_to_goal": 7.0, "mean_path_to_goal": 9.5,
        "mean_optimal_to_goal": 6.5, "mean_optimal_all": 6.5,
    })
    assert row == {"reached": 1, "steps_to_goal": 7, "path_to_goal": 9.5,
                   "optimal_to_goal": 6.5, "optimal_all": 6.5}

    miss = _to_emit_metrics({
        "nav_det": 0.0, "mean_steps_to_goal": float("nan"),
        "mean_path_to_goal": float("nan"),
        "mean_optimal_to_goal": float("nan"), "mean_optimal_all": 4.0,
    })
    assert miss["optimal_to_goal"] is None
    assert miss["optimal_all"] == 4.0
