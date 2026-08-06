"""Do the evaluators compute the right number? Checked against hand arithmetic.

The golden fixtures pin each evaluator against its own previous output on an
untrained network. That is a regression test: it catches a metric that *moves*,
and is blind to one that was wrong from the start. If `mean_coverage` had been
off by a factor of two, or `store_efficiency` had divided by the wrong count,
every golden would have pinned the mistake and every test would have passed.

These tests know the answer. A `ScriptedAgent` walks a fixed direction on a
small grid, so which cells it visits and when it reaches the goal follow from
arithmetic, and the expected value of every metric can be written down.

Determinism: the policy is a constant, `deterministic=True` everywhere, and the
starts are passed explicitly rather than drawn -- so nothing here depends on a
seed. The two evaluator-level tests that do go through `random_start` assert
identities between the returned metrics and the per-trial records, which hold
whatever the draws were.

Geometry, once, so the arithmetic below is checkable:
    CARDINAL_ACTIONS = [(0,1), (1,0), (0,-1), (-1,0)]  ->  N, E, S, W
    `GridEnv.step` clamps to [0, size-1], so an east-walker starting at
    (x0, y) visits (x0, y), (x0+1, y) ... (size-1, y) and then stays put:
    `size - x0` distinct cells, reaching the east wall after `size-1-x0` steps.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from hopfield import Hopfield
from hopfield_nav.evaluation.batched import (
    batched_exploration_trials, batched_navigation_trials,
)
from hopfield_nav.evaluation.metrics import (
    evaluate_exploration, evaluate_goal_discovery,
)
from hopfield_nav.tests.fixtures import ScriptedAgent, make_collector, make_stub_cfg
from hopfield_nav.world.env import make_env

EMBED_DIM = 8
EAST = 1
SIZE = 5
DEVICE = torch.device("cpu")


def _world(goal: tuple[int, int], *, size: int = SIZE, n_envs: int = 1):
    """A stub world with the goal placed where the test wants it."""
    cfg = make_stub_cfg(movement_mode="discrete")
    cfg.env.size = size
    cfg.env.goal_radius = 0.5          # exactly the goal cell, nothing else
    _c, _agent, vh = make_collector(cfg, EMBED_DIM, seed=0)
    vh.env_offsets = [(0, 0), (8, 8)][:n_envs]
    envs = []
    for i in range(n_envs):
        env = make_env(cfg.env, "discrete", seed=100 + i)
        env._goal = goal               # override the seeded placement
        envs.append(env)
    return cfg, vh, envs


def _empty(cfg, n: int) -> list[Hopfield]:
    return [Hopfield(EMBED_DIM, beta=cfg.hopfield.beta, device="cpu")
            for _ in range(n)]


# ---------------------------------------------------------------------------
# exploration: which cells, and how the coverage numbers follow from them
# ---------------------------------------------------------------------------

def test_exploration_visits_exactly_the_cells_the_walk_covers():
    """An east-walker from (1, 2) on a 5-wide grid covers (1..4, 2). Four cells."""
    cfg, vh, envs = _world(goal=(0, 0))
    agent = ScriptedAgent(move=EAST); agent.eval()
    visited, found, steps = batched_exploration_trials(
        agent=agent, env=envs[0], env_offset=vh.env_offsets[0], vectorhash=vh,
        hopfields=_empty(cfg, 1), cfg=cfg, device=DEVICE,
        starts=[(1, 2)], max_steps=10, deterministic=True)

    assert visited[0] == {(1, 2), (2, 2), (3, 2), (4, 2)}
    # It reaches the east wall after 3 steps and is clamped for the other 7,
    # so more budget does not mean more cells.
    assert len(visited[0]) == SIZE - 1
    assert found[0] is False and steps[0] == -1


def test_exploration_reports_the_step_the_walk_hits_the_goal():
    """Goal at (3, 2), start at (0, 2): east-walker arrives on step 3."""
    cfg, vh, envs = _world(goal=(3, 2))
    agent = ScriptedAgent(move=EAST); agent.eval()
    _v, found, steps = batched_exploration_trials(
        agent=agent, env=envs[0], env_offset=vh.env_offsets[0], vectorhash=vh,
        hopfields=_empty(cfg, 1), cfg=cfg, device=DEVICE,
        starts=[(0, 2)], max_steps=10, deterministic=True)
    assert found[0] is True
    assert steps[0] == 3
    # ...and it keeps walking past it: the goal is inert, not an endpoint.
    assert steps[0] < 10


def test_exploration_redundancy_is_one_when_rollouts_are_disjoint():
    """Four east-walkers in four different rows share no cell: redundancy 1.0.

    This is the case the metric exists to reward, and the one neither coverage
    number can distinguish from its opposite below.
    """
    cfg, vh, envs = _world(goal=(0, 0))
    agent = ScriptedAgent(move=EAST); agent.eval()
    starts = [(0, y) for y in range(4)]
    visited, _f, _s = batched_exploration_trials(
        agent=agent, env=envs[0], env_offset=vh.env_offsets[0], vectorhash=vh,
        hopfields=_empty(cfg, 4), cfg=cfg, device=DEVICE,
        starts=starts, max_steps=10, deterministic=True)

    union = set().union(*visited)
    summed = sum(len(v) for v in visited)
    assert summed == 4 * SIZE            # each walker covers a full row
    assert len(union) == 4 * SIZE        # no overlap at all
    assert len(union) / summed == 1.0


def test_exploration_redundancy_is_one_over_n_when_rollouts_coincide():
    """Four east-walkers from the *same* cell retrace one path: redundancy 1/N.

    Coverage per rollout is identical to the disjoint case above -- same walk,
    same cell count -- so `mean_coverage` cannot tell the two apart and
    `redundancy` is the only thing that does.
    """
    cfg, vh, envs = _world(goal=(0, 0))
    agent = ScriptedAgent(move=EAST); agent.eval()
    visited, _f, _s = batched_exploration_trials(
        agent=agent, env=envs[0], env_offset=vh.env_offsets[0], vectorhash=vh,
        hopfields=_empty(cfg, 4), cfg=cfg, device=DEVICE,
        starts=[(0, 2)] * 4, max_steps=10, deterministic=True)

    union = set().union(*visited)
    summed = sum(len(v) for v in visited)
    assert summed == 4 * SIZE
    assert len(union) == SIZE
    assert len(union) / summed == pytest.approx(0.25)


def test_exploration_metrics_match_the_records_they_are_built_from():
    """Aggregation identities, whatever the draws happened to be.

    Checks the formulas rather than the rollouts: every returned number is
    recomputed from `per_trial` and must agree.
    """
    cfg, vh, envs = _world(goal=(4, 4), n_envs=2)
    agent = ScriptedAgent(move=EAST); agent.eval()
    n_trials, max_steps = 5, 12
    records: list[tuple] = []
    res = evaluate_exploration(
        agent, envs, vh, [0, 1], cfg, DEVICE, num_trials=n_trials,
        max_steps=max_steps, n_distractors_list=[0], per_trial=records)
    m = res[0]

    assert len(records) == n_trials * len(envs)
    cells = np.array([r[3] for r in records], dtype=float)
    found = np.array([r[4] for r in records], dtype=bool)
    total_cells = cfg.env.size ** 2

    assert m["mean_coverage"] == pytest.approx(cells.mean() / total_cells)
    assert m["cells_per_step"] == pytest.approx(cells.mean() / max_steps)
    assert m["goal_find_rate"] == pytest.approx(found.mean())
    assert m["num_trials_per_env"] == n_trials
    assert m["max_steps"] == max_steps
    # Every walker ends on the east wall, so every trial covers a whole row.
    assert set(cells.tolist()) <= {float(c) for c in range(1, SIZE + 1)}


# ---------------------------------------------------------------------------
# navigation: the step count is the arithmetic
# ---------------------------------------------------------------------------

def test_navigation_returns_the_exact_step_count_and_minus_one_on_failure():
    """Start (0,2), goal (4,2): an east-walker takes exactly 4 steps.

    A north-walker in the same world never gets there and must report -1
    rather than, say, 0 or max_steps.
    """
    cfg, vh, envs = _world(goal=(4, 2))
    east = ScriptedAgent(move=EAST); east.eval()
    got = batched_navigation_trials(
        agent=east, env=envs[0], env_offset=vh.env_offsets[0], vectorhash=vh,
        hopfields=_empty(cfg, 1), cfg=cfg, device=DEVICE,
        starts=[(0, 2)], goal=(4, 2), max_steps=10, deterministic=True)
    assert got == [4]

    north = ScriptedAgent(move=0); north.eval()   # 0 = N
    got = batched_navigation_trials(
        agent=north, env=envs[0], env_offset=vh.env_offsets[0], vectorhash=vh,
        hopfields=_empty(cfg, 1), cfg=cfg, device=DEVICE,
        starts=[(0, 2)], goal=(4, 2), max_steps=10, deterministic=True)
    assert got == [-1]


def test_navigation_scores_each_row_from_its_own_start():
    """Four starts at different distances give four different step counts."""
    cfg, vh, envs = _world(goal=(4, 2))
    agent = ScriptedAgent(move=EAST); agent.eval()
    starts = [(0, 2), (1, 2), (2, 2), (3, 2)]
    got = batched_navigation_trials(
        agent=agent, env=envs[0], env_offset=vh.env_offsets[0], vectorhash=vh,
        hopfields=_empty(cfg, 4), cfg=cfg, device=DEVICE,
        starts=starts, goal=(4, 2), max_steps=10, deterministic=True)
    assert got == [4, 3, 2, 1]


# ---------------------------------------------------------------------------
# goal discovery: arrivals, stores, and the teleport between them
# ---------------------------------------------------------------------------

def test_goal_discovery_credits_every_arrival_when_the_agent_always_stores():
    """store_efficiency is stores/arrivals, so an always-firing head scores 1.0.

    It also has to *have* arrivals for that to mean anything, which the
    assertion below insists on rather than accepting a vacuous 0/0.
    """
    # 2x2 with the goal on the east wall: an east-walker started in row 0
    # arrives, one started in row 1 clamps beside it. Measured 7 arrivals
    # across 12 trials, so the assertions below are not vacuous by luck.
    cfg, vh, envs = _world(goal=(1, 0), size=2, n_envs=1)
    agent = ScriptedAgent(move=EAST, store=1.0); agent.eval()
    records: list[tuple] = []
    res = evaluate_goal_discovery(
        agent, envs, vh, [0], cfg, DEVICE, num_trials=12, max_steps=20,
        n_distractors_list=[0], per_trial=records)

    arrivals = sum(r[6] for r in records)
    stores = sum(r[7] for r in records)
    assert arrivals > 0, "vacuous: the walker never reached the goal"
    assert stores == arrivals
    assert res[0]["store_efficiency"] == pytest.approx(1.0)
    assert res[0]["store_success_rate"] == pytest.approx(
        np.mean([r[4] for r in records]))


def test_goal_discovery_credits_nothing_when_the_agent_never_stores():
    """Arrivals still counted, stores zero -- the two are independent."""
    cfg, vh, envs = _world(goal=(1, 0), size=2, n_envs=1)
    agent = ScriptedAgent(move=EAST, store=0.0); agent.eval()
    records: list[tuple] = []
    res = evaluate_goal_discovery(
        agent, envs, vh, [0], cfg, DEVICE, num_trials=12, max_steps=20,
        n_distractors_list=[0], per_trial=records)

    assert sum(r[6] for r in records) > 0, "vacuous: no arrivals"
    assert sum(r[7] for r in records) == 0
    assert res[0]["store_efficiency"] == 0.0
    assert res[0]["store_success_rate"] == 0.0
    assert res[0]["reach_success_rate"] > 0.0    # reaching != storing


def test_goal_discovery_teleport_stops_an_arrival_every_step():
    """A walker parked on the goal would arrive every step without the teleport.

    An east-walker that reaches the east wall is clamped there forever. Put the
    goal on that wall and, with no relocation, every remaining step would begin
    at the goal -- `max_steps` arrivals from one trial. The teleport throws it
    to a random non-goal cell instead, so arrivals are bounded by how often the
    walk finds its way back.
    """
    cfg, vh, envs = _world(goal=(1, 0), size=2, n_envs=1)
    agent = ScriptedAgent(move=EAST, store=0.0); agent.eval()
    max_steps = 20
    records: list[tuple] = []
    evaluate_goal_discovery(
        agent, envs, vh, [0], cfg, DEVICE, num_trials=12, max_steps=max_steps,
        n_distractors_list=[0], per_trial=records)

    arrivals = [r[6] for r in records]
    assert sum(arrivals) > 0, "vacuous: never reached the goal"
    assert max(arrivals) < max_steps, (
        f"a trial recorded {max(arrivals)} arrivals in {max_steps} steps -- "
        f"the walker was left parked on the goal")
