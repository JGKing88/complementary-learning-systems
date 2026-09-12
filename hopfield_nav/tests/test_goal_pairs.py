"""Tests for the goal-conditioned pair experiment (plan §5.10).

Small world (size 8, 3 + 2 envs, lambdas 5 6 7) so the whole file runs in
seconds. What is pinned:

- the layout: widths sum to `compute_rnn_input_dim`; with every new flag at
  its default the assembled tensor is what it was; enabled-but-missing raises
- `FeedForwardCore`: T-step == T single-steps; state shape (L, B, H)
- the split: region ⊂ goal_cells_val; round-trip; a world.json without the
  field loads with an empty region
- the sampler: every p in starts, g in goals, p != g; enumeration count
- `pair_inputs` == `build_rnn_input` at prev_action = 0 (the A/B bridge)
- the optimal set: 1 action iff aligned, else 2; random acc ≈ mean|set|/4
- the NN decoder: exact on train × train in every mode
- `PairRegressor`: loss is the metric (continuous); learns xy in a few steps
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch

from hopfield_nav.config import EnvConfig, RNNAgentConfig, RNNTrainConfig
from hopfield_nav.evaluation.goal_pairs import (
    NearestNeighbourDecoder, aggregate_tables, angular_error_deg, enumerate_pairs,
    evaluate_pairs, optimal_action_set, pair_inputs, pair_targets, sample_pairs,
    score_pairs, unit_vectors, diff_tables)
from hopfield_nav.policy.agent_rnn import RNNAgent, compute_rnn_input_dim, rnn_input_layout
from hopfield_nav.policy.pair_regressor import PairRegressor
from hopfield_nav.policy.recurrent import FeedForwardCore, build_recurrent_core
from hopfield_nav.rollout.rnn import (
    build_rnn_input, goal_sensory_vec, grid_state_vec, sensory_vec, xy_vec)
from hopfield_nav.training.goal_pairs_setup import agent_cfg_for_mode, build_env_sets
from hopfield_nav.world.spec import CellSets, GeneratedSplit, WorldSpec

SIZE, OBS = 8, 12


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    os.environ.setdefault("CLS_RUNS", str(tmp_path_factory.mktemp("runs")))
    cfg = RNNTrainConfig(
        env=EnvConfig(size=SIZE, observation_size=OBS),
        agent=RNNAgentConfig(input_grid_state=True),
        n_envs=3, n_val_envs=2, env_generator=True, place_margin=4,
        goal_val_frac=0.25, region_val_frac=0.1, lambdas=[5, 6, 7], seed=3)
    train, heldout, same, split, vh, sgb = build_env_sets(
        cfg, np.random.RandomState(3), n_same=1, keep_field=True)
    return dict(cfg=cfg, train=train, heldout=heldout, same=same, split=split,
                vh=vh, sgb=sgb, cells=split.cell_sets())


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def test_layout_default_is_historical():
    cfg = RNNAgentConfig(input_prev_action=True, input_grid_state=True, goal_channel="abs")
    names = [n for n, _ in rnn_input_layout(cfg, OBS, 10)]
    assert names == ["sensory", "prev_action", "grid_state", "goal_vec"]
    assert compute_rnn_input_dim(cfg, OBS, 10) == OBS + 4 + 10 + 2


def test_layout_new_channels_append_in_order():
    cfg = RNNAgentConfig(input_sensory=True, sensory_mode="omni", input_grid_state=True,
                         input_xy_state=True, input_goal_grid_state=True, goal_sensory="omni")
    names = [n for n, _ in rnn_input_layout(cfg, OBS, 10)]
    assert names == ["sensory", "grid_state", "xy_state", "goal_grid_state", "goal_sensory"]
    assert compute_rnn_input_dim(cfg, OBS, 10) == 4 * OBS + 10 + 2 + 10 + 4 * OBS


def test_layout_sensory_off_removes_first_slot():
    cfg = RNNAgentConfig(input_sensory=False, input_grid_state=True, input_goal_grid_state=True)
    assert [n for n, _ in rnn_input_layout(cfg, OBS, 10)] == ["grid_state", "goal_grid_state"]
    assert compute_rnn_input_dim(cfg, OBS, 10) == 20


def test_build_rnn_input_raises_on_missing_new_channel():
    cfg = RNNAgentConfig(input_sensory=False, input_grid_state=True, input_goal_grid_state=True)
    with pytest.raises(KeyError, match="goal_grid_state"):
        build_rnn_input(None, None, None, np.zeros((2, 10), np.float32), cfg, "cpu")


def test_build_rnn_input_historical_lenient_guard_kept():
    # prev_action enabled but None still silently skipped: callers rely on it.
    cfg = RNNAgentConfig(input_prev_action=True)
    x = build_rnn_input(np.zeros((2, OBS), np.float32), None, None, None, cfg, "cpu")
    assert x.shape == (2, 1, OBS)


# ---------------------------------------------------------------------------
# Trunk
# ---------------------------------------------------------------------------

def test_feedforward_core_contracts():
    cfg = RNNAgentConfig(rnn_cell="mlp", num_rnn_layers=3, hidden_size=16, rnn_nonlinearity="relu")
    core = build_recurrent_core(cfg, 5)
    assert isinstance(core, FeedForwardCore)
    x = torch.randn(4, 6, 5)
    y, h = core(x)
    assert y.shape == (4, 6, 16) and h.shape == (3, 4, 16) and (h == 0).all()
    ys = torch.cat([core(x[:, t:t + 1])[0] for t in range(6)], dim=1)
    assert torch.allclose(y, ys)


def test_mlp_softplus_rejected():
    with pytest.raises(ValueError, match="softplus"):
        build_recurrent_core(RNNAgentConfig(rnn_cell="mlp", rnn_nonlinearity="softplus"), 5)


def test_rnn_agent_runs_on_mlp_trunk():
    cfg = RNNAgentConfig(rnn_cell="mlp", num_rnn_layers=2, hidden_size=16, movement_mode="continuous")
    agent = RNNAgent(cfg, OBS)
    dist, h = agent(torch.randn(3, 4, OBS), None)
    assert dist.mean.shape == (3, 4, 2) and h.shape == (2, 3, 16)


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def test_region_subset_and_partition(world):
    split, cells = world["split"], world["cells"]
    assert cells.region and cells.region <= split.goal_cells_val
    assert not (cells.region & cells.start_train)
    arena = {(x, y) for x in range(SIZE) for y in range(SIZE)}
    assert cells.goal_train | cells.goal_heldout | cells.region == arena
    assert len(cells.region) == round(0.1 * SIZE * SIZE)


def test_split_roundtrip_and_legacy_load(world, tmp_path):
    split = world["split"]
    d = split.to_json()
    back = GeneratedSplit.from_json(json.loads(json.dumps(d)))
    assert back.region_cells == split.region_cells
    del d["region_cells"]
    legacy = GeneratedSplit.from_json(d)
    assert legacy.region_cells == frozenset()
    # And cell_sets on a legacy split: region empty, every cell a start.
    cs = legacy.cell_sets()
    assert cs.region == frozenset() and len(cs.start_train) == SIZE * SIZE


def test_cellsets_rejects_region_outside_goal_val(world):
    split = world["split"]
    bad = GeneratedSplit(**{**split.__dict__, "region_cells": frozenset(list(split.goal_cells_train)[:1])})
    with pytest.raises(ValueError, match="never be a goal"):
        bad.cell_sets()


# ---------------------------------------------------------------------------
# Sampler / enumeration / targets
# ---------------------------------------------------------------------------

def test_sampler_respects_sets(world):
    cells = world["cells"]
    rng = np.random.RandomState(0)
    for s in CellSets.START_SETS:
        for g in CellSets.GOAL_SETS:
            p, q = sample_pairs(cells, s, g, 500, rng)
            ps = {(int(i // SIZE), int(i % SIZE)) for i in p}
            gs = {(int(i // SIZE), int(i % SIZE)) for i in q}
            assert ps <= cells.starts(s) and gs <= cells.goals(g)
            assert (p != q).all()
            P, G = enumerate_pairs(cells, s, g)
            n_s, n_g = len(cells.starts(s)), len(cells.goals(g))
            assert len(P) == n_s * n_g - len(cells.starts(s) & cells.goals(g))


def test_optimal_set_sizes_and_random_acc():
    rng = np.random.RandomState(0)
    p = rng.randint(0, SIZE * SIZE, 5000)
    g = rng.randint(0, SIZE * SIZE, 5000)
    keep = p != g
    p, g = p[keep], g[keep]
    opt = optimal_action_set(p, g, SIZE)
    aligned = (p // SIZE == g // SIZE) | (p % SIZE == g % SIZE)
    assert (opt[aligned].sum(1) == 1).all() and (opt[~aligned].sum(1) == 2).all()
    t = pair_targets(p, g, SIZE, "discrete")
    assert np.allclose(t.sum(1), 1.0)
    rnd = score_pairs(rng.standard_normal((len(p), 4)).astype(np.float32), p, g, SIZE, "discrete")
    assert abs(rnd["metric"] - opt.sum(1).mean() / 4) < 0.03
    u = unit_vectors(p, g, SIZE)
    assert np.allclose(np.linalg.norm(u, axis=1), 1.0)
    assert angular_error_deg(u, u).max() < 0.1   # float32 arccos near 1


# ---------------------------------------------------------------------------
# The A/B bridge
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["xy", "grid", "regular"])
def test_pair_inputs_matches_build_rnn_input(world, mode):
    tr, sgb, cells = world["train"], world["sgb"], world["cells"]
    env, off, t = tr.envs[0], tr.offsets[0], tr.tensors[0]
    acfg = agent_cfg_for_mode(mode, "continuous")
    p, g = sample_pairs(cells, "train", "train", 64, np.random.RandomState(1))
    xa = pair_inputs(t, acfg, p, g)
    pos = np.stack([p // SIZE, p % SIZE], 1)
    goals = np.stack([g // SIZE, g % SIZE], 1)
    xb = build_rnn_input(
        sensory=sensory_vec(env, pos, "omni") if acfg.input_sensory else None,
        prev_action=None, prev_reward=None,
        grid_state=grid_state_vec(pos, off, sgb) if acfg.input_grid_state else None,
        cfg=acfg, device="cpu",
        goal_vec=xy_vec(goals, SIZE) if acfg.goal_channel == "abs" else None,
        xy_state=xy_vec(pos, SIZE) if acfg.input_xy_state else None,
        goal_grid_state=grid_state_vec(goals, off, sgb) if acfg.input_goal_grid_state else None,
        goal_sensory=goal_sensory_vec(env, goals, "omni") if acfg.goal_sensory == "omni" else None,
        sensory_dim=OBS,
    )[:, 0].numpy()
    assert xa.shape == xb.shape and np.array_equal(xa, xb)
    assert xa.shape[1] == compute_rnn_input_dim(acfg, OBS, world["vh"].Ng)


# ---------------------------------------------------------------------------
# NN decoder and the table
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["xy", "grid", "regular"])
@pytest.mark.parametrize("mm", ["continuous", "discrete"])
def test_nn_decoder_exact_on_train_train(world, mode, mm):
    t, cells = world["train"].tensors[0], world["cells"]
    dec = NearestNeighbourDecoder(t, agent_cfg_for_mode(mode, mm), cells, mm)
    p, g = enumerate_pairs(cells, "train", "train")
    sc = score_pairs(dec.predict_for_pairs(p, g), p, g, SIZE, mm)["metric"]
    assert (sc < 0.1) if mm == "continuous" else (sc > 0.9999)


def test_table_reference_lines_and_keys(world):
    t, cells, vh = world["train"].tensors[0], world["cells"], world["vh"]
    acfg = agent_cfg_for_mode("grid", "continuous")
    model = PairRegressor(compute_rnn_input_dim(acfg, OBS, vh.Ng), 16, 1, "continuous")
    tab = evaluate_pairs(model, t, acfg, cells, movement_mode="continuous", device="cpu",
                         env_set="train", n_per_quadrant=256, rng=np.random.RandomState(0))
    for q, row in tab.items():
        assert row["teacher"]["metric"] < 0.1   # float32 arccos near 1
        assert 60 < row["random"]["metric"] < 120
        assert row["key"] == ("train", q[0], q[1], 256)
    agg = aggregate_tables([tab, tab])
    assert agg[("train", "train")]["n_envs"] == 2
    # C20: mismatched keys refuse to subtract.
    other = evaluate_pairs(model, t, acfg, cells, movement_mode="continuous", device="cpu",
                           env_set="train", n_per_quadrant=128, rng=np.random.RandomState(0))
    with pytest.raises(ValueError, match="keys differ"):
        diff_tables(tab, other)


# ---------------------------------------------------------------------------
# PairRegressor
# ---------------------------------------------------------------------------

def test_regressor_loss_is_metric_continuous():
    m = PairRegressor(4, 8, 1, "continuous")
    x = torch.randn(32, 4)
    tgt = torch.nn.functional.normalize(torch.randn(32, 2), dim=-1)
    loss = m.loss(x, tgt).item()
    d = m.predict_direction(x).detach().numpy()
    ang = angular_error_deg(d, tgt.numpy())
    assert abs(loss - float(np.mean(1 - np.cos(np.radians(ang))))) < 1e-5


def test_regressor_learns_xy_quickly(world):
    t, cells = world["train"].tensors[0], world["cells"]
    acfg = agent_cfg_for_mode("xy", "continuous")
    m = PairRegressor(4, 64, 2, "continuous", nonlinearity="relu")
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    rng = np.random.RandomState(0)
    for _ in range(300):
        p, g = sample_pairs(cells, "train", "train", 512, rng)
        x = torch.from_numpy(pair_inputs(t, acfg, p, g))
        y = torch.from_numpy(pair_targets(p, g, SIZE, "continuous"))
        loss = m.loss(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
    p, g = enumerate_pairs(cells, "region", "region")
    x = torch.from_numpy(pair_inputs(t, acfg, p, g))
    sc = score_pairs(m.predict_direction(x).detach().numpy(), p, g, SIZE, "continuous")
    assert sc["metric"] < 15.0, sc


# ---------------------------------------------------------------------------
# Experiment B: per-row goals in the vec env and the collector
# ---------------------------------------------------------------------------

def test_vec_goals_default_is_scalar_goal(world):
    from hopfield_nav.world.vec_env import make_vec
    env = world["train"].envs[0]
    vec = make_vec(env, 5, "discrete")
    assert vec._goals.shape == (5, 2) and (vec._goals == np.asarray(env._goal)).all()
    # No pool: a reset keeps every row's goal.
    vec.reset_indices(np.array([1, 3]))
    assert (vec._goals == np.asarray(env._goal)).all()


def test_vec_goal_pool_redraws_on_reset(world):
    from hopfield_nav.world.env import at_goal
    from hopfield_nav.world.vec_env import make_vec
    env, cells = world["train"].envs[0], world["cells"]
    vec = make_vec(env, 16, "discrete")
    vec.set_goal_pool(cells.goal_train)
    vec.reset_all()
    pool = set(map(tuple, vec._goal_pool.tolist()))
    for b in range(16):
        assert tuple(vec._goals[b]) in pool
        assert tuple(vec._pos[b]) != tuple(vec._goals[b])
    before = vec._goals.copy()
    vec.reset_indices(np.array([2, 5]))
    # Rows not reset keep their goal; reset rows draw from the pool.
    keep = np.ones(16, bool); keep[[2, 5]] = False
    assert (vec._goals[keep] == before[keep]).all()
    assert all(tuple(vec._goals[b]) in pool for b in (2, 5))
    # at_goal is row-wise on _goals.
    vec.set_positions(vec._goals.copy())
    assert at_goal(vec).all()


def test_collector_follows_per_row_goals(world):
    from hopfield_nav.rollout.rnn import collect_rollout_rnn
    from hopfield_nav.world.vec_env import make_vec
    env, off, sgb, cells = (world["train"].envs[0], world["train"].offsets[0],
                            world["sgb"], world["cells"])
    cfg = RNNAgentConfig(rnn_cell="gru", hidden_size=8, movement_mode="continuous",
                         input_sensory=False, input_grid_state=True,
                         input_goal_grid_state=True, input_prev_action=True)
    agent = RNNAgent(cfg, compute_rnn_input_dim(cfg, OBS, world["vh"].Ng))
    vec = make_vec(env, 6, "continuous")
    vec.set_goal_pool(cells.goal_train)
    vec.reset_all()
    g0 = vec._goals.copy()
    # Place every row ON its goal so the first step reaches and resets it.
    vec.set_positions(g0.astype(np.float64))
    # Two steps: t=0 is the at-goal step (reset after it), t=1 reads the new
    # goal. Stopping there means no row can reach again and be redrawn twice.
    r = collect_rollout_rnn(vec, agent, cfg, 2, "cpu", sgb=sgb, env_offset=off,
                            carry_across_episodes=True, episode_max_steps=50)
    assert (r.move_label_mask[:, 0] == 0).all()
    assert (r.episodes_completed >= 1).all()
    g1 = vec._goals.copy()
    assert (g1 != g0).any(axis=1).sum() >= 4, "most rows should have a new goal"
    from hopfield_nav.rollout.rnn import grid_state_vec
    D_gs = world["vh"].Ng
    # Layout: prev_action(2) | grid_state(Ng) | goal_grid_state(Ng).
    lo = 2 + D_gs
    chan = r.obs[:, 1, lo:lo + D_gs].numpy()
    # Rows that did not reach again at t=1 still hold g1; compare those.
    still = ~(r.goal_reached[:, 1].numpy().astype(bool))
    assert still.sum() >= 3
    assert np.allclose(chan[still], grid_state_vec(g1[still], off, sgb), atol=1e-6)


def test_trajectory_sampler_is_short_range_weighted(world):
    from hopfield_nav.evaluation.goal_pairs import sample_trajectory_pairs
    cells = world["cells"]
    rng = np.random.RandomState(0)
    pi, gi = sample_pairs(cells, "train", "train", 4000, rng)
    pt, gt = sample_trajectory_pairs(cells, "train", "train", 4000, rng)
    assert len(pt) == 4000 and (pt != gt).all()
    gs = {(int(i // SIZE), int(i % SIZE)) for i in gt}
    assert gs <= cells.goal_train
    cheb = lambda p, g: np.maximum(np.abs(p // SIZE - g // SIZE), np.abs(p % SIZE - g % SIZE))
    d_iid, d_tr = cheb(pi, gi), cheb(pt, gt)
    # The point of the sampler: mass moves to short range.
    assert d_tr.mean() < d_iid.mean() - 0.5
    assert (d_tr == 1).mean() > (d_iid == 1).mean() * 1.5
