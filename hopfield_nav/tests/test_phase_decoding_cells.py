"""The phase-decoding analysis, driven under every recurrent cell.

`analysis/phase_decoding` is the one consumer that reads the trunk's hidden
state directly rather than only its heads: `rollout.py` slices
``out["h_rnn"][-1, 0]`` into a per-timestep feature matrix, and every metric in
`metrics.py` is a geometric statement about that matrix. So a change to what
the trunk *is* lands here harder than anywhere else in the tree, and the
default-stays-GRU argument does not cover it -- the point of the option is that
somebody will run the analysis on a softplus policy.

Rather than restate the pipeline, this patches `_make_cfg` in the existing
suite and re-runs its end-to-end cases against each cell. What is being pinned
is not the *values* -- those should differ, that is the experiment -- but that
the pipeline runs, the shapes hold, and the metrics stay in their defined
ranges when the features are unbounded and strictly positive rather than
bounded and zero-centred.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import hopfield_nav.tests.test_phase_decoding as tpd
from hopfield_nav.policy.agent import NavAgent, compute_input_dim
from hopfield_nav.policy.recurrent import SoftplusRNN

# (rnn_cell, rnn_nonlinearity)
CELLS = [
    pytest.param(("gru", "tanh"), id="gru"),
    pytest.param(("rnn", "tanh"), id="rnn-tanh"),
    pytest.param(("rnn", "relu"), id="rnn-relu"),
    pytest.param(("rnn", "softplus"), id="rnn-softplus"),
]


@pytest.fixture(params=CELLS)
def cell(request, monkeypatch):
    """Point the borrowed suite's config factory at one cell type."""
    rnn_cell, nonlinearity = request.param
    original = tpd._make_cfg

    def _patched():
        cfg = original()
        cfg.agent.rnn_cell = rnn_cell
        cfg.agent.rnn_nonlinearity = nonlinearity
        return cfg

    monkeypatch.setattr(tpd, "_make_cfg", _patched)
    return request.param


def test_trunk_is_the_requested_cell(cell):
    """The knob reaches the module the analysis will read from."""
    rnn_cell, nonlinearity = cell
    cfg = tpd._make_cfg()
    agent = NavAgent(
        cfg.agent, compute_input_dim(cfg.agent, tpd.EMBED_DIM,
                                     cfg.env.observation_size))
    if rnn_cell == "gru":
        assert isinstance(agent.rnn, torch.nn.GRU)
    elif nonlinearity == "softplus":
        assert isinstance(agent.rnn, SoftplusRNN)
    else:
        assert isinstance(agent.rnn, torch.nn.RNN)
        assert not isinstance(agent.rnn, SoftplusRNN)
        assert agent.rnn.nonlinearity == nonlinearity


def test_collector_returns_per_arena_trials(cell):
    tpd.TestExploreExploitCollector().test_returns_per_arena_trials()


def test_pooled_and_centroids(cell):
    tpd.TestExploreExploitCollector().test_pooled_and_centroids()


def test_dataset_save_load_roundtrip(cell, tmp_path):
    tpd.TestExploreExploitCollector().test_save_load_roundtrip(tmp_path)


def test_stochastic_vs_deterministic(cell):
    tpd.TestExploreExploitCollector().test_stochastic_vs_deterministic_flag_runs()


def test_two_episode_trajectory(cell):
    """Carries h across a teleport without resetting it -- the case where an
    unbounded state would show up as drift rather than as an exception."""
    tpd.TestTwoEpisodeTrajectoryCollector().test_run_or_skip()


def test_exp1_pipeline(cell, tmp_path):
    """Collect -> splits -> parallelism + decodability -> metrics.json."""
    tpd.TestEndToEnd().test_exp1_pipeline(tmp_path)


def test_hidden_features_are_finite_and_in_range(cell):
    """The feature matrix the metrics consume: right width, finite, and for
    softplus strictly positive (the property that distinguishes it from tanh).
    """
    rnn_cell, nonlinearity = cell
    engine, bundle = tpd._make_engine_and_bundle(n_envs=2, seed=0)
    collector = tpd.ExploreExploitCollector(engine)
    ds = collector.collect(bundle, n_starts=2, max_steps=6, n_dist_min=0,
                           n_dist_max=2, deterministic=False, seed=0)
    h, phase, arena_id = ds.pooled()

    assert h.shape[1] == bundle.cfg.agent.hidden_size
    assert h.shape[0] == phase.shape[0] == arena_id.shape[0]
    assert np.isfinite(h).all(), "non-finite activations reached the decoder"
    if nonlinearity == "softplus":
        assert (h > 0).all(), "softplus features must be strictly positive"
    elif rnn_cell == "gru" or nonlinearity == "tanh":
        assert np.abs(h).max() <= 1.0 + 1e-5, "tanh-family state is bounded"


def test_metrics_stay_in_range_on_real_features(cell):
    """Parallelism is a cosine and decodability a balanced accuracy. Neither
    standardizes before computing the centroid difference, so an unbounded,
    off-centre feature space is exactly where a silent domain violation would
    appear -- as a nan or an out-of-range score, not as a crash.
    """
    engine, bundle = tpd._make_engine_and_bundle(n_envs=4, seed=0)
    collector = tpd.ExploreExploitCollector(engine)
    ds = collector.collect(bundle, n_starts=3, max_steps=6, n_dist_min=0,
                           n_dist_max=2, deterministic=False, seed=0)
    h, phase, arena_id = ds.pooled()
    if len(np.unique(phase)) < 2:
        pytest.skip("stub rollout produced a single phase")

    train, test = {0, 1}, {2, 3}
    par = tpd.parallelism_score(h, phase, arena_id, train, test)
    dec = tpd.decodability(h, phase, arena_id, train, test)
    assert np.isnan(par) or -1.0 - 1e-6 <= par <= 1.0 + 1e-6
    assert np.isnan(dec) or 0.0 <= dec <= 1.0
