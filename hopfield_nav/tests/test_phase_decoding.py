"""Tests for analysis.phase_decoding.

Strategy: bypass ckpt loading by constructing ``RolloutEngine`` via ``__new__``
and wiring its fields manually with the same ``StubVectorHash`` used by
``test_phase_decoding``. Drives the full pipeline (collect → metrics → MLP →
PCA-shape) on tiny synthetic data.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from hopfield_nav.policy.agent import NavAgent, compute_input_dim
from hopfield_nav.config import (
    AgentConfig, BCConfig, EnvConfig, HopfieldConfig, PPOConfig, TrainConfig,
    VectorHashConfig,
)
from hopfield_nav.world.env import GridEnv
from analysis.phase_decoding.classifier import (
    MLPPhaseClassifier, extract_hidden, train_mlp,
)
from analysis.phase_decoding.collect_trajectory import (
    TwoEpisodeTrajectoryCollector,
)
from analysis.phase_decoding.collect_trials import (
    ExploreExploitCollector, TrialsDataset,
)
from analysis.phase_decoding.metrics import (
    decodability, parallelism_score, within_arena_baseline,
)
from analysis.phase_decoding.rollout import EnvBundle, RolloutEngine
from analysis.phase_decoding.splits import (
    all_splits,
    split_loo,
    split_quadrant_one_vs_rest,
    split_quadrant_three_vs_one,
    split_random,
)
from hopfield_nav.tests.test_audit import StubVectorHash


# ---------------------------------------------------------------------------
# Synthetic engine + bundle (no ckpt)
# ---------------------------------------------------------------------------

ENV_SIZE = 4
NPOS = 12
EMBED_DIM = 16
HIDDEN = 16


def _make_cfg() -> TrainConfig:
    return TrainConfig(
        env=EnvConfig(size=ENV_SIZE, observation_size=12, time_penalty=0.01,
                      movement_mode="discrete"),
        vectorhash=VectorHashConfig(),
        hopfield=HopfieldConfig(beta=1.0, alpha=1.0, steps=1),
        agent=AgentConfig(
            hidden_size=HIDDEN, num_rnn_layers=1,
            hopfield_mode="discrete", movement_mode="discrete",
            input_encoded_state=False,
            input_hopfield_signal=True,
            input_prev_action=False,
            input_prev_reward=False,
            input_sensory=False,
            input_goal_in_memory=False,
        ),
        ppo=PPOConfig(),
        bc=BCConfig(),
        encoder_checkpoint="dummy",
        batch_envs=1,
        steps_per_rollout=8,
        device="cpu",
        num_val_envs=1,
    )


def _make_engine_and_bundle(n_envs: int = 4, seed: int = 0):
    cfg = _make_cfg()
    torch.manual_seed(seed)
    np.random.seed(seed)

    vh = StubVectorHash(Npos=NPOS, embed_dim=EMBED_DIM)
    offsets = [(0, 0), (0, ENV_SIZE + 1),
               (ENV_SIZE + 1, 0), (ENV_SIZE + 1, ENV_SIZE + 1)][:n_envs]
    vh.env_offsets = offsets

    val_envs = [GridEnv(size=ENV_SIZE, observation_size=12, seed=seed + i)
                for i in range(n_envs)]
    val_offsets = offsets

    input_dim = compute_input_dim(cfg.agent, EMBED_DIM, cfg.env.observation_size)
    agent = NavAgent(cfg.agent, input_dim).to("cpu")
    agent.eval()

    engine = RolloutEngine.__new__(RolloutEngine)
    engine.ckpt_path = "stub"
    engine.encoder_path = "stub"
    engine.embed_dim = EMBED_DIM
    engine.device = torch.device("cpu")
    engine.cfg = cfg
    engine.val_envs = val_envs
    engine.vh = vh
    engine.val_offsets = val_offsets
    engine.agent = agent
    engine.num_arenas = n_envs

    bundle = engine.build_bundle()
    return engine, bundle


# ---------------------------------------------------------------------------
# Splits unit tests
# ---------------------------------------------------------------------------

class TestSplits:
    def test_loo_covers_every_arena(self):
        s = split_loo([0, 1, 2, 3])
        assert s.name == "LOO"
        assert len(s.folds) == 4
        for f in s.folds:
            assert len(f.train) == 3 and len(f.test) == 1
            assert f.train.isdisjoint(f.test)

    def test_random_split_size_and_disjoint(self):
        s = split_random(list(range(10)), n_splits=5, test_frac=0.2, seed=0)
        assert len(s.folds) == 5
        for f in s.folds:
            assert len(f.test) == 2
            assert len(f.train) == 8
            assert f.train.isdisjoint(f.test)

    def test_quadrant_one_vs_rest_covers_all(self):
        # 8 arenas, 2 per quadrant
        quadrants = {i: i % 4 for i in range(8)}
        s = split_quadrant_one_vs_rest(quadrants)
        assert s.name == "Quadrant 1v3"
        assert len(s.folds) == 4
        for f in s.folds:
            assert len(f.train) == 2  # 1 quadrant × 2 arenas
            assert len(f.test) == 6

    def test_quadrant_three_vs_one_covers_all(self):
        quadrants = {i: i % 4 for i in range(8)}
        s = split_quadrant_three_vs_one(quadrants)
        assert len(s.folds) == 4
        for f in s.folds:
            assert len(f.test) == 2
            assert len(f.train) == 6

    def test_all_splits_returns_four_families(self):
        ids = list(range(8))
        quadrants = {i: i % 4 for i in range(8)}
        out = all_splits(ids, quadrants, n_random=3, test_frac=0.25, seed=0)
        names = [s.name for s in out]
        assert names == ["LOO", "Random 80/20", "Quadrant 1v3", "Quadrant 3v1"]


# ---------------------------------------------------------------------------
# Metrics unit tests
# ---------------------------------------------------------------------------

def _synthetic_trials(n_arenas=4, n_per_phase=20, hidden=8, sep=5.0, seed=0):
    """Build a TrialsDataset with cleanly separated explore/exploit means."""
    from analysis.phase_decoding.collect_trials import TrialData
    rng = np.random.RandomState(seed)
    per_arena: dict[int, TrialData] = {}
    for a in range(n_arenas):
        h0 = rng.randn(n_per_phase, hidden).astype(np.float32)
        h1 = rng.randn(n_per_phase, hidden).astype(np.float32) + sep
        per_arena[a] = TrialData(
            arena_id=a,
            goal_local=(0, 0),
            quadrant=a % 4,
            h_explore=h0, h_exploit=h1,
            trial_explore=np.zeros(n_per_phase, dtype=np.int64),
            trial_exploit=np.zeros(n_per_phase, dtype=np.int64),
            summaries_explore=[], summaries_exploit=[],
        )
    return TrialsDataset(per_arena=per_arena, meta={"hidden_size": hidden})


class TestMetrics:
    def test_parallelism_high_for_separated_data(self):
        data = _synthetic_trials(n_arenas=4, sep=5.0)
        h, phase, arena = data.pooled()
        score = parallelism_score(h, phase, arena, frozenset({0, 1}), frozenset({2, 3}))
        assert 0.9 < score <= 1.0

    def test_parallelism_in_range(self):
        data = _synthetic_trials(n_arenas=4)
        h, phase, arena = data.pooled()
        s = parallelism_score(h, phase, arena, frozenset({0}), frozenset({1, 2, 3}))
        assert -1.0 <= s <= 1.0

    def test_decodability_high_for_separated_data(self):
        data = _synthetic_trials(n_arenas=4, sep=5.0)
        h, phase, arena = data.pooled()
        acc = decodability(h, phase, arena, frozenset({0, 1}), frozenset({2, 3}))
        assert acc > 0.9

    def test_decodability_in_range(self):
        data = _synthetic_trials(n_arenas=4, sep=0.05)
        h, phase, arena = data.pooled()
        acc = decodability(h, phase, arena, frozenset({0, 1}), frozenset({2, 3}))
        assert 0.0 <= acc <= 1.0

    def test_decodability_subsample_train_runs(self):
        data = _synthetic_trials(n_arenas=4, sep=5.0, n_per_phase=50)
        h, phase, arena = data.pooled()
        acc = decodability(h, phase, arena, frozenset({0, 1}), frozenset({2, 3}),
                           subsample_train=20)
        assert 0.0 <= acc <= 1.0

    def test_within_arena_baseline_high_on_separated_data(self):
        data = _synthetic_trials(n_arenas=4, sep=5.0, n_per_phase=50)
        h, phase, arena = data.pooled()
        rows = within_arena_baseline(h, phase, arena, test_frac=0.2, seed=0)
        assert len(rows) == 4
        for r in rows:
            assert r["decodability"] > 0.9
            assert r["parallelism"] > 0.5  # within-arena cosine should be high

    def test_within_arena_baseline_skips_single_class(self):
        from analysis.phase_decoding.collect_trials import (
            TrialData, TrialsDataset,
        )
        td = TrialData(
            arena_id=0, goal_local=(0, 0), quadrant=0,
            h_explore=np.zeros((20, 4), dtype=np.float32),
            h_exploit=np.zeros((1, 4), dtype=np.float32),  # below min_per_phase
            trial_explore=np.zeros(20, dtype=np.int64),
            trial_exploit=np.zeros(1, dtype=np.int64),
            summaries_explore=[], summaries_exploit=[],
        )
        ds = TrialsDataset(per_arena={0: td}, meta={"hidden_size": 4})
        h, phase, arena = ds.pooled()
        rows = within_arena_baseline(h, phase, arena, min_per_phase=5)
        assert len(rows) == 1
        assert rows[0]["skipped"] is True
        assert np.isnan(rows[0]["decodability"])


# ---------------------------------------------------------------------------
# Classifier unit tests
# ---------------------------------------------------------------------------

class TestMLP:
    def test_overfits_separable_toy_data(self):
        rng = np.random.RandomState(0)
        n = 200
        h0 = rng.randn(n, 16).astype(np.float32)
        h1 = rng.randn(n, 16).astype(np.float32) + 5.0
        h = np.concatenate([h0, h1], axis=0)
        y = np.concatenate([np.zeros(n), np.ones(n)]).astype(np.int64)
        model, m = train_mlp(
            h, y, hidden_dim=16, epochs=10, batch_size=64, seed=0, device="cpu",
        )
        assert m["val_balanced_acc"] > 0.95
        hidden = extract_hidden(model, h, m["scaler"], device="cpu")
        assert hidden.shape == (h.shape[0], 16)

    def test_forward_returns_logits_and_hidden(self):
        m = MLPPhaseClassifier(in_dim=8, hidden_dim=4)
        x = torch.randn(3, 8)
        logits, hidden = m(x)
        assert logits.shape == (3, 2)
        assert hidden.shape == (3, 4)
        # ReLU is non-negative
        assert (hidden >= 0).all()


# ---------------------------------------------------------------------------
# Collectors integration tests
# ---------------------------------------------------------------------------

class TestExploreExploitCollector:
    def test_returns_per_arena_trials(self):
        engine, bundle = _make_engine_and_bundle(n_envs=4, seed=42)
        collector = ExploreExploitCollector(engine)
        data = collector.collect(
            bundle,
            n_starts=2, max_steps=8,
            n_dist_min=0, n_dist_max=2,
            deterministic=False, seed=0,
        )
        assert set(data.per_arena.keys()) == {0, 1, 2, 3}
        for td in data.per_arena.values():
            assert td.h_explore.ndim == 2 and td.h_explore.shape[1] == HIDDEN
            assert td.h_exploit.ndim == 2 and td.h_exploit.shape[1] == HIDDEN
            assert len(td.summaries_explore) == 2
            assert len(td.summaries_exploit) == 2

    def test_pooled_and_centroids(self):
        engine, bundle = _make_engine_and_bundle(n_envs=3, seed=7)
        data = ExploreExploitCollector(engine).collect(
            bundle, n_starts=2, max_steps=6,
            n_dist_min=0, n_dist_max=1, deterministic=True, seed=0,
        )
        h, phase, arena = data.pooled()
        assert h.shape[0] == phase.shape[0] == arena.shape[0]
        assert set(np.unique(phase).tolist()).issubset({0, 1})
        c0, c1 = data.centroids()
        assert c0.shape == (HIDDEN,) and c1.shape == (HIDDEN,)
        assert np.isfinite(c0).all() and np.isfinite(c1).all()

    def test_save_load_roundtrip(self, tmp_path: Path):
        engine, bundle = _make_engine_and_bundle(n_envs=2, seed=3)
        data = ExploreExploitCollector(engine).collect(
            bundle, n_starts=2, max_steps=6,
            n_dist_min=0, n_dist_max=1, deterministic=True, seed=0,
        )
        out = tmp_path / "trials"
        data.save(out)
        loaded = TrialsDataset.load(out)
        assert set(loaded.per_arena.keys()) == set(data.per_arena.keys())
        for a in data.per_arena:
            np.testing.assert_array_equal(
                loaded.per_arena[a].h_explore, data.per_arena[a].h_explore,
            )

    def test_stochastic_vs_deterministic_flag_runs(self):
        """Both flag values should run end-to-end without crashing."""
        engine, bundle = _make_engine_and_bundle(n_envs=2, seed=11)
        c = ExploreExploitCollector(engine)
        for det in (True, False):
            data = c.collect(
                bundle, n_starts=1, max_steps=6,
                n_dist_min=0, n_dist_max=1, deterministic=det, seed=0,
            )
            assert len(data.per_arena) == 2


class TestTwoEpisodeTrajectoryCollector:
    def test_run_or_skip(self):
        """With small env_size and high max_steps, ep1 should reach goal often
        enough that at least one trajectory is captured."""
        engine, bundle = _make_engine_and_bundle(n_envs=2, seed=5)
        trajs = TwoEpisodeTrajectoryCollector(engine).collect(
            bundle,
            n_traj_per_arena=4,
            max_steps_ep1=80, max_steps_ep2=40,
            n_dist_min=0, n_dist_max=1,
            deterministic=False, seed=0,
        )
        # If the agent never reaches goal in ep1, trajs will be empty; the
        # collector still must return a valid dataset object.
        assert trajs.meta["n_dropped_no_ep1_goal"] >= 0
        for traj in trajs.trajectories:
            T = traj.h.shape[0]
            assert traj.phase.shape == (T,)
            assert 0 < traj.switch_t <= T
            assert (traj.phase[:traj.switch_t] == 0).all()
            assert (traj.phase[traj.switch_t:] == 1).all()


# ---------------------------------------------------------------------------
# End-to-end smoke test
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_exp1_pipeline(self, tmp_path: Path):
        from analysis.phase_decoding.exp1 import _evaluate

        engine, bundle = _make_engine_and_bundle(n_envs=4, seed=99)
        # Collected hidden states are unlikely to be linearly separable for an
        # untrained agent, but the pipeline must run end-to-end.
        data = ExploreExploitCollector(engine).collect(
            bundle, n_starts=2, max_steps=10,
            n_dist_min=0, n_dist_max=1, deterministic=True, seed=0,
        )
        quadrants = {a: bundle.quadrants[a] for a in bundle.arena_ids()}
        splits = all_splits(bundle.arena_ids(), quadrants,
                            n_random=3, test_frac=0.25, seed=0)
        h, phase, arena = data.pooled()
        results = _evaluate(h, phase, arena, splits)
        for split in splits:
            assert split.name in results
            for fold in results[split.name]:
                assert "parallelism" in fold
                assert "decodability" in fold
                # Either a real value or NaN — never crashes.
                p = fold["parallelism"]
                d = fold["decodability"]
                assert (np.isnan(p) or -1.0 <= p <= 1.0)
                assert (np.isnan(d) or 0.0 <= d <= 1.0)

    def test_viz_bars_writes_png(self, tmp_path: Path):
        from analysis.phase_decoding.viz import plot_bars
        results = {
            "LOO": [{"parallelism": 0.8, "decodability": 0.9}] * 5,
            "Random 80/20": [{"parallelism": 0.7, "decodability": 0.85}] * 5,
            "Quadrant 1v3": [{"parallelism": 0.5, "decodability": 0.7}] * 4,
            "Quadrant 3v1": [{"parallelism": 0.6, "decodability": 0.78}] * 4,
        }
        out = tmp_path / "bars.png"
        plot_bars(results, out)
        assert out.exists() and out.stat().st_size > 0

    def test_viz_bars_grouped_writes_png(self, tmp_path: Path):
        from analysis.phase_decoding.viz import plot_bars_grouped
        a = {
            "Within-arena": [{"parallelism": 0.95, "decodability": 0.99}] * 4,
            "LOO":          [{"parallelism": 0.8,  "decodability": 0.9}]  * 5,
        }
        b = {
            "Within-arena": [{"parallelism": 0.93, "decodability": 0.98}] * 4,
            "LOO":          [{"parallelism": 0.1,  "decodability": 0.55}] * 5,
        }
        out = tmp_path / "grouped.png"
        plot_bars_grouped([("trained", a), ("random_init", b)], out)
        assert out.exists() and out.stat().st_size > 0

    def test_plot_module_loads_metrics_json(self, tmp_path: Path):
        from analysis.phase_decoding.plot import _load_results, _resolve

        run_dir = tmp_path / "run_a"
        run_dir.mkdir()
        results = {
            "LOO": [{"parallelism": 0.8, "decodability": 0.9}],
        }
        (run_dir / "metrics.json").write_text(json.dumps({
            "per_fold": results,
            "summary": {},
        }))
        # Resolve a directory → metrics.json under it.
        p = _resolve(str(run_dir))
        assert p.name == "metrics.json"
        loaded = _load_results(p)
        assert loaded == results
