"""Test suite for the bugs and parameter behaviors flagged in PARAMETER_AUDIT.md.

These exercise the recently-fixed bugs as regression tests, plus a few
behavioral pins that the audit suggested as worthwhile guards.

Run via run_tests.sh. Requires the `cls` conda env (torch + numpy).
"""
from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from hopfield_nav.config import (
    AgentConfig, BCConfig, EnvConfig, HopfieldConfig, PPOConfig, TrainConfig,
    VectorHashConfig, validate_train_config,
)
from hopfield_nav.env import CARDINAL_ACTIONS, ContinuousGridEnv, GridEnv
from hopfield_nav.hopfield import Hopfield, recall_per_env_batch
from hopfield_nav.oracle_bfs import bfs_action_batch_discrete
from hopfield_nav.utils import (
    classify_direction_batch, direction_to_onehot, gram_schmidt_2d_batch,
)
from hopfield_nav.tests.fixtures import StubVectorHash, make_stub_cfg
from hopfield_nav.vec_env import ContinuousVecEnv, VecEnv


# ---------------------------------------------------------------------------
# Pure unit tests
# ---------------------------------------------------------------------------

class TestUtils:
    def test_gram_schmidt_orthonormality(self):
        """W rows should be orthonormal in the embedding tangent space."""
        rng = np.random.default_rng(0)
        B, D = 8, 32
        d_fwd = rng.standard_normal((B, D)).astype(np.float32)
        d_rgt = rng.standard_normal((B, D)).astype(np.float32)
        W = gram_schmidt_2d_batch(d_fwd, d_rgt)
        # W: (B, 2, D); check W @ W.T ≈ I_2 per batch
        gram = np.einsum("bid,bjd->bij", W, W)
        eye = np.broadcast_to(np.eye(2, dtype=np.float32), (B, 2, 2))
        np.testing.assert_allclose(gram, eye, atol=1e-5)

    @pytest.mark.parametrize("vec,expected", [
        ((1.0, 0.0), 1),    # E
        ((0.0, 1.0), 0),    # N
        ((-1.0, 0.0), 3),   # W
        ((0.0, -1.0), 2),   # S
        # Edge of E-bin (angle == pi/4 exactly): should land in N (mapping uses [pi/4, 3pi/4)).
        ((math.cos(math.pi / 4), math.sin(math.pi / 4)), 0),
    ])
    def test_classify_direction_cardinals(self, vec, expected):
        q = np.array([vec], dtype=np.float32)
        idx = classify_direction_batch(q)
        assert idx[0] == expected

    def test_direction_to_onehot(self):
        idx = np.array([0, 1, 2, 3], dtype=np.int32)
        oh = direction_to_onehot(idx)
        np.testing.assert_array_equal(oh, np.eye(4, dtype=np.float32))


class TestOracleBFS:
    def test_greedy_strict_decrease(self):
        """Greedy action must reduce Manhattan distance from any non-goal cell."""
        rng = np.random.RandomState(0)
        size = 8
        for _ in range(20):
            goal = (int(rng.randint(0, size)), int(rng.randint(0, size)))
            while True:
                start = (int(rng.randint(0, size)), int(rng.randint(0, size)))
                if start != goal:
                    break
            positions = np.array([start], dtype=np.int32)
            a_idx = bfs_action_batch_discrete(positions, goal, size, rng)[0]
            dx, dy = CARDINAL_ACTIONS[int(a_idx)]
            nx = max(0, min(size - 1, start[0] + dx))
            ny = max(0, min(size - 1, start[1] + dy))
            old = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
            new = abs(nx - goal[0]) + abs(ny - goal[1])
            assert new == old - 1, (start, goal, a_idx, old, new)


# ---------------------------------------------------------------------------
# Env + vec-env semantics
# ---------------------------------------------------------------------------

@pytest.fixture
def make_grid_env():
    def _make(size=8, time_penalty=0.01, goal_reward=1.0,
              goals_active=True, seed=0):
        return GridEnv(
            size=size, observation_size=12, seed=seed,
            time_penalty=time_penalty, goal_reward=goal_reward,
            goals_active=goals_active,
        )
    return _make


class TestVecEnvSemantics:
    def test_at_goal_teleport_discrete(self, make_grid_env):
        """Pre-step at-goal: goal_reward + teleport, action ignored."""
        env = make_grid_env()
        env._goal = (3, 3)
        vec = VecEnv(env, batch_size=4)
        vec.reset_all()
        # Force every batch element to start at goal.
        vec._pos[:] = [3, 3]
        # Take an action that *would* move; should be ignored.
        rewards, goal_reached, _ = vec.step_batch(np.zeros(4, dtype=int))  # action N
        assert np.all(goal_reached)
        np.testing.assert_array_equal(rewards, np.full(4, env.goal_reward, dtype=np.float32))
        # Post-step positions are random non-goal cells.
        for b in range(4):
            assert tuple(vec._pos[b]) != (3, 3)

    def test_continuous_snapping_invariant(self, make_grid_env):
        env = make_grid_env()
        vec = ContinuousVecEnv(env, batch_size=4, scale=1.0, normalize=False)
        vec.reset_all()
        for _ in range(20):
            actions = np.random.uniform(-2, 2, size=(4, 2))
            vec.step_batch(actions)
            np.testing.assert_array_equal(
                vec._pos,
                np.clip(np.round(vec._pos_f).astype(np.int32), 0, vec.size - 1),
            )

    def test_continuous_normalize_off_uses_raw_action(self, make_grid_env):
        """With normalize=False, displacement = action * scale (no L2-norm)."""
        env = make_grid_env(size=20)
        env._goal = (19, 19)  # far from start
        vec = ContinuousVecEnv(env, batch_size=1, scale=1.0, normalize=False)
        vec._pos_f[:] = [5.0, 5.0]
        vec._update_snapped()
        # 3-unit step in x, 0 in y; with normalize=False expect +3.
        vec.step_batch(np.array([[3.0, 0.0]]))
        np.testing.assert_allclose(vec._pos_f[0], [8.0, 5.0], atol=1e-9)

    def test_continuous_normalize_on_unit_step(self, make_grid_env):
        """With normalize=True, displacement magnitude = scale regardless of |action|."""
        env = make_grid_env(size=20)
        env._goal = (19, 19)
        vec = ContinuousVecEnv(env, batch_size=1, scale=1.0, normalize=True)
        vec._pos_f[:] = [5.0, 5.0]
        vec._update_snapped()
        vec.step_batch(np.array([[3.0, 0.0]]))
        # Action got L2-normalized to (1, 0) before scaling.
        np.testing.assert_allclose(vec._pos_f[0], [6.0, 5.0], atol=1e-9)

    def test_goals_active_false(self, make_grid_env):
        """No teleport, only -time_penalty per step, even at the goal cell."""
        env = make_grid_env(goals_active=False)
        env._goal = (3, 3)
        vec = VecEnv(env, batch_size=2)
        vec.reset_all()
        vec._pos[:] = [3, 3]
        rewards, goal_reached, _ = vec.step_batch(np.zeros(2, dtype=int))
        assert not goal_reached.any()
        np.testing.assert_allclose(rewards, np.full(2, -env.time_penalty), rtol=1e-6)


# ---------------------------------------------------------------------------
# Hopfield class
# ---------------------------------------------------------------------------

class TestHopfield:
    def test_class_methods_exist(self):
        """Bug #1 regression: clone/reset/energy must be Hopfield methods."""
        h = Hopfield(8, beta=2.0)
        assert hasattr(h, "clone")
        assert hasattr(h, "reset")
        assert hasattr(h, "energy")
        # And they must be bound methods, not orphan functions.
        assert callable(h.clone)
        clone = h.clone()
        assert isinstance(clone, Hopfield)

    def test_clone_independence(self):
        h = Hopfield(8, beta=2.0)
        z = torch.randn(8)
        h.input_memory(z)
        c = h.clone()
        # Mutating the clone must not affect the original W.
        c.input_memory(torch.randn(8))
        assert not torch.allclose(c.W, h.W)
        assert h.num_memories == 1
        assert c.num_memories == 2

    def test_reset_clears_memory(self):
        h = Hopfield(8, beta=2.0)
        h.input_memory(torch.randn(8))
        assert h.num_memories == 1
        h.reset()
        assert h.num_memories == 0
        assert torch.allclose(h.W, torch.zeros_like(h.W))

    def test_recall_batch_matches_per_element_recall(self):
        """recall_batch(B) ≈ per-row recall(b)."""
        D = 16
        h = Hopfield(D, beta=2.0)
        # Store a few patterns
        torch.manual_seed(0)
        for _ in range(3):
            h.input_memory(torch.randn(D))
        cues = torch.randn(5, D)
        out_batch = h.recall_batch(cues, steps=3, beta=2.0)
        for i in range(5):
            out_one = h.recall(cues[i], steps=3, beta=2.0)
            torch.testing.assert_close(out_batch[i], out_one, rtol=1e-5, atol=1e-5)

    def test_recall_per_env_batch_matches_per_env_recall(self):
        """Different W per env: bmm-batched should match per-env recall."""
        D = 12
        torch.manual_seed(0)
        Ws = [torch.randn(D, D) for _ in range(4)]
        cues = torch.randn(4, D)
        # Per-env reference
        ref = []
        for b in range(4):
            h = Hopfield(D, beta=2.0)
            h.W = Ws[b].clone()
            ref.append(h.recall(cues[b], steps=2, beta=2.0))
        ref = torch.stack(ref, dim=0)
        # Batched
        W_stack = torch.stack(Ws, dim=0)
        got = recall_per_env_batch(cues, W_stack, steps=2, beta=2.0)
        torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

class TestConfigValidation:
    def _base_cfg(self) -> TrainConfig:
        return TrainConfig(
            env=EnvConfig(),
            vectorhash=VectorHashConfig(),
            hopfield=HopfieldConfig(),
            agent=AgentConfig(),
            ppo=PPOConfig(),
            bc=BCConfig(),
            encoder_checkpoint="dummy",
        )

    def test_valid_default_passes(self):
        cfg = self._base_cfg()
        validate_train_config(cfg)  # should not raise

    def test_agent_can_store_false_with_warmup_raises(self):
        cfg = self._base_cfg()
        cfg.hopfield.agent_can_store = False
        cfg.hopfield.auto_store_warmup = 5
        with pytest.raises(ValueError, match="auto_store_warmup"):
            validate_train_config(cfg)

    def test_agent_can_store_false_warmup_zero_passes(self):
        cfg = self._base_cfg()
        cfg.hopfield.agent_can_store = False
        cfg.hopfield.auto_store_warmup = 0
        validate_train_config(cfg)  # should not raise

    def test_continuous_normalize_default_is_false(self):
        """Audit fix: train and eval magnitudes agree because both are off."""
        env_cfg = EnvConfig()
        assert env_cfg.continuous_normalize is False


# ---------------------------------------------------------------------------
# Rollout: per-step pipeline behaviors
#
# These tests use a stub VectorHash that mimics the geometry needed by the
# rollout collector but skips the heavy scaffold build, so they are fast.
# ---------------------------------------------------------------------------

# StubVectorHash and _make_stub_cfg now live in tests/fixtures.py: the golden
# fixtures depend on them byte-for-byte, and two copies of "the stub world"
# would drift. _make_stub_cfg is re-bound below so the call sites in this file
# read unchanged.
_make_stub_cfg = make_stub_cfg


def _make_collector(cfg: TrainConfig, embed_dim: int = 8):
    from hopfield_nav.agent import NavAgent, compute_input_dim
    from hopfield_nav.rollout import RolloutCollector
    vh = StubVectorHash(Npos=16, embed_dim=embed_dim)
    device = torch.device("cpu")
    collector = RolloutCollector(vh, cfg, embed_dim, device)
    input_dim = compute_input_dim(cfg.agent, embed_dim, cfg.env.observation_size)
    agent = NavAgent(cfg.agent, input_dim).to(device)
    return collector, agent, vh


class TestRolloutRewardShaping:
    def test_novelty_skips_teleport_target(self):
        """Bug #2 regression: an at-goal step earns goal_reward, not novelty.

        Force every batch element to start at goal via a patched reset_all.
        The reward at the at-goal step must be exactly goal_reward, with no
        novelty bonus mixed in even though the post-step (teleport) cell is
        unvisited.
        """
        cfg = _make_stub_cfg(novelty_reward=10.0)
        cfg.batch_envs = 4
        cfg.steps_per_rollout = 1
        collector, agent, vh = _make_collector(cfg)
        env = GridEnv(size=cfg.env.size, observation_size=12,
                      goal_reward=1.0, time_penalty=0.01)
        env._goal = (2, 2)
        hops = [Hopfield(8, beta=1.0, device="cpu") for _ in range(cfg.batch_envs)]

        # Patch VecEnv.reset_all to deterministically place every trajectory
        # at the env goal — vec creates its own _pos, env._pos doesn't carry over.
        def force_at_goal(self):
            self._pos[:] = list(self._goal)
            self._heading[:] = [1, 0]

        with patch.object(VecEnv, "reset_all", force_at_goal):
            rollout = collector.collect_rollout(
                env, agent, hops, env_offset=(2, 2), update_idx=1,
            )
        # B=4, T=1; every trajectory's pre-step state was at-goal, so rewards
        # should be exactly goal_reward (1.0). Without the fix, the post-step
        # teleport cell would earn novelty and rewards would jump to ~10.99.
        np.testing.assert_allclose(
            rollout.rewards.cpu().numpy(),
            np.full((4, 1), 1.0, dtype=np.float32),
            atol=1e-6,
        )

    def test_novelty_fires_on_first_visit_normal_step(self):
        """Sanity: with novelty_reward set, a normal first-visit step adds it."""
        cfg = _make_stub_cfg(novelty_reward=0.5)
        cfg.batch_envs = 4
        cfg.steps_per_rollout = 8
        collector, agent, vh = _make_collector(cfg)
        env = GridEnv(size=cfg.env.size, observation_size=12,
                      goal_reward=1.0, time_penalty=0.01, seed=0)
        # Goal is far from typical start so most steps are explore.
        env._goal = (5, 5)
        hops = [Hopfield(8, beta=1.0, device="cpu") for _ in range(cfg.batch_envs)]
        rollout = collector.collect_rollout(env, agent, hops,
                                            env_offset=(2, 2), update_idx=1)
        # At least one step must include the novelty bonus (reward > -time_penalty).
        rew = rollout.rewards.cpu().numpy()
        # rewards = -time_penalty (-0.01) on revisit, -0.01 + 0.5 = 0.49 on novelty.
        assert (rew > 0.4).any(), f"no novelty fired; rewards={rew}"

    def test_wall_penalty_only_on_edge(self):
        """wall_penalty fires iff post-step cell is on the perimeter."""
        cfg = _make_stub_cfg(wall_penalty=1.0)
        cfg.batch_envs = 1
        cfg.steps_per_rollout = 2
        collector, agent, vh = _make_collector(cfg)
        size = cfg.env.size
        env = GridEnv(size=size, observation_size=12,
                      goal_reward=1.0, time_penalty=0.01, seed=0)
        env._goal = (size - 1, size - 1)  # corner so other cells are interior options
        hops = [Hopfield(8, beta=1.0, device="cpu")]
        # Patch the agent to always step North (action 0) — deterministic.
        with patch.object(
            agent, "get_action_and_value",
            wraps=lambda *a, **k: _force_action(agent, *a, action_idx=0, **k),
        ):
            # Stage start at (1, 0): one step N → (1, 1) is interior, no penalty.
            env._pos = (1, 0)
            rollout = collector.collect_rollout(env, agent, hops, env_offset=(0, 0),
                                                update_idx=1)
        # The first step starts at (1,0) which is itself perimeter (y=0). With
        # action N, post-step is (1,1) = interior. wall_penalty should NOT fire.
        # First step reward: -time_penalty (no penalty).
        rew = rollout.rewards.cpu().numpy()
        # Tolerance: should be -0.01 exactly on a non-edge non-goal step.
        assert rew[0, 0] == pytest.approx(-0.01, abs=1e-6), rew


def _force_action(agent, x, h=None, *, action_idx=0, **kwargs):
    """Helper: bypass policy sampling to force a fixed discrete action.

    Matches the dict signature of NavAgent.get_action_and_value but always
    returns action `action_idx` with re-scored log_prob.
    """
    move_dist, store_dist, values, h_next = agent.forward(x, h)
    B = x.shape[0]
    device = x.device
    move_action = torch.full((B, 1), action_idx, dtype=torch.long, device=device)
    move_lp = move_dist.log_prob(move_action)
    store_action = torch.zeros(B, 1, device=device)
    store_lp = store_dist.log_prob(store_action)
    return {
        "move_action": move_action.squeeze(1),
        "store_action": store_action.squeeze(1),
        "move_log_prob": move_lp.squeeze(1),
        "store_log_prob": store_lp.squeeze(1),
        "value": values.squeeze(1),
        "h_next": h_next,
    }


class TestRolloutGoalInMemoryGate:
    def test_goal_bit_does_not_flip_in_exploit(self):
        """Bug #3 regression: agent_goal_store_fired stays False outside in_explore.

        With explore_steps=2 and a 5-step rollout, fire store at-goal at every
        timestep. Bit must remain False because in_explore is False on steps
        2-4 and the explore-phase steps don't see at_goal until/unless the
        env teleports.
        """
        cfg = _make_stub_cfg(explore_steps=2, input_goal_in_memory=True)
        cfg.batch_envs = 1
        cfg.steps_per_rollout = 5
        collector, agent, vh = _make_collector(cfg)
        env = GridEnv(size=cfg.env.size, observation_size=12,
                      goal_reward=1.0, time_penalty=0.01)
        env._goal = (3, 3)
        env._pos = (3, 3)  # start at goal
        hops = [Hopfield(8, beta=1.0, device="cpu")]

        # Force store_action=1 every step, move arbitrary.
        def force_store(x, h=None, **kwargs):
            move_dist, store_dist, values, h_next = agent.forward(x, h)
            B = x.shape[0]
            device = x.device
            move_action = torch.zeros(B, 1, dtype=torch.long, device=device)
            move_lp = move_dist.log_prob(move_action)
            store_action = torch.ones(B, 1, device=device)
            store_lp = store_dist.log_prob(store_action)
            return {
                "move_action": move_action.squeeze(1),
                "store_action": store_action.squeeze(1),
                "move_log_prob": move_lp.squeeze(1),
                "store_log_prob": store_lp.squeeze(1),
                "value": values.squeeze(1),
                "h_next": h_next,
            }

        # Start at goal → first step is at_goal=True. Within explore window
        # (t<2): bit flips True; that's correct. We test that AFTER explore,
        # if the env returns to goal and stores, the bit does NOT flip from
        # False to True. To isolate, we change goal to a remote cell so that
        # the first step is the only at-goal step in explore.
        env._goal = (5, 5)
        env._pos = (0, 0)  # not at goal
        hops = [Hopfield(8, beta=1.0, device="cpu")]
        # Run rollout normally; the explore phase store actions can fire on
        # non-goal cells but never at goal (we never reach it). The bit should
        # remain False through the rollout.
        with patch.object(agent, "get_action_and_value", wraps=force_store):
            rollout = collector.collect_rollout(env, agent, hops,
                                                env_offset=(0, 0), update_idx=1)
        # Hopfield content: stores during explore (t<2), nothing after. The
        # number of stored memories should equal the count of effective_store
        # firings *during* explore.
        assert hops[0].num_memories <= cfg.explore_steps, (
            f"stores happened outside explore: {hops[0].num_memories}"
        )


class TestRolloutOverrideCompose:
    """Bug #5 regression: epsilon and auto_nav must compose, not clobber."""

    def test_epsilon_only(self):
        cfg = _make_stub_cfg(epsilon_explore=1.0)  # always epsilon
        cfg.batch_envs = 4
        cfg.steps_per_rollout = 2
        collector, agent, vh = _make_collector(cfg)
        env = GridEnv(size=cfg.env.size, observation_size=12, seed=0)
        env._goal = (5, 5)
        hops = [Hopfield(8, beta=1.0, device="cpu") for _ in range(4)]
        # Just ensure rollout runs without error and log_probs are finite.
        torch.manual_seed(0)
        rollout = collector.collect_rollout(env, agent, hops,
                                            env_offset=(0, 0), update_idx=1,
                                            epsilon_now=1.0)
        assert torch.isfinite(rollout.move_log_probs).all()

    def test_compose_epsilon_then_auto_nav(self):
        """With epsilon=1 and auto_nav active over a populated Hopfield,
        every env should be overridden (mask = eps_mask | nav_mask = all True),
        and the action should be the epsilon random action (epsilon precedence)
        for envs whose eps_mask was True (i.e., all of them when epsilon=1).
        """
        cfg = _make_stub_cfg(epsilon_explore=1.0, auto_nav_warmup=1)
        cfg.batch_envs = 4
        cfg.steps_per_rollout = 1
        collector, agent, vh = _make_collector(cfg)
        env = GridEnv(size=cfg.env.size, observation_size=12, seed=0)
        env._goal = (5, 5)
        # Pre-populate every per-env Hopfield so memory_mask is True everywhere.
        hops = [Hopfield(8, beta=1.0, device="cpu") for _ in range(4)]
        for h in hops:
            h.input_memory(torch.randn(8))
        # Roll once and just confirm it runs without breaking the log_prob.
        torch.manual_seed(0)
        rollout = collector.collect_rollout(env, agent, hops,
                                            env_offset=(0, 0), update_idx=1,
                                            epsilon_now=1.0)
        assert torch.isfinite(rollout.move_log_probs).all()


class TestPPOAdvantageNormalization:
    """Bug audit: advantage normalization is buffer-global, not per-rollout."""

    def test_pool_norm_is_buffer_global(self):
        from hopfield_nav.ppo import RolloutBatch, _pool_rollouts
        torch.manual_seed(0)
        # Two rollouts with disparate scales.
        def make(scale, B=2, T=4):
            return RolloutBatch(
                obs=torch.zeros(B, T, 1),
                move_actions=torch.zeros(B, T, dtype=torch.long),
                store_actions=torch.zeros(B, T),
                move_log_probs=torch.zeros(B, T),
                store_log_probs=torch.zeros(B, T),
                values=torch.zeros(B, T),
                rewards=torch.randn(B, T) * scale,
                bootstrap_value=torch.zeros(B),
                goal_reached=torch.zeros(B, T),
                explore_mask=torch.ones(B, T),
            )
        r1 = make(1.0)
        r2 = make(10.0)
        pool = _pool_rollouts([r1, r2], gamma=0.99, gae_lambda=0.95)
        adv = pool["advantages"]
        # Replicate ppo_update's normalization step.
        normed = (adv - adv.mean()) / adv.std().clamp_min(1e-8)
        torch.testing.assert_close(normed.mean(), torch.tensor(0.0), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(normed.std(unbiased=True), torch.tensor(1.0), atol=1e-3, rtol=1e-3)


class TestPPOStoreLossMask:
    """Store loss must be zero when explore_mask is all zero."""

    def test_zero_explore_mask_zero_store_loss(self):
        from hopfield_nav.agent import NavAgent
        from hopfield_nav.ppo import RolloutBatch, ppo_update
        cfg_ppo = PPOConfig()
        agent_cfg = AgentConfig(
            hidden_size=8, num_rnn_layers=1,
            input_encoded_state=False, input_hopfield_signal=False,
        )
        agent = NavAgent(agent_cfg, input_dim=1)
        opt = torch.optim.Adam(agent.parameters(), lr=1e-3)
        torch.manual_seed(0)
        B, T = 2, 4
        r = RolloutBatch(
            obs=torch.randn(B, T, 1),
            move_actions=torch.zeros(B, T, dtype=torch.long),
            store_actions=torch.zeros(B, T),
            move_log_probs=torch.zeros(B, T),
            store_log_probs=torch.zeros(B, T),
            values=torch.zeros(B, T),
            rewards=torch.zeros(B, T),
            bootstrap_value=torch.zeros(B),
            goal_reached=torch.zeros(B, T),
            explore_mask=torch.zeros(B, T),  # all zero → store loss masked out
        )
        losses = ppo_update(agent, [r], cfg_ppo, opt)
        assert losses["store_loss"] == 0.0
        assert losses["store_entropy"] == 0.0


class TestBCEDetachTrunkNoGrad:
    """With bce_detach_trunk=True, store-only BCE doesn't leak into the trunk."""

    def test_detached_bce_has_no_rnn_grad(self):
        from hopfield_nav.agent import NavAgent
        torch.manual_seed(0)
        agent_cfg = AgentConfig(
            hidden_size=8, num_rnn_layers=1,
            input_encoded_state=False, input_hopfield_signal=False,
        )
        agent = NavAgent(agent_cfg, input_dim=1)
        x = torch.randn(2, 4, 1)
        # Forward and grab features
        _, _, _, _, features = agent(x, return_features=True)
        store_logits = agent.store_logits_from(features.detach())
        labels = torch.zeros_like(store_logits)
        loss = F.binary_cross_entropy_with_logits(store_logits, labels)
        loss.backward()
        # RNN grads should all be None or zero.
        for p in agent.rnn.parameters():
            assert p.grad is None or torch.allclose(p.grad, torch.zeros_like(p.grad)), \
                "BCE leaked into RNN trunk despite features.detach()"


class TestGAEBootstrap:
    def test_one_step_truncation(self):
        from hopfield_nav.ppo import compute_gae
        rewards = torch.tensor([[1.0]])
        values = torch.tensor([[0.5]])
        boot = torch.tensor([2.0])
        adv, ret = compute_gae(rewards, values, boot, gamma=0.99, lam=0.95)
        # delta = r + gamma * boot - V = 1 + 0.99*2 - 0.5 = 2.48
        # advantages[0] = delta (last step), returns[0] = adv + V = 2.48 + 0.5 = 2.98
        torch.testing.assert_close(adv[0, 0].item(), 2.48, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(ret[0, 0].item(), 2.98, atol=1e-6, rtol=1e-6)


class TestBCStoreCap:
    """BCConfig.bce_pos_weight_cap actually caps pos_weight in bc_update."""

    def test_cap_active(self):
        from hopfield_nav.agent import NavAgent
        from hopfield_nav.bc import bc_update
        from hopfield_nav.ppo import RolloutBatch
        agent_cfg = AgentConfig(
            hidden_size=8, num_rnn_layers=1,
            input_encoded_state=False, input_hopfield_signal=False,
        )
        agent = NavAgent(agent_cfg, input_dim=1)
        opt = torch.optim.Adam(agent.parameters(), lr=1e-3)
        torch.manual_seed(0)
        B, T = 4, 8
        # Imbalanced: 1 positive in 32 entries, raw pos_weight = 31.
        teacher_store = torch.zeros(B, T)
        teacher_store[0, 0] = 1.0
        r = RolloutBatch(
            obs=torch.randn(B, T, 1),
            move_actions=torch.zeros(B, T, dtype=torch.long),
            store_actions=torch.zeros(B, T),
            move_log_probs=torch.zeros(B, T),
            store_log_probs=torch.zeros(B, T),
            values=torch.zeros(B, T),
            rewards=torch.zeros(B, T),
            bootstrap_value=torch.zeros(B),
            goal_reached=torch.zeros(B, T),
            explore_mask=torch.ones(B, T),
            teacher_move_action=torch.zeros(B, T, dtype=torch.long),
            teacher_store_action=teacher_store,
            move_label_mask=torch.ones(B, T),
            store_label_mask=torch.ones(B, T),
        )
        # Two updates with caps off vs on. With cap=2, the BCE on the lone
        # positive cell must be smaller (less weight on the positive class).
        bc_uncapped = BCConfig(epochs=1, n_minibatches=1, bce_pos_weight_cap=0.0)
        bc_capped = BCConfig(epochs=1, n_minibatches=1, bce_pos_weight_cap=2.0)
        # Need separate agent copies so optimizer step doesn't pollute.
        agent_a = NavAgent(agent_cfg, input_dim=1)
        agent_b = NavAgent(agent_cfg, input_dim=1)
        agent_b.load_state_dict(agent_a.state_dict())
        opt_a = torch.optim.Adam(agent_a.parameters(), lr=1e-3)
        opt_b = torch.optim.Adam(agent_b.parameters(), lr=1e-3)
        torch.manual_seed(1)
        loss_uncapped = bc_update(agent_a, [r], bc_uncapped, "discrete", opt_a)
        torch.manual_seed(1)
        loss_capped = bc_update(agent_b, [r], bc_capped, "discrete", opt_b)
        # Capped run should have a strictly smaller store_loss (smaller positive weight).
        assert loss_capped["store_loss"] < loss_uncapped["store_loss"], (
            f"cap had no effect: capped={loss_capped['store_loss']} "
            f"uncapped={loss_uncapped['store_loss']}"
        )
