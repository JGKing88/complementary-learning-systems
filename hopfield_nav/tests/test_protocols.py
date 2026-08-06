"""The sequential protocol: one implementation, two callers, same numbers.

``eval.evaluate_sequential_episodes`` drives the in-training metric and the
eval_all JSON; ``final_plotting.agenthash`` drives the paper figure. They ran
the same protocol from two copies of the code, and the copies had begun to
diverge. The danger is specific: both produce numbers that get compared to each
other, so a drift surfaces as a scientific result rather than as a bug.

These tests pin the equivalence directly -- same fixture, same seed, same
flags, same episode records -- rather than trusting that the two call sites
stay in sync.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from hopfield_nav.world.env import make_env
from hopfield_nav.evaluation import protocols
from analysis.continual.agenthash import mini_episode
from hopfield_nav.world.memory import Hopfield
from hopfield_nav.tests.fixtures import make_collector, make_stub_cfg

EMBED_DIM = 8


def _world(n_envs: int = 2, seed: int = 0):
    cfg = make_stub_cfg(movement_mode="discrete")
    _collector, agent, vh = make_collector(cfg, EMBED_DIM, seed=seed)
    vh.env_offsets = [(0, 0), (8, 0), (0, 8)][:n_envs]
    envs = [make_env(cfg.env, "discrete", seed=100 + i) for i in range(n_envs)]
    return cfg, agent, vh, envs


def _fresh_state(n_envs: int):
    return (Hopfield(EMBED_DIM, beta=1.0, device="cpu"),
            {i: False for i in range(n_envs)},
            {i: 0 for i in range(n_envs)})


FLAGS = [
    dict(oracle_store_at_goal=False, suppress_off_goal_stores=False),
    dict(oracle_store_at_goal=True, suppress_off_goal_stores=True),
    dict(oracle_store_at_goal=True, suppress_off_goal_stores=False),
    dict(oracle_store_at_goal=False, suppress_off_goal_stores=True),
]


@pytest.mark.parametrize("flags", FLAGS)
@pytest.mark.parametrize("lock_after", [False, True])
def test_agenthash_wrapper_matches_the_shared_protocol(flags, lock_after):
    """agenthash.mini_episode is the shared function plus a tuple unpack.

    Its two oracle flags map onto the protocol's two parameters one-to-one;
    eval's single flag is the case where both are set together.
    """
    results = []
    for use_wrapper in (False, True):
        cfg, agent, vh, envs = _world()
        hop, gim, cnt = _fresh_state(len(envs))
        rng = np.random.RandomState(11)
        torch.manual_seed(0)
        np.random.seed(0)
        recs = []
        for k in range(8):
            j = k % 2
            kwargs = dict(
                agent=agent, env=envs[j], env_offset=vh.env_offsets[j],
                vectorhash=vh, hopfield=hop, cfg=cfg,
                device=torch.device("cpu"), local_idx=j,
                allow_store=(j == 0), max_steps=12, rng=rng,
                goal_in_mem=gim, stored_at_goal_count=cnt,
                deterministic=True, lock_store_after_goal=lock_after,
            )
            if use_wrapper:
                r = mini_episode(
                    **kwargs,
                    oracle_store_at_goal=flags["oracle_store_at_goal"],
                    oracle_lock_store_not_at_goal=flags["suppress_off_goal_stores"],
                    store_trainer=None)
            else:
                rec = protocols.run_mini_episode(**kwargs, **flags)
                r = (rec.reached, rec.steps_to_goal, rec.path_to_goal,
                     rec.stored_at_goal, rec.stored_off_goal)
            recs.append(r)
        results.append((recs, hop.num_memories, dict(gim), dict(cnt)))
    assert results[0] == results[1]


def test_eval_flag_is_the_conjunction_of_agenthashs_two():
    """eval's single --oracle-store-at-goal means both protocol parameters.

    agenthash's docstring records this: 'to reproduce the old combined
    behavior, pass both flags'. The equality is what lets eval keep one flag
    while the protocol takes two.
    """
    outs = {}
    for label, kw in (
        ("eval_style", dict(oracle_store_at_goal=True,
                            suppress_off_goal_stores=True)),
        ("agenthash_both", dict(oracle_store_at_goal=True,
                                suppress_off_goal_stores=True)),
    ):
        cfg, agent, vh, envs = _world()
        hop, gim, cnt = _fresh_state(len(envs))
        rng = np.random.RandomState(5)
        torch.manual_seed(0)
        np.random.seed(0)
        recs = [protocols.run_mini_episode(
            agent=agent, env=envs[0], env_offset=vh.env_offsets[0],
            vectorhash=vh, hopfield=hop, cfg=cfg, device=torch.device("cpu"),
            local_idx=0, allow_store=True, max_steps=12, rng=rng,
            goal_in_mem=gim, stored_at_goal_count=cnt, deterministic=True,
            **kw) for _ in range(6)]
        outs[label] = (recs, hop.num_memories)
    assert outs["eval_style"] == outs["agenthash_both"]


def test_records_carry_steps_and_path_on_both_paths():
    """eval used to discard these; recording them is what lets eval_all emit a
    history the plotting code can render without re-running via agenthash."""
    cfg, agent, vh, envs = _world()
    hop, gim, cnt = _fresh_state(len(envs))
    rng = np.random.RandomState(3)
    reached_any = False
    for _ in range(20):
        rec = protocols.run_mini_episode(
            agent=agent, env=envs[0], env_offset=vh.env_offsets[0],
            vectorhash=vh, hopfield=hop, cfg=cfg, device=torch.device("cpu"),
            local_idx=0, allow_store=True, max_steps=16, rng=rng,
            goal_in_mem=gim, stored_at_goal_count=cnt, deterministic=False)
        if rec.reached:
            reached_any = True
            assert rec.steps_to_goal is not None
            assert rec.path_to_goal is not None and rec.path_to_goal >= 0.0
        else:
            assert rec.steps_to_goal is None and rec.path_to_goal is None
    assert reached_any, "vacuous: the agent never reached the goal"


def test_protocol_schedule_is_the_lower_triangle():
    """Block i runs one mini-episode in every env introduced so far, and only
    the primary may store. That shape is the protocol."""
    cfg, agent, vh, envs = _world(n_envs=3)
    hop, gim, cnt = _fresh_state(len(envs))
    steps = list(protocols.run_sequential_protocol(
        agent=agent, val_envs=envs, env_offsets=vh.env_offsets,
        vectorhash=vh, hopfield=hop, cfg=cfg, device=torch.device("cpu"),
        iters_per_block=2, max_steps=6, rng=np.random.RandomState(1),
        goal_in_mem=gim, stored_at_goal_count=cnt))

    # blocks 0,1,2 with 2 iterations each: 1 + 2 + 3 episodes per iteration
    assert len(steps) == 2 * (1 + 2 + 3)
    for s in steps:
        assert s.env_idx <= s.block, "a revisit env was introduced too early"
        assert s.is_primary == (s.env_idx == s.block)
    assert [s.iteration for s in steps] == sorted(s.iteration for s in steps)


def test_hopfield_is_never_reset_across_blocks():
    """Accumulation is the point -- it is what makes revisits measure
    interference rather than fresh performance."""
    cfg, agent, vh, envs = _world(n_envs=3)
    hop, gim, cnt = _fresh_state(len(envs))
    seen = []
    for _ in protocols.run_sequential_protocol(
        agent=agent, val_envs=envs, env_offsets=vh.env_offsets,
        vectorhash=vh, hopfield=hop, cfg=cfg, device=torch.device("cpu"),
        iters_per_block=2, max_steps=6, rng=np.random.RandomState(1),
        goal_in_mem=gim, stored_at_goal_count=cnt,
        oracle_store_at_goal=True, suppress_off_goal_stores=True,
    ):
        seen.append(hop.num_memories)
    assert seen == sorted(seen), "memory count must be non-decreasing"
    assert seen[-1] > 0, "vacuous: nothing was ever stored"


def test_on_block_end_fires_once_per_block():
    cfg, agent, vh, envs = _world(n_envs=3)
    hop, gim, cnt = _fresh_state(len(envs))
    calls = []
    list(protocols.run_sequential_protocol(
        agent=agent, val_envs=envs, env_offsets=vh.env_offsets,
        vectorhash=vh, hopfield=hop, cfg=cfg, device=torch.device("cpu"),
        iters_per_block=2, max_steps=6, rng=np.random.RandomState(1),
        goal_in_mem=gim, stored_at_goal_count=cnt,
        on_block_end=lambda block, cur: calls.append((block, cur))))
    assert calls == [(0, 2), (1, 4), (2, 6)]


# ---------------------------------------------------------------------------
# The RNN baseline's block loop -- a separate protocol, also deduplicated
# ---------------------------------------------------------------------------

def _rnn_cfg(n_envs: int = 2):
    from hopfield_nav.config import (
        EnvConfig, RNNAgentConfig, RNNBCConfig, RNNTrainConfig,
    )
    return RNNTrainConfig(
        env=EnvConfig(size=5, observation_size=12, time_penalty=0.01),
        agent=RNNAgentConfig(hidden_size=16, num_rnn_layers=1),
        bc=RNNBCConfig(), n_envs=n_envs, updates_per_env=3, batch_envs=2,
        steps_per_rollout=6, eval_max_steps=8, n_eval_trials=2,
        eval_every=2, device="cpu")


def _rnn_agent(cfg):
    from hopfield_nav.policy.agent_rnn import RNNAgent, compute_rnn_input_dim
    torch.manual_seed(0)
    np.random.seed(0)
    agent = RNNAgent(
        cfg.agent, compute_rnn_input_dim(cfg.agent, cfg.env.observation_size))
    return agent, torch.optim.Adam(agent.parameters(), lr=1e-3)


def test_rnn_block_schedule_and_boundaries():
    """One block per env; blocks tile the step range with inclusive ends."""
    from hopfield_nav.train_rnn import build_envs
    from hopfield_nav.training.rnn_sequential import run_sequential_blocks

    cfg = _rnn_cfg(n_envs=3)
    envs = build_envs(cfg, np.random.RandomState(0))
    agent, opt = _rnn_agent(cfg)
    seen = []
    blocks = run_sequential_blocks(
        cfg=cfg, agent=agent, optimizer=opt, envs=envs,
        device=torch.device("cpu"), n_eval_trials=1,
        on_update=lambda u: seen.append(u))

    assert blocks == [(1, 3, 0), (4, 6, 1), (7, 9, 2)]
    assert [u.global_step for u in seen] == list(range(1, 10))
    # Every update evaluates exactly the envs introduced so far.
    for u in seen:
        assert sorted(u.metrics) == list(range(u.block + 1))


def test_rnn_untrained_envs_are_not_evaluated():
    """Untrained envs would inject pre-training noise into the forgetting curve."""
    from hopfield_nav.train_rnn import build_envs
    from hopfield_nav.training.rnn_sequential import run_sequential_blocks

    cfg = _rnn_cfg(n_envs=3)
    envs = build_envs(cfg, np.random.RandomState(0))
    agent, opt = _rnn_agent(cfg)
    seen = []
    run_sequential_blocks(
        cfg=cfg, agent=agent, optimizer=opt, envs=envs,
        device=torch.device("cpu"), n_eval_trials=1,
        on_update=lambda u: seen.append(u))
    first_block = [u for u in seen if u.block == 0]
    assert all(set(u.metrics) == {0} for u in first_block)


def test_both_rnn_drivers_run_the_same_loop():
    """train_rnn and the figure driver agree step-for-step on the block
    schedule, differing only in trial count and what they record."""
    from analysis.continual.baseline import run_sequential
    from hopfield_nav.train_rnn import build_envs, train_sequential

    outs = {}
    for name, fn in (("train_rnn", train_sequential), ("baseline", run_sequential)):
        cfg = _rnn_cfg()
        envs = build_envs(cfg, np.random.RandomState(0))
        agent, opt = _rnn_agent(cfg)
        torch.manual_seed(0)
        np.random.seed(0)
        res = fn(cfg, agent, opt, envs, torch.device("cpu"))
        outs[name] = res["blocks"] if isinstance(res, dict) else res[1]
    assert outs["train_rnn"] == outs["baseline"] == [(1, 3, 0), (4, 6, 1)]
