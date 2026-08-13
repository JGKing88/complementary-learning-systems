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
from hopfield import Hopfield
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
    from hopfield_nav.training.rnn_setup import build_envs_from_config
    from hopfield_nav.training.rnn_sequential import run_sequential_blocks

    cfg = _rnn_cfg(n_envs=3)
    envs = build_envs_from_config(cfg, np.random.RandomState(0))
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
    from hopfield_nav.training.rnn_setup import build_envs_from_config
    from hopfield_nav.training.rnn_sequential import run_sequential_blocks

    cfg = _rnn_cfg(n_envs=3)
    envs = build_envs_from_config(cfg, np.random.RandomState(0))
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
    from hopfield_nav.train_rnn import train_sequential
    from hopfield_nav.training.rnn_setup import build_envs_from_config

    outs = {}
    for name, fn in (("train_rnn", train_sequential), ("baseline", run_sequential)):
        cfg = _rnn_cfg()
        envs = build_envs_from_config(cfg, np.random.RandomState(0))
        agent, opt = _rnn_agent(cfg)
        torch.manual_seed(0)
        np.random.seed(0)
        res = fn(cfg, agent, opt, envs, torch.device("cpu"))
        outs[name] = res["blocks"] if isinstance(res, dict) else res[1]
    assert outs["train_rnn"] == outs["baseline"] == [(1, 3, 0), (4, 6, 1)]


# ---------------------------------------------------------------------------
# The RNN stack on the same world record as train_navigate (Phase 5.5)
# ---------------------------------------------------------------------------

def _rnn_gen_cfg(**over):
    cfg = _rnn_cfg(n_envs=3)
    cfg.lambdas = [5, 7]                 # Npos = 35
    cfg.env.size = 4
    cfg.agent.input_grid_state = True
    cfg.env_generator = True
    cfg.place_margin = 3
    cfg.n_val_envs = 2
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def test_both_rnn_drivers_build_the_same_world():
    """The point of moving this stack onto the generator.

    Before, `analysis.continual.baseline` and `train_rnn` agreed with
    `agenthash` only through the hand-matched draw-order convention documented
    at agenthash.py:325-333 -- a comment, enforced by nothing. Now they call one
    function and can be pointed at one record, so agreement is a property rather
    than a convention.
    """
    from hopfield_nav.training.rnn_setup import rnn_world

    made = []
    for _ in range(2):
        cfg = _rnn_gen_cfg()
        envs, offsets, split, field, kind = rnn_world(cfg, np.random.RandomState(cfg.seed))
        made.append((kind, [e.seed for e in envs], offsets,
                     [s.goal for s in split.train]))
    assert made[0] == made[1]
    assert made[0][0] == "declared"


def test_the_generated_rnn_world_records_its_offsets():
    """The §1.4 bug, in this stack: `place_envs(..., np.random, ...)` drew from
    a global stream whose state depended on everything built before it, so the
    offsets a baseline used were unrecoverable afterwards."""
    from hopfield_nav.training.rnn_setup import rnn_world

    cfg = _rnn_gen_cfg()
    envs, offsets, split, field, _ = rnn_world(cfg, np.random.RandomState(cfg.seed))
    assert offsets == [s.offset for s in split.train]
    assert len(split.base_val) == cfg.n_val_envs
    assert split.margin == cfg.place_margin
    from hopfield_nav.world import generate as gen
    for i, a in enumerate(split.train + split.base_val):
        for b in (split.train + split.base_val)[i + 1:]:
            assert gen.toroidal_gap(a.offset, a.size, b.offset, b.size,
                                    split.period) >= split.margin


def test_the_legacy_rnn_path_is_still_describable():
    """Off by default, and an unconstrained draw still gets recorded -- which is
    the first time this stack could say what envs it used at all."""
    from hopfield_nav.training.rnn_setup import rnn_world

    cfg = _rnn_gen_cfg(env_generator=False)
    envs, offsets, split, field, kind = rnn_world(cfg, np.random.RandomState(cfg.seed))
    assert kind == "legacy" and split.margin == 0
    assert [s.wall_seed for s in split.train] == [e.seed for e in envs]
    assert [s.goal for s in split.train] == [e.goal_location for e in envs]
    assert offsets == [s.offset for s in split.train]


def test_a_generated_rnn_world_round_trips_through_disk(tmp_path):
    """Same file and same reader as train_navigate's, which is what makes a
    baseline and an agent-hash run comparable."""
    from hopfield_nav.training.rnn_setup import rnn_world, write_rnn_world_spec
    from hopfield_nav.world import generate as gen
    from hopfield_nav.world.spec import WorldSpec

    cfg = _rnn_gen_cfg()
    envs, offsets, split, field, kind = rnn_world(cfg, np.random.RandomState(cfg.seed))
    write_rnn_world_spec(cfg, split, field, generator=kind, save_dir=tmp_path)

    back = WorldSpec.read(tmp_path)
    assert back.split.train == split.train
    rebuilt = gen.build_envs(back.split.train, cfg.env, "discrete")
    for a, b in zip(envs, rebuilt):
        assert np.array_equal(a._wall_code, b._wall_code)
        assert a.goal_location == b.goal_location
    # No encoder in this stack, so the embedding diagnostic is empty rather than
    # invented -- and the geometric ones are still there.
    assert back.split.diagnostics["cosine"] == {}
    assert back.split.diagnostics["min_place_gap"] >= split.margin


def test_the_scaffold_is_built_for_the_generator_not_only_for_the_agent():
    """Placement is recorded even when the agent cannot observe it.

    Under `input_grid_state=False` the RNN never sees where its envs sit -- but
    the placement is still part of the world's identity, and an agent-hash run
    pointed at the same world.json *does* observe those offsets. Refusing to
    generate here would mean the two stacks could not share a world at all.
    """
    from hopfield_nav.training.rnn_setup import rnn_world
    from hopfield_nav.world import generate as gen

    cfg = _rnn_gen_cfg()
    cfg.agent.input_grid_state = False
    envs, offsets, split, field, kind = rnn_world(cfg, np.random.RandomState(0))
    assert kind == "declared"
    assert field is not None, "the generator needs a coordinate system"
    assert offsets == [s.offset for s in split.train]
    for i, a in enumerate(split.train + split.base_val):
        for b in (split.train + split.base_val)[i + 1:]:
            assert gen.toroidal_gap(a.offset, a.size, b.offset, b.size,
                                    split.period) >= split.margin


def test_grid_state_does_not_change_which_envs_the_generator_draws():
    """The declared world is a property of the config, not of what the agent
    happens to observe -- otherwise two stacks could not be given the same one."""
    from hopfield_nav.training.rnn_setup import rnn_world

    made = []
    for grid in (True, False):
        cfg = _rnn_gen_cfg()
        cfg.agent.input_grid_state = grid
        envs, offsets, split, _, _ = rnn_world(cfg, np.random.RandomState(cfg.seed))
        made.append(([e.seed for e in envs], offsets,
                     [s.goal for s in split.train]))
    assert made[0] == made[1]


def test_the_generator_will_not_invent_a_margin():
    """`derive_margin` reads the scaffold's cosine-vs-distance curve, which
    needs an encoder. This stack has none, and a borrowed constant would be
    wrong at a different Npos."""
    from hopfield_nav.training.rnn_setup import rnn_world

    cfg = _rnn_gen_cfg(place_margin=None)
    with pytest.raises(SystemExit, match="explicit --place_margin"):
        rnn_world(cfg, np.random.RandomState(0))
