"""``memory_backend='none'``: idea 3 of GRID_MLP_NAV_PLAN (§7).

A plain GRU with no memory module, fed the grid code at its cell. Its hidden
state is the only memory, so it is carried across the visits of a task
sequence, and PPO replays each sequence as one trajectory. Covered here:

1. ``validate`` refuses inputs that read a memory.
2. The ``grid_code`` channel is last and carries the grid code at the cell.
3. ``TaskRegime`` hands out no memories.
4. Carried state: a second visit starts from the first visit's final state,
   and re-forwarding the joined sequence from a zero state reproduces every
   collected log-prob and value (what PPO's ratio relies on). Re-forwarding
   the second visit alone does not.
5. The CLI trains a tiny task schedule end to end.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from hopfield_nav.memory import backend as mb
from hopfield_nav.policy import channels
from hopfield_nav.tests.fixtures import make_collector, make_stub_cfg

REPO_ROOT = Path(__file__).resolve().parents[2]


def none_cfg(**kw):
    cfg = make_stub_cfg(movement_mode="continuous", input_hopfield_signal=False,
                        input_prev_reward=True, input_prev_action=True,
                        input_sensory=True, novelty_reward=0.3, **kw)
    cfg.agent.memory_backend = "none"
    cfg.agent.input_grid_code = True
    cfg.agent.grid_code_dim = mb.grid_code_dim(cfg)
    cfg.env.distal_amp = 1.0
    cfg.env.goal_radius = 1.0
    return cfg


# 1 ---------------------------------------------------------------------------

@pytest.mark.parametrize("breaks,needle", [
    (lambda c: setattr(c.agent, "input_hopfield_signal", True), "hopfield_signal"),
    (lambda c: setattr(c.agent, "input_hopfield_multistep", [1]), "multistep"),
    (lambda c: setattr(c.agent, "input_goal_in_memory", True), "goal_in_memory"),
    (lambda c: setattr(c.hopfield, "auto_nav_warmup", 5), "auto_nav"),
    (lambda c: setattr(c.env, "reset_state_on_teleport", True), "reset_state"),
])
def test_validate_refuses_memory_inputs(breaks, needle):
    cfg = none_cfg()
    mb.validate(cfg)
    assert mb.is_none(cfg) and not mb.is_kv(cfg)
    breaks(cfg)
    with pytest.raises(ValueError, match=needle):
        mb.validate(cfg)


def test_require_hopfield_refuses_none():
    with pytest.raises(NotImplementedError, match="none"):
        mb.require_hopfield(none_cfg(), "evaluate_realistic")


# 2 ---------------------------------------------------------------------------

def test_grid_code_channel_is_last():
    cfg = none_cfg()
    specs = channels.channel_specs(cfg.agent, 8, cfg.env.observation_size)
    assert specs[-1] == channels.ChannelSpec("grid_code", mb.grid_code_dim(cfg))
    assert "hopfield_signal" not in [s.name for s in specs]
    cfg.agent.grid_code_dim = 0
    with pytest.raises(ValueError, match="grid_code_dim"):
        channels.channel_specs(cfg.agent, 8, cfg.env.observation_size)


# 3 ---------------------------------------------------------------------------

def test_task_regime_has_no_memory():
    from hopfield_nav.training.stages import Knobs
    from hopfield_nav.training.task import TaskRegime
    from hopfield_nav.world.env import make_env
    cfg = none_cfg()
    env = make_env(cfg.env, "continuous", seed=3)
    reg = TaskRegime(cfg, 8, torch.device("cpu"), np.random.RandomState(0),
                     use_distractors=True, batch_size=5)
    knobs = Knobs(lr=1e-4, empty_frac=0.0, novelty=0.3, eps=0.0,
                  dist_min=2, dist_max=4, emp_dist_min=0, emp_dist_max=0)
    assert reg.new_memory(None, env, (4, 5), knobs) == [None] * 5


# 4 ---------------------------------------------------------------------------

def _visit(collector, agent, env, cfg, h, fired, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    return collector.collect_rollout(
        env, agent, [None] * cfg.batch_envs, allow_store=True, h_rnn=h,
        env_offset=(3, 4), task_mode=True, store_fired_init=fired)


def test_state_carried_across_visits_replays_exactly():
    from hopfield_nav.training.task import merge_visits
    from hopfield_nav.world.env import make_env
    cfg = none_cfg(batch_envs=6, steps_per_rollout=40)
    collector, agent, _ = make_collector(cfg, seed=0)
    env = make_env(cfg.env, "continuous", seed=5)
    env.set_goal((2, 3))
    r1 = _visit(collector, agent, env, cfg, None, None, seed=0)
    r2 = _visit(collector, agent, env, cfg, r1.h_final, r1.store_fired_final, seed=1)

    specs = channels.channel_specs(cfg.agent, 8, cfg.env.observation_size)
    off = np.cumsum([0] + [s.width for s in specs])
    gc = slice(off[-2], off[-1])
    # The channel is the grid code at the cell, which teleports and moves.
    gcode = r1.obs[..., gc].numpy()
    assert gcode.shape[-1] == mb.grid_code_dim(cfg)
    assert np.allclose(gcode.max(-1), 1.0)   # one peak per module, peak 1
    assert len(np.unique(gcode.reshape(-1, gcode.shape[-1]), axis=0)) > 5

    merged = merge_visits([r1, r2])
    assert merged.obs.shape[1] == 2 * cfg.steps_per_rollout
    assert torch.equal(merged.bootstrap_value, r2.bootstrap_value)
    with torch.no_grad():
        dist, _, values, _ = agent(merged.obs)
        lp = dist.log_prob(merged.move_actions).sum(-1)
        assert torch.allclose(lp, merged.move_log_probs, atol=1e-4)
        assert torch.allclose(values, merged.values, atol=1e-5)
        # Visit 2 alone from a zero state is a different policy input history.
        d2, _, v2, _ = agent(r2.obs)
        assert not torch.allclose(v2, r2.values, atol=1e-3)
        d2c, _, v2c, _ = agent(r2.obs, r1.h_final)
        assert torch.allclose(v2c, r2.values, atol=1e-5)


def test_merge_refuses_ends_on_goal_rollouts():
    from hopfield_nav.training.task import merge_visits
    from hopfield_nav.world.env import make_env
    cfg = none_cfg(batch_envs=2, steps_per_rollout=6)
    collector, agent, _ = make_collector(cfg)
    env = make_env(cfg.env, "continuous", seed=5)
    r = _visit(collector, agent, env, cfg, None, None, seed=0)
    r.alive_mask = torch.ones_like(r.rewards)
    with pytest.raises(ValueError, match="ends_on_goal"):
        merge_visits([r, r])


# 5 ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("schedule", ["task:3,visits=2",
                                      "task:2,visits=2 ; lifetime:2,envs=2,iters=2"])
def test_cli_task_run_under_none(tmp_path, schedule):
    root = tmp_path
    env = dict(os.environ, CLS_RUNS=str(root), WANDB_MODE="disabled",
               PYTHONPATH=str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    enc = root / "enc.pt"
    subprocess.run([sys.executable, "-m", "encoder_training.save_untrained_encoder",
                    "--encoder-type", "mlp", "--out-dim", "8", "--hidden-dim", "32",
                    "--num-hidden-layers", "2", "--gain", "5.0", "--lambdas", "3", "4",
                    "--out", str(enc)], cwd=REPO_ROOT, env=env, check=True,
                   capture_output=True)
    from hopfield_nav.tests.golden_task_run import TASK_ARGS
    args = list(TASK_ARGS)
    i = args.index("--input_hopfield_multistep")
    del args[i:i + 2]
    args.remove("--input_hopfield_raw")
    args[args.index("--schedule") + 1] = schedule
    save = root / "run"
    proc = subprocess.run(
        [sys.executable, "-m", "hopfield_nav.train_navigate",
         "--encoder_checkpoint", str(enc), *args, "--distal_amp", "1.0",
         "--memory_backend", "none", "--input_grid_code",
         "--no-input_hopfield_signal", "--n_train_distractors_max", "0",
         "--print_every", "1", "--save_dir", str(save)],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert "task={0:" in proc.stdout
    assert "lifetime={'first_found'" in proc.stdout
    assert "] nav=" not in proc.stdout
    if "lifetime:" in schedule:
        assert "| lifetime: first_found=" in proc.stdout
    ck = torch.load(save / "navigate_final.pt", map_location="cpu", weights_only=False)
    assert ck["config"]["agent"]["memory_backend"] == "none"
    assert ck["config"]["agent"]["grid_code_dim"] == 3 * 3 + 4 * 4


# 6 (stage 2) -----------------------------------------------------------------

def test_protocol_order_is_the_continual_protocol():
    from hopfield_nav.rollout.lifetime import protocol_order
    assert protocol_order(3, 2) == [0, 0, 0, 1, 0, 1, 0, 1, 2, 0, 1, 2]
    assert protocol_order(1, 3) == [0, 0, 0]


def test_lifetime_packs_and_replays_exactly():
    from hopfield_nav.rollout import lifetime
    from hopfield_nav.world.env import make_env
    cfg = none_cfg(batch_envs=5, steps_per_rollout=30)
    collector, agent, _ = make_collector(cfg, seed=0)
    envs = [make_env(cfg.env, "continuous", seed=s) for s in (5, 6)]
    envs[0].set_goal((2, 3))
    envs[1].set_goal((1, 1))
    torch.manual_seed(0)
    np.random.seed(0)
    rs, order = lifetime.run_lifetime(collector, agent, envs, [(3, 4), (40, 9)],
                                      n_envs=2, iters=2, epsilon=0.0)
    assert order == [0, 0, 0, 1, 0, 1]
    # A finished row's steps are dead to the end of the trial: alive steps
    # are a prefix of each trial.
    alive = [r.alive_mask for r in rs]
    for a in alive:
        lens = a.sum(1)
        for b in range(5):
            n = int(lens[b])
            assert a[b, :n].all() and not a[b, n:].any()
    out = lifetime.trial_outcomes(rs, order)
    assert out["reached"].any(), "no row reached a goal; lengthen the trial"
    assert out["first"].tolist() == [True, False, False, True, False, False]
    assert not out["known"][:, 0].any()
    # known <=> reached that arena's goal on an earlier trial
    for b in range(5):
        for k in range(len(order)):
            prev = [kk for kk in range(k) if order[kk] == order[k]]
            assert out["known"][b, k] == any(out["reached"][b, kk] for kk in prev)

    packed = lifetime.pack(rs)
    assert packed.obs.shape[1] == int(max(sum(a[b].sum() for a in alive)
                                          for b in range(5)))
    assert torch.equal(packed.alive_mask.sum(1), sum(a.sum(1) for a in alive))
    with torch.no_grad():
        dist, _, values, _ = agent(packed.obs)
        lp = dist.log_prob(packed.move_actions).sum(-1)
    keep = packed.alive_mask > 0
    assert torch.allclose(lp[keep], packed.move_log_probs[keep], atol=1e-4)
    assert torch.allclose(values[keep], packed.values[keep], atol=1e-5)
    # The packed row is the trials' alive steps, in order, then zeros.
    b = 0
    want = torch.cat([r.obs[b][r.alive_mask[b] > 0] for r in rs])
    assert torch.equal(packed.obs[b, :len(want)], want)
    assert not packed.obs[b, len(want):].any()
    s = lifetime.summarise([out])
    assert set(s) == {"first_found", "known_reached", "known_steps",
                      "unknown_found", "known_frac"}


def test_freeze_done_state_needs_ends_on_goal():
    from hopfield_nav.world.env import make_env
    cfg = none_cfg(batch_envs=2, steps_per_rollout=6)
    collector, agent, _ = make_collector(cfg)
    env = make_env(cfg.env, "continuous", seed=5)
    with pytest.raises(ValueError, match="ends_on_goal"):
        collector.collect_rollout(env, agent, [None, None], allow_store=True,
                                  task_mode=True, freeze_done_state=True)


def test_continual_protocol_carries_state_without_memory():
    """agenthash's mini-episodes under 'none': no memory object, the oracle
    store only marks the goal found, and the state carries from one
    mini-episode into the next."""
    from analysis.continual.agenthash import mini_episode
    from hopfield_nav.evaluation import protocols
    from hopfield_nav.world.env import make_env
    cfg = none_cfg()
    _c, agent, vh = make_collector(cfg, 8, seed=0)
    env = make_env(cfg.env, "continuous", seed=100)
    env.set_goal((2, 2))
    common = dict(agent=agent, env=env, env_offset=(0, 0), vectorhash=vh,
                  hopfield=None, cfg=cfg, device=torch.device("cpu"),
                  local_idx=0, allow_store=True, max_steps=40,
                  deterministic=False, oracle_store_at_goal=True)

    def one(h_init, seed):
        torch.manual_seed(seed)
        return protocols.run_mini_episode(
            **common, rng=np.random.RandomState(seed), goal_in_mem={0: False},
            stored_at_goal_count={0: 0}, h_init=h_init)

    first = one(None, 0)
    assert first.h_final is not None
    carried, fresh = one(first.h_final, 1), one(None, 1)
    assert not torch.allclose(carried.h_final, fresh.h_final)

    carry = {"h": None}
    gim, cnt = {0: False}, {0: 0}
    rng = np.random.RandomState(3)
    for _ in range(3):
        mini_episode(**{k: v for k, v in common.items()},
                     rng=rng, goal_in_mem=gim, stored_at_goal_count=cnt,
                     oracle_lock_store_not_at_goal=True,
                     lock_store_after_goal=True, carry=carry)
    assert carry["h"].shape == (1, 1, cfg.agent.hidden_size)


@pytest.mark.parametrize("name", ["evaluate_navigation", "evaluate_goal_discovery",
                                  "evaluate_sequential_episodes"])
def test_memory_evaluators_refuse_none(name):
    import inspect
    from hopfield_nav.evaluation import metrics
    f = getattr(metrics, name)
    kwargs = {n: None for n, prm in inspect.signature(f).parameters.items()
              if prm.default is inspect.Parameter.empty}
    kwargs["cfg"] = none_cfg()
    with pytest.raises(NotImplementedError, match=name):
        f(**kwargs)


def test_last_reward_code_targets():
    from hopfield_nav.updates.ppo import last_reward_code
    obs = torch.arange(2 * 5 * 3, dtype=torch.float32).view(2, 5, 3)
    goal = torch.tensor([[0, 1, 0, 0, 1], [0, 0, 0, 0, 0]], dtype=torch.float32)
    tgt, mask = last_reward_code(obs, goal, ng=2)
    assert mask.tolist() == [[0, 1, 1, 1, 1], [0, 0, 0, 0, 0]]
    assert torch.equal(tgt[0, 1], obs[0, 1, -2:])
    assert torch.equal(tgt[0, 3], obs[0, 1, -2:])
    assert torch.equal(tgt[0, 4], obs[0, 4, -2:])


def test_aux_goal_head_trains_in_ppo():
    from hopfield_nav.training.task import merge_visits
    from hopfield_nav.updates.ppo import ppo_update
    from hopfield_nav.world.env import make_env
    cfg = none_cfg(batch_envs=6, steps_per_rollout=40)
    cfg.agent.aux_goal_weight = 0.1
    collector, agent, _ = make_collector(cfg, seed=0)
    assert agent.goal_code_head.out_features == mb.grid_code_dim(cfg)
    env = make_env(cfg.env, "continuous", seed=5)
    env.set_goal((2, 3))
    r1 = _visit(collector, agent, env, cfg, None, None, seed=0)
    r2 = _visit(collector, agent, env, cfg, r1.h_final, r1.store_fired_final, seed=1)
    assert r1.goal_reached.any(), "no goal reached; lengthen the rollout"
    before = agent.goal_code_head.weight.detach().clone()
    opt = torch.optim.Adam(agent.parameters(), lr=1e-3)
    agent.train()
    stats = ppo_update(agent, [merge_visits([r1, r2])], cfg.ppo, opt)
    assert np.isfinite(stats["aux_goal_loss"]) and stats["aux_goal_loss"] > 0
    assert not torch.equal(before, agent.goal_code_head.weight)


def test_lifetime_schedule_parses():
    from hopfield_nav.training.stages import (ScheduleError, format_schedule,
                                               parse_schedule)
    st = parse_schedule("lifetime:10,envs=3,iters=2 ; task:5,visits=2")
    assert (st[0].kind, st[0].envs, st[0].iters) == ("lifetime", 3, 2)
    assert format_schedule(st) == "lifetime:10,envs=3,iters=2 ; task:5,visits=2"
    with pytest.raises(ScheduleError, match="lifetime"):
        parse_schedule("task:5,envs=3")
