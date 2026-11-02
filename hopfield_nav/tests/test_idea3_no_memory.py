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
def test_cli_task_run_under_none(tmp_path):
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
    save = root / "run"
    proc = subprocess.run(
        [sys.executable, "-m", "hopfield_nav.train_navigate",
         "--encoder_checkpoint", str(enc), *args, "--distal_amp", "1.0",
         "--memory_backend", "none", "--input_grid_code",
         "--no-input_hopfield_signal", "--n_train_distractors_max", "0",
         "--save_dir", str(save)],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert "task={0:" in proc.stdout
    assert "] nav=" not in proc.stdout
    ck = torch.load(save / "navigate_final.pt", map_location="cpu", weights_only=False)
    assert ck["config"]["agent"]["memory_backend"] == "none"
    assert ck["config"]["agent"]["grid_code_dim"] == 3 * 3 + 4 * 4
