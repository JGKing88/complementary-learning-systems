"""``memory_backend='sensory_kv'`` wired through the task training path.

The Hopfield default is pinned bit-for-bit elsewhere (``test_golden.py``,
``test_golden_task_run.py``). This covers the new branch:

1. ``validate`` refuses what the backend does not implement.
2. The ``memory_conf`` channel exists only under the backend, and last.
3. ``TaskRegime`` / ``evaluate_task`` build sensory stores with foreign goals.
4. A task-mode rollout: ``c = 0`` and ``d = 0`` before the store, one oracle
   write per trajectory at the goal, a unit ``d`` and ``c > 0`` after, and a
   second visit arriving with the memory intact.
5. Evaluators not yet taught the backend raise instead of misreading it.
6. The CLI runs a tiny task schedule end to end and records the backend.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from hopfield_nav.memory import backend as mb
from hopfield_nav.memory.grid_mlp import GridDirectionMLP
from hopfield_nav.memory.sensory_kv import SensoryKVMemory
from hopfield_nav.policy import channels
from hopfield_nav.tests.fixtures import make_collector, make_stub_cfg

REPO_ROOT = Path(__file__).resolve().parents[2]


def fake_grid_mlp(path: Path, lambdas, hidden: int = 32, layers: int = 2) -> Path:
    """A random grid MLP in the Phase-1 checkpoint format, sized for ``lambdas``."""
    ng = sum(l * l for l in lambdas)
    torch.manual_seed(0)
    m = GridDirectionMLP(2 * ng, hidden, layers)
    torch.save({"model_state_dict": m.state_dict(),
                "argv": {"movement_mode": "continuous", "nonlinearity": "relu"}}, path)
    return path


def kv_cfg(tmp_path, **kw):
    cfg = make_stub_cfg(movement_mode="continuous", input_hopfield_raw=True,
                        input_prev_reward=True, input_sensory=True,
                        novelty_reward=0.3, **kw)
    cfg.agent.memory_backend = "sensory_kv"
    cfg.agent.grid_mlp_checkpoint = str(fake_grid_mlp(
        tmp_path / "gmlp.pt", cfg.vectorhash.lambdas))
    cfg.env.distal_amp = 1.0
    cfg.env.goal_radius = 1.0
    cfg.env.observation_size = 60      # one ray per panorama slice (validate)
    return cfg


# 1 ---------------------------------------------------------------------------

def test_hopfield_is_the_default_and_passes_validation():
    cfg = make_stub_cfg()
    assert cfg.agent.memory_backend == "hopfield" and not mb.is_kv(cfg)
    mb.validate(cfg)


@pytest.mark.parametrize("breaks,needle", [
    (lambda c: setattr(c.env, "distal_amp", 0.0), "distal_amp"),
    (lambda c: setattr(c.agent, "input_hopfield_multistep", [1]), "multistep"),
    (lambda c: setattr(c.agent, "grid_mlp_checkpoint", None), "grid_mlp_checkpoint"),
    (lambda c: setattr(c.agent, "input_goal_in_memory", True), "goal_in_memory"),
    (lambda c: setattr(c.hopfield, "auto_nav_warmup", 5), "auto_nav"),
    (lambda c: setattr(c.env, "observation_size", 12), "observation_size 60"),
])
def test_validate_refuses_unsupported(tmp_path, breaks, needle):
    cfg = kv_cfg(tmp_path)
    mb.validate(cfg)
    breaks(cfg)
    with pytest.raises(ValueError, match=needle):
        mb.validate(cfg)


def test_unknown_backend_is_refused():
    cfg = make_stub_cfg()
    cfg.agent.memory_backend = "nope"
    with pytest.raises(ValueError):
        mb.is_kv(cfg)


# 2 ---------------------------------------------------------------------------

def test_memory_conf_channel_is_last_and_only_under_kv(tmp_path):
    hop = make_stub_cfg(movement_mode="continuous", input_sensory=True)
    kv = kv_cfg(tmp_path)
    h_specs = channels.channel_specs(hop.agent, 8, 60)
    k_specs = channels.channel_specs(kv.agent, 8, 60)
    assert "memory_conf" not in [s.name for s in h_specs]
    assert k_specs[-1] == channels.ChannelSpec("memory_conf", 1)
    kv.agent.memory_backend = "hopfield"
    assert channels.channel_specs(kv.agent, 8, 60) == k_specs[:-1]


# 3 ---------------------------------------------------------------------------

def test_task_regime_builds_sensory_stores_with_foreign_goals(tmp_path):
    from hopfield_nav.training.stages import Knobs
    from hopfield_nav.training.task import TaskRegime
    from hopfield_nav.world.env import make_env
    cfg = kv_cfg(tmp_path)
    env = make_env(cfg.env, "continuous", seed=3)
    reg = TaskRegime(cfg, 8, torch.device("cpu"), np.random.RandomState(0),
                     use_distractors=True, batch_size=6)
    knobs = Knobs(lr=1e-4, empty_frac=0.0, novelty=0.3, eps=0.0,
                  dist_min=2, dist_max=4, emp_dist_min=0, emp_dist_max=0)
    mems = reg.new_memory(None, env, (40, 50), knobs)
    assert len(mems) == 6 and all(isinstance(m, SensoryKVMemory) for m in mems)
    assert all(2 <= m.num_memories <= 4 for m in mems)
    ro = mb.readout_for(cfg)
    own = ro.goal_value(env._goal, (40, 50))
    for m in mems:
        for v in m.values:
            assert v.shape == (ro.grid_code.Ng,) and not np.array_equal(v, own)
        for k in m.keys:            # pool keys: other arenas' goals
            assert any(np.array_equal(k, pk) for pk in mb.foreign_pool_for(cfg).keys)


def test_offarena_cells_are_outside():
    cells = mb.sample_offarena_cells(30, (5, 5), 10, 500, np.random.RandomState(0))
    inside = (cells >= 5) & (cells < 15)
    assert not inside.all(axis=1).any()


# 4 ---------------------------------------------------------------------------

def _rollout(cfg, mems, fired=None, seed=0):
    from hopfield_nav.world.env import make_env
    collector, agent, _ = make_collector(cfg, seed=seed)
    env = make_env(cfg.env, "continuous", seed=5)
    env.set_goal((2, 3))
    torch.manual_seed(seed)
    np.random.seed(seed)
    r = collector.collect_rollout(env, agent, mems, allow_store=True,
                                  env_offset=(3, 4), task_mode=True,
                                  store_fired_init=fired)
    return r, agent, env


def test_task_rollout_under_kv(tmp_path):
    cfg = kv_cfg(tmp_path, batch_envs=8, steps_per_rollout=60)
    mems = [SensoryKVMemory() for _ in range(8)]
    r, agent, env = _rollout(cfg, mems)
    specs = channels.channel_specs(cfg.agent, 8, cfg.env.observation_size)
    names = [s.name for s in specs]
    off = np.cumsum([0] + [s.width for s in specs])
    sl = lambda n: slice(off[names.index(n)], off[names.index(n) + 1])
    obs = r.obs.cpu().numpy()
    assert obs.shape[-1] == off[-1]
    conf, d = obs[..., sl("memory_conf")][..., 0], obs[..., sl("hopfield_signal")]
    fired = r.store_fired_final
    assert fired.any(), "no trajectory reached the goal; lengthen the rollout"
    # One write per trajectory that touched the goal, none elsewhere.
    for b in range(8):
        assert mems[b].num_memories == int(fired[b])
    # Before the store: d = 0 and c = 0 exactly (empty memory). From the
    # store on: |d| = 1 every step. c is a cosine, so a stored goal can still
    # read c = 0 or below at a given step -- it is |d|, not c, that marks an
    # empty memory -- but on average it sits near the own-goal similarity.
    after_c = []
    for b in range(8):
        norm = np.linalg.norm(d[b], axis=-1)
        if not fired[b]:
            assert not conf[b].any() and not norm.any()
            continue
        if not norm.any():
            # Stored on the rollout's last step: no later observation shows it.
            assert not conf[b].any()
            continue
        t_store = int(np.flatnonzero(norm > 0)[0])
        assert not conf[b, :t_store].any()
        assert np.allclose(norm[t_store:], 1, atol=1e-4)
        after_c.append(conf[b, t_store:])
    assert np.concatenate(after_c).mean() > 0.35   # offline own-goal c ~ 0.5
    # Second visit: the carried memory is read from step 0.
    r2, *_ = _rollout(cfg, mems, fired=fired, seed=1)
    d2 = r2.obs.cpu().numpy()[:, 0, sl("hopfield_signal")]
    assert np.allclose(np.linalg.norm(d2[fired], axis=-1), 1, atol=1e-4)
    assert not d2[~fired].any()
    # Stored once, ever: a trajectory that found the goal on visit 1 is not
    # written again; one that finds it on visit 2 is written then.
    fired2 = r2.store_fired_final
    assert (fired2 >= fired).all()
    assert all(m.num_memories == int(f) for m, f in zip(mems, fired2))


def test_kv_refuses_non_task_rollouts(tmp_path):
    cfg = kv_cfg(tmp_path)
    collector, agent, _ = make_collector(cfg)
    from hopfield_nav.world.env import make_env
    env = make_env(cfg.env, "continuous", seed=5)
    with pytest.raises(NotImplementedError):
        collector.collect_rollout(env, agent, [SensoryKVMemory()] * cfg.batch_envs,
                                  allow_store=True, task_mode=False)


# 5 ---------------------------------------------------------------------------

@pytest.mark.parametrize("where", [
    "metrics.agent_step", "metrics.evaluate_navigation",
    "metrics.evaluate_goal_discovery", "metrics.evaluate_exploration",
    "metrics.evaluate_realistic", "metrics.evaluate_repeat",
    "metrics.evaluate_sequential_episodes", "protocols.run_mini_episode",
    "protocols.run_sequential_protocol", "batched.batched_navigation_trials",
    "batched.batched_exploration_trials",
])
def test_unsupported_evaluators_refuse_kv(tmp_path, where):
    """The guard is each function's first statement, so every other argument
    can be None: it must raise before touching any of them."""
    import importlib
    import inspect
    mod, fn = where.split(".")
    f = getattr(importlib.import_module(f"hopfield_nav.evaluation.{mod}"), fn)
    kwargs = {n: None for n, prm in inspect.signature(f).parameters.items()
              if prm.default is inspect.Parameter.empty}
    kwargs["cfg"] = kv_cfg(tmp_path)
    with pytest.raises(NotImplementedError, match=fn):
        out = f(**kwargs)
        if inspect.isgenerator(out):     # run_sequential_protocol yields
            next(out)


# 6 ---------------------------------------------------------------------------

@pytest.mark.slow
def test_cli_task_run_under_kv(tmp_path):
    import os
    root = tmp_path
    env = dict(os.environ, CLS_RUNS=str(root), WANDB_MODE="disabled",
               PYTHONPATH=str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    enc = root / "enc.pt"
    subprocess.run([sys.executable, "-m", "encoder_training.save_untrained_encoder",
                    "--encoder-type", "mlp", "--out-dim", "8", "--hidden-dim", "32",
                    "--num-hidden-layers", "2", "--gain", "5.0", "--lambdas", "3", "4",
                    "--out", str(enc)], cwd=REPO_ROOT, env=env, check=True,
                   capture_output=True)
    gmlp = fake_grid_mlp(root / "gmlp.pt", [3, 4])
    from hopfield_nav.tests.golden_task_run import TASK_ARGS
    args = [a for a in TASK_ARGS]
    i = args.index("--input_hopfield_multistep")
    del args[i:i + 2]
    save = root / "run"
    proc = subprocess.run(
        [sys.executable, "-m", "hopfield_nav.train_navigate",
         "--encoder_checkpoint", str(enc), *args, "--distal_amp", "1.0",
         "--observation_size", "60",
         "--memory_backend", "sensory_kv", "--grid_mlp_checkpoint", str(gmlp),
         "--save_dir", str(save)],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert "task={" in proc.stdout
    ck = torch.load(save / "navigate_final.pt", map_location="cpu", weights_only=False)
    assert ck["config"]["agent"]["memory_backend"] == "sensory_kv"
    assert ck["config"]["env"]["distal_amp"] == 1.0
    # current_reward, prev_reward, d (2), prev_action (2), prev_displacement
    # (2), sensory (60), memory_conf (1)
    w = ck["agent_state_dict"]["rnn.weight_ih_l0"].shape[1]
    assert w == 1 + 1 + 2 + 2 + 2 + 60 + 1


def test_cli_refuses_kv_outside_task(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "hopfield_nav.train_navigate",
         "--encoder_checkpoint", "x", "--schedule", "explore:2",
         "--memory_backend", "sensory_kv", "--grid_mlp_checkpoint", "x",
         "--distal_amp", "1", "--eval_scope", "task", "--device", "cpu"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
    assert proc.returncode != 0 and "task stages only" in proc.stderr
