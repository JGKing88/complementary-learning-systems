"""``input_q_magnitude``: ||q|| as its own channel (GRID_MLP_NAV_PLAN §9.3).

With ``--no-input_hopfield_raw`` the q slot is unit and ||q|| arrives apart --
the Hopfield analog of the sensory_kv backend's (d, c). Off by default, so the
goldens pin that nothing else moved; this pins the new channel:

1. The channel exists only with the flag, and last.
2. In a rollout its value is exactly ||q|| (checked against the raw q slot),
   and 0 wherever memory is empty.
3. Every assembly site supplies it: a tiny CLI run under eval scope ``all``
   (batched nav + expl, agent_step disc) and ``task`` (evaluate_task).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from hopfield import Hopfield
from hopfield_nav.policy import channels
from hopfield_nav.rollout.signal import q_magnitude
from hopfield_nav.tests.fixtures import make_collector, make_stub_cfg

REPO_ROOT = Path(__file__).resolve().parents[2]


def _cfg(raw: bool):
    cfg = make_stub_cfg(movement_mode="continuous", input_hopfield_raw=raw,
                        input_prev_reward=True, batch_envs=6, steps_per_rollout=40,
                        novelty_reward=0.3)
    cfg.agent.input_q_magnitude = True
    return cfg


def test_channel_only_with_the_flag_and_last():
    cfg = _cfg(raw=False)
    specs = channels.channel_specs(cfg.agent, 8, 12)
    assert specs[-1] == channels.ChannelSpec("q_magnitude", 1)
    cfg.agent.input_q_magnitude = False
    assert channels.channel_specs(cfg.agent, 8, 12) == specs[:-1]


def test_q_magnitude_helper():
    q = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    assert np.allclose(q_magnitude(q), [[5.0], [0.0]])


def _rollout(cfg, populated: bool):
    from hopfield_nav.world.env import make_env
    collector, agent, vh = make_collector(cfg)
    env = make_env(cfg.env, "continuous", seed=5)
    hops = []
    for b in range(cfg.batch_envs):
        h = Hopfield(8, beta=cfg.hopfield.beta, device="cpu")
        if populated and b % 2 == 0:       # half the rows hold a memory
            h.input_memory(torch.from_numpy(vh.encoded_Phi[2 + b, 3]).float())
        hops.append(h)
    torch.manual_seed(0)
    r = collector.collect_rollout(env, agent, hops, allow_store=False)
    specs = channels.channel_specs(cfg.agent, 8, cfg.env.observation_size)
    off = np.cumsum([0] + [s.width for s in specs])
    names = [s.name for s in specs]
    sl = lambda n: slice(off[names.index(n)], off[names.index(n) + 1])
    obs = r.obs.numpy()
    return obs[..., sl("hopfield_signal")], obs[..., sl("q_magnitude")][..., 0]


def test_value_is_norm_of_q_and_zero_when_empty():
    # Raw q in the slot: the new channel must equal its norm exactly.
    q, mag = _rollout(_cfg(raw=True), populated=True)
    assert np.allclose(mag, np.linalg.norm(q, axis=-1), atol=1e-6)
    assert not mag[1::2].any()                 # empty rows: 0
    assert (mag[0::2] > 0).any()
    # Unit q in the slot: |slot| is 1 where memory exists, 0 where not; the
    # magnitude rides apart.
    u, mag_u = _rollout(_cfg(raw=False), populated=True)
    n = np.linalg.norm(u, axis=-1)
    assert np.allclose(n[0::2][mag_u[0::2] > 0], 1.0, atol=1e-5)
    assert not n[1::2].any() and not mag_u[1::2].any()
    # Same starts and memories, so step 0 sees the same ||q|| either way;
    # after that the two policies (raw vs unit input) walk differently.
    assert np.allclose(mag_u[:, 0], mag[:, 0], atol=1e-6)


@pytest.mark.slow
@pytest.mark.parametrize("scope,schedule", [("all", "interleave:2"),
                                            ("task", "task:2,visits=2")])
def test_cli_every_assembly_site_supplies_it(tmp_path, scope, schedule):
    env = dict(os.environ, CLS_RUNS=str(tmp_path), WANDB_MODE="disabled",
               PYTHONPATH=str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    enc = tmp_path / "enc.pt"
    subprocess.run([sys.executable, "-m", "encoder_training.save_untrained_encoder",
                    "--encoder-type", "mlp", "--out-dim", "8", "--hidden-dim", "32",
                    "--num-hidden-layers", "2", "--gain", "5.0", "--lambdas", "3", "4",
                    "--out", str(enc)], cwd=REPO_ROOT, env=env, check=True,
                   capture_output=True)
    args = ["--encoder_checkpoint", str(enc), "--lambdas", "3", "4", "--Np", "40",
            "--size", "4", "--observation_size", "16", "--batch_envs", "2",
            "--steps_per_rollout", "8", "--schedule", schedule,
            "--envs_per_world", "2", "--num_worlds", "1", "--num_val_envs", "1",
            "--n_val_trials", "2", "--eval_every", "1", "--eval_scope", scope,
            "--movement_mode", "continuous", "--no-input_hopfield_raw",
            "--input_q_magnitude", "--device", "cpu", "--static-vectorhash",
            "--save_dir", str(tmp_path / "run")]
    if schedule.startswith("task"):
        args += ["--env_repeats", "2"]
    proc = subprocess.run([sys.executable, "-m", "hopfield_nav.train_navigate", *args],
                          cwd=REPO_ROOT, env=env, capture_output=True, text=True,
                          timeout=900)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    ck = torch.load(tmp_path / "run" / "navigate_final.pt", map_location="cpu",
                    weights_only=False)
    assert ck["config"]["agent"]["input_q_magnitude"] is True
    assert ck["config"]["agent"]["input_hopfield_raw"] is False
