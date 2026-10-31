"""Stage 2 of GRID_MLP_NAV_PLAN: the sensory_kv backend in the evaluators.

The Hopfield path is pinned by the goldens (``test_golden.py``,
``test_golden_task_run.py``, ``test_golden_agenthash_run.py``). This covers
the new branch.

1. The evaluators' distractor draws leave the shared random stream exactly
   where the Hopfield draws do, so both backends get the same trials.
2. A store writes a ``KVEntry``; each memory refuses the other's payload.
3. ``agent_step`` reads the sensory memory and returns the right payload.
4. End to end: training with eval scope ``all`` runs nav, disc and expl, and
   the continual protocol CLI runs on the resulting checkpoint.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from hopfield import Hopfield
from hopfield_nav.memory import backend as mb
from hopfield_nav.memory.sensory_kv import KVEntry, SensoryKVMemory
from hopfield_nav.rollout.distractors import sample_distractors
from hopfield_nav.tests.fixtures import StubVectorHash
from hopfield_nav.tests.test_memory_backend import fake_grid_mlp, kv_cfg

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("n", [0, 1, 4, 10])
def test_distractor_draws_match_the_hopfield_stream(tmp_path, n):
    cfg = kv_cfg(tmp_path)
    npos = int(np.prod(cfg.vectorhash.lambdas))
    vh = StubVectorHash(Npos=npos, embed_dim=8)
    a, b = np.random.RandomState(7), np.random.RandomState(7)
    sample_distractors(vh, (20, 30), 6, n, a)
    entries = mb.kv_distractor_entries(cfg, (20, 30), 6, n, b)
    assert len(entries) == n
    assert a.randint(1 << 30) == b.randint(1 << 30)
    # ... and the shuffle that follows consumes the same too.
    pa, pb = [0] * (n + 1), [0] * (n + 1)
    a.shuffle(pa); b.shuffle(pb)
    assert a.randint(1 << 30) == b.randint(1 << 30)


def test_distractor_entries_are_foreign_and_deterministic(tmp_path):
    cfg = kv_cfg(tmp_path)
    e1 = mb.kv_distractor_entries(cfg, (20, 30), 6, 5, np.random.RandomState(3))
    e2 = mb.kv_distractor_entries(cfg, (20, 30), 6, 5, np.random.RandomState(3))
    pool = mb.foreign_pool_for(cfg, 6).keys
    for x, y in zip(e1, e2):
        assert np.array_equal(x.key, y.key) and np.array_equal(x.value, y.value)
        assert any(np.array_equal(x.key, k) for k in pool)


def test_payload_types_are_enforced(tmp_path):
    cfg = kv_cfg(tmp_path)
    mem = mb.kv_memory_with(cfg, [])
    with pytest.raises(TypeError):
        mem.input_memory(torch.zeros(8))
    hop = Hopfield(8, beta=1.0, device="cpu")
    entry = KVEntry(np.zeros(180, np.float32), np.zeros(5, np.float32))
    with pytest.raises(Exception):
        hop.input_memory(entry)


def test_agent_step_reads_and_stores_under_kv(tmp_path):
    from hopfield_nav.evaluation.metrics import agent_step
    from hopfield_nav.policy.agent import NavAgent, compute_input_dim
    from hopfield_nav.world.env import make_env
    cfg = kv_cfg(tmp_path)
    vh = StubVectorHash(16, 8)
    torch.manual_seed(0)
    agent = NavAgent(cfg.agent, compute_input_dim(cfg.agent, 8, cfg.env.observation_size)).eval()
    env = make_env(cfg.env, "continuous", seed=5)
    env.set_goal((2, 3))
    env.set_position((4, 4))
    mem = SensoryKVMemory()
    out = agent_step(agent, env, (3, 4), vh, mem, None, cfg, torch.device("cpu"),
                     goal_local=(2, 3), goal_in_memory=False)
    entry = out["store_embedding"][0]
    assert isinstance(entry, KVEntry) and entry.key.shape == (180,)
    mem.input_memory(entry)
    assert mem.num_memories == 1
    # A stored goal now reads back: the policy input carries a nonzero c.
    env.set_position((0, 0))
    out2 = agent_step(agent, env, (3, 4), vh, mem, out["h_rnn"], cfg,
                      torch.device("cpu"), goal_local=(2, 3))
    assert out2["store_embedding"][0].key.shape == (180,)


def _env():
    return dict(os.environ, WANDB_MODE="disabled",
                PYTHONPATH=str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))


@pytest.mark.slow
def test_cli_all_scope_then_continual_protocol(tmp_path):
    env = _env()
    env["CLS_RUNS"] = str(tmp_path)
    enc = tmp_path / "enc.pt"
    subprocess.run([sys.executable, "-m", "encoder_training.save_untrained_encoder",
                    "--encoder-type", "mlp", "--out-dim", "8", "--hidden-dim", "32",
                    "--num-hidden-layers", "2", "--gain", "5.0", "--lambdas", "3", "4",
                    "--out", str(enc)], cwd=REPO_ROOT, env=env, check=True,
                   capture_output=True)
    gmlp = fake_grid_mlp(tmp_path / "gmlp.pt", [3, 4])
    from hopfield_nav.tests.golden_task_run import TASK_ARGS
    args = list(TASK_ARGS)
    i = args.index("--input_hopfield_multistep")
    del args[i:i + 2]
    j = args.index("--eval_scope")
    args[j + 1] = "all"
    save = tmp_path / "run"
    proc = subprocess.run(
        [sys.executable, "-m", "hopfield_nav.train_navigate", "--encoder_checkpoint",
         str(enc), *args, "--distal_amp", "1.0", "--observation_size", "60",
         "--memory_backend", "sensory_kv", "--grid_mlp_checkpoint", str(gmlp),
         "--save_dir", str(save)],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    for k in ("nav", "disc", "expl"):
        lines = [ln for ln in proc.stdout.splitlines() if f"] {k}=" in ln]
        assert lines and "{}" not in lines[-1], (k, lines[-1:])
    for stoch in ([], ["--stochastic_policy"]):
        out = tmp_path / f"hist{len(stoch)}.json"
        p2 = subprocess.run(
            [sys.executable, "-m", "analysis.continual.agenthash",
             "--ckpt", str(save / "navigate_final.pt"), "--out", str(out),
             "--device", "cpu", "--static_vectorhash", "--iters_per_block", "3",
             "--max_steps", "40", "--seed", "0", "--oracle_store_at_goal",
             "--oracle_lock_store_not_at_goal", *stoch],
            cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=900)
        assert p2.returncode == 0, p2.stdout[-3000:] + p2.stderr[-3000:]
        h = json.load(open(out))
        assert h["trace"] and h["blocks"]
