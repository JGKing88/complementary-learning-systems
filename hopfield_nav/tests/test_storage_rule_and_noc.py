"""Two knobs for GRID_MLP_NAV_PLAN §9.6, both off by default.

1. ``HopfieldConfig.storage_rule``: every Hopfield the trainers and evaluators
   build uses it -- "hebb" by default (the goldens pin that nothing moved),
   "proj" when asked -- and a proj memory stores its pattern as an exact fixed
   point.
2. ``AgentConfig.input_memory_conf``: off drops the sensory_kv backend's c
   channel; with ``scale_q_by_c`` the policy then sees only c * d.
Both end to end through the CLI.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from hopfield_nav.policy import channels
from hopfield_nav.tests.fixtures import StubVectorHash, make_stub_cfg
from hopfield_nav.tests.test_memory_backend import fake_grid_mlp, kv_cfg

REPO_ROOT = Path(__file__).resolve().parents[2]


def _task_regime_hops(rule):
    from hopfield_nav.training.stages import Knobs
    from hopfield_nav.training.task import TaskRegime
    from hopfield_nav.world.env import make_env
    cfg = make_stub_cfg(movement_mode="continuous")
    cfg.hopfield.storage_rule = rule
    vh = StubVectorHash(16, 8)
    reg = TaskRegime(cfg, 8, torch.device("cpu"), np.random.RandomState(0),
                     use_distractors=True, batch_size=4)
    knobs = Knobs(lr=1e-4, empty_frac=0.0, novelty=0.3, eps=0.0, dist_min=2,
                  dist_max=3, emp_dist_min=0, emp_dist_max=0)
    return reg.new_memory(vh, make_env(cfg.env, "continuous", seed=1), (0, 0), knobs), vh


def test_default_rule_is_hebb():
    hops, _ = _task_regime_hops("hebb")
    assert make_stub_cfg().hopfield.storage_rule == "hebb"
    assert all(h.storage_rule == "hebb" for h in hops)


def test_proj_reaches_the_task_regime_and_is_a_projector():
    hops, vh = _task_regime_hops("proj")
    assert all(h.storage_rule == "proj" for h in hops)
    h = hops[0]
    z = torch.from_numpy(vh.encoded_Phi[3, 4]).float()
    h.input_memory(z)
    zn = z / z.norm()
    # W is scale * projector onto span(stored): a stored pattern maps to a
    # multiple of itself.
    Wz = h.W @ zn
    cos = float(Wz @ zn / (Wz.norm() * zn.norm()))
    assert cos > 1 - 1e-5


def test_every_production_hopfield_takes_the_rule():
    """No construction site was missed: each passes storage_rule."""
    import re
    files = ["hopfield_nav/train_phased.py", "hopfield_nav/train.py",
             "hopfield_nav/evaluation/metrics.py", "hopfield_nav/training/explore.py",
             "hopfield_nav/training/exploit.py", "hopfield_nav/training/world_setup.py",
             "hopfield_nav/training/task.py", "analysis/continual/agenthash.py"]
    for f in files:
        src = (REPO_ROOT / f).read_text()
        calls = [m.start() for m in re.finditer(r"\bHopfield\(", src)]
        for c in calls:
            window = src[c:c + 300]
            assert "storage_rule=" in window, f"{f}: Hopfield( at {c} has no storage_rule"


def test_memory_conf_off_drops_the_channel(tmp_path):
    cfg = kv_cfg(tmp_path)
    on = channels.channel_specs(cfg.agent, 8, 60)
    cfg.agent.input_memory_conf = False
    off = channels.channel_specs(cfg.agent, 8, 60)
    assert on[-1].name == "memory_conf" and off == on[:-1]


def test_kv_refuses_a_storage_rule(tmp_path):
    from hopfield_nav.memory import backend as mb
    cfg = kv_cfg(tmp_path)
    cfg.hopfield.storage_rule = "proj"
    with pytest.raises(ValueError, match="storage_rule"):
        mb.validate(cfg)


def _cli(tmp_path, extra, kv=False):
    env = dict(os.environ, CLS_RUNS=str(tmp_path), WANDB_MODE="disabled",
               PYTHONPATH=str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    enc = tmp_path / "enc.pt"
    subprocess.run([sys.executable, "-m", "encoder_training.save_untrained_encoder",
                    "--encoder-type", "mlp", "--out-dim", "8", "--hidden-dim", "32",
                    "--num-hidden-layers", "2", "--gain", "5.0", "--lambdas", "3", "4",
                    "--out", str(enc)], cwd=REPO_ROOT, env=env, check=True,
                   capture_output=True)
    from hopfield_nav.tests.golden_task_run import TASK_ARGS
    args = list(TASK_ARGS)
    if kv:
        i = args.index("--input_hopfield_multistep")
        del args[i:i + 2]
        args += ["--distal_amp", "1.0", "--observation_size", "60",
                 "--memory_backend", "sensory_kv",
                 "--grid_mlp_checkpoint", str(fake_grid_mlp(tmp_path / "g.pt", [3, 4]))]
    proc = subprocess.run([sys.executable, "-m", "hopfield_nav.train_navigate",
                           "--encoder_checkpoint", str(enc), *args, *extra,
                           "--save_dir", str(tmp_path / "run")],
                          cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return torch.load(tmp_path / "run" / "navigate_final.pt", map_location="cpu",
                      weights_only=False)


@pytest.mark.slow
def test_cli_qmag_proj(tmp_path):
    ck = _cli(tmp_path, ["--no-input_hopfield_raw", "--input_q_magnitude",
                         "--hopfield_storage_rule", "proj"])
    assert ck["config"]["hopfield"]["storage_rule"] == "proj"


@pytest.mark.slow
def test_cli_gmlp_scq_noc(tmp_path):
    ck = _cli(tmp_path, ["--scale_q_by_c", "--no-input_memory_conf"], kv=True)
    a = ck["config"]["agent"]
    assert a["scale_q_by_c"] is True and a["input_memory_conf"] is False
    # current_reward, prev_reward, c*d (2), prev_action (2), prev_disp (2), sensory (60)
    assert ck["agent_state_dict"]["rnn.weight_ih_l0"].shape[1] == 1 + 1 + 2 + 2 + 2 + 60
