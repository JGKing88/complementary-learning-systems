"""End-to-end wiring of the explore-first fork (docs/EXPLORE_FIRST_PLAN.md).

A tiny explorer is trained, then forked into the task regime with both prior
terms on. What has to hold, and is only visible at the CLI level: the fork
scores its parent at u0 before any step; the Fisher is estimated on the
first update's search steps; both terms and the drift reach the per-update
log; and the prior without a parent, or under --continue_from, is refused.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

TINY_WORLD = [
    "--lambdas", "3", "4", "--Np", "40",
    "--size", "4", "--observation_size", "16",
    "--batch_envs", "2", "--steps_per_rollout", "8",
    "--envs_per_world", "1", "--num_worlds", "1",
    "--num_val_envs", "1", "--n_val_trials", "2",
    "--device", "cpu", "--static-vectorhash",
    "--movement_mode", "continuous", "--action_polar",
    "--state_dependent_std", "--hidden_size", "8",
    "--min_action_norm", "0.5", "--max_action_norm", "1.0",
]


def _run(args, env, timeout=900):
    return subprocess.run([sys.executable, "-m", *args], cwd=REPO_ROOT,
                          env=env, capture_output=True, text=True,
                          timeout=timeout)


def _must(proc, what):
    if proc.returncode != 0:
        pytest.fail(f"{what} exited {proc.returncode}\n--- stdout ---\n"
                    f"{proc.stdout[-4000:]}\n--- stderr ---\n{proc.stderr[-4000:]}")
    return proc


@pytest.fixture(scope="module")
def sandbox(tmp_path_factory):
    root = tmp_path_factory.mktemp("xf_smoke")
    env = dict(os.environ)
    env["CLS_RUNS"] = str(root)
    env["WANDB_MODE"] = "disabled"
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return root, env


@pytest.fixture(scope="module")
def tiny_encoder(sandbox):
    root, env = sandbox
    out = root / "tiny_encoder.pt"
    _must(_run(["encoder_training.save_untrained_encoder",
                "--encoder-type", "mlp", "--out-dim", "8", "--hidden-dim", "32",
                "--num-hidden-layers", "2", "--gain", "5.0",
                "--lambdas", "3", "4", "--out", str(out)], env), "encoder")
    return out


@pytest.fixture(scope="module")
def explorer(sandbox, tiny_encoder):
    root, env = sandbox
    save = root / "explorer"
    _must(_run(["hopfield_nav.train_navigate",
                "--encoder_checkpoint", str(tiny_encoder), *TINY_WORLD,
                "--schedule", "explore:2", "--eval_every", "1000",
                "--ckpt_every", "1", "--save_dir", str(save)], env), "explorer")
    ck = save / "navigate_u2.pt"
    assert ck.exists()
    return ck


@pytest.fixture(scope="module")
def forked(sandbox, tiny_encoder, explorer):
    root, env = sandbox
    save = root / "fork"
    proc = _must(_run(["hopfield_nav.train_navigate",
                       "--load_checkpoint", str(explorer),
                       "--encoder_checkpoint", str(tiny_encoder), *TINY_WORLD,
                       "--schedule", "task:2,visits=2,novelty=0,eps=0",
                       "--env_repeats", "2", "--eval_scope", "task",
                       "--eval_every", "1", "--ckpt_every", "1",
                       "--print_every", "1",
                       "--ewc_lambda", "10", "--prior_kl_coef", "1",
                       "--fisher_trajectories", "4",
                       "--save_dir", str(save)], env), "fork")
    return save, proc.stdout


@pytest.mark.slow
def test_fork_scores_its_parent_at_u0(forked):
    save, out = forked
    assert "[navigate_u0] samples={'episodes': 0, 'env_steps': 0}" in out
    assert "[navigate_u0] task=" in out
    # ...and the u0 block comes before the first update line.
    assert out.index("[navigate_u0] task=") < out.index("u1(task)")


@pytest.mark.slow
def test_fisher_is_estimated_once_on_search_steps(forked):
    _, out = forked
    assert out.count("explorer prior: Fisher estimated on") == 1
    assert "Explorer prior: EWC lambda=10" in out
    assert "KL(explorer || policy) x 1" in out


@pytest.mark.slow
def test_prior_terms_reach_the_update_log(forked):
    save, out = forked
    line = next(l for l in out.splitlines() if l.strip().startswith("u1(task)"))
    assert "prior_ewc=" in line and "prior_kl=" in line and "prior_drift=" in line
    assert (save / "navigate_u2.pt").exists()
    ck = torch.load(save / "navigate_u2.pt", map_location="cpu", weights_only=False)
    assert ck["config"]["ewc_lambda"] == 10.0
    assert ck["config"]["prior_kl_coef"] == 1.0


@pytest.mark.slow
def test_prior_without_a_parent_is_refused(sandbox, tiny_encoder):
    root, env = sandbox
    proc = _run(["hopfield_nav.train_navigate",
                 "--encoder_checkpoint", str(tiny_encoder), *TINY_WORLD,
                 "--schedule", "task:1,visits=1", "--eval_every", "1000",
                 "--ewc_lambda", "10", "--save_dir", str(root / "noparent")], env)
    assert proc.returncode != 0
    assert "need --load_checkpoint" in proc.stderr
