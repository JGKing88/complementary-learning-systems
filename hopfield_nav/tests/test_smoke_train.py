"""End-to-end wiring guard: encoder -> scaffold -> training -> checkpoint.

The golden fixtures pin numerical behavior but run on a stub scaffold, so they
never touch the encoder loader, the real VectorHash build, checkpoint writing,
or the CLI plumbing. This test exercises the whole chain on a deliberately tiny
world -- the failure it catches is a phase that leaves two modules disagreeing
about a name or a shape, which shows up only at import or wiring time.

Kept under a minute on CPU: out_dim=8, lambdas 3 4 (Npos=12), size=4,
batch_envs=2, steps_per_rollout=8, 2 updates.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(args: list[str], env: dict, timeout: int = 600):
    """Run a module in a subprocess so import-time wiring bugs surface."""
    proc = subprocess.run(
        [sys.executable, "-m", *args],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"`python -m {' '.join(args)}` exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout[-4000:]}\n"
            f"--- stderr ---\n{proc.stderr[-4000:]}"
        )
    return proc


@pytest.fixture(scope="module")
def sandbox(tmp_path_factory):
    """An isolated CLS_RUNS so the smoke test never writes to real outputs."""
    root = tmp_path_factory.mktemp("cls_smoke")
    env = dict(os.environ)
    env["CLS_RUNS"] = str(root)
    env["WANDB_MODE"] = "disabled"
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return root, env


@pytest.fixture(scope="module")
def tiny_encoder(sandbox):
    root, env = sandbox
    out = root / "tiny_encoder.pt"
    _run(["encoder_training.save_untrained_encoder",
          "--encoder-type", "mlp", "--out-dim", "8", "--hidden-dim", "32",
          "--num-hidden-layers", "2", "--gain", "5.0",
          "--lambdas", "3", "4", "--out", str(out)], env)
    assert out.exists()
    return out


@pytest.mark.slow
def test_save_untrained_encoder_is_loadable(tiny_encoder):
    """The checkpoint the rest of the chain depends on round-trips.

    Loaded through hopfield_nav.encoder.load_encoder -- the adapter the
    training entry points use, which resolves the encoder gain as well.
    """
    from hopfield_nav.encoder import load_encoder
    encoder, enc_cfg, gain = load_encoder(str(tiny_encoder), "cpu")
    assert enc_cfg.out_dim == 8
    assert gain == pytest.approx(5.0)


@pytest.mark.slow
def test_train_phase_a_only_end_to_end(sandbox, tiny_encoder):
    """encoder -> scaffold -> rollouts -> PPO update -> checkpoint on disk."""
    root, env = sandbox
    save_dir = root / "phase_a_ckpt"
    _run(["hopfield_nav.train_phase_a_only",
          "--encoder_checkpoint", str(tiny_encoder),
          "--lambdas", "3", "4", "--Np", "40",
          "--size", "4", "--observation_size", "16",
          "--batch_envs", "2", "--steps_per_rollout", "8",
          "--phase_a_updates", "2", "--envs_per_world", "1", "--num_worlds", "1",
          "--num_val_envs", "1", "--eval_every", "1000",
          "--device", "cpu", "--static-vectorhash",
          "--save_dir", str(save_dir)], env)

    ckpts = sorted(save_dir.glob("*.pt"))
    assert ckpts, f"no checkpoint written to {save_dir}"

    ck = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
    assert "agent_state_dict" in ck
    for name, tensor in ck["agent_state_dict"].items():
        assert torch.isfinite(tensor).all(), f"non-finite parameter: {name}"


@pytest.mark.slow
def test_train_ppo_end_to_end_and_losses_are_finite(sandbox, tiny_encoder):
    """The train.py path, which phase 2 switched onto make_env."""
    root, env = sandbox
    save_dir = root / "train_ckpt"
    proc = _run(["hopfield_nav.train",
                 "--encoder_checkpoint", str(tiny_encoder),
                 "--lambdas", "3", "4", "--Np", "40",
                 "--size", "4", "--observation_size", "16",
                 "--batch_envs", "2", "--steps_per_rollout", "8",
                 "--n_updates", "2", "--envs_per_world", "1", "--num_worlds", "1",
                 "--num_val_envs", "1", "--eval_every", "1000",
                 # default is every 100 updates, which never fires in 2
                 "--save_every", "1",
                 "--device", "cpu", "--static-vectorhash",
                 "--save_dir", str(save_dir)], env)

    update_lines = [l for l in proc.stdout.splitlines() if l.startswith("update ")]
    assert update_lines, f"no update lines in output:\n{proc.stdout[-2000:]}"
    for line in update_lines:
        for field in line.split("|"):
            if "=" not in field:
                continue
            key, _, value = field.strip().partition("=")
            assert value.lower() not in ("nan", "inf", "-inf"), \
                f"non-finite {key} in: {line}"

    assert sorted(save_dir.glob("*.pt")), f"no checkpoint written to {save_dir}"


@pytest.mark.slow
def test_outputs_land_under_cls_runs(sandbox, tiny_encoder):
    """CLS_RUNS really does relocate defaults -- nothing leaks into the tree."""
    root, env = sandbox
    from cls_paths import runs_root
    import os as _os

    old = _os.environ.get("CLS_RUNS")
    _os.environ["CLS_RUNS"] = str(root)
    try:
        assert runs_root() == root
    finally:
        if old is None:
            _os.environ.pop("CLS_RUNS", None)
        else:
            _os.environ["CLS_RUNS"] = old
