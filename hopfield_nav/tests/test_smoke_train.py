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

import json
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

    Loaded through hopfield_nav.encoder_io.load_encoder -- the adapter the
    training entry points use, which resolves the encoder gain as well.
    """
    from hopfield_nav.encoder_io import load_encoder
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


# ---------------------------------------------------------------------------
# Downstream entry points
# ---------------------------------------------------------------------------
#
# These four had no coverage at all. Phase 6 of the 2026-08 refactor moves every
# module they import, and an import that goes stale surfaces only when the
# script is run -- which, for the analysis scripts, may be weeks later. Each
# test below is a wiring guard, not a behavioral one: it asserts the chain runs
# end to end and produces its artifact.


@pytest.fixture(scope="module")
def phase_a_checkpoint(sandbox, tiny_encoder):
    """A real checkpoint for the downstream entry points to consume."""
    root, env = sandbox
    save_dir = root / "downstream_ckpt"
    if not sorted(save_dir.glob("*.pt")):
        _run(["hopfield_nav.train_phase_a_only",
              "--encoder_checkpoint", str(tiny_encoder),
              "--lambdas", "3", "4", "--Np", "40",
              "--size", "4", "--observation_size", "16",
              "--batch_envs", "2", "--steps_per_rollout", "8",
              "--phase_a_updates", "2", "--envs_per_world", "1",
              "--num_worlds", "1", "--num_val_envs", "2",
              # eval_every 1: the per-update checkpoint is written inside the
              # eval branch, and analysis.trajectories only accepts files whose
              # basename carries an update number -- it deliberately skips
              # phase_a_only_final.pt. Kept cheap by n_val_trials 1.
              "--eval_every", "1", "--n_val_trials", "1",
              "--val_distractors", "0", "--device", "cpu",
              "--static-vectorhash", "--save_dir", str(save_dir)], env)
    ckpts = sorted(save_dir.glob("*.pt"))
    assert ckpts, f"no checkpoint written to {save_dir}"
    return save_dir, ckpts[-1]


@pytest.mark.slow
def test_eval_all_cli_end_to_end(sandbox, phase_a_checkpoint):
    """eval_all's main(): checkpoint -> rebuilt eval world -> evaluators -> JSON.

    Its helpers are covered by the goldens; this covers the CLI path that
    stitches them together, including the config-from-checkpoint rebuild.
    """
    root, env = sandbox
    _save_dir, ckpt = phase_a_checkpoint
    out_json = root / "eval_all_out.json"
    _run(["hopfield_nav.eval_all",
          "--ckpt", str(ckpt), "--device", "cpu",
          "--num_trials", "2", "--max_steps", "8",
          "--n_distractors", "0",
          "--repeat-trials", "0", "--skip-realistic", "--no-nav-stoch",
          "--num-val-envs", "2",
          "--output-json", str(out_json)], env)

    assert out_json.exists(), "eval_all wrote no JSON"
    payload = json.loads(out_json.read_text())
    assert "navigation" in payload or "nav_det" in json.dumps(payload)


@pytest.mark.slow
def test_ckpt_every_beats_eval_every_in_a_real_run(sandbox, tiny_encoder):
    """Drive the real trainer, not a copy of its arithmetic.

    test_ckpt_cadence.py pins the schedule with local helpers, which cannot
    catch the trainer's own expression changing -- and the expression is the
    thing that was wrong. So this runs phase A with the two cadences set apart
    and counts what lands on disk: 4 checkpoints from 4 updates, while
    --eval_every 4 permits only one eval.
    """
    import run_manifest

    root, env = sandbox
    save_dir = root / "cadence_ckpt"
    _run(["hopfield_nav.train_phase_a_only",
          "--encoder_checkpoint", str(tiny_encoder),
          "--lambdas", "3", "4", "--Np", "40",
          "--size", "4", "--observation_size", "16",
          "--batch_envs", "2", "--steps_per_rollout", "8",
          "--phase_a_updates", "4", "--envs_per_world", "1",
          "--num_worlds", "1", "--num_val_envs", "2",
          "--eval_every", "4", "--ckpt_every", "1",
          "--n_val_trials", "1", "--val_distractors", "0", "--device", "cpu",
          "--static-vectorhash", "--save_dir", str(save_dir)], env)

    updates = [u for u, _ in run_manifest.checkpoints_in(save_dir)]
    assert updates == [1, 2, 3, 4], (
        f"--ckpt_every 1 over 4 updates should save 4 times, got {updates}. "
        f"[1, 4] means the save is still gated on --eval_every.")


@pytest.mark.slow
def test_phase_a_writes_a_usable_manifest(phase_a_checkpoint, tiny_encoder):
    """A real training run leaves a manifest that identifies it.

    The unit tests in test_run_manifest.py drive the module directly; this is
    the one that catches a trainer forgetting to call it, or calling it before
    `cfg.encoder_checkpoint` is resolved.
    """
    import run_manifest

    save_dir, _ckpt = phase_a_checkpoint
    m = run_manifest.read(save_dir)
    assert m is not None, "phase A wrote no run.json"
    assert m["kind"] == "phase_a_only"
    assert m["status"] == run_manifest.STATUS_DONE
    assert m["config"]["env"]["size"] == 4          # the flag this run passed
    assert m["encoder"]["path"] == str(tiny_encoder)
    assert m["encoder"]["sha256"] == run_manifest.file_digest(tiny_encoder)

    # The manifest's checkpoint list agrees with what is on disk, which is the
    # property `analysis.trajectories` now depends on instead of a basename regex.
    from_manifest = {os.path.basename(p) for _u, p in
                     run_manifest.checkpoints_in(save_dir)}
    on_disk = {p.name for p in save_dir.glob("*_u*.pt")}
    assert from_manifest == on_disk, f"manifest {from_manifest} != disk {on_disk}"


@pytest.mark.slow
def test_train_phase_b_only_end_to_end(sandbox, tiny_encoder, phase_a_checkpoint):
    """Phase B resumes from a phase-A checkpoint and trains the store head."""
    root, env = sandbox
    _save_dir, ckpt = phase_a_checkpoint
    proc = _run(["hopfield_nav.train_phase_b_only",
                 "--load_checkpoint", str(ckpt),
                 "--encoder_checkpoint", str(tiny_encoder),
                 "--phase_b_updates", "2", "--steps_per_rollout", "8",
                 "--device", "cpu", "--eval_every", "1000"], env)
    assert proc.returncode == 0


@pytest.mark.slow
def test_visualize_trajectories_renders(sandbox, phase_a_checkpoint):
    """The figure path: checkpoint dir -> rollouts -> PNG + PDF on disk.

    Also exercises _resolve_encoder_path, which reads the encoder location out
    of the checkpoint and has to find it after the storage migration.
    """
    root, env = sandbox
    save_dir, _ckpt = phase_a_checkpoint
    out_stem = root / "viz_smoke"
    _run(["analysis.trajectories",
          "--checkpoint_dir", str(save_dir),
          "--mode", "combined", "--trials", "2",
          "--explore_steps", "6", "--nav_steps", "6",
          "--device", "cpu", "--out", str(out_stem)], env)
    produced = list(root.glob("viz_smoke*"))
    assert produced, f"no figure written; dir holds {[p.name for p in root.iterdir()]}"


@pytest.mark.slow
def test_agenthash_run_sequential_outer_loop():
    """agenthash's outer protocol loop, in-process.

    run_mini_episode is covered by test_protocols; the loop around it -- block
    scheduling, history accumulation, the stored-at-goal counters -- was not.
    """
    import numpy as np
    import torch
    from hopfield_nav.world.env import make_env
    from analysis.continual.agenthash import run_sequential
    from hopfield_nav.tests.fixtures import make_collector, make_stub_cfg

    cfg = make_stub_cfg(movement_mode="discrete")
    cfg.env.size = 4
    cfg.env.goal_radius = 1.5
    _c, agent, vh = make_collector(cfg, 8, seed=0)
    vh.env_offsets = [(0, 0), (8, 0)]
    envs = [make_env(cfg.env, "discrete", seed=300 + i) for i in range(2)]

    torch.manual_seed(0)
    np.random.seed(0)
    trace, blocks, stored = run_sequential(
        agent=agent, val_envs=envs, vectorhash=vh, val_idxs=[0, 1], cfg=cfg,
        device=torch.device("cpu"), iters_per_block=3, max_steps=10, seed=5,
        deterministic=True, oracle_store_at_goal=True,
        oracle_lock_store_not_at_goal=True, lock_store_after_goal=False)

    # 2 blocks x 3 iterations; block 0 evaluates env 0 only, block 1 both.
    assert len(trace) == 6
    assert blocks == [(0, 2, 0), (3, 5, 1)]
    assert sorted(trace[0][2]) == [0]
    assert sorted(trace[-1][2]) == [0, 1]
    assert set(stored) == {0, 1}
    # Non-vacuous: the oracle stores whenever the agent sits on the goal.
    assert sum(stored.values()) > 0, "no store ever fired; fixture is vacuous"
