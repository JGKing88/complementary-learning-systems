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
@pytest.mark.parametrize("cell,nonlinearity", [
    ("gru", "tanh"),
    # The softplus trunk is the one whose failure mode is arithmetic rather
    # than structural: the unit tests pin the recurrence, but only a real
    # update loop shows whether an unbounded positive state stays finite once
    # PPO is pushing on it.
    ("rnn", "softplus"),
], ids=["gru", "rnn-softplus"])
def test_train_navigate_end_to_end(sandbox, tiny_encoder, cell, nonlinearity):
    """encoder -> scaffold -> rollouts -> PPO update -> checkpoint on disk."""
    root, env = sandbox
    save_dir = root / f"navigate_ckpt_{cell}_{nonlinearity}"
    _run(["hopfield_nav.train_navigate",
          "--encoder_checkpoint", str(tiny_encoder),
          "--lambdas", "3", "4", "--Np", "40",
          "--size", "4", "--observation_size", "16",
          "--batch_envs", "2", "--steps_per_rollout", "8",
          "--schedule", "interleave:2",
          "--envs_per_world", "1", "--num_worlds", "1",
          "--num_val_envs", "1", "--eval_every", "1000",
          "--rnn_cell", cell, "--rnn_nonlinearity", nonlinearity,
          "--device", "cpu", "--static-vectorhash",
          "--save_dir", str(save_dir)], env)

    ckpts = sorted(save_dir.glob("*.pt"))
    assert ckpts, f"no checkpoint written to {save_dir}"

    ck = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
    assert "agent_state_dict" in ck
    for name, tensor in ck["agent_state_dict"].items():
        assert torch.isfinite(tensor).all(), f"non-finite parameter: {name}"
    # The trunk the run actually built is recorded, so a resume rebuilds it.
    assert ck["config"]["agent"]["rnn_cell"] == cell
    assert ck["config"]["agent"]["rnn_nonlinearity"] == nonlinearity


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
def navigate_checkpoint(sandbox, tiny_encoder):
    """A real checkpoint for the downstream entry points to consume."""
    root, env = sandbox
    save_dir = root / "downstream_ckpt"
    if not sorted(save_dir.glob("*.pt")):
        _run(["hopfield_nav.train_navigate",
              "--encoder_checkpoint", str(tiny_encoder),
              "--lambdas", "3", "4", "--Np", "40",
              "--size", "4", "--observation_size", "16",
              "--batch_envs", "2", "--steps_per_rollout", "8",
              "--schedule", "interleave:2", "--envs_per_world", "1",
              "--num_worlds", "1", "--num_val_envs", "2",
              # eval_every 1: analysis.trajectories indexes its rows by update
              # number, so it skips navigate_final.pt and needs at least one
              # navigate_u{N}.pt. Cheap via n_val_trials 1. (--ckpt_every would
              # now do this without paying for the evals; left as-is so the
              # fixture still exercises the default coupled cadence.)
              "--eval_every", "1", "--n_val_trials", "1",
              "--val_distractors", "0", "--device", "cpu",
              "--static-vectorhash", "--save_dir", str(save_dir)], env)
    ckpts = sorted(save_dir.glob("*.pt"))
    assert ckpts, f"no checkpoint written to {save_dir}"
    return save_dir, ckpts[-1]


@pytest.mark.slow
def test_eval_all_cli_end_to_end(sandbox, navigate_checkpoint):
    """eval_all's main(): checkpoint -> rebuilt eval world -> evaluators -> JSON.

    Its helpers are covered by the goldens; this covers the CLI path that
    stitches them together, including the config-from-checkpoint rebuild.
    """
    root, env = sandbox
    _save_dir, ckpt = navigate_checkpoint
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
    thing that was wrong. So this runs train_navigate with the two cadences set apart
    and counts what lands on disk: 4 checkpoints from 4 updates, while
    --eval_every 4 permits only one eval.
    """
    import run_manifest

    root, env = sandbox
    save_dir = root / "cadence_ckpt"
    _run(["hopfield_nav.train_navigate",
          "--encoder_checkpoint", str(tiny_encoder),
          "--lambdas", "3", "4", "--Np", "40",
          "--size", "4", "--observation_size", "16",
          "--batch_envs", "2", "--steps_per_rollout", "8",
          "--schedule", "interleave:4", "--envs_per_world", "1",
          "--num_worlds", "1", "--num_val_envs", "2",
          "--eval_every", "4", "--ckpt_every", "1",
          "--n_val_trials", "1", "--val_distractors", "0", "--device", "cpu",
          "--static-vectorhash", "--save_dir", str(save_dir)], env)

    updates = [u for u, _ in run_manifest.checkpoints_in(save_dir)]
    assert updates == [1, 2, 3, 4], (
        f"--ckpt_every 1 over 4 updates should save 4 times, got {updates}. "
        f"[1, 4] means the save is still gated on --eval_every.")


@pytest.mark.slow
def test_env_refresh_end_to_end(sandbox, tiny_encoder):
    """A real refreshing run, through the CLI, and the record it leaves behind.

    The unit tests drive `Refresher` directly; this is the only thing that
    checks the wiring -- that the flags reach a cadence, the cadence reaches the
    update loop, and `world.json` gets rewritten on the checkpoint cadence with
    the union grown rather than the startup draw frozen. A run that refreshed
    but recorded only its first draw would look identical from every angle
    except this file.
    """
    import json

    root, env = sandbox
    save_dir = root / "refresh_ckpt"
    _run(["hopfield_nav.train_navigate",
          "--encoder_checkpoint", str(tiny_encoder),
          "--lambdas", "3", "4", "--Np", "40",
          "--size", "3", "--observation_size", "16",
          "--batch_envs", "2", "--steps_per_rollout", "8",
          "--schedule", "interleave:4", "--envs_per_world", "2",
          "--num_worlds", "1", "--num_val_envs", "1",
          "--eval_every", "100", "--ckpt_every", "2",
          "--n_val_trials", "1", "--val_distractors", "0", "--device", "cpu",
          "--static-vectorhash", "--env_generator", "--place_margin", "1",
          "--refresh_place", "1", "--refresh_goal", "1", "--refresh_wall", "2",
          "--save_dir", str(save_dir)], env)

    spec = json.loads((save_dir / "world.json").read_text())
    split = spec["split"]
    n_train = len(split["train"])
    rep = split["diagnostics"]["refresh"]
    assert rep["cadence"] == {"place": 1, "wall": 2, "goal": 1, "size": None}
    assert rep["ticks"] == 4 and rep["counts"]["wall"] == 2
    # 4 updates x 2 envs of fresh wall seeds, plus the startup draw.
    assert len(split["used"]["wall"]) == n_train * 3, (
        "the used union did not grow with the refresh ticks -- a later "
        "held_out val set would treat training's own walls as unseen")
    # Validation is drawn once and held: an eval curve is only readable if the
    # thing being evaluated on stood still.
    assert len(split["base_val"]) == 1


@pytest.mark.slow
def test_refresh_without_the_generator_fails_fast(sandbox, tiny_encoder):
    """And says which flag is missing, before the 12 GB scaffold gets built."""
    root, env = sandbox
    proc = subprocess.run(
        [sys.executable, "-m", "hopfield_nav.train_navigate",
         "--encoder_checkpoint", str(tiny_encoder),
         "--lambdas", "3", "4", "--size", "3", "--schedule", "explore:1",
         "--device", "cpu", "--refresh_place", "1",
         "--save_dir", str(root / "no_generator")],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode != 0
    assert "needs --env_generator" in proc.stdout + proc.stderr


@pytest.mark.slow
def test_navigate_writes_a_usable_manifest(navigate_checkpoint, tiny_encoder):
    """A real training run leaves a manifest that identifies it.

    The unit tests in test_run_manifest.py drive the module directly; this is
    the one that catches a trainer forgetting to call it, or calling it before
    `cfg.encoder_checkpoint` is resolved.
    """
    import run_manifest

    save_dir, _ckpt = navigate_checkpoint
    m = run_manifest.read(save_dir)
    assert m is not None, "train_navigate wrote no run.json"
    assert m["kind"] == "navigate"
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
def test_resuming_inherits_the_parents_recipe(sandbox, tiny_encoder):
    """--load_checkpoint carries the whole config, not just the weights.

    The trap this guards: a parent trained with non-default reward shaping, a
    child that re-states only its schedule, and every unmentioned flag silently
    reverting to an argparse default. So the child here passes *nothing* but
    --schedule and --save_dir, and has to come out with the parent's
    goal_reward and wall_penalty -- while `size`, which it does pass, overrides.
    """
    import run_manifest

    root, env = sandbox
    parent_dir = root / "inherit_parent"
    base = [
        "hopfield_nav.train_navigate",
        "--encoder_checkpoint", str(tiny_encoder),
        "--lambdas", "3", "4", "--Np", "40",
        "--size", "4", "--observation_size", "16",
        "--batch_envs", "2", "--steps_per_rollout", "8",
        "--envs_per_world", "1", "--num_worlds", "1", "--num_val_envs", "1",
        "--eval_every", "1000", "--device", "cpu", "--static-vectorhash",
    ]
    _run(base + ["--schedule", "explore:2",
                 "--goal_reward", "5.0", "--wall_penalty", "0.1",
                 # Passed rather than left to the flag's default: the assertion
                 # below distinguishes the run's novelty from the 0.0 the loop
                 # parks in cfg between rollouts, which it cannot do if the
                 # default happens to be 0.0 too.
                 "--novelty_reward", "0.25",
                 "--save_dir", str(parent_dir)], env)

    child_dir = root / "inherit_child"
    _run(["hopfield_nav.train_navigate",
          "--encoder_checkpoint", str(tiny_encoder),
          "--load_checkpoint", str(parent_dir / "navigate_final.pt"),
          "--schedule", "exploit:2",
          "--device", "cpu",
          "--save_dir", str(child_dir)], env)

    child = run_manifest.read(child_dir)["config"]
    assert child["env"]["goal_reward"] == 5.0, "shaping reverted to the CLI default"
    assert child["hopfield"]["wall_penalty"] == 0.1
    assert child["env"]["size"] == 4                # architecture came along too
    assert child["schedule"] == "exploit:2"         # but the schedule is the child's
    # save_dir is the one field never inherited: reusing it would have the
    # child overwrite its own parent.
    assert child["save_dir"] == str(child_dir)

    parent = run_manifest.read(parent_dir)["config"]
    assert parent["schedule"] == "explore:2"
    # The parent's own novelty_reward survived into its manifest, rather than
    # being recorded as the 0.0 the loop parks it at between rollouts.
    assert parent["hopfield"]["novelty_reward"] == pytest.approx(0.25)
    # ...and the child inherited that too, having said nothing about it.
    assert child["hopfield"]["novelty_reward"] == pytest.approx(0.25)


@pytest.mark.slow
def test_train_store_end_to_end(sandbox, tiny_encoder, navigate_checkpoint):
    """train_store resumes from a train_navigate checkpoint and trains the store head."""
    root, env = sandbox
    _save_dir, ckpt = navigate_checkpoint
    proc = _run(["hopfield_nav.train_store",
                 "--load_checkpoint", str(ckpt),
                 "--encoder_checkpoint", str(tiny_encoder),
                 "--phase_b_updates", "2", "--steps_per_rollout", "8",
                 "--device", "cpu", "--eval_every", "1000"], env)
    assert proc.returncode == 0


@pytest.mark.slow
def test_train_writes_a_world_on_both_paths(sandbox, tiny_encoder):
    """`train` could not say which envs it used until now.

    Both paths write it -- the point of recording is not that a run opted into
    declared domains, it is that a post-hoc eval can rebuild the envs a
    checkpoint was actually scored against (docs/EVAL_SPLITS_DESIGN.md 1.4).
    """
    import json

    root, env = sandbox
    common = ["hopfield_nav.train",
              "--encoder_checkpoint", str(tiny_encoder),
              "--lambdas", "3", "4", "--Np", "40", "--size", "3",
              "--observation_size", "16", "--batch_envs", "2",
              "--steps_per_rollout", "8", "--envs_per_world", "2",
              "--num_worlds", "1", "--num_val_envs", "2", "--n_updates", "2",
              "--eval_every", "100", "--save_every", "2",
              "--device", "cpu", "--static-vectorhash"]
    seen = {}
    for kind, extra in (("legacy", []),
                        ("declared", ["--env_generator", "--place_margin", "1"])):
        out = root / f"train_world_{kind}"
        _run(common + extra + ["--save_dir", str(out)], env)
        spec = json.loads((out / "world.json").read_text())
        assert spec["generator"] == kind
        assert len(spec["split"]["train"]) == 2
        assert len(spec["split"]["base_val"]) == 2
        seen[kind] = spec

    # The declared path asserts separation; the legacy path only measures it.
    # That difference is the reason `--env_generator` exists, so pin it rather
    # than assume both records mean the same thing.
    assert seen["declared"]["split"]["diagnostics"]["min_place_gap"] >= 1
    assert seen["declared"]["split"]["margin"] == 1
    assert seen["legacy"]["split"]["margin"] == 0

    # And a checkpoint names the file, which is what makes a post-hoc eval
    # resolve the right envs instead of replaying an RNG.
    ck = torch.load(root / "train_world_declared" / "hopfield_nav_update2.pt",
                    map_location="cpu", weights_only=False)
    assert ck["world_spec"]["spec_hash"] == seen["declared"]["spec_hash"]


@pytest.mark.slow
def test_train_resumes_a_navigate_checkpoint_with_no_flags_restated(
        sandbox, generator_checkpoint):
    """Cross-trainer resume, which did not work at all until 2026-08-12.

    `train.py` built its config from argv only, so every architecture flag had
    to be restated by hand -- and 8 of 17 agent fields differ between a
    `train_navigate` checkpoint and `train.py`'s defaults, `movement_mode` among
    them, which is a different policy head rather than a width mismatch. The
    invocation below is the one that used to die at `load_state_dict`.
    """
    import json

    root, env = sandbox
    parent_dir, ckpt = generator_checkpoint
    out = root / "train_resume_navigate"
    _run(["hopfield_nav.train",
          "--load_checkpoint", str(ckpt),
          "--encoder_checkpoint", str(tiny_encoder_path(parent_dir)),
          "--n_updates", "1", "--eval_every", "100", "--save_every", "1",
          "--device", "cpu", "--seed", "5", "--save_dir", str(out)], env)

    ck = torch.load(out / "hopfield_nav_update1.pt", map_location="cpu",
                    weights_only=False)
    parent = torch.load(ckpt, map_location="cpu", weights_only=False)["config"]
    child = ck["config"]

    # Inherited, not defaulted. These are the fields whose train.py default
    # differs from what train_navigate wrote.
    from hopfield_nav.config import TrainConfig
    d = TrainConfig()
    for blk, field in (("agent", "movement_mode"), ("env", "movement_mode"),
                       ("agent", "input_sensory"), ("agent", "hopfield_mode")):
        assert child[blk][field] == parent[blk][field], (
            f"{blk}.{field} was not inherited from the parent")
    assert child["agent"]["movement_mode"] != d.agent.movement_mode, (
        "the parent and train.py's default agree here, so this cannot show "
        "inheritance happening")

    # Both worlds recorded, which is now every trainer's behaviour and not
    # train_store's privilege.
    assert (out / "world.json").exists()
    assert (out / "world_parent.json").exists()
    block = json.loads((out / "world.json").read_text())
    assert "parent" in block["split"]["diagnostics"]
    assert ck["world_spec"]["spec_hash"] == block["spec_hash"]

    # The child inherited its parent's world, which is what makes a later
    # `--split held_out` mean "unseen by either stage" rather than "unseen by
    # the last one". Two halves, and the CLI wiring is the only place both are
    # visible at once: placed clear of the parent, and recording it.
    parent_world = json.loads((out / "world_parent.json").read_text())
    overlap = block["split"]["diagnostics"]["parent"]["overlap"]
    margin = block["split"]["margin"]
    assert overlap["min_place_gap_vs_parent"] >= margin, (
        f"child train envs came within {overlap['min_place_gap_vs_parent']} of "
        f"the parent's, below margin {margin} -- the inherited exclusion did "
        f"not reach the placement")
    assert overlap["n_wall_seeds_shared"] == 0
    for trait in ("place", "wall", "goal", "size"):
        p_used = {tuple(v) if isinstance(v, list) else v
                  for v in parent_world["split"]["used"][trait]}
        c_used = {tuple(v) if isinstance(v, list) else v
                  for v in block["split"]["used"][trait]}
        assert p_used <= c_used, (
            f"world.json dropped the parent's used {trait}; a val set minted "
            f"from this record would treat stage one's envs as unseen")


def tiny_encoder_path(run_dir):
    """The encoder a fixture run was trained against, read back from its cfg."""
    import json
    return json.loads((run_dir / "run.json").read_text())["encoder"]["path"]


@pytest.mark.slow
def test_train_survives_its_own_eval_logging(sandbox, tiny_encoder):
    """The wandb path, which nothing drove -- so nothing caught it going dead.

    `train.py --use_wandb` raised `NameError: name 'union' is not defined` at its
    first eval from 2026-08-06, when `evaluate_union_coverage` was absorbed into
    `evaluate_exploration` and this logging line outlived its producer. Every run
    with logging and evals both on -- which is how the script is normally run --
    died there. `WANDB_MODE=disabled` still evaluates the expression, so this
    reaches the bug without needing a wandb account.
    """
    root, env = sandbox
    out = root / "train_wandb_eval"
    _run(["hopfield_nav.train",
          "--encoder_checkpoint", str(tiny_encoder),
          "--lambdas", "3", "4", "--Np", "40", "--size", "3",
          "--observation_size", "16", "--batch_envs", "2",
          "--steps_per_rollout", "8", "--envs_per_world", "2",
          "--num_worlds", "1", "--num_val_envs", "2", "--n_updates", "1",
          "--eval_every", "1", "--n_val_trials", "1", "--save_every", "1",
          "--device", "cpu", "--static-vectorhash", "--use_wandb",
          "--save_dir", str(out)], env)
    assert list(out.glob("*.pt")), "the run did not reach its first checkpoint"


@pytest.mark.slow
def test_train_refuses_both_refresh_mechanisms_at_once(sandbox, tiny_encoder):
    """The old one re-places over the whole scaffold and records nothing, so it
    undoes exactly what the new one guarantees. Silently letting both run would
    leave world.json describing envs the run had stopped using."""
    root, env = sandbox
    proc = subprocess.run(
        [sys.executable, "-m", "hopfield_nav.train",
         "--encoder_checkpoint", str(tiny_encoder),
         "--lambdas", "3", "4", "--size", "3", "--device", "cpu",
         "--n_updates", "1", "--env_generator", "--refresh_place", "1",
         "--refresh_envs_each_update", "--save_dir", str(root / "both")],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode != 0
    assert "--refresh_place" in proc.stdout + proc.stderr
    assert not list((root / "both").glob("*.pt")) if (root / "both").exists() else True


@pytest.mark.slow
def test_train_store_records_both_its_world_and_its_parents(
        sandbox, tiny_encoder, generator_checkpoint):
    """Phase B draws its own envs, so its eval numbers are not on the same axes
    as Phase A's. The run directory says so, on its own, without needing the
    parent to still exist: a verbatim copy of the parent's world.json plus an
    overlap block, both reachable from the checkpoint.
    """
    import filecmp
    import json

    root, env = sandbox
    parent_dir, ckpt = generator_checkpoint
    out = root / "store_both_worlds"
    _run(["hopfield_nav.train_store",
          "--load_checkpoint", str(ckpt),
          # The parent's own encoder, read back from its manifest: the fixture's
          # scaffold is lambdas 3 4 5, and `tiny_encoder` is built for 3 4.
          "--encoder_checkpoint", str(tiny_encoder_path(parent_dir)),
          "--phase_b_updates", "2", "--steps_per_rollout", "8",
          "--device", "cpu", "--eval_every", "1000", "--ckpt_every", "2",
          "--seed", "7", "--save_dir", str(out)], env)

    # Verbatim, so the copy's own spec_hash still verifies. A re-serialized
    # copy would be a different file claiming to be the parent's.
    assert filecmp.cmp(parent_dir / "world.json", out / "world_parent.json",
                       shallow=False)

    child = json.loads((out / "world.json").read_text())
    parent = json.loads((out / "world_parent.json").read_text())
    block = child["split"]["diagnostics"]["parent"]
    assert block["spec_hash"] == parent["spec_hash"]
    assert block["checkpoint"] == str(ckpt)

    # Phase B used --seed 7 against a parent trained at another seed, so the
    # worlds differ -- and the record says so rather than leaving it to be
    # discovered by whoever plots the two curves together.
    overlap = block["overlap"]
    assert overlap["val_envs_identical"] is False
    assert overlap["min_place_gap_vs_parent"] is not None
    assert child["split"]["base_val"] != parent["split"]["base_val"], (
        "the two worlds are identical, so this test cannot show the difference "
        "being recorded")

    ck = torch.load(out / "store_final.pt", map_location="cpu",
                    weights_only=False)
    assert ck["world_spec"]["spec_hash"] == child["spec_hash"]
    assert ck["parent_world_spec"]["spec_hash"] == parent["spec_hash"]


@pytest.mark.slow
def test_visualize_trajectories_renders(sandbox, navigate_checkpoint):
    """The figure path: checkpoint dir -> rollouts -> PNG + PDF on disk.

    Also exercises _resolve_encoder_path, which reads the encoder location out
    of the checkpoint and has to find it after the storage migration.
    """
    root, env = sandbox
    save_dir, _ckpt = navigate_checkpoint
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
        agent=agent, val_envs=envs, vectorhash=vh,
        env_offsets=vh.env_offsets, cfg=cfg,
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


@pytest.mark.slow
def test_eval_all_output_keeps_its_shape_for_a_single_split(sandbox,
                                                           navigate_checkpoint):
    """Nine readers consume this file's top-level keys.

    `analysis.continual.plotting`, `train.py`, `train_rnn.py`,
    `evaluation/rnn.py`, three `run_*.sh` scripts and this suite all index
    `nav_det` / `discovery` / `exploration` directly. Splits are additive: the
    first combination stays at the top level, so a one-split run is what it
    always was.
    """
    import json

    root, env = sandbox
    _save_dir, ckpt = navigate_checkpoint
    out = root / "eval_shape.json"
    _run(["hopfield_nav.eval_all", "--ckpt", str(ckpt),
          "--device", "cpu", "--num_trials", "1", "--max_steps", "4",
          "--skip-realistic", "--no-nav-stoch",
          "--output-json", str(out)], env)
    res = json.loads(out.read_text())
    for key in ("ckpt_path", "encoder_path", "Npos", "movement_mode",
                "num_val_envs", "nav_det", "discovery", "exploration",
                "scaffold_layout", "tag"):
        assert key in res, f"eval_all output lost the top-level key {key!r}"
    assert set(res["splits"]) == {"recorded"}
    assert res["splits"]["recorded"]["nav_det"] == res["nav_det"]


@pytest.fixture(scope="module")
def roomy_encoder(sandbox):
    """lambdas 3 4 5 -> Npos = 60, against `tiny_encoder`'s 3 4 -> 12.

    Anything a *continuation* is built on needs the larger scaffold. Since
    2026-08-12 a run resuming from `--load_checkpoint` places clear of every env
    its parent trained on, and on a 12x12 torus two size-3 parent envs at margin
    1 leave three mutually-clear slots where the child wants four -- so the
    child fails to place, correctly and unhelpfully. The world has to be big
    enough to hold two stages before it can test two stages.
    """
    root, env = sandbox
    out = root / "roomy_encoder.pt"
    _run(["encoder_training.save_untrained_encoder",
          "--encoder-type", "mlp", "--out-dim", "8", "--hidden-dim", "32",
          "--num-hidden-layers", "2", "--gain", "5.0",
          "--lambdas", "3", "4", "5", "--out", str(out)], env)
    assert out.exists()
    return out


@pytest.fixture(scope="module")
def generator_checkpoint(sandbox, roomy_encoder):
    """A checkpoint with a `world.json`, so a split can actually be minted.

    `navigate_checkpoint` runs without `--env_generator` on purpose -- most of
    the suite should keep exercising the path a run without a declared world
    takes -- but `--split` and `--val_size` both need the recorded domains.
    """
    root, env = sandbox
    save_dir = root / "generator_ckpt"
    if not sorted(save_dir.glob("*.pt")):
        _run(["hopfield_nav.train_navigate",
              "--encoder_checkpoint", str(roomy_encoder),
              "--lambdas", "3", "4", "5", "--Np", "40",
              "--size", "3", "--observation_size", "16",
              "--batch_envs", "2", "--steps_per_rollout", "8",
              "--schedule", "interleave:2", "--envs_per_world", "2",
              "--num_worlds", "1", "--num_val_envs", "2",
              "--eval_every", "100", "--ckpt_every", "2",
              "--n_val_trials", "1", "--val_distractors", "0", "--device", "cpu",
              "--static-vectorhash", "--env_generator", "--place_margin", "1",
              "--save_dir", str(save_dir)], env)
    ckpts = sorted(save_dir.glob("*.pt"))
    assert ckpts, f"no checkpoint written to {save_dir}"
    assert (save_dir / "world.json").exists()
    return save_dir, ckpts[-1]


@pytest.mark.slow
def test_val_size_needs_a_minted_split(sandbox, generator_checkpoint):
    """`recorded` is the fixed list of envs the run was scored against; there is
    no size-N version of it. Ignoring the flag is the silent failure this whole
    phase exists to end, so it is an error instead."""
    root, env = sandbox
    _save_dir, ckpt = generator_checkpoint
    proc = subprocess.run(
        [sys.executable, "-m", "hopfield_nav.eval_all", "--ckpt",
         str(ckpt), "--device", "cpu", "--val_size", "6"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode != 0
    assert "every --split is 'recorded'" in proc.stdout + proc.stderr


@pytest.mark.slow
def test_val_size_at_the_trained_size_returns_the_same_results(
        sandbox, generator_checkpoint):
    """The Phase 6 acceptance gate, at the CLI.

    Every evaluator's numbers, for a `--val_size` equal to the size the run
    trained at, must equal the ones the same `--split` produces without it. That
    is what says the size plumbing changed nothing for the runs that do not use
    it -- a claim no amount of unit testing inside the generator can make.
    """
    import json

    root, env = sandbox
    _save_dir, ckpt = generator_checkpoint
    levels = "place=held_out,wall=held_out,goal=held_out"
    base_args = ["hopfield_nav.eval_all", "--ckpt", str(ckpt),
                 "--device", "cpu", "--num_trials", "1", "--max_steps", "4",
                 "--skip-realistic", "--no-nav-stoch", "--split", levels]

    plain = root / "size_noop_plain.json"
    sized = root / "size_noop_sized.json"
    _run(base_args + ["--output-json", str(plain)], env)
    _run(base_args + ["--val_size", "3", "--output-json", str(sized)], env)

    a, b = json.loads(plain.read_text()), json.loads(sized.read_text())
    assert a["split"] == b["split"] == levels, (
        "an equal --val_size decorated the key, so it was not a no-op")
    for key in ("nav_det", "discovery", "exploration", "split_report",
                "scaffold_layout", "splits"):
        assert a[key] == b[key], f"--val_size at the trained size changed {key}"
    # Non-vacuous: the run really did evaluate something.
    assert a["exploration"]["0"]["mean_coverage"] > 0
    # And the extra budget pass does not fire when there is nothing to scale.
    assert "step_budget" not in a and "step_budget" not in b
    assert "exploration_scaled" not in b
