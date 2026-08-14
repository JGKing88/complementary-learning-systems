"""Continuing a run, as against forking one from its weights.

The headline test is `test_continued_run_matches_uninterrupted`: train four
updates straight through, train two and continue to four, and require the two to
land on bit-identical weights. That is the only check that actually pins the
feature, because every piece of state a continuation forgets shows up the same
way -- as a run that trains fine and quietly isn't the run you asked for.

It is also how the missing pieces were found. With the weights, the optimizer
moments, the config, the world spec and both global RNG streams verified equal
at the resume point, the continuation still diverged at 4e-3 on its first
update: `GridEnv` owns a `RandomState` that draws the agent's start cell on
every reset, and rebuilding the envs restarts all of them. A test that only
asserted "it runs and the loss is finite" would have passed throughout.
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

from hopfield_nav.training import resume as resume_io

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Unit: the resume contract
# ---------------------------------------------------------------------------

class _Agent(torch.nn.Module):
    def __init__(self, n=4):
        super().__init__()
        self.lin = torch.nn.Linear(n, n)


def _stepped(agent):
    """An optimizer with populated Adam state, so shapes exist to compare."""
    opt = torch.optim.Adam(agent.parameters(), lr=1e-3)
    sum(p.sum() for p in agent.parameters()).backward()
    opt.step()
    return opt


def test_save_is_atomic_and_leaves_no_temp(tmp_path):
    agent = _Agent()
    path = resume_io.save(str(tmp_path), kind="navigate", agent=agent,
                          optimizer=_stepped(agent), update=7, config={"a": 1})
    assert Path(path) == tmp_path / resume_io.RESUME_FILE
    # The sibling temp is renamed over the target, never left behind: a stray
    # .tmp would be mistaken for a resume point by anything globbing the dir.
    assert not list(tmp_path.rglob("*.tmp"))


def test_resume_point_is_not_visible_as_a_checkpoint(tmp_path):
    """`gc_runs`, `backfill_manifests` and `checkpoints_in`'s legacy fallback
    all read "every *.pt in the run dir" as "every checkpoint of the run". The
    resume point is not one -- it carries optimizer state no evaluator wants,
    and it is the largest file in the run."""
    agent = _Agent()
    resume_io.save(str(tmp_path), kind="navigate", agent=agent,
                   optimizer=_stepped(agent), update=1, config={})
    assert list(tmp_path.glob("*.pt")) == []


def test_save_overwrites_in_place(tmp_path):
    """The resume point rolls forward rather than accumulating one file per tick."""
    agent = _Agent()
    opt = _stepped(agent)
    for u in (1, 2, 3):
        resume_io.save(str(tmp_path), kind="navigate", agent=agent,
                       optimizer=opt, update=u, config={})
    assert len(list(tmp_path.rglob("*.pt"))) == 1
    assert torch.load(tmp_path / resume_io.RESUME_FILE,
                      weights_only=False)["update"] == 3


def test_load_accepts_a_directory(tmp_path):
    agent = _Agent()
    resume_io.save(str(tmp_path), kind="navigate", agent=agent,
                   optimizer=_stepped(agent), update=2, config={})
    assert resume_io.load(str(tmp_path), "cpu", kind="navigate")["update"] == 2


def test_load_rejects_the_wrong_script(tmp_path):
    agent = _Agent()
    resume_io.save(str(tmp_path), kind="navigate", agent=agent,
                   optimizer=_stepped(agent), update=2, config={})
    with pytest.raises(SystemExit, match="written by `navigate`"):
        resume_io.load(str(tmp_path), "cpu", kind="store")


def test_load_names_a_missing_point_and_points_at_the_fork_flag(tmp_path):
    with pytest.raises(SystemExit, match="no resume point"):
        resume_io.load(str(tmp_path / "nothing.pt"), "cpu", kind="navigate")


def test_load_refuses_a_periodic_checkpoint(tmp_path):
    """A navigate_u*.pt is a fork point; it has no optimizer state to continue."""
    fork = tmp_path / "navigate_u10.pt"
    torch.save({"agent_state_dict": {}, "config": {}, "update": 10}, fork)
    with pytest.raises(SystemExit, match="missing"):
        resume_io.load(str(fork), "cpu", kind="navigate")


def test_load_accepts_the_resume_subdir_too(tmp_path):
    agent = _Agent()
    resume_io.save(str(tmp_path), kind="navigate", agent=agent,
                   optimizer=_stepped(agent), update=4, config={})
    sub = tmp_path / resume_io.RESUME_SUBDIR
    assert resume_io.load(str(sub), "cpu", kind="navigate")["update"] == 4


def test_restore_optimizer_names_a_changed_freeze_set(tmp_path):
    """A count mismatch is what a moved --freeze_store looks like from here."""
    big = _Agent()
    state = _stepped(big).state_dict()

    small = torch.nn.Linear(4, 4)
    opt = torch.optim.Adam([small.weight], lr=1e-3)   # one param, not two
    with pytest.raises(SystemExit, match="different number of trainable"):
        resume_io.restore_optimizer(opt, state, source="x.pt")


def test_restore_optimizer_catches_a_shape_change_before_the_step(tmp_path):
    """Torch would let this load and raise inside the first `.step()` instead.

    That error names a broadcast, not a config field, and arrives after the
    scaffold has built -- which on a real run is minutes in.
    """
    a = _Agent(n=4)
    state = _stepped(a).state_dict()

    wide = _Agent(n=8)                       # same param *count*, new shapes
    opt = torch.optim.Adam(wide.parameters(), lr=1e-3)
    with pytest.raises(SystemExit, match="architecture moved"):
        resume_io.restore_optimizer(opt, state, source="x.pt")


def test_reject_overrides_lists_exactly_the_offending_flags():
    with pytest.raises(SystemExit) as exc:
        resume_io.reject_overrides(
            {"continue_from", "goal_reward", "hidden_size"},
            allowed={"continue_from", "device"})
    msg = str(exc.value)
    assert "--goal_reward" in msg and "--hidden_size" in msg
    assert "--continue_from continues an existing run" in msg


def test_reject_overrides_passes_the_allowed_ones():
    resume_io.reject_overrides({"continue_from", "device"},
                               allowed={"continue_from", "device", "schedule"})


def test_rng_round_trips():
    torch.manual_seed(0)
    np.random.seed(0)
    state = resume_io.rng_state()
    before = (torch.randn(3), np.random.rand(3))

    torch.manual_seed(999)
    np.random.seed(999)
    resume_io.restore_rng(state)
    after = (torch.randn(3), np.random.rand(3))

    assert torch.equal(before[0], after[0])
    assert np.array_equal(before[1], after[1])


class _FakeEnv:
    def __init__(self, seed):
        self.rng = np.random.RandomState(seed)


class _FakeWorld:
    def __init__(self, seeds):
        self.envs = [_FakeEnv(s) for s in seeds]


def test_env_rng_round_trips():
    """The stream that broke the first working version of this feature."""
    worlds = [_FakeWorld([1, 2]), _FakeWorld([3])]
    for w in worlds:                       # advance them off their seeds
        for e in w.envs:
            e.rng.rand(5)

    saved = resume_io.env_rng_states(worlds)
    expected = [e.rng.rand(4) for w in worlds for e in w.envs]

    fresh = [_FakeWorld([1, 2]), _FakeWorld([3])]
    resume_io.restore_env_rng(saved, fresh)
    got = [e.rng.rand(4) for w in fresh for e in w.envs]

    for want, have in zip(expected, got):
        assert np.array_equal(want, have)


def test_env_rng_mismatch_warns_rather_than_raising(capsys):
    """A changed env count must not kill a run hours in -- but must not be silent."""
    saved = resume_io.env_rng_states([_FakeWorld([1, 2])])
    resume_io.restore_env_rng(saved, [_FakeWorld([1])])
    out = capsys.readouterr().out
    assert "2 train-env RNG states" in out and "not be bit-identical" in out


# ---------------------------------------------------------------------------
# Unit: the refresher fast-forward
# ---------------------------------------------------------------------------

SIZE, OBS = 8, 12


@pytest.fixture(scope="module")
def refresh_field():
    from hopfield_nav.config import VectorHashConfig
    from hopfield_nav.world.scaffold import VectorHash

    vh = VectorHash(VectorHashConfig(lambdas=[11, 12], static_vectorhash=True))
    vh.build_scaffold()
    torch.manual_seed(0)
    enc = torch.nn.Linear(vh.Ng, 16)
    enc.eval()
    vh.precompute_encoded_phi(enc, 0.25, device="cpu")
    return vh


def _draw_only_refresher(field, cadence, seed=0):
    """A refresher that draws and records but builds nothing (`worlds=None`).

    The same mode `preflight` runs in, and the right one here: fast_forward's
    contract is about the *draw* sequence, which is the path-dependent part.
    """
    from hopfield_nav.config import EnvConfig
    from hopfield_nav.training.refresh import Refresher
    from hopfield_nav.world import domains as dom
    from hopfield_nav.world import generate as gen
    from hopfield_nav.world.spec import TraitDomains

    env_cfg = EnvConfig(size=SIZE, observation_size=OBS)
    domains = TraitDomains(place=dom.Anywhere(),
                           wall=dom.SeedRange(0, 10_000_000),
                           goal=dom.AnyCells(), size=dom.Sizes((SIZE,)))
    split = gen.generate_split(field, env_cfg, domains, 8, 3, seed=seed,
                               margin=12,
                               refresh_goal=cadence.goal is not None,
                               diagnostics=False)
    return Refresher(cadence, split, None, env_cfg, "discrete", seed)


def test_fast_forward_matches_tick_by_tick_replay(refresh_field):
    """Fast-forwarding to N must land exactly where N maybe_refresh calls land.

    Not merely "somewhere plausible": the wall draw excludes `split.used`, so
    tick N's seeds depend on every tick before it, and a fast-forward that
    skipped to the last due tick would hand the second segment walls the first
    segment never trained on.
    """
    from hopfield_nav.training.refresh import Cadence

    cad = Cadence(goal=2, wall=3)

    a = _draw_only_refresher(refresh_field, cad)
    for tick in range(1, 13):
        a.maybe_refresh(tick)

    b = _draw_only_refresher(refresh_field, cad)
    b.fast_forward(12)

    assert a.ticks == b.ticks
    assert a.counts == b.counts
    assert [s.wall_seed for s in a.split.train] == [s.wall_seed for s in b.split.train]
    assert [s.goal for s in a.split.train] == [s.goal for s in b.split.train]
    assert a.split.used["wall"] == b.split.used["wall"]


def test_fast_forward_to_zero_is_a_no_op(refresh_field):
    from hopfield_nav.training.refresh import Cadence

    r = _draw_only_refresher(refresh_field, Cadence(goal=1))
    before = [s.goal for s in r.split.train]
    assert r.fast_forward(0) == 0
    assert [s.goal for s in r.split.train] == before


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

def _run(args, env, timeout=900):
    proc = subprocess.run([sys.executable, "-m", *args], cwd=REPO_ROOT, env=env,
                          capture_output=True, text=True, timeout=timeout)
    return proc


def _must(proc, what):
    if proc.returncode != 0:
        pytest.fail(f"{what} exited {proc.returncode}\n"
                    f"--- stdout ---\n{proc.stdout[-4000:]}\n"
                    f"--- stderr ---\n{proc.stderr[-4000:]}")
    return proc


@pytest.fixture(scope="module")
def sandbox(tmp_path_factory):
    root = tmp_path_factory.mktemp("cls_continue")
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


def _navigate_args(enc, schedule, save_dir):
    return ["hopfield_nav.train_navigate",
            "--encoder_checkpoint", str(enc), "--lambdas", "3", "4", "--Np", "40",
            "--size", "4", "--observation_size", "16",
            "--batch_envs", "2", "--steps_per_rollout", "8",
            "--envs_per_world", "1", "--num_worlds", "1", "--num_val_envs", "1",
            "--eval_every", "1000", "--ckpt_every", "1",
            "--device", "cpu", "--static-vectorhash",
            "--schedule", schedule, "--save_dir", str(save_dir)]


def _weights(path):
    return torch.load(path, map_location="cpu",
                      weights_only=False)["agent_state_dict"]


@pytest.mark.slow
def test_continued_run_matches_uninterrupted(sandbox, tiny_encoder):
    """The whole feature, in one assertion.

    Anything a continuation fails to carry -- Adam's moments, either global RNG
    stream, the distractor stream, a single env's own stream, the refresher's
    tick history, the update counter the anneals key off -- moves these weights.
    """
    root, env = sandbox

    _must(_run(_navigate_args(tiny_encoder, "interleave:4", root / "ref"), env),
          "reference run")
    _must(_run(_navigate_args(tiny_encoder, "interleave:2", root / "seg"), env),
          "first segment")
    _must(_run(["hopfield_nav.train_navigate",
                "--continue_from", str(root / "seg" / resume_io.RESUME_FILE),
                "--schedule", "interleave:4"], env), "continuation")

    for u in (3, 4):
        ref = _weights(root / "ref" / f"navigate_u{u}.pt")
        got = _weights(root / "seg" / f"navigate_u{u}.pt")
        assert ref.keys() == got.keys()
        for k in ref:
            assert torch.equal(ref[k], got[k]), (
                f"u{u} parameter {k} diverged: max abs diff "
                f"{(ref[k].float() - got[k].float()).abs().max().item():.3e}")


@pytest.mark.slow
def test_continuation_continues_the_same_run_dir_and_manifest(sandbox,
                                                              tiny_encoder):
    """One run in two segments, not two runs -- `created` and the checkpoint
    list survive, and `segments` records where the second picked up."""
    root, env = sandbox
    manifest = json.loads((root / "seg" / "run.json").read_text())

    assert manifest["segments"], "continuation left no segment record"
    assert manifest["segments"][-1]["resumed_at_update"] == 2
    # The first segment's checkpoints are still listed alongside the second's.
    updates = {e["update"] for e in manifest["checkpoints"]}
    assert {1, 2, 3, 4} <= updates


@pytest.mark.slow
def test_continue_refuses_a_config_override(sandbox, tiny_encoder):
    root, env = sandbox
    proc = _run(["hopfield_nav.train_navigate",
                 "--continue_from", str(root / "seg" / resume_io.RESUME_FILE),
                 "--schedule", "interleave:4",
                 "--goal_reward", "9.0"], env)
    assert proc.returncode != 0
    assert "--goal_reward" in proc.stderr


@pytest.mark.slow
def test_continue_refuses_a_schedule_that_rewrites_the_past(sandbox,
                                                            tiny_encoder):
    """`explore` for updates the run already ran as `interleave`."""
    root, env = sandbox
    proc = _run(["hopfield_nav.train_navigate",
                 "--continue_from", str(root / "seg" / resume_io.RESUME_FILE),
                 "--schedule", "explore:2 ; interleave:2"], env)
    assert proc.returncode != 0
    assert "u1" in proc.stderr and "explore" in proc.stderr


@pytest.mark.slow
def test_continue_refuses_a_shorter_schedule(sandbox, tiny_encoder):
    root, env = sandbox
    proc = _run(["hopfield_nav.train_navigate",
                 "--continue_from", str(root / "seg" / resume_io.RESUME_FILE),
                 "--schedule", "interleave:1"], env)
    assert proc.returncode != 0
    assert "cannot be shorter" in proc.stderr


@pytest.mark.slow
def test_continue_and_load_checkpoint_are_mutually_exclusive(sandbox,
                                                             tiny_encoder):
    root, env = sandbox
    proc = _run(["hopfield_nav.train_navigate",
                 "--continue_from", str(root / "seg" / resume_io.RESUME_FILE),
                 "--load_checkpoint", str(root / "seg" / "navigate_u1.pt"),
                 "--schedule", "interleave:4"], env)
    assert proc.returncode != 0
    assert "different operations" in proc.stderr


@pytest.mark.slow
def test_fork_says_when_it_drops_adam(sandbox, tiny_encoder):
    """train.py forking a navigate checkpoint gets fresh moments -- out loud."""
    root, env = sandbox
    proc = _must(_run(["hopfield_nav.train",
                       "--load_checkpoint", str(root / "seg" / "navigate_u1.pt"),
                       "--n_updates", "1", "--save_every", "1",
                       "--eval_every", "1000", "--device", "cpu",
                       "--static-vectorhash",
                       "--save_dir", str(root / "forked")], env), "fork")
    assert "no optimizer_state_dict" in proc.stdout


@pytest.mark.slow
def test_periodic_checkpoints_stay_free_of_optimizer_state(sandbox,
                                                           tiny_encoder):
    """The reason the resume point is its own file: Adam doubles the bytes and
    eval/analysis read these."""
    root, _ = sandbox
    ck = torch.load(root / "seg" / "navigate_u2.pt", map_location="cpu",
                    weights_only=False)
    assert "optimizer_state_dict" not in ck
    assert "optimizer_state_dict" in torch.load(
        root / "seg" / resume_io.RESUME_FILE, map_location="cpu",
        weights_only=False)
