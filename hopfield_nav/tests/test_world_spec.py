"""Tests for the recorded world: `world.json`, its hash, and rebuilding from it.

The property that matters here is that a world can be *read back*, not replayed.
`build_eval_world` recovers a checkpoint's val wall codes and goals by replaying
the seed stream but not their offsets, because placement drew from global
`np.random` (docs/EVAL_SPLITS_DESIGN.md §1.4) — so every post-hoc eval has scored
checkpoints on scaffold patches training never used. These tests pin the fix.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from hopfield_nav.config import EnvConfig, TrainConfig, VectorHashConfig
from hopfield_nav.evaluation import checkpoint_io as cio
from hopfield_nav.world import domains as dom
from hopfield_nav.world import generate as gen
from hopfield_nav.world.scaffold import VectorHash
from hopfield_nav.world.spec import (
    EnvSpec, GeneratedSplit, TraitDomains, WorldSpec,
)
from hopfield_nav.training.world_setup import (
    build_field, legacy_split, setup_world, specs_from_world, write_world_spec,
)

LAMBDAS = [5, 7]        # Npos = 35
SIZE = 4
OBS = 16


def _cfg(save_dir=None) -> TrainConfig:
    cfg = TrainConfig(
        env=EnvConfig(size=SIZE, observation_size=OBS),
        vectorhash=VectorHashConfig(lambdas=LAMBDAS, Np=40,
                                    static_vectorhash=True),
        envs_per_world=2, num_worlds=1, num_val_envs=2, device="cpu",
    )
    cfg.save_dir = str(save_dir) if save_dir else None
    return cfg


@pytest.fixture(scope="module")
def field():
    cfg = _cfg()
    torch.manual_seed(0)
    enc = torch.nn.Linear(int(np.sum(np.square(LAMBDAS))), 8)
    enc.eval()
    return build_field(cfg, enc)


def _split(field, seed=0):
    domains = TraitDomains(place=dom.Anywhere(), wall=dom.SeedRange(0, 100_000),
                           goal=dom.AnyCells(), size=dom.Sizes((SIZE,)))
    return gen.generate_split(field, EnvConfig(size=SIZE, observation_size=OBS),
                              domains, 2, 2, seed=seed, margin=5)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def test_world_spec_round_trips_through_disk(field, tmp_path):
    spec = WorldSpec(scaffold={"lambdas": LAMBDAS, "Npos": field.Npos},
                     generator="declared", split=_split(field))
    spec.write(tmp_path)
    back = WorldSpec.read(tmp_path)
    assert back.split.train == spec.split.train
    assert back.split.base_val == spec.split.base_val
    assert back.split.goal_cells_val == spec.split.goal_cells_val
    assert back.split.domains == spec.split.domains
    assert back.spec_hash() == spec.spec_hash()


def test_spec_hash_is_insensitive_to_key_order(field):
    spec = WorldSpec(scaffold={"a": 1, "b": 2}, generator="legacy",
                     split=_split(field))
    reordered = WorldSpec(scaffold={"b": 2, "a": 1}, generator="legacy",
                          split=spec.split)
    assert spec.spec_hash() == reordered.spec_hash()


def test_spec_hash_catches_a_hand_edit(field, tmp_path):
    spec = WorldSpec(scaffold={"lambdas": LAMBDAS}, generator="declared",
                     split=_split(field))
    path = spec.write(tmp_path)
    raw = json.load(open(path))
    raw["split"]["train"][0]["offset"] = [999, 999]
    json.dump(raw, open(path, "w"))
    with pytest.raises(ValueError, match="hash mismatch"):
        WorldSpec.read(tmp_path)


def test_checkpoint_summary_omits_the_resolved_lists(field):
    """A checkpoint carries domains and a hash, never the union.

    The union grows with every refresh tick once Phase 4 lands, and checkpoints
    are written far more often than a world changes.
    """
    spec = WorldSpec(scaffold={}, generator="declared", split=_split(field))
    summary = spec.summary("/tmp/world.json")
    assert set(summary) == {"spec_version", "generator", "spec_hash",
                            "domains", "world_json"}
    blob = json.dumps(summary)
    assert "base_val" not in blob and "used" not in blob


# ---------------------------------------------------------------------------
# Rebuilding
# ---------------------------------------------------------------------------

def test_rebuilt_envs_are_bit_identical(field, tmp_path):
    """Equal seeds is not the claim -- the wall arrays and goals must match."""
    spec = WorldSpec(scaffold={"Npos": field.Npos}, generator="declared",
                     split=_split(field))
    spec.write(tmp_path)
    env_cfg = EnvConfig(size=SIZE, observation_size=OBS)
    before = gen.build_envs(spec.split.base_val, env_cfg, "discrete")
    after = gen.build_envs(WorldSpec.read(tmp_path).split.base_val,
                           env_cfg, "discrete")
    for a, b in zip(before, after):
        assert np.array_equal(a._wall_code, b._wall_code)
        assert np.array_equal(a._codebook, b._codebook)
        assert a.goal_location == b.goal_location


def test_world_spec_for_finds_and_misses(tmp_path, field):
    assert cio.world_spec_for(tmp_path) is None
    spec = WorldSpec(scaffold={}, generator="legacy", split=_split(field))
    spec.write(tmp_path)
    assert cio.world_spec_for(tmp_path) is not None
    # A checkpoint path resolves to its own run directory.
    ckpt = tmp_path / "navigate_u1.pt"
    ckpt.write_bytes(b"")
    assert cio.world_spec_for(ckpt) is not None


# ---------------------------------------------------------------------------
# The legacy path records what it drew -- this is the §1.4 fix for ordinary runs
# ---------------------------------------------------------------------------

def test_legacy_split_matches_the_envs_it_describes(field):
    cfg = _cfg()
    rng = np.random.RandomState(0)
    worlds = [setup_world(cfg, None, 8, rng, role="train", field=field)]
    eval_world = setup_world(cfg, None, 8, rng, role="eval", field=field)
    split = legacy_split(cfg, field, worlds, eval_world)

    for spec, env, off in zip(split.train, worlds[0].envs, worlds[0].offsets):
        assert spec.wall_seed == env.seed
        assert spec.goal == env.goal_location
        assert spec.offset == tuple(off)
    assert len(split.base_val) == cfg.num_val_envs
    # Honest about what the historical path guaranteed, which is nothing.
    assert split.margin == 0
    assert isinstance(split.domains.place, dom.Anywhere)


def test_legacy_world_json_rebuilds_the_same_envs(field, tmp_path):
    """An ordinary run's envs become recoverable, offsets included."""
    cfg = _cfg(tmp_path)
    rng = np.random.RandomState(1)
    worlds = [setup_world(cfg, None, 8, rng, role="train", field=field)]
    eval_world = setup_world(cfg, None, 8, rng, role="eval", field=field)
    split = legacy_split(cfg, field, worlds, eval_world)
    write_world_spec(cfg, field, split, {"path": "x"}, generator="legacy")

    back = cio.world_spec_for(tmp_path)
    rebuilt = gen.build_envs(back.split.base_val,
                             EnvConfig(size=SIZE, observation_size=OBS),
                             "discrete")
    for env, off, spec in zip(rebuilt, eval_world.offsets, back.split.base_val):
        assert spec.offset == tuple(off), "the offset is the part replay loses"
    for env, orig in zip(rebuilt, eval_world.envs):
        assert np.array_equal(env._codebook, orig._codebook)
        assert env.goal_location == orig.goal_location


def test_legacy_diagnostics_expose_overlap(field, tmp_path):
    """The recording is useful, not just complete.

    The historical path enforces no train/val separation, so `min_place_gap` can
    come back negative -- the overlap measured in §1.3, now surfaced by any run
    that writes a spec rather than having to be discovered by an audit.
    """
    cfg = _cfg(tmp_path)
    rng = np.random.RandomState(2)
    worlds = [setup_world(cfg, None, 8, rng, role="train", field=field)]
    eval_world = setup_world(cfg, None, 8, rng, role="eval", field=field)
    split = legacy_split(cfg, field, worlds, eval_world)
    spec, _ = write_world_spec(cfg, field, split, {}, generator="legacy")
    d = spec.split.diagnostics
    assert "min_place_gap" in d and "min_wall_hamming" in d
    assert isinstance(d["min_place_gap"], int)
    assert 0.0 <= d["cosine"]["max"] <= 1.0001


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------

def test_missing_spec_falls_back_and_says_what_is_approximate(field, capsys):
    """No world.json must degrade, not raise -- 355 run dirs predate it."""
    cfg = _cfg()
    cfg.encoder_checkpoint = "nonexistent.pt"
    torch.manual_seed(0)
    enc = torch.nn.Linear(int(np.sum(np.square(LAMBDAS))), 8)
    enc.eval()
    envs, vh, offsets = cio.build_eval_world(cfg, enc, "cpu", spec=None)
    assert len(envs) == cfg.num_val_envs and len(offsets) == cfg.num_val_envs
    out = capsys.readouterr().out
    assert "no world.json" in out
    assert "offsets are not" in out, "the warning must name what is unreliable"


def test_spec_path_does_not_warn(field, tmp_path, capsys):
    cfg = _cfg(tmp_path)
    torch.manual_seed(0)
    enc = torch.nn.Linear(int(np.sum(np.square(LAMBDAS))), 8)
    enc.eval()
    spec = WorldSpec(scaffold={"Npos": field.Npos}, generator="declared",
                     split=_split(field))
    envs, vh, offsets = cio.build_eval_world(cfg, enc, "cpu", spec=spec)
    assert offsets == [s.offset for s in spec.split.base_val]
    assert "no world.json" not in capsys.readouterr().out


def test_npos_mismatch_is_refused(field, tmp_path):
    """Offsets recorded against one scaffold must not silently index another."""
    cfg = _cfg(tmp_path)
    torch.manual_seed(0)
    enc = torch.nn.Linear(int(np.sum(np.square(LAMBDAS))), 8)
    enc.eval()
    spec = WorldSpec(scaffold={"Npos": field.Npos + 100}, generator="declared",
                     split=_split(field))
    with pytest.raises(ValueError, match="different scaffold"):
        cio.build_eval_world(cfg, enc, "cpu", spec=spec)


# ---------------------------------------------------------------------------
# Discovery -- the §1.4 regression
# ---------------------------------------------------------------------------

def test_a_recorded_world_is_found_from_the_checkpoint_path(field, tmp_path,
                                                            capsys):
    """The bug this parameter exists for.

    Between Phase 3 and Phase 5 every caller passed three arguments, so a run
    with a truthful world.json beside it still took the RNG replay -- and the
    replay cannot recover offsets, which is the whole point of the file. Nothing
    failed; the numbers were just computed against envs training never used.
    """
    cfg = _cfg(tmp_path)
    torch.manual_seed(0)
    enc = torch.nn.Linear(int(np.sum(np.square(LAMBDAS))), 8)
    enc.eval()
    spec = WorldSpec(scaffold={"Npos": field.Npos}, generator="declared",
                     split=_split(field))
    spec.write(tmp_path)
    ckpt = tmp_path / "navigate_u1.pt"
    ckpt.write_bytes(b"")

    envs, vh, offsets = cio.build_eval_world(cfg, enc, "cpu", ckpt_path=ckpt)
    assert offsets == [s.offset for s in spec.split.base_val]
    assert [e.seed for e in envs] == [s.wall_seed for s in spec.split.base_val]
    assert "no world.json" not in capsys.readouterr().out

    # ...and the run directory works as well as a checkpoint inside it.
    _, _, from_dir = cio.build_eval_world(cfg, enc, "cpu", ckpt_path=tmp_path)
    assert from_dir == offsets


def test_the_replay_still_runs_and_still_warns(field, tmp_path, capsys):
    """A pre-Phase-3 run has no record; it must still evaluate, and still say so."""
    cfg = _cfg(tmp_path)
    torch.manual_seed(0)
    enc = torch.nn.Linear(int(np.sum(np.square(LAMBDAS))), 8)
    enc.eval()
    ckpt = tmp_path / "navigate_u1.pt"
    ckpt.write_bytes(b"")
    envs, vh, offsets = cio.build_eval_world(cfg, enc, "cpu", ckpt_path=ckpt)
    assert len(envs) == cfg.num_val_envs == len(offsets)
    assert "no world.json" in capsys.readouterr().out


def test_an_explicit_spec_beats_discovery(field, tmp_path):
    """For evaluating one run's checkpoint against another run's world."""
    cfg = _cfg(tmp_path)
    torch.manual_seed(0)
    enc = torch.nn.Linear(int(np.sum(np.square(LAMBDAS))), 8)
    enc.eval()
    WorldSpec(scaffold={"Npos": field.Npos}, generator="declared",
              split=_split(field, seed=0)).write(tmp_path)
    other = WorldSpec(scaffold={"Npos": field.Npos}, generator="declared",
                      split=_split(field, seed=9))
    ckpt = tmp_path / "navigate_u1.pt"
    ckpt.write_bytes(b"")
    _, _, offsets = cio.build_eval_world(cfg, enc, "cpu", spec=other,
                                         ckpt_path=ckpt)
    assert offsets == [s.offset for s in other.split.base_val]


def test_no_eval_entry_point_resolves_a_world_without_the_record():
    """Discovery is only automatic if the call sites actually feed it.

    The four here were each silently on the replay path for two phases -- they
    called `build_eval_world` with three arguments and got a fresh offset draw.
    A fifth landing in the same state is the failure mode this guards, and it
    reads the source because there is no runtime moment at which "nobody passed
    it" is distinguishable from "this run has no record".

    Three spellings are sanctioned, and all three go through `world_spec_for`:
    `build_eval_world(..., ckpt_path=)`, and the `eval_world_for_split` /
    `eval_env_set` helpers the split-aware drivers use. A bare
    `build_eval_world(cfg, encoder, device)` is the shape that was wrong.
    """
    import inspect
    import hopfield_nav.eval_all as eval_all
    from analysis import trajectories
    from analysis.continual import agenthash
    from analysis.phase_decoding import rollout

    def _args(text: str) -> str:
        """The call's argument list, to its *balanced* close paren.

        Naive `.index(")")` stops inside `str(device)`, which every one of these
        calls contains -- and the test then fails on correct code.
        """
        depth, out = 1, []
        for ch in text:
            depth += (ch == "(") - (ch == ")")
            if depth == 0:
                break
            out.append(ch)
        return "".join(out)

    for mod in (eval_all, trajectories, agenthash, rollout):
        src = inspect.getsource(mod)
        # Every bare build_eval_world call must name where the record lives.
        for call in src.split("build_eval_world(")[1:]:
            head = _args(call)
            assert "ckpt_path=" in head or "spec=" in head, (
                f"{mod.__name__} calls build_eval_world without a ckpt_path, so "
                f"it silently evaluates on a fresh offset draw: ...{head}")
        # ...and each driver resolves its world through one of the three.
        assert any(f"{name}(" in src for name in
                   ("build_eval_world", "eval_world_for_split", "eval_env_set")), (
            f"{mod.__name__} resolves its eval world some other way, which means "
            "it may not be reading the run's world.json at all")
