"""The run manifest: what it records, and that it can never break a run.

Two things are being tested, and they pull in opposite directions.

The first is that the manifest is *correct* -- statuses, checkpoint lists,
the encoder digest.

The second is that it is *inert*. The manifest is an index, not the source of
truth: ~350 run directories predate it, the config embedded in each ``.pt``
remains authoritative, and a metadata file that can abort a seven-day training
job is worse than no metadata file. So every writer swallows its own errors,
and every reader falls back. Those failure paths are where most of these tests
are, because they are the ones nothing else would notice breaking.

The equivalence test (`test_fallback_matches_the_regex_it_replaced`) is the
load-bearing one: `analysis.trajectories` used to own a regex over checkpoint
basenames, and 2503 existing checkpoint files across ten naming conventions are
still discovered by it. Replacing that with a manifest lookup is only safe if
the fallback path is byte-identical to what it replaced, so it is checked
against the real filename population rather than against invented examples.
"""
from __future__ import annotations

import json
import os
import re

import pytest

import run_manifest


# The ten naming conventions actually present in the run tree, with the counts
# observed on 2026-08-06. `phase_a_only_final.pt` and friends carry no update
# number and must stay excluded -- the trajectory figure indexes rows by update.
REAL_CHECKPOINT_NAMES = [
    "phase_a_u1495.pt",            # x1495
    "hopfield_nav_update829.pt",   # x829
    "phase_a_only_final.pt",       # x56   -- no update number
    "phase_b_only_final.pt",       # x40   -- no update number
    "phase_b_u35.pt",              # x35
    "phase_c_u31.pt",              # x31
    "final.pt",                    # x8    -- no update number
    "phased_v3_final.pt",          # x5    -- no update number
    "phased_final.pt",             # x3    -- no update number
    "phase_c_only_final.pt",       # x1    -- no update number
]

# Verbatim from analysis/trajectories.py before the manifest replaced it.
ORIGINAL_CKPT_RE = re.compile(r"(?:_u|_update)(\d+)\.pt$")


def _touch(d, name, body=b"x"):
    p = os.path.join(str(d), name)
    with open(p, "wb") as fh:
        fh.write(body)
    return p


@pytest.fixture
def run(tmp_path):
    """A run directory with a manifest, as a trainer would leave it mid-run."""
    run_manifest.begin(tmp_path, kind="phase_a_only", name="testy-run-1",
                       config={"seed": 7, "env": {"size": 8}})
    return tmp_path


# ---------------------------------------------------------------------------
# What it records
# ---------------------------------------------------------------------------

def test_begin_records_a_running_run(run):
    m = run_manifest.read(run)
    assert m["kind"] == "phase_a_only"
    assert m["name"] == "testy-run-1"
    assert m["status"] == run_manifest.STATUS_RUNNING
    assert m["finished"] is None
    assert m["config"]["seed"] == 7
    assert m["checkpoints"] == []
    # Written at the *start*, so a run that dies is still identifiable. If this
    # ever moves to the end, the runs you most need to identify lose their
    # manifest -- which is the failure mode the whole module exists to fix.
    assert m["created"]


def test_finish_marks_done(run):
    run_manifest.finish(run)
    m = run_manifest.read(run)
    assert m["status"] == run_manifest.STATUS_DONE
    assert m["finished"] is not None


def test_a_killed_run_stays_running(run):
    """No finish() call => status stays `running`, which is the crash signal.

    Nothing can be done about SIGKILL, so `running` means "not known to have
    finished" rather than "alive". gc_runs resolves it by mtime.
    """
    assert run_manifest.read(run)["status"] == run_manifest.STATUS_RUNNING


def test_record_checkpoint_appends_sorted_and_deduped(run):
    run_manifest.record_checkpoint(run, "phase_a_u4.pt", 4)
    run_manifest.record_checkpoint(run, "phase_a_u2.pt", 2)
    run_manifest.record_checkpoint(run, "phase_a_only_final.pt")
    # A rerun overwriting the same update must not double up.
    run_manifest.record_checkpoint(run, "phase_a_u2.pt", 2)

    entries = run_manifest.read(run)["checkpoints"]
    assert [e["file"] for e in entries] == [
        "phase_a_u2.pt", "phase_a_u4.pt", "phase_a_only_final.pt",
    ]
    assert [e["update"] for e in entries] == [2, 4, None]


def test_record_checkpoint_stores_basenames_not_paths(run):
    """Manifests must survive their directory being moved or renamed."""
    run_manifest.record_checkpoint(run, "/somewhere/else/phase_a_u9.pt", 9)
    assert run_manifest.read(run)["checkpoints"][0]["file"] == "phase_a_u9.pt"


def test_encoder_identity_distinguishes_encoders(tmp_path):
    a = _touch(tmp_path, "a.pt", b"encoder-alpha")
    b = _touch(tmp_path, "b.pt", b"encoder-beta")
    assert run_manifest.file_digest(a) != run_manifest.file_digest(b)
    assert run_manifest.file_digest(a) == run_manifest.file_digest(a)
    # This is the point: a path can be resolved to the wrong file by
    # _resolve_encoder_path's six-candidate search, and the digest is what
    # makes that detectable rather than silent.
    assert run_manifest.encoder_identity(a)["sha256"] == run_manifest.file_digest(a)


def test_encoder_identity_survives_a_missing_encoder(tmp_path):
    """4 of 25 sampled checkpoints point at an encoder that no longer exists."""
    ident = run_manifest.encoder_identity(str(tmp_path / "gone.pt"))
    assert ident["path"].endswith("gone.pt")
    assert ident["sha256"] is None


# ---------------------------------------------------------------------------
# Reading, and the legacy fallback
# ---------------------------------------------------------------------------

def test_checkpoints_in_prefers_the_manifest(run):
    for name, update in [("phase_a_u2.pt", 2), ("phase_a_u4.pt", 4)]:
        _touch(run, name)
        run_manifest.record_checkpoint(run, name, update)
    # A file the manifest does not list is not returned, even though the
    # glob fallback would have found it. The manifest is authoritative when
    # present -- that is what makes a future naming scheme free.
    _touch(run, "phase_a_u6.pt")
    assert [u for u, _ in run_manifest.checkpoints_in(run)] == [2, 4]


def test_checkpoints_in_drops_deleted_files(run):
    _touch(run, "phase_a_u2.pt")
    run_manifest.record_checkpoint(run, "phase_a_u2.pt", 2)
    run_manifest.record_checkpoint(run, "phase_a_u4.pt", 4)   # never written
    assert [u for u, _ in run_manifest.checkpoints_in(run)] == [2]


def test_fallback_matches_the_regex_it_replaced(tmp_path):
    """The no-manifest path reproduces analysis.trajectories' old regex exactly.

    Checked against the real filename population, not invented names: 2503
    checkpoints across ten conventions live in directories that will never have
    a manifest, and they are read through this path forever.
    """
    for name in REAL_CHECKPOINT_NAMES:
        _touch(tmp_path, name)

    expected = sorted(
        (int(ORIGINAL_CKPT_RE.search(n).group(1)), n)
        for n in REAL_CHECKPOINT_NAMES if ORIGINAL_CKPT_RE.search(n)
    )
    got = [(u, os.path.basename(p)) for u, p in run_manifest.checkpoints_in(tmp_path)]
    assert got == expected
    # Guard against a vacuous pass: the population must contain both kinds.
    assert 0 < len(expected) < len(REAL_CHECKPOINT_NAMES)


def test_read_returns_none_for_a_legacy_directory(tmp_path):
    assert run_manifest.read(tmp_path) is None


def test_corrupt_manifest_degrades_to_the_fallback(tmp_path):
    """A truncated run.json must not take the figure pipeline down with it."""
    _touch(tmp_path, "phase_a_u3.pt")
    run_manifest.path_for(tmp_path).write_text("{not json")
    assert run_manifest.read(tmp_path) is None
    assert [u for u, _ in run_manifest.checkpoints_in(tmp_path)] == [3]


def test_manifest_with_empty_checkpoint_list_falls_back(tmp_path):
    """A run killed before its first checkpoint still lists its files.

    `begin` writes `checkpoints: []`, so preferring the manifest blindly would
    report no checkpoints for any run whose manifest write survived but whose
    record_checkpoint calls did not.
    """
    run_manifest.begin(tmp_path, kind="train", name="x", config={})
    _touch(tmp_path, "hopfield_nav_update50.pt")
    assert [u for u, _ in run_manifest.checkpoints_in(tmp_path)] == [50]


# ---------------------------------------------------------------------------
# Inertness: no manifest failure may reach the training loop
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("call", [
    lambda p: run_manifest.begin(p, kind="train", name="n", config={}),
    lambda p: run_manifest.record_checkpoint(p, "x.pt", 1),
    lambda p: run_manifest.finish(p),
    lambda p: run_manifest.read(p),
])
def test_writers_never_raise_on_an_unwritable_directory(tmp_path, call):
    """A read-only or vanished output directory warns; it does not abort.

    Checked for every entry point rather than one, because the failure is a
    seven-day job dying on its metadata file.
    """
    blocked = tmp_path / "ro"
    blocked.mkdir()
    blocked.chmod(0o500)
    try:
        call(blocked / "run")            # cannot be created
    finally:
        blocked.chmod(0o700)


def test_read_of_a_directory_that_is_a_file(tmp_path):
    not_a_dir = _touch(tmp_path, "file.pt")
    assert run_manifest.read(not_a_dir) is None


def test_write_is_atomic(run):
    """os.replace, so a reader never sees a half-written manifest."""
    run_manifest.record_checkpoint(run, "phase_a_u1.pt", 1)
    leftovers = [f for f in os.listdir(run) if f.endswith(".tmp")]
    assert not leftovers, f"temp files left behind: {leftovers}"
    json.loads(run_manifest.path_for(run).read_text())     # parses
