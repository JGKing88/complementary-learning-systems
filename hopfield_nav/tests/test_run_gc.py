"""Backfill and GC: the two tools that read the run tree and can delete from it.

`gc_runs.classify` decides what `--delete` removes, and the tree it runs on
holds every result this project has produced. So the tests here are weighted
towards the ways a classifier can be *confidently wrong*: a directory it cannot
read, a category label that invites the wrong deletion, and the one bug this
pair actually had -- backfilling destroyed the signal the classifier reads.

`scripts/` is not a package (no `__init__.py`, no layer in the table), so the
modules are loaded by path.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time

import pytest

import run_manifest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load(name: str):
    path = os.path.join(REPO_ROOT, "scripts", f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gc_runs = _load("gc_runs")
backfill = _load("backfill_manifests")


def _run_dir(tmp_path, name, files=("phase_a_u2.pt",), *, age_days=0.0):
    d = tmp_path / name
    d.mkdir()
    when = time.time() - age_days * 86400
    for f in files:
        p = d / f
        p.write_bytes(b"x")
        os.utime(p, (when, when))
    return d


# ---------------------------------------------------------------------------
# backfill: reading a legacy directory
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dirname,kind,name", [
    ("navigate_amber-plant-56", "navigate", "amber-plant-56"),
    ("store_20260806_125705", "store", "20260806_125705"),
    # The pre-2026-08-06 spellings. ~150 directories are named this way and
    # backfill still has to parse them.
    ("phase_a_only_amber-plant-56", "phase_a_only", "amber-plant-56"),
    ("phase_b_only_20260806_125705", "phase_b_only", "20260806_125705"),
    ("phased_lively-surf-104", "phased", "lively-surf-104"),
    # No prefix at all: hopfield_nav.train writes a bare wandb run name, and
    # ~100 directories look like this.
    ("revived-water-17", "train", "revived-water-17"),
])
def test_infer_kind_takes_the_longest_prefix(dirname, kind, name):
    """`store_x` must not be read as kind `train` with a funny name.

    RUN_KINDS has an empty prefix for `train`, so a naive scan matches it
    against everything.
    """
    assert backfill.infer_kind(dirname, rnn=False) == (kind, name)


def test_a_new_kind_needs_no_entry_in_run_kinds():
    """Adding a trainer must not require editing cls_paths.

    RUN_KINDS describes how the *existing* tree is laid out -- it is read
    backwards by `infer_kind` to parse legacy directory names, which is why
    rows are listed even when `default_layout` reproduces them.
    It is not a registry of permitted kinds, and `run_dir` used to raise on an
    unlisted one, which made every new trainer edit a file it has no other
    reason to touch.
    """
    from cls_paths import RUN_KINDS, run_dir

    assert "phase_c_only" not in RUN_KINDS
    got = run_dir("phase_c_only", "zesty-fog-9")
    assert got.name == "phase_c_only_zesty-fog-9"
    assert got.parent.name == "agent_ckpts"


def test_the_regular_rows_of_run_kinds_agree_with_the_default():
    """The listed-but-regular kinds must not drift from the convention.

    If someone edits one of those rows, `run_dir` and `default_layout` start
    disagreeing for a kind that looks ordinary, and new runs land beside the
    old ones instead of in them.
    """
    from cls_paths import RUN_KINDS, default_layout

    irregular = {k for k, v in RUN_KINDS.items() if v != default_layout(k)}
    # Exactly two rows deviate, and both are historical. Everything else in the
    # table is listed only so backfill can parse legacy directory names, and
    # must therefore still agree with the convention.
    assert irregular == {"train", "rnn"}
    assert RUN_KINDS["train"] == ("agent_ckpts", "")
    assert RUN_KINDS["rnn"] == ("checkpoint_rnn", "")


def test_infer_kind_is_ordered_not_just_lucky(monkeypatch):
    """The longest-prefix sort has to be tested with prefixes that overlap.

    No real prefix is a prefix of another (`phased_` does not match
    `phase_a_only_...`), so the ordering happens to be irrelevant today and
    every mutation of it survives -- the sort reads as tested when it is not.
    It still guards a real hazard, and one that is now closer: the table gained
    `navigate_` and `store_` in the 2026-08-06 rename, so a future `store_bc_`
    or `navigate_v2_` would make unordered scanning silently file every run of
    the longer kind under the shorter one. This pins the property directly.
    """
    # Built with the same key the module uses, so mutating that key breaks
    # this test rather than being papered over by a hand-ordered list.
    overlapping = sorted(
        [("phase_c_", "phase_c"), ("phase_c_only_", "phase_c_only")],
        key=backfill._PREFIX_ORDER,
    )
    monkeypatch.setattr(backfill, "_PREFIXES", overlapping)
    assert backfill.infer_kind("phase_c_only_zesty-fog-9", rnn=False) == (
        "phase_c_only", "zesty-fog-9")
    assert backfill.infer_kind("phase_c_zesty-fog-9", rnn=False) == (
        "phase_c", "zesty-fog-9")


def test_backfilled_checkpoints_match_the_legacy_regex(tmp_path):
    """The backfilled list is what checkpoints_in would have globbed."""
    d = _run_dir(tmp_path, "phase_a_only_x",
                 ("phase_a_u2.pt", "phase_a_u10.pt", "phase_a_only_final.pt"))
    data = backfill.build(str(d), rnn=False)
    assert [(e["file"], e["update"]) for e in data["checkpoints"]] == [
        ("phase_a_u2.pt", 2), ("phase_a_u10.pt", 10),
        ("phase_a_only_final.pt", None),
    ]
    run_manifest.write(d, data)
    assert [u for u, _ in run_manifest.checkpoints_in(d)] == [2, 10]


def test_backfill_marks_provenance(tmp_path):
    """A backfilled manifest must be distinguishable from a live one.

    argv/git/wandb are unrecoverable, and are left null rather than guessed.
    A reader has to be able to tell "no SHA was recorded" from "the tree was
    clean", which is what provenance is for.
    """
    d = _run_dir(tmp_path, "phase_a_only_x")
    data = backfill.build(str(d), rnn=False)
    assert data["provenance"] == "backfilled"
    assert data["git"] is None and data["argv"] is None and data["wandb"] is None


def test_backfill_status_from_the_final_checkpoint(tmp_path):
    finished = _run_dir(tmp_path, "a", ("phase_a_u2.pt", "phase_a_only_final.pt"))
    partial = _run_dir(tmp_path, "b", ("phase_a_u2.pt",))
    assert backfill.build(str(finished), rnn=False)["status"] == run_manifest.STATUS_DONE
    assert backfill.build(str(partial), rnn=False)["status"] == run_manifest.STATUS_RUNNING


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------

def test_a_directory_with_no_manifest_is_kept(tmp_path):
    """Unreadable => untouchable. This is the safety property that matters.

    Anything the classifier cannot explain must land in `keep`, never in a
    deletable category, because the alternative is deleting a run whose figure
    is in a paper.
    """
    d = _run_dir(tmp_path, "mystery")
    category, reason = gc_runs.classify(str(d), stale_days=7)
    assert category == "keep"
    assert "no manifest" in reason


def test_pytest_droppings_are_identified(tmp_path):
    d = _run_dir(tmp_path, "phase_b_only_20260806_002747")
    run_manifest.begin(d, kind="phase_b_only", name="x", config={},
                       encoder={"path": "/tmp/pytest-of-jackking/pytest-48/"
                                        "cls_smoke0/tiny_encoder.pt"})
    run_manifest.finish(d)
    assert gc_runs.classify(str(d), stale_days=7)[0] == "test"


def test_running_run_is_unfinished_only_once_stale(tmp_path):
    fresh = _run_dir(tmp_path, "fresh", age_days=0)
    stale = _run_dir(tmp_path, "stale", age_days=90)
    for d in (fresh, stale):
        run_manifest.begin(d, kind="phase_a_only", name=d.name, config={},
                           encoder={"path": __file__})       # exists
    assert gc_runs.classify(str(fresh), stale_days=7)[0] == "keep"
    assert gc_runs.classify(str(stale), stale_days=7)[0] == "unfinished"


def test_finished_run_with_a_missing_encoder_is_orphaned_not_junk(tmp_path):
    d = _run_dir(tmp_path, "phase_a_only_old")
    run_manifest.begin(d, kind="phase_a_only", name="old", config={},
                       # Not tmp_path: that IS a pytest tmpdir, so it would classify
                       # as `test` and this would test the wrong branch.
                       encoder={"path": "/orcd/pool/003/jackking/cls_runs/"
                                        "encoders/deleted_run/encoder_best.pt"})
    run_manifest.finish(d)
    category, _ = gc_runs.classify(str(d), stale_days=7)
    assert category == "orphaned"
    assert category not in gc_runs.SAFE_TO_DELETE, (
        "an orphaned run still holds real checkpoints and real numbers")


def test_only_test_and_empty_are_deletable_without_force():
    """The categories that hold results must not be removable by accident.

    241 of ~350 runs are `unfinished`, holding 6.9 GB -- and cancelling a doomed
    phase-A variant partway while keeping its checkpoints is the normal workflow
    here, so most of those are results.
    """
    assert set(gc_runs.SAFE_TO_DELETE) == {"test", "empty"}
    for c in ("unfinished", "orphaned", "keep"):
        assert c not in gc_runs.SAFE_TO_DELETE


# ---------------------------------------------------------------------------
# The bug: backfilling destroyed the staleness signal
# ---------------------------------------------------------------------------

def test_last_activity_ignores_the_manifest(tmp_path):
    """Writing run.json must not make a three-month-old run look active.

    This is a regression test for a real failure: the first gc_runs run after
    backfilling all 354 directories reported *zero* unfinished runs, because
    the freshly written run.json (and the directory mtime it bumped) were the
    newest thing in every directory. Reading checkpoint mtimes only is the fix.
    """
    d = _run_dir(tmp_path, "old_run", ("phase_a_u2.pt",), age_days=90)
    run_manifest.begin(d, kind="phase_a_only", name="old", config={})  # now
    age_days = (time.time() - gc_runs._last_activity(str(d), run_manifest.read(d))) / 86400
    assert age_days > 80, f"manifest mtime leaked into the age ({age_days:.1f}d)"
    assert gc_runs.classify(str(d), stale_days=7)[0] == "unfinished"


def test_last_activity_falls_back_to_created_without_checkpoints(tmp_path):
    d = tmp_path / "no_ckpts"
    d.mkdir()
    run_manifest.begin(d, kind="train", name="x", config={})
    data = run_manifest.read(d)
    data["created"] = "2026-01-01T00:00:00"
    run_manifest.write(d, data)
    age_days = (time.time() - gc_runs._last_activity(str(d), run_manifest.read(d))) / 86400
    assert age_days > 100


def test_backfill_created_predates_the_backfill(tmp_path):
    """`created` is read from the directory mtime *before* run.json is written.

    If it were read afterwards it would be the backfill time, and the fallback
    in `_last_activity` would carry the same bug the main path had.
    """
    d = _run_dir(tmp_path, "phase_a_only_x", age_days=90)
    os.utime(d, (time.time() - 90 * 86400,) * 2)
    data = backfill.build(str(d), rnn=False)
    run_manifest.write(d, data)
    import datetime
    created = datetime.datetime.fromisoformat(json.loads(
        run_manifest.path_for(d).read_text())["created"])
    assert (datetime.datetime.now() - created).days > 80
