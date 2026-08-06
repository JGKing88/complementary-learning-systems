#!/usr/bin/env python3
"""Classify run directories so junk can be told apart from results.

The agent-checkpoint tree is 8.5 GB across ~350 directories and there has never
been a way to ask anything about it. Answering "which of these are junk" meant
`torch.load` on every one. With manifests it is a `json.loads`, which is what
makes a classifier practical at all.

Categories
----------
``test``       Written by the test suite. Identified by an encoder path in
               pytest's tmp_path_factory layout. 36 existed on 2026-08-06:
               every `./run_tests.sh` used to leave one behind, because
               `train_phase_b_only` had no `--save_dir` with which to honour the
               sandbox fixture's CLS_RUNS. Each holds an agent trained against a
               `/tmp/pytest-of-.../tiny_encoder.pt` that no longer exists.
``empty``      No `.pt` at all.
``unfinished`` status "running" with no checkpoint written for `--stale-days`.
               Nothing runs on SIGKILL, so "running" means "not known to have
               finished"; checkpoint mtime distinguishes a dead job from a live
               one. Explicitly NOT junk: cancelling a doomed phase-A variant
               partway and keeping its checkpoints is the normal workflow here,
               so 241 of ~350 runs are in this state and many are results.
               Listed so the tree can be understood, not so it can be emptied.
``orphaned``   Finished, but its encoder is gone -- so the run cannot be
               re-evaluated or reproduced. NOT junk: the checkpoints are still
               loadable and the numbers still real. Reported for triage only.
``keep``       Everything else.

Nothing is deleted without `--delete`, and `--delete` refuses any category but
`test` and `empty` unless it is named explicitly. The default output is a
report: this is a tool for looking, and the expensive mistake it could make is
removing a run whose figure is in a paper.

Usage:
    python scripts/gc_runs.py                       # report
    python scripts/gc_runs.py --category test -v    # list one category
    python scripts/gc_runs.py --delete test         # remove the pytest junk
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run_manifest                                          # noqa: E402
from cls_paths import checkpoints_dir, rnn_checkpoints_dir   # noqa: E402

CATEGORIES = ("test", "empty", "unfinished", "orphaned", "keep")

# Only these may be removed without being named. A run that merely lost its
# encoder still holds real results.
SAFE_TO_DELETE = ("test", "empty")

# pytest's tmp_path_factory layout, which is what all 36 known droppings carry:
#   /tmp/pytest-of-jackking/pytest-48/cls_smoke0/tiny_encoder.pt
# Deliberately narrower than "the path is under /tmp": staging an encoder in
# /tmp for a one-off run is a legitimate thing to do, and this decides what
# `--delete test` removes. Matching the layout rather than the root also keeps
# it working when TMPDIR points somewhere else.
_PYTEST_TMPDIR_RE = re.compile(r"/pytest-of-[^/]+/pytest-\d+/")


def _dir_size(path: str) -> int:
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for f in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total


def _last_activity(run_dir: str, manifest: dict | None) -> float:
    """When this run last did something, as a unix timestamp.

    Reads the *checkpoint* mtimes only, deliberately. The obvious version --
    newest mtime of anything in the directory -- is wrong now that manifests
    exist: `backfill_manifests.py` writes a `run.json` into every legacy
    directory, which updates both that file's mtime and the directory's, so
    every run in the tree looks like it was active seconds ago. The first run
    of this script after a backfill reported zero unfinished runs for exactly
    that reason.

    The `.pt` files are the artifacts, and nothing rewrites them. Their mtimes
    survive backfilling, re-evaluation, and anything else done to a run
    afterwards. `created` from the manifest is the fallback for a directory
    whose checkpoints are unreadable -- backfill recorded it from the directory
    mtime *before* writing, so it too predates the backfill.
    """
    stamps = []
    for p in glob.glob(os.path.join(run_dir, "*.pt")):
        try:
            stamps.append(os.path.getmtime(p))
        except OSError:
            pass
    if stamps:
        return max(stamps)
    if manifest and manifest.get("created"):
        try:
            from datetime import datetime
            return datetime.fromisoformat(manifest["created"]).timestamp()
        except ValueError:
            pass
    return os.path.getmtime(run_dir)


def classify(run_dir: str, *, stale_days: float) -> tuple[str, str]:
    """(category, one-line reason) for a run directory."""
    if not glob.glob(os.path.join(run_dir, "*.pt")):
        return "empty", "no .pt files"

    m = run_manifest.read(run_dir)
    if m is None:
        # Pre-manifest and un-backfilled. Refusing to guess is the point: a
        # directory this tool cannot read is a directory it must not delete.
        return "keep", "no manifest (run backfill_manifests.py first)"

    enc = (m.get("encoder") or {}).get("path") or ""
    if _PYTEST_TMPDIR_RE.search(enc):
        return "test", f"encoder is a pytest tmpdir: {enc}"

    if m.get("status") == run_manifest.STATUS_RUNNING:
        age_days = (time.time() - _last_activity(run_dir, m)) / 86400.0
        if age_days > stale_days:
            return "unfinished", f"status=running, no checkpoint written in {age_days:.0f}d"
        return "keep", "status=running, recently active"

    if enc and not os.path.exists(enc):
        return "orphaned", f"encoder missing: {enc}"

    return "keep", ""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=None,
                   help="A single tree of run directories. Default: both the "
                        "agent-checkpoint and RNN-checkpoint trees.")
    p.add_argument("--stale-days", type=float, default=7.0,
                   help="A status=running run with no checkpoint written for "
                        "longer than this counts as unfinished rather than "
                        "live. Default 7.")
    p.add_argument("--category", choices=CATEGORIES, default=None,
                   help="List only this category.")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="One line per run, with its reason.")
    p.add_argument("--delete", metavar="CATEGORY", default=None,
                   help=f"Delete a category. Without --force, only "
                        f"{'/'.join(SAFE_TO_DELETE)} are allowed.")
    p.add_argument("--force", action="store_true",
                   help="Permit --delete of a category that holds real results.")
    args = p.parse_args()

    roots = [args.root] if args.root else [str(checkpoints_dir()),
                                           str(rnn_checkpoints_dir())]

    found: dict[str, list[tuple[str, str, int]]] = {c: [] for c in CATEGORIES}
    for root in roots:
        if not os.path.isdir(root):
            continue
        for d in sorted(glob.glob(os.path.join(root, "*/"))):
            d = d.rstrip("/")
            category, reason = classify(d, stale_days=args.stale_days)
            found[category].append((d, reason, _dir_size(d)))

    print(f"{'category':<12} {'runs':>5} {'size':>9}")
    print("-" * 28)
    for c in CATEGORIES:
        rows = found[c]
        if not rows:
            continue
        size = sum(r[2] for r in rows) / 1e9
        print(f"{c:<12} {len(rows):>5} {size:>8.2f}G")

    show = [args.category] if args.category else (
        [c for c in CATEGORIES if c != "keep"] if args.verbose else [])
    for c in show:
        if not found[c]:
            continue
        print(f"\n--- {c} ---")
        for d, reason, size in found[c]:
            print(f"  {os.path.basename(d):<48} {size/1e6:>8.1f}M  {reason}")

    if args.delete:
        if args.delete not in CATEGORIES:
            print(f"\nunknown category {args.delete!r}")
            return 2
        if args.delete not in SAFE_TO_DELETE and not args.force:
            print(f"\nrefusing to delete {args.delete!r}: it may hold real "
                  f"results. Re-run with --force if you are sure.")
            return 2
        rows = found[args.delete]
        freed = sum(r[2] for r in rows) / 1e9
        for d, _reason, _size in rows:
            shutil.rmtree(d)
        print(f"\ndeleted {len(rows)} {args.delete} runs, freed {freed:.2f} GB")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
