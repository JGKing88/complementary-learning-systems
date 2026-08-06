#!/usr/bin/env python3
"""Import every entry point and, where it has a parser, run it with --help.

A stale import in a script that nobody runs day-to-day surfaces only when
somebody runs it, which in this repo has historically been months later and in
the middle of a figure deadline. The refactor's file moves rewrite nearly every
import in the tree, so this walks the whole set rather than trusting the test
suite -- most of these modules are executed by no test.

An entry point is a module with an ``if __name__ == "__main__":`` guard, outside
``tests/``. `--help` is the cheapest execution that still runs module-level
imports *and* argparse construction; three scripts once died at the latter with
a `TypeError` on an unescaped `%` in a help string.

The five ``analysis/schematics/make_*.py`` have no guard at all -- their bodies
run at import -- so the guard test skipped them entirely, which is the worst
case: figure scripts nobody runs for months are exactly what this exists to
catch. They are executed in full instead, with ``CLS_RUNS`` pointed at a scratch
directory so they write nowhere real. That is stronger coverage than `--help`,
not weaker.

Usage:
    python scripts/check_entry_points.py [--list]
"""
from __future__ import annotations

import argparse
import ast
import os
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints", ".git", "notebooks", "docs"}
SKIP_PARTS = {"tests"}


def has_main_guard(path: str) -> bool:
    try:
        tree = ast.parse(open(path, encoding="utf-8").read())
    except (SyntaxError, UnicodeDecodeError):
        return False
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "__name__"
            and any(
                isinstance(c, ast.Constant) and c.value == "__main__"
                for c in test.comparators
            )
        ):
            return True
    return False


# Guard-less script directories: every module here runs its body on import, so
# it is checked by running it rather than by `--help`.
RUN_WHOLE = ("analysis/schematics",)


def find_entry_points() -> list[tuple[str, bool]]:
    """(dotted module, needs_full_run) for every non-test script."""
    found: list[tuple[str, bool]] = []
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        rel_dir = os.path.relpath(dirpath, REPO_ROOT)
        parts = [] if rel_dir == "." else rel_dir.split(os.sep)
        if SKIP_PARTS & set(parts):
            continue
        guardless_dir = rel_dir.replace(os.sep, "/") in RUN_WHOLE
        for filename in filenames:
            if not filename.endswith(".py") or filename == "__init__.py":
                continue
            path = os.path.join(dirpath, filename)
            guarded = has_main_guard(path)
            if not guarded and not guardless_dir:
                continue
            found.append((".".join(parts + [filename[:-3]]), not guarded))
    return sorted(found)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true",
                        help="Print the entry points and exit.")
    args = parser.parse_args()

    modules = find_entry_points()
    if args.list:
        for m, full in modules:
            print(f"{m}{'  (run in full: no __main__ guard)' if full else ''}")
        print(f"\n{len(modules)} entry points")
        return 0

    scratch = tempfile.mkdtemp(prefix="cls_entrypoint_check_")
    env = dict(os.environ, MPLBACKEND="Agg", CLS_RUNS=scratch,
               WANDB_MODE="disabled")
    failures: list[tuple[str, str]] = []
    for module, needs_full_run in modules:
        argv = [sys.executable, "-m", module] + ([] if needs_full_run else ["--help"])
        proc = subprocess.run(
            argv,
            cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=300,
        )
        # A module with no argparse exits non-zero on --help, or ignores it and
        # starts working. Either is fine -- what must not happen is a failure
        # during import, which shows up as a traceback naming an import error.
        blob = proc.stdout + proc.stderr
        if needs_full_run:
            # No argparse to hide behind: it either completed or it did not.
            broken = proc.returncode != 0
        else:
            broken = (
                "ModuleNotFoundError" in blob
                or "ImportError" in blob
                or "AttributeError: module" in blob
                or ("Traceback" in blob and "argparse" not in blob and proc.returncode != 0
                    and "usage:" not in blob)
            )
        status = "BROKEN" if broken else ("ran" if needs_full_run else "ok")
        if broken:
            failures.append((module, blob.strip().splitlines()[-1] if blob.strip() else "?"))
        print(f"  {status:<6} {module}", flush=True)

    shutil.rmtree(scratch, ignore_errors=True)
    print(f"\n{len(modules)} entry points, {len(failures)} broken")
    for module, last in failures:
        print(f"  {module}: {last}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
