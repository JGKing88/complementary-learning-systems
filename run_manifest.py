"""What a training run *was*, recorded as JSON beside its checkpoints.

Why this module is top-level
----------------------------
Same reason as ``cls_paths``: the five trainers write manifests, the figure
pipelines and the maintenance scripts read them, and ``scripts/`` is not a
package with a layer. A module inside ``hopfield_nav`` would be unreachable
from half of those. This one sits below everything with no project import but
``cls_paths``.

Why it exists
-------------
Before this, a run was an *emergent* property of three unrelated things: a
directory name encoding phase and identity, a filename convention inside it,
and a path string buried in a pickle. Everything downstream was a parser
reconstructing what should have been recorded. Concretely, all of these were
true at once:

- **Ten checkpoint filename conventions** across 2503 files (``phase_a_uN.pt``
  x1495, ``hopfield_nav_updateN.pt`` x829, ``phase_c_uN.pt``, ``phased_vN_final.pt``,
  ...). ``analysis.trajectories`` regex-parses them and carries a deliberate
  rule to skip ``phase_a_only_final.pt`` because it has no update number.
- **The encoder is referenced by path**, and paths moved once already, so
  ``_resolve_encoder_path`` tries six candidate roots -- two of them hardcoded
  home directories -- and takes the first that exists. It cannot tell a
  *correct* hit from a coincidental one.
- **Asking anything about the run tree meant loading torch.** Answering "which
  of these 346 directories are junk" required ``torch.load`` on every one.
- **A crashed run looks exactly like a finished one.**
- **Nothing recorded the git SHA**, so no figure was reproducible to a commit.

The manifest is an *index*, never the source of truth
-----------------------------------------------------
The config embedded in each ``.pt`` stays authoritative -- ``cfg_from_checkpoint``
still reads it, and nothing here is required to load a checkpoint. A missing or
corrupt ``run.json`` degrades to exactly the pre-manifest behavior: readers fall
back to globbing (:func:`checkpoints_in`), and the 346 pre-existing run
directories keep working untouched. This is deliberate. A metadata file that
can break training or evaluation is worse than no metadata file, which is also
why every writer here swallows its own errors and warns rather than raising.

``status``
----------
``running`` -> ``done`` on a clean finish. A run that is killed (SIGKILL, node
failure, walltime) leaves ``running`` behind, because no handler runs. So
``running`` means "not known to have finished", not "alive" -- ``scripts/gc_runs.py``
resolves it by mtime rather than trusting the field.
"""
from __future__ import annotations

import getpass
import hashlib
import json
import os
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from cls_paths import REPO_ROOT

MANIFEST_NAME = "run.json"

# Bumped when a field changes meaning. Readers must tolerate an unknown value:
# a newer writer's manifest should degrade to the glob fallback, not crash.
SCHEMA_VERSION = 1

STATUS_RUNNING = "running"
STATUS_DONE = "done"

# Only the leading N bytes of the encoder file are hashed. A full hash of a
# 25 MB encoder x every run is pointless: the first megabyte already separates
# any two distinct encoders in this tree, and the field exists to *detect a
# mismatch*, not to defend against a forged checkpoint.
_HASH_BYTES = 1 << 20


def _warn(what: str, exc: BaseException) -> None:
    print(f"[run_manifest] {what} failed ({type(exc).__name__}: {exc}); "
          f"continuing without it", file=sys.stderr, flush=True)


def _git_state() -> dict[str, Any]:
    """Current commit and whether the tree is dirty. Empty dict if not a repo."""
    def _git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    try:
        return {
            "sha": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(_git("status", "--porcelain")),
        }
    except Exception as exc:                       # pragma: no cover - env dependent
        _warn("reading git state", exc)
        return {}


def file_digest(path: str | os.PathLike[str] | None) -> str | None:
    """sha256 of the first megabyte of a file, or None if unreadable.

    Used to give the encoder an identity that survives being moved, so that a
    reader can tell "I found *the* encoder" from "I found *an* encoder at that
    path" -- the distinction ``_resolve_encoder_path``'s six-candidate search
    could never make.
    """
    if not path:
        return None
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read(_HASH_BYTES)).hexdigest()
    except (FileNotFoundError, NotADirectoryError):
        # Silent: a checkpoint pointing at an encoder that no longer exists is
        # a normal state of this tree, not an error. 35 of the ~350 runs are in
        # it. `sha256: null` in the manifest is the record of that, and
        # gc_runs classifies on it -- a warning per run would just be noise.
        return None
    except Exception as exc:
        _warn(f"hashing {path}", exc)
        return None


def encoder_identity(path: str | None, enc_cfg: Any = None,
                     gain: float | None = None) -> dict[str, Any]:
    """The `encoder` block: where it was, what it hashes to, what shape it is."""
    ident: dict[str, Any] = {"path": path, "sha256": file_digest(path)}
    if enc_cfg is not None:
        for field in ("out_dim", "lambdas", "encoder_type"):
            if hasattr(enc_cfg, field):
                value = getattr(enc_cfg, field)
                ident[field] = list(value) if isinstance(value, (list, tuple)) else value
    if gain is not None:
        ident["gain"] = float(gain)
    return ident


def path_for(run_dir: str | os.PathLike[str]) -> Path:
    return Path(run_dir) / MANIFEST_NAME


def read(run_dir: str | os.PathLike[str]) -> dict[str, Any] | None:
    """The manifest for a run directory, or None if it has none.

    None is the normal answer for the 346 directories written before this
    existed; callers fall back rather than fail.
    """
    p = path_for(run_dir)
    try:
        if not p.exists():
            return None
        return json.loads(p.read_text())
    except Exception as exc:
        _warn(f"reading {p}", exc)
        return None


def write(run_dir: str | os.PathLike[str], data: dict[str, Any]) -> None:
    """Write a complete manifest, atomically.

    Public because `scripts/backfill_manifests.py` legitimately writes whole
    manifests for the ~350 legacy run directories; the incremental helpers
    below cover every other caller. Atomic via os.replace, so a reader never
    sees a half-written file.
    """
    p = path_for(run_dir)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=str))
    os.replace(tmp, p)


def begin(
    run_dir: str | os.PathLike[str],
    *,
    kind: str,
    name: str,
    config: dict[str, Any],
    encoder: dict[str, Any] | None = None,
    parent: str | None = None,
    wandb_run: Any = None,
    provenance: str = "live",
) -> None:
    """Record a run at its start, before the first update.

    At the *start* specifically: a manifest written at the end would be absent
    from exactly the runs you most need to identify -- the ones that died.

    `parent` is the checkpoint this run resumed from (``--load_checkpoint``),
    which is what makes the resume chains in the 101 sweep variants traversable
    instead of grep-able.
    """
    try:
        os.makedirs(run_dir, exist_ok=True)
        wandb_block = None
        if wandb_run is not None:
            wandb_block = {
                "id": getattr(wandb_run, "id", None),
                "name": getattr(wandb_run, "name", None),
                "project": getattr(wandb_run, "project", None),
                "url": getattr(wandb_run, "url", None),
            }
        write(run_dir, {
            "schema": SCHEMA_VERSION,
            "kind": kind,
            "name": name,
            "status": STATUS_RUNNING,
            "provenance": provenance,
            "created": datetime.now().isoformat(timespec="seconds"),
            "finished": None,
            "host": socket.gethostname(),
            "user": getpass.getuser(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "argv": list(sys.argv),
            "git": _git_state(),
            "wandb": wandb_block,
            "parent": parent,
            "encoder": encoder,
            "config": config,
            "checkpoints": [],
        })
    except Exception as exc:
        _warn(f"writing manifest in {run_dir}", exc)


def record_checkpoint(run_dir: str | os.PathLike[str], filename: str,
                      update: int | None = None) -> None:
    """Append one checkpoint to the manifest's list.

    This is what replaces basename parsing: the update number is *recorded*
    rather than recovered from a filename by regex, so a new naming scheme
    costs a reader nothing.
    """
    try:
        data = read(run_dir)
        if data is None:
            return
        entry = {"file": os.path.basename(filename), "update": update}
        entries = [e for e in data.get("checkpoints", [])
                   if e.get("file") != entry["file"]]
        entries.append(entry)
        entries.sort(key=lambda e: (e["update"] is None, e["update"] or 0, e["file"]))
        data["checkpoints"] = entries
        write(run_dir, data)
    except Exception as exc:
        _warn(f"recording {filename} in {run_dir}", exc)


def finish(run_dir: str | os.PathLike[str], status: str = STATUS_DONE) -> None:
    """Mark the run finished. Never called for a killed job -- see the docstring."""
    try:
        data = read(run_dir)
        if data is None:
            return
        data["status"] = status
        data["finished"] = datetime.now().isoformat(timespec="seconds")
        write(run_dir, data)
    except Exception as exc:
        _warn(f"finishing {run_dir}", exc)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

# The legacy fallback: the update suffix on every naming scheme in the tree.
#   hopfield_nav_update{N}.pt   phase_a_u{N}.pt   phase_b_u{N}.pt   phase_c_u{N}.pt
# Files with no update number (phase_a_only_final.pt, final.pt) match nothing
# and are excluded, which is the behavior `analysis.trajectories` relied on.
LEGACY_CKPT_RE = r"(?:_u|_update)(\d+)\.pt$"


def checkpoints_in(run_dir: str | os.PathLike[str]) -> list[tuple[int, str]]:
    """[(update, absolute path)] for a run, sorted by update.

    Prefers the manifest; falls back to globbing + the legacy regex when there
    is none, which is the case for every run directory written before 2026-08.
    Entries whose file has since been deleted are dropped either way, so a
    manifest listing a pruned checkpoint does not hand back a dead path.
    """
    import glob
    import re

    run_dir = str(run_dir)
    data = read(run_dir)
    if data is not None:
        out = []
        for entry in data.get("checkpoints", ()):
            if entry.get("update") is None:
                continue
            full = os.path.join(run_dir, entry["file"])
            if os.path.exists(full):
                out.append((int(entry["update"]), full))
        # One guard, deliberately: `if out` is the only condition, so a run
        # whose manifest exists but yields nothing usable -- killed before its
        # first checkpoint, or every listed file since pruned -- still gets the
        # glob. Adding a second emptiness check above would be redundant with
        # this one, which sounds harmless but means neither can be mutated
        # independently and both read as untested.
        if out:
            out.sort(key=lambda x: x[0])
            return out

    pattern = re.compile(LEGACY_CKPT_RE)
    out = []
    for p in glob.glob(os.path.join(run_dir, "*.pt")):
        m = pattern.search(os.path.basename(p))
        if m:
            out.append((int(m.group(1)), p))
    out.sort(key=lambda x: x[0])
    return out


__all__ = [
    "MANIFEST_NAME",
    "SCHEMA_VERSION",
    "STATUS_RUNNING",
    "STATUS_DONE",
    "LEGACY_CKPT_RE",
    "write",
    "begin",
    "record_checkpoint",
    "finish",
    "read",
    "path_for",
    "checkpoints_in",
    "file_digest",
    "encoder_identity",
]
