"""Canonical filesystem locations for run outputs.

Why this module is top-level
----------------------------
Both ``hopfield_nav`` and ``encoder_training`` need these paths, and the
dependency between those two packages is one-way: ``hopfield_nav`` may import
``encoder_training``, never the reverse. A paths module inside either package
would therefore create a cycle, so it sits below both as a leaf with no
project imports of its own.

Why this module exists
----------------------
Until 2026-08 every output directory lived inside the source tree: encoder
checkpoints in ``encoders/``, agent checkpoints in ``checkpoint/``, continual
histories in ``analysis/continual/histories/``, and so on. That put
~64 GB of run artifacts on a 200 GB HOME quota shared with everything else, so
a long training job could fail on write. Outputs now live under a single root
on pool storage; the in-repo names survive as symlinks.

The symlinks are load-bearing, not cosmetic
-------------------------------------------
Saved checkpoints and the sweep scripts store *relative* encoder paths such as
``encoders/run_20260422_185816/encoder_best.pt``, resolved against the repo
root at load time. Keeping ``encoders`` as a symlink means every existing
checkpoint keeps resolving. New code should prefer the accessors here so that
the symlinks can eventually be dropped.

Overriding the root
-------------------
Set ``CLS_RUNS`` to relocate every output at once (a scratch dir for a smoke
test, a different pool allocation, a second machine)::

    CLS_RUNS=/tmp/cls_smoke python -m hopfield_nav.train ...

Note that ``/orcd/pool/003/jackking`` and ``/home/jackking/orcd/pool`` are the
same directory (the latter is a symlink), as are ``/orcd/home/002/jackking/cls``
and ``/home/jackking/cls``. The canonical spellings are used here so that a
resolved path compares equal to itself.
"""

from __future__ import annotations

import os
from pathlib import Path

# Repo root: this module lives at the top of the source tree.
REPO_ROOT = Path(__file__).resolve().parent

DEFAULT_RUNS_ROOT = Path("/orcd/pool/003/jackking/cls_runs")


def runs_root() -> Path:
    """Root of all run outputs. Override with the CLS_RUNS env var."""
    return Path(os.environ.get("CLS_RUNS", DEFAULT_RUNS_ROOT)).expanduser()


# Read once at import so that a single process cannot see two different roots
# mid-run, which would scatter one job's outputs across two trees.
RUNS_ROOT = runs_root()


def _sub(name: str, *, ensure: bool = False) -> Path:
    p = RUNS_ROOT / name
    if ensure:
        p.mkdir(parents=True, exist_ok=True)
    return p


# --- model artifacts -------------------------------------------------------

def encoders_dir(*, ensure: bool = False) -> Path:
    """Encoder training runs (``run_<timestamp>/encoder_best.pt``)."""
    return _sub("encoders", ensure=ensure)


def checkpoints_dir(*, ensure: bool = False) -> Path:
    """Agent checkpoints, one directory per wandb run name."""
    return _sub("agent_ckpts", ensure=ensure)


def legacy_checkpoints_dir(*, ensure: bool = False) -> Path:
    """Pre-April-2026 agent checkpoints (the old ``checkpoints/``, plural)."""
    return _sub("agent_ckpts_legacy", ensure=ensure)


def rnn_checkpoints_dir(*, ensure: bool = False) -> Path:
    """RNN-baseline checkpoints."""
    return _sub("checkpoint_rnn", ensure=ensure)


# --- experiment outputs ----------------------------------------------------

def histories_dir(*, ensure: bool = False) -> Path:
    """Continual-learning history JSON written by the final_plotting drivers."""
    return _sub("histories", ensure=ensure)


def scaffold_cache_dir(*, ensure: bool = False) -> Path:
    """Cached VectorHash scaffolds keyed by (encoder, lambdas, size)."""
    return _sub("scaffold_cache", ensure=ensure)


def results_dir(*, ensure: bool = False) -> Path:
    """Evaluation results: eval_all JSON, phase-decoding outputs, diagnostics."""
    return _sub("results", ensure=ensure)


def figures_dir(*, ensure: bool = False) -> Path:
    """Rendered figures (schematics, continual-learning panels, trajectories)."""
    return _sub("figures", ensure=ensure)


def sweeps_dir(*, ensure: bool = False) -> Path:
    """Encoder sweep outputs (one subdirectory per sweep)."""
    return _sub("sweeps", ensure=ensure)


def logs_dir(*, ensure: bool = False) -> Path:
    """SLURM stdout/stderr."""
    return _sub("logs", ensure=ensure)


def wandb_dir(*, ensure: bool = False) -> Path:
    """WANDB_DIR. wandb creates its own ``wandb/`` subdirectory beneath this."""
    return _sub("wandb", ensure=ensure)


def archive_dir(*, ensure: bool = False) -> Path:
    """Superseded material kept for provenance (notebooks, reference PDFs)."""
    return _sub("archive", ensure=ensure)


__all__ = [
    "REPO_ROOT",
    "DEFAULT_RUNS_ROOT",
    "RUNS_ROOT",
    "runs_root",
    "encoders_dir",
    "checkpoints_dir",
    "legacy_checkpoints_dir",
    "rnn_checkpoints_dir",
    "histories_dir",
    "scaffold_cache_dir",
    "results_dir",
    "figures_dir",
    "sweeps_dir",
    "logs_dir",
    "wandb_dir",
    "archive_dir",
]
