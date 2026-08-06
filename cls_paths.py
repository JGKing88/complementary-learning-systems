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
from datetime import datetime
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


# --- run identity and run directories --------------------------------------
#
# A training run's output is one directory named ``<prefix><run name>``. The
# prefix records which trainer wrote it; the name is the run's identity -- the
# wandb run name, or a timestamp when wandb is off. That name is the only key
# linking wandb <-> checkpoint <-> slurm log, which is why it is worth having
# one function produce it.
#
# Why this is here and not in each trainer
# ----------------------------------------
# Until 2026-08 all five trainers built this path by concatenating the string
# literal ``"checkpoint"``. That is a *relative* path, so it resolved against
# the CWD, then through a repo symlink, to the default runs root -- no matter
# what ``CLS_RUNS`` said. Three consequences, all observed rather than
# theorised:
#
#   1. ``CLS_RUNS`` relocated every output except the one there are 8.5 GB of.
#   2. Run from anywhere but the repo root, a job silently created a stray
#      ``./checkpoint/`` next to wherever it was started. This is why every
#      sbatch script has to ``cd`` to the repo root first.
#   3. ``test_smoke_train.py``'s sandbox fixture sets ``CLS_RUNS`` to a tmpdir
#      "so the smoke test never writes to real outputs", and it worked for
#      every trainer that accepted ``--save_dir`` -- but ``train_phase_b_only``
#      had no such flag, so each test run deposited one more junk directory in
#      the real tree. 36 of the 346 run directories arrived that way.
#
# ``test_no_hardcoded_output_dirs`` in the layering suite is what stops a sixth
# trainer reintroducing the literal.
# The convention every new kind gets for free.
DEFAULT_RUN_SUBDIR = "agent_ckpts"


def default_layout(kind: str) -> tuple[str, str]:
    """(subdirectory, name prefix) for a kind with no entry in RUN_KINDS."""
    return DEFAULT_RUN_SUBDIR, f"{kind}_"


# How the *existing* tree is laid out -- not a registry of permitted kinds.
#
# Three of these five rows are just `default_layout`: `phased_`, `phase_a_only_`
# and `phase_b_only_` are each `kind + "_"`. They are listed anyway because
# `scripts/backfill_manifests.py` reads this table backwards, to parse a legacy
# directory name into (kind, name) -- `phase_a_only_amber-plant-56` cannot be
# split any other way. Removing them would silently reclassify ~150 directories
# as kind "train".
#
# The two that carry real information are the irregular ones, both historical:
# `train` writes a bare wandb run name (`agent_ckpts/revived-water-17/`, ~100
# directories), and `rnn` writes a bare name into a *different* subdirectory.
#
# A new kind needs no entry here. `run_dir("phase_c_only", name)` gives
# `agent_ckpts/phase_c_only_<name>/` on its own; add a row only to deviate from
# that, or -- once the run has legacy directories of its own -- so that backfill
# can parse them.
RUN_KINDS: dict[str, tuple[str, str]] = {
    # kind           (subdirectory of RUNS_ROOT, run-directory name prefix)
    "train":         ("agent_ckpts", ""),            # irregular: no prefix
    "phased":        ("agent_ckpts", "phased_"),
    "navigate":      ("agent_ckpts", "navigate_"),
    "store":         ("agent_ckpts", "store_"),
    # The pre-rename spellings of navigate/store. `train_phase_a_only` and
    # `train_phase_b_only` became `train_navigate` and `train_store` on
    # 2026-08-06; ~150 directories are named the old way and backfill parses
    # them through these two rows. Nothing writes them any more.
    "phase_a_only":  ("agent_ckpts", "phase_a_only_"),
    "phase_b_only":  ("agent_ckpts", "phase_b_only_"),
    "rnn":           ("checkpoint_rnn", ""),         # irregular: own subdir
}


def run_name(wandb_name: str | None = None, wandb_id: str | None = None) -> str:
    """Identity for a run: the wandb run name, its id, else a timestamp.

    Callers pass ``wandb.run.name`` / ``wandb.run.id`` when wandb is on and
    nothing when it is off. The precedence is the one all five trainers already
    used independently; centralising it means the fallback is the same
    everywhere and can be tested without a wandb session.
    """
    return wandb_name or wandb_id or datetime.now().strftime("%Y%m%d_%H%M%S")


def run_dir(kind: str, name: str, *, ensure: bool = False) -> Path:
    """Output directory for one training run.

    ``name`` is the run identity from :func:`run_name`. ``kind`` names the
    trainer: it takes its layout from :data:`RUN_KINDS` if it has a row there,
    and otherwise from :func:`default_layout` -- so a new trainer picks a kind
    string and needs nothing here.

    For the five existing kinds the resulting directory name is unchanged from
    the pre-2026-08 layout (``phase_a_only_<name>``, ``phased_<name>``, a bare
    ``<name>`` for ``train`` and ``rnn``), so existing run directories are still
    addressed by the same string -- only the root moves, and only when
    ``CLS_RUNS`` is set.

    An unknown kind is not an error. It used to raise, on the theory that this
    caught typos (``"phase_a"`` for ``"phase_a_only"``), but that made every new
    trainer edit a file it otherwise has no reason to touch -- real coupling
    paid for a weak guard, since a mistyped kind produces a visibly wrong
    directory that is printed at startup.
    """
    subdir, prefix = RUN_KINDS.get(kind) or default_layout(kind)
    p = _sub(subdir) / f"{prefix}{name}"
    if ensure:
        p.mkdir(parents=True, exist_ok=True)
    return p


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
    "RUN_KINDS",
    "DEFAULT_RUN_SUBDIR",
    "default_layout",
    "run_name",
    "run_dir",
]
