#!/usr/bin/env python3
"""Reconstruct run manifests for run directories written before they existed.

~350 agent run directories and ~9 RNN ones predate `run_manifest`. Everything
recoverable about them is already on disk -- the config is embedded in every
`.pt`, the checkpoint list is the directory listing, the phase is the directory
name prefix -- it just costs a `torch.load` to reach. This writes it out once
so that afterwards it costs a `json.loads`.

What can and cannot be recovered
--------------------------------
Recovered: kind, name, config, encoder path (and its digest, when the encoder
still exists), the checkpoint list with update numbers, and `created` from the
directory mtime.

Not recovered, and left absent rather than guessed: `argv`, `git`, `wandb`,
`parent`, `host`, `slurm_job_id`. A backfilled manifest is marked
``provenance: "backfilled"`` precisely so that a reader can tell "this run had
no git SHA recorded" from "this run was on a clean tree" -- inventing a plausible
value for either would be worse than the gap.

`status` is set to "done" only when the directory holds a `*final*.pt`, which is
the artifact every trainer writes last. Otherwise "running", which for a
directory nobody has touched in months means the job died -- see gc_runs.py.

Usage:
    python scripts/backfill_manifests.py --dry-run          # report only
    python scripts/backfill_manifests.py                    # write
    python scripts/backfill_manifests.py --root <dir>       # a specific tree
    python scripts/backfill_manifests.py --force            # rewrite existing
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run_manifest                                          # noqa: E402
from cls_paths import RUN_KINDS, checkpoints_dir, rnn_checkpoints_dir  # noqa: E402

# Longest prefix wins. No two current prefixes overlap, so this is latent
# today -- but `train` has an empty prefix and would match everything, which is
# why it is the fallback below rather than a loop entry, and the moment
# RUN_KINDS gains a `phase_c_` / `phase_c_only_` pair unordered scanning
# misfiles every run of the longer kind.
def _PREFIX_ORDER(prefix_kind: tuple[str, str]) -> int:
    return -len(prefix_kind[0])


_PREFIXES = sorted(
    ((prefix, kind) for kind, (_sub, prefix) in RUN_KINDS.items() if prefix),
    key=_PREFIX_ORDER,
)

_FINAL_RE = re.compile(r"final\.pt$")


def infer_kind(dirname: str, *, rnn: bool) -> tuple[str, str]:
    """(kind, name) from a run directory's name."""
    if rnn:
        return "rnn", dirname
    for prefix, kind in _PREFIXES:
        if dirname.startswith(prefix):
            return kind, dirname[len(prefix):]
    return "train", dirname


def read_embedded_config(run_dir: str) -> tuple[dict | None, str | None]:
    """(config dict, checkpoint it came from). Loads the smallest `.pt` there.

    The smallest because every checkpoint in a run carries the same config and
    the smallest is the cheapest to unpickle -- some of these directories hold
    360 checkpoints.
    """
    import torch

    pts = sorted(glob.glob(os.path.join(run_dir, "*.pt")), key=os.path.getsize)
    for p in pts:
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False)
        except Exception as exc:
            print(f"    unreadable {os.path.basename(p)}: "
                  f"{type(exc).__name__}: {exc}")
            continue
        cfg = ck.get("config")
        if isinstance(cfg, dict):
            return cfg, p
    return None, None


def build(run_dir: str, *, rnn: bool) -> dict | None:
    """The manifest for one legacy run directory, or None if it has no .pt."""
    name_on_disk = os.path.basename(run_dir.rstrip("/"))
    kind, name = infer_kind(name_on_disk, rnn=rnn)

    files = sorted(os.path.basename(p)
                   for p in glob.glob(os.path.join(run_dir, "*.pt")))
    if not files:
        return None

    cfg, _src = read_embedded_config(run_dir)
    encoder = None
    if cfg:
        encoder = {"path": cfg.get("encoder_checkpoint"),
                   "sha256": run_manifest.file_digest(cfg.get("encoder_checkpoint"))}
        if cfg.get("encoder_gain") is not None:
            encoder["gain"] = cfg["encoder_gain"]
        vh = cfg.get("vectorhash")
        if isinstance(vh, dict) and vh.get("lambdas"):
            encoder["lambdas"] = list(vh["lambdas"])

    pattern = re.compile(run_manifest.LEGACY_CKPT_RE)
    checkpoints = []
    for f in files:
        m = pattern.search(f)
        checkpoints.append({"file": f,
                            "update": int(m.group(1)) if m else None})
    checkpoints.sort(key=lambda e: (e["update"] is None, e["update"] or 0, e["file"]))

    return {
        "schema": run_manifest.SCHEMA_VERSION,
        "kind": kind,
        "name": name,
        "status": (run_manifest.STATUS_DONE if any(_FINAL_RE.search(f) for f in files)
                   else run_manifest.STATUS_RUNNING),
        "provenance": "backfilled",
        "created": datetime.fromtimestamp(
            os.path.getmtime(run_dir)).isoformat(timespec="seconds"),
        "finished": None,
        "argv": None,
        "git": None,
        "wandb": None,
        "parent": None,
        "encoder": encoder,
        "config": cfg,
        "checkpoints": checkpoints,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=None,
                   help="A single directory of run directories. Default: both "
                        "the agent-checkpoint and RNN-checkpoint trees.")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would be written; touch nothing.")
    p.add_argument("--force", action="store_true",
                   help="Rewrite manifests that already exist. Refuses to "
                        "overwrite a live one -- see below.")
    args = p.parse_args()

    roots = ([(args.root, False)] if args.root else
             [(str(checkpoints_dir()), False), (str(rnn_checkpoints_dir()), True)])

    written = skipped = empty = failed = 0
    for root, rnn in roots:
        if not os.path.isdir(root):
            print(f"no such tree: {root}")
            continue
        print(f"\n=== {root} ===")
        for d in sorted(glob.glob(os.path.join(root, "*/"))):
            existing = run_manifest.read(d)
            if existing is not None and not args.force:
                skipped += 1
                continue
            if existing is not None and existing.get("provenance") == "live":
                # A live manifest holds argv, git and wandb, none of which can be
                # reconstructed. Overwriting it would destroy the only copy.
                print(f"  refusing to overwrite live manifest: "
                      f"{os.path.basename(d.rstrip('/'))}")
                skipped += 1
                continue
            try:
                data = build(d, rnn=rnn)
            except Exception as exc:
                print(f"  FAILED {os.path.basename(d.rstrip('/'))}: "
                      f"{type(exc).__name__}: {exc}")
                failed += 1
                continue
            if data is None:
                empty += 1
                continue
            n_ck = len(data["checkpoints"])
            enc = (data.get("encoder") or {}).get("path") or "?"
            print(f"  {data['kind']:<13} {data['status']:<8} {n_ck:>4} ckpt  "
                  f"{os.path.basename(d.rstrip('/'))}  <- {enc}")
            if not args.dry_run:
                run_manifest.write(d, data)
            written += 1

    verb = "would write" if args.dry_run else "wrote"
    print(f"\n{verb} {written} manifests; {skipped} already had one; "
          f"{empty} directories held no .pt; {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
