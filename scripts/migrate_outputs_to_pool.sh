#!/usr/bin/env bash
# One-shot migration of run outputs from the source tree to pool storage.
#
# Each directory is copied with rsync, verified by file count, removed from the
# source tree, and replaced by a symlink pointing at its new home. The script is
# idempotent and resumable: a directory that is already a symlink is skipped, and
# an interrupted rsync simply resumes on the next run.
#
# The symlinks are required, not cosmetic -- saved checkpoints and sweep scripts
# store relative encoder paths (encoders/run_.../encoder_best.pt) resolved
# against the repo root. See hopfield_nav/paths.py.
#
#   ./scripts/migrate_outputs_to_pool.sh            # migrate
#   DRY_RUN=1 ./scripts/migrate_outputs_to_pool.sh  # print the plan only
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLS_RUNS="${CLS_RUNS:-/orcd/pool/003/jackking/cls_runs}"
DRY_RUN="${DRY_RUN:-0}"

# src-relative-to-repo : dest-relative-to-CLS_RUNS
# Two directories are renamed on the way: checkpoint -> agent_ckpts and the
# older checkpoints -> agent_ckpts_legacy. The repo-side symlinks keep the old
# names so nothing that hardcodes them breaks.
MAPPINGS=(
    "encoders:encoders"
    "checkpoint:agent_ckpts"
    "checkpoints:agent_ckpts_legacy"
    "checkpoint_rnn:checkpoint_rnn"
    "wandb:wandb"
    "analysis/continual/histories:histories"
    "analysis/continual/scaffold_cache:scaffold_cache"
    "analysis/continual/figures:figures/final_plotting"
    "analysis/continual/final_figures:figures/final"
    "analysis/continual/model_comparison:figures/model_comparison"
    "analysis/phase_decoding/results:results/phase_decoding_v2"
    "analysis/phase_decoding_v1_results:results/phase_decoding_v1"
    "hopfield_nav/logs:logs"
    "hopfield_nav/eval_results:results/eval_results"
    "hopfield_nav/eval_all:results/eval_all"
    "hopfield_nav/diagnostics:results/diagnostics"
    "hopfield_nav/inspect:results/inspect"
    "hopfield_nav/trajectory_plots:figures/trajectory_plots"
    "encoder_training/sweeps:sweeps"
    "plots:figures/plots"
    "images:figures/images"
    "npos_sweep:results/npos_sweep"
    "displacement_plots:figures/displacement_plots"
    "smoke_pd2:results/smoke_pd2"
    "smoke_seq:results/smoke_seq"
)

log() { printf '%s  %s\n' "$(date +%H:%M:%S)" "$*"; }

mkdir -p "$CLS_RUNS"
log "target root: $CLS_RUNS"

for mapping in "${MAPPINGS[@]}"; do
    src_rel="${mapping%%:*}"
    dest_rel="${mapping##*:}"
    src="$REPO_ROOT/$src_rel"
    dest="$CLS_RUNS/$dest_rel"

    if [[ -L "$src" ]]; then
        log "skip (already a symlink): $src_rel"
        continue
    fi
    if [[ ! -d "$src" ]]; then
        log "skip (absent): $src_rel"
        continue
    fi
    if [[ -e "$dest" && ! -d "$dest" ]]; then
        log "ERROR: destination exists and is not a directory: $dest"
        exit 1
    fi

    n_src=$(find "$src" -type f | wc -l)
    size=$(du -sh "$src" | cut -f1)
    log "migrate $src_rel -> $dest_rel  ($size, $n_src files)"
    if [[ "$DRY_RUN" == "1" ]]; then
        continue
    fi

    mkdir -p "$(dirname "$dest")"
    rsync -a --partial "$src/" "$dest/"

    n_dest=$(find "$dest" -type f | wc -l)
    if [[ "$n_src" -ne "$n_dest" ]]; then
        log "ERROR: file-count mismatch for $src_rel (src=$n_src dest=$n_dest); source left in place"
        exit 1
    fi

    rm -rf "$src"
    ln -s "$dest" "$src"
    log "done $src_rel  ($n_dest files verified)"
done

log "migration complete"
log "freed on HOME: check with 'du -sh $REPO_ROOT'"
