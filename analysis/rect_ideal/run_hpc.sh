#!/bin/bash
# Place fields of the Vector-HaSH place layer at the grid's six sparsities, one task per module set
# (analysis/rect_ideal/hpc.py): 0 = 11-12-13, 1 = 11-15-23.
#
#   bash analysis/rect_ideal/run_hpc.sh [ARRAY]            # submits; default 0-1
#
# Runs from THIS worktree: do not delete it while jobs are live.
set -euo pipefail
WT=${RECT_WT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}
export RECT_WT=$WT
PY=/home/jackking/.conda/envs/cls/bin/python
OUT=${OUT:-/orcd/pool/003/jackking/cls_runs/results/rect_ideal_grid/hpc}
VARS=(m111213 m111523)

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    mkdir -p "$OUT/logs"
    exec sbatch --job-name=rect_hpc --partition=pi_evelina9,ou_bcs_normal \
        --time=2:00:00 --cpus-per-task=16 --mem=24G \
        --array="${1:-0-1}" --output="$OUT/logs/hpc_%A_%a.out" "$0"
fi

cd "$WT"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
V=${VARS[$SLURM_ARRAY_TASK_ID]}
echo "host $(hostname)  commit $(git rev-parse --short HEAD)  variant $V"
"$PY" -m analysis.rect_ideal.hpc fields --variant "$V" --out "$OUT"
echo "DONE hpc $V"
