#!/bin/bash
# Place fields of the rectified-ideal grid over the whole scaffold (analysis/rect_ideal/fields.py):
# 30 tasks, one per (f, D).
#
#   bash analysis/rect_ideal/run_fields.sh [ARRAY]         # submits; default 0-29%4
#
# Runs from THIS worktree: do not delete it while jobs are live.
set -euo pipefail
WT=${RECT_WT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}
export RECT_WT=$WT
PY=/home/jackking/.conda/envs/cls/bin/python
OUT=${OUT:-/orcd/pool/003/jackking/cls_runs/results/rect_ideal_grid/fields}

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    mkdir -p "$OUT/logs"
    exec sbatch --job-name=rect_fields --partition=pi_evelina9,ou_bcs_normal \
        --time=2:00:00 --cpus-per-task=16 --mem=48G \
        --array="${1:-0-29%4}" --output="$OUT/logs/fields_%A_%a.out" "$0"
fi

cd "$WT"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
i=$SLURM_ARRAY_TASK_ID
echo "host $(hostname)  commit $(git rev-parse --short HEAD)"
IFS='|' read -r SPEC label <<< "$("$PY" -m analysis.rect_ideal.codes field "$i")"
echo "task $i  $label  $SPEC"
"$PY" -m analysis.rect_ideal.fields --spec "$SPEC" --label "$label" --out "$OUT"
echo "DONE fields $i  $label"
