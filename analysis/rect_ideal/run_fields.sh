#!/bin/bash
# Place fields of the rectified-ideal grid over the whole scaffold (analysis/rect_ideal/fields.py):
# 30 tasks, one per (f, D).
#
#   bash analysis/rect_ideal/run_fields.sh [ARRAY]         # submits; default 0-29%4
#   VARIANT=m111523_r16 bash analysis/rect_ideal/run_fields.sh  # other grid modules (codes.VARIANTS)
#
# Runs from THIS worktree: do not delete it while jobs are live.
set -euo pipefail
WT=${RECT_WT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}
export RECT_WT=$WT
PY=/home/jackking/.conda/envs/cls/bin/python
VARIANT=${VARIANT:-m111213}
export VARIANT
# Results folder, codes.VARIANTS[...]["results"] (no Python at submit time).
case "$VARIANT" in m111213) RES=rect_ideal_grid ;; *) RES=rect_ideal_grid_$VARIANT ;; esac
OUT=${OUT:-/orcd/pool/003/jackking/cls_runs/results/$RES/fields}
# The 11-15-23 scaffold has 4.9 times the cells: ~1 h and ~25 GB for the 4096-unit codes.
if [[ "$VARIANT" == m111213 ]]; then MEM=48G; TIME=2:00:00; else MEM=64G; TIME=4:00:00; fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    mkdir -p "$OUT/logs"
    exec sbatch --job-name=rect_fields_$VARIANT --partition=pi_evelina9,ou_bcs_normal \
        --time=$TIME --cpus-per-task=16 --mem=$MEM \
        --array="${1:-0-29%4}" --output="$OUT/logs/fields_%A_%a.out" "$0"
fi

cd "$WT"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
i=$SLURM_ARRAY_TASK_ID
echo "host $(hostname)  commit $(git rev-parse --short HEAD)"
IFS='|' read -r SPEC label <<< "$("$PY" -m analysis.rect_ideal.codes field "$i" "$VARIANT")"
echo "task $i  $label  $SPEC"
"$PY" -m analysis.rect_ideal.fields --spec "$SPEC" --label "$label" --out "$OUT" --variant "$VARIANT"
echo "DONE fields $i  $label"
