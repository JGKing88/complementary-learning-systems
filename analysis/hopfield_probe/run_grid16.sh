#!/usr/bin/env bash
# The 16-condition grid (EXPERIMENTS_IDEAL_ENCODER Sec 18): one array task per
# (memory condition, layout, K) in grid16_check.TASKS -- 12 memory conditions
# (8 + 4 recall-only-saturation side rows) x {whole, region 0 0 500} x K {5, 20}.
#
#   bash analysis/hopfield_probe/run_grid16.sh [ARRAY]      # default 0-47
#   python -m analysis.hopfield_probe.grid16_summary OUT    # tables
#
# Runs from THIS worktree: do not delete it while jobs are live.
set -euo pipefail

WT=${GRID16_WT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}
export GRID16_WT=$WT
PY=/home/jackking/.conda/envs/cls/bin/python
OUT=${OUT:-/orcd/pool/003/jackking/cls_runs/results/hopfield_probe/ideal_encoder/grid16}

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    mkdir -p "$OUT/logs"
    exec sbatch --job-name=grid16 --partition=ou_bcs_normal \
        --exclude=node3804 --time=4:00:00 --cpus-per-task=16 --mem=32G \
        --array="${1:-0-47}" \
        --output="$OUT/logs/grid16_%A_%a.out" "$0"
fi

cd "$WT"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
"$PY" -m analysis.hopfield_probe.grid16_check \
    --index "$SLURM_ARRAY_TASK_ID" --out "$OUT/json"
echo "DONE grid16 $SLURM_ARRAY_TASK_ID -> $OUT/json"
