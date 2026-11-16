#!/usr/bin/env bash
# The coding-comparison measures (run.py), one Slurm array task per population.
#
#   bash analysis/coding_comparison/run.sh [ARRAY] [MEASURES]
#
# ARRAY defaults to every population (0-6); MEASURES to the phase-1 set
# (kernels,fields,remapping). TAG (default v1) names the results directory,
# <results>/coding_comparison/$TAG/<population>/.
#
# Runs from THIS worktree: do not delete it while jobs are live.
set -euo pipefail

WT=${CC_WT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}
export CC_WT=$WT
PY=/home/jackking/.conda/envs/cls/bin/python
TAG=${TAG:-v1}
OUT=/orcd/pool/003/jackking/cls_runs/results/coding_comparison/$TAG
POPS=(grid ideal ideal_rot ideal_sparse trained untrained hippocampus)
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    MEASURES=${2:-kernels,fields,remapping}
    mkdir -p "$OUT/logs"
    # MEASURES travels as an argument: sbatch --export would split it at the commas.
    exec sbatch --job-name=coding_cmp --partition=ou_bcs_normal \
        --time=${CC_TIME:-4:00:00} --cpus-per-task=4 --mem=${CC_MEM:-40G} \
        --array="${1:-0-6}" --export=ALL,TAG="$TAG" \
        --output="$OUT/logs/%x_%A_%a.out" "$0" "$MEASURES"
fi

MEASURES=$1
cd "$WT"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
POP=${POPS[$SLURM_ARRAY_TASK_ID]}
"$PY" -m analysis.coding_comparison.run --pops "$POP" --tag "$TAG" --measures "$MEASURES"
echo "DONE $POP $MEASURES -> $OUT/$POP"
