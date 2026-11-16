#!/usr/bin/env bash
# The ideal-family grid (run.py), as one Slurm array over interleaved slices of the configs.
#
#   bash analysis/ideal_family/run.sh <snapshot-tag> [N_TASKS]
#
# Launch from a commit: the script snapshots HEAD with scripts/snapshot_code.sh
# (which refuses a dirty tree), and every task runs from that snapshot, so each
# result JSON records the commit in its "git" field. Configs whose JSON exists
# are skipped, so resubmitting with a new tag resumes.
set -euo pipefail

PY=/home/jackking/.conda/envs/cls/bin/python
OUT=${IF_OUT:-/orcd/pool/003/jackking/cls_runs/results/ideal_family/v1}
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    TAG=${1:?usage: run.sh <snapshot-tag> [N_TASKS]}
    N=${2:-48}
    ROOT=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
    "$ROOT/scripts/snapshot_code.sh" "$TAG"
    CODE=/orcd/pool/003/jackking/cls_runs/code/$TAG
    mkdir -p "$OUT/logs"
    exec sbatch --job-name=ideal_family --partition=ou_bcs_normal \
        --time=${IF_TIME:-2:30:00} --cpus-per-task=1 --mem=${IF_MEM:-20G} \
        --array="0-$((N - 1))" --export=ALL,IF_CODE="$CODE",IF_N="$N",IF_OUT="$OUT" \
        --output="$OUT/logs/%x_%A_%a.out" "$CODE/analysis/ideal_family/run.sh"
fi

cd "$IF_CODE"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PY" -m analysis.ideal_family.run --task "$SLURM_ARRAY_TASK_ID" --n-tasks "$IF_N" --out "$IF_OUT"
echo "TASK DONE $SLURM_ARRAY_TASK_ID"
