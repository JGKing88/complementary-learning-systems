#!/bin/bash
# Hopfield probe of the independent-bump place codes (analysis/rect_ideal/bump.py), with exactly
# run_probe.sh's probe settings: tasks 0-15 are the 8 codes, plain and with inhib:; task 16 runs
# the exactness and kernel checks (bump.py check) into ../kernel.
#
#   bash analysis/rect_ideal/run_bump_probe.sh [ARRAY]     # submits; default 0-16%4
#   python -m analysis.rect_ideal.bump_summary             # the tables
#
# Runs from THIS worktree: do not delete it while jobs are live.
set -euo pipefail
WT=${RECT_WT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}
export RECT_WT=$WT
PY=/home/jackking/.conda/envs/cls/bin/python
OUT=${OUT:-/orcd/pool/003/jackking/cls_runs/results/rect_ideal_grid_bump/probe}

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    mkdir -p "$OUT/logs"
    exec sbatch --job-name=bump_probe --partition=pi_evelina9,ou_bcs_normal \
        --time=4:00:00 --cpus-per-task=16 --mem=32G \
        --array="${1:-0-16%4}" --output="$OUT/logs/probe_%A_%a.out" "$0"
fi

cd "$WT"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
i=$SLURM_ARRAY_TASK_ID
echo "host $(hostname)  commit $(git rev-parse --short HEAD)"
NT=$("$PY" -m analysis.rect_ideal.bump ntasks)
if (( i == NT )); then
    "$PY" -m analysis.rect_ideal.bump check "$(dirname "$OUT")/kernel"
    echo "DONE checks -> $(dirname "$OUT")/kernel"
    exit 0
fi
IFS='|' read -r CK label <<< "$("$PY" -m analysis.rect_ideal.bump probe "$i")"
echo "task $i  $label  $CK"

"$PY" -m analysis.hopfield_probe.run \
    --ckpt "$CK" --label "$label" \
    --n_worlds 8 --n_envs_per_world 20 --k 1 3 5 10 20 \
    --steps 1 2 3 5 10 15 --env_size 20 --Npos 1716 \
    --n_alias 5000 --n_cont_samples 60000 --n_cont_annulus 20000 \
    --seed 0 --device cpu --fwhm_fallback 0.25 \
    --storage_rule hebb \
    --out "$OUT/t$i"

echo "DONE probe $i  $label -> $OUT/t$i"
