#!/bin/bash
# Hopfield probe of the rectified-ideal grid (analysis/rect_ideal/codes.py): 65 tasks, whole
# scaffold, K = 1, 3, 5, 10, 20, the same probe settings as hopfield_probe/run_relu_mix.sh.
#
#   bash analysis/rect_ideal/run_probe.sh [ARRAY]          # submits; default 0-64%4
#   VARIANT=m111523_r16 bash analysis/rect_ideal/run_probe.sh   # other grid modules (codes.VARIANTS)
#   python -m analysis.rect_ideal.summary                  # the tables
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
OUT=${OUT:-/orcd/pool/003/jackking/cls_runs/results/$RES/probe}

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    mkdir -p "$OUT/logs"
    exec sbatch --job-name=rect_probe_$VARIANT --partition=pi_evelina9,ou_bcs_normal \
        --time=4:00:00 --cpus-per-task=16 --mem=32G \
        --array="${1:-0-64%4}" --output="$OUT/logs/probe_%A_%a.out" "$0"
fi

cd "$WT"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
i=$SLURM_ARRAY_TASK_ID
echo "host $(hostname)  commit $(git rev-parse --short HEAD)"
IFS='|' read -r CK label <<< "$("$PY" -m analysis.rect_ideal.codes probe "$i" "$VARIANT")"
NPOS=$("$PY" -m analysis.rect_ideal.codes var npos "$VARIANT")
echo "task $i  $VARIANT  $label  $CK  Npos $NPOS"

"$PY" -m analysis.hopfield_probe.run \
    --ckpt "$CK" --label "$label" \
    --n_worlds 8 --n_envs_per_world 20 --k 1 3 5 10 20 \
    --steps 1 2 3 5 10 15 --env_size 20 --Npos "$NPOS" \
    --n_alias 5000 --n_cont_samples 60000 --n_cont_annulus 20000 \
    --seed 0 --device cpu --fwhm_fallback 0.25 \
    --storage_rule hebb \
    --out "$OUT/t$i"

echo "DONE probe $i  $label -> $OUT/t$i"
