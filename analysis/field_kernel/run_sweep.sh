#!/usr/bin/env bash
# analysis.field_kernel.sweep over unit counts and both kinds of fast input.
#   scripts/snapshot_code.sh <tag>; CODE=<tag> bash analysis/field_kernel/run_sweep.sh [ARRAY]
# Task i: kind = (gauss, grid)[i % 2], n_freq = (512 1024 2048 4096 8192)[i / 2].
set -euo pipefail
CODE=${CODE:?CODE=<snapshot tag under cls_runs/code>}
WT=/orcd/pool/003/jackking/cls_runs/code/$CODE
OUT=${OUT:-/orcd/pool/003/jackking/cls_runs/results/field_kernel/sweep}
EXTRA=${EXTRA:-}
export CODE OUT EXTRA
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    mkdir -p "$OUT/logs"
    exec sbatch --job-name=fk_sweep --partition=pi_evelina9,ou_bcs_normal,mit_normal \
        --time=5:00:00 --cpus-per-task=16 --mem=64G --array="${1:-0-9}%${THROTTLE:-5}" \
        --output="$OUT/logs/sweep_%A_%a.out" "$0"
fi
cd "$WT"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
i=$SLURM_ARRAY_TASK_ID
KINDS=(gauss grid); MS=(512 1024 2048 4096 8192)
M=${MS[$(( i / 2 ))]}
T=1500; [[ $M -ge 4096 ]] && T=600
echo "host $(hostname)  code $CODE  $(cat GIT_COMMIT)"
/home/jackking/.conda/envs/cls/bin/python -m analysis.field_kernel.sweep \
    --n_freq "$M" --kind "${KINDS[$(( i % 2 ))]}" --trials $T --out "$OUT" $EXTRA
echo "DONE sweep $i"
