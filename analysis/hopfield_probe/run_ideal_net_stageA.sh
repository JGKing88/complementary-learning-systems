#!/usr/bin/env bash
# Stage A (docs/EXPERIMENTS_IDEAL_ENCODER.md): train IdealNet's harmonic layer
# against the r=16 Gaussian kernel on the att0.5 recipe's patches, from three
# inits x three seeds. Task i: init = (integer, noisy, random)[i / 3], seed = i % 3.
#
#   bash analysis/hopfield_probe/run_ideal_net_stageA.sh [ARRAY]   # default 0-8
#
# Runs from THIS worktree: do not delete it while jobs are live.
set -euo pipefail
WT=${IDEAL_WT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}
export IDEAL_WT=$WT
PY=/home/jackking/.conda/envs/cls/bin/python
OUT=${OUT:-/orcd/pool/003/jackking/cls_runs/results/ideal_net/stageA}
EPOCHS=${EPOCHS:-500}
LR=${LR:-1e-2}
RATE_LAMBDA=${RATE_LAMBDA:-0}       # Sec 12: 0.5 with the recipe's eps = 1

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    mkdir -p "$OUT/logs"
    exec sbatch --job-name=idealnet_A --partition=ou_bcs_normal \
        --gres=gpu:1 --time=3:00:00 --cpus-per-task=4 --mem=24G \
        --array="${1:-0-8}" --output="$OUT/logs/A_%A_%a.out" "$0"
fi

cd "$WT"
INITS=(integer noisy random)
i=$SLURM_ARRAY_TASK_ID
init=${INITS[$(( i / 3 ))]}
seed=$(( i % 3 ))
"$PY" -m encoder_training.train_ideal_net --r 16 --init "$init" --seed "$seed" \
    --epochs "$EPOCHS" --lr "$LR" --rate_lambda "$RATE_LAMBDA" --rate_eps 1.0 \
    --out "$OUT/${init}_s${seed}"
echo "DONE $init s$seed"
