#!/usr/bin/env bash
# Translation-invariance loss test (docs/EXPERIMENTS_IDEAL_ENCODER.md Sec 15).
# Kernel MSE (r=16) + 0.5 x coding rate + INV x translation-invariance penalty,
# optionally with near-pair weighting of the MSE.
#
#   TEST=a: start at the exact ideal integers, 100 epochs. Does the ideal stay
#           integral? (Rate alone pushes half the rows off within 10 epochs.)
#   TEST=b: random starts, 500 epochs, for the configs that pass (a).
#
# Task i indexes CONFIGS below; INIT/SEED/EPOCHS come from the environment.
#
#   TEST=a bash analysis/hopfield_probe/run_ideal_net_inv.sh [ARRAY]   # 0-7
#
# Runs from THIS worktree: do not delete it while jobs are live.
set -euo pipefail
WT=${IDEAL_WT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}
export IDEAL_WT=$WT
PY=/home/jackking/.conda/envs/cls/bin/python
TEST=${TEST:-a}
OUT=${OUT:-/orcd/pool/003/jackking/cls_runs/results/ideal_net/inv_${TEST}}

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    mkdir -p "$OUT/logs"
    exec sbatch --job-name=idealnet_inv --partition=ou_bcs_normal \
        --gres=gpu:1 --time=3:00:00 --cpus-per-task=4 --mem=24G \
        --exclude=node3804 --array="${1:-0-7}" \
        --output="$OUT/logs/inv_%A_%a.out" "$0"
fi

cd "$WT"
#        inv_lambda  near_weight
CONFIGS=("10 0" "30 0" "100 0" "300 0" "10 1" "30 1" "100 1" "300 1"
         "1000 0" "1000 1")
read -r INV NEAR <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
if [[ "$TEST" == a ]]; then INIT=${INIT:-integer}; EPOCHS=${EPOCHS:-100}
else INIT=${INIT:-random}; EPOCHS=${EPOCHS:-500}; fi
SEED=${SEED:-0}
RAMP=${RAMP:-0}                       # epochs over which inv_lambda ramps from 0
DELAY=${DELAY:-0}                     # epochs with inv_lambda = 0 before the ramp
NOISE=${NOISE:-0}                     # per-step weight noise, decays to 0
NW=(); [[ "$NEAR" == 1 ]] && NW=(--near_weight)
name="inv${INV}_near${NEAR}_d${DELAY}_r${RAMP}_n${NOISE}_${INIT}_s${SEED}"
"$PY" -m encoder_training.train_ideal_net --r 16 --init "$INIT" --seed "$SEED" \
    --epochs "$EPOCHS" --lr 1e-2 --rate_lambda 0.5 --rate_eps 1.0 \
    --inv_lambda "$INV" --inv_ramp_epochs "$RAMP" --inv_delay_epochs "$DELAY" \
    --noise_std "$NOISE" ${NW[@]+"${NW[@]}"} --log_every 10 \
    --save_at 0 "$EPOCHS" --out "$OUT/$name"
echo "DONE $name"
