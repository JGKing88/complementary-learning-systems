#!/bin/bash -l
#SBATCH --job-name=i4replay
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=pi_fiete,ou_bcs_normal
#SBATCH --mem=48G
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/logs/i4replay_%j.out

# Idea 4's memory trained on recorded agent movement
# (analysis/idea4/train_replay.py, GRID_MLP_NAV_PLAN §7b.15).
#
#   INIT=<memory.pt> RECORDS='<glob>' OUT=<dir> [STEPS=3000] [LR=1e-4] \
#     [BATCH=32] [EVAL_EVERY=1000] REPO=<code snapshot> sbatch hopfield_nav/run_idea4_replay.sh
#
# Writes <OUT>/memory.pt and <OUT>/log.json.

set -euo pipefail
REPO=${REPO:?set REPO to a code snapshot}
cd "$REPO"
module load miniforge/24.3.0-0
module load cuda/13.0.1
source activate cls
unset CUDA_VISIBLE_DEVICES
source scripts/cls_env.sh
python -u -m analysis.idea4.train_replay \
    --init_from "${INIT:?}" --records "${RECORDS:?}" --out "${OUT:?}" \
    --steps "${STEPS:-3000}" --lr "${LR:-1e-4}" --batch "${BATCH:-32}" \
    --eval_every "${EVAL_EVERY:-1000}"
echo "[i4replay] done $(date)"
