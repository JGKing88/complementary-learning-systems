#!/bin/bash -l
#SBATCH --job-name=i4push
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=pi_fiete,ou_bcs_normal
#SBATCH --mem=48G
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/logs/i4push_%j.out

# Idea 4: training pushes on the clean associative-recall control
# (analysis/idea4/recall_grid.py): longer training, width, and pair-count
# schedules. GROUP=w512 (six arms) or GROUP=w2048 (two arms); every arm tests at
# 2, 6, 10 and 20 pairs and checkpoints, so resubmitting resumes it.
#   REPO=<code snapshot> GROUP=w512 sbatch hopfield_nav/run_idea4_recall_push.sh

set -uo pipefail
REPO=${REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/sensory-key-retrieval}
GROUP=${GROUP:-w512}
STEPS=${STEPS:-300000}
OUT=/orcd/pool/003/jackking/cls_runs/idea4/recall
cd "$REPO"
module load miniforge/24.3.0-0
module load cuda/13.0.1
source activate cls
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=1
nvidia-smi --query-gpu=name --format=csv,noheader
COMMON="--steps $STEPS --eval_every 25000 --eval_pairs 2 6 10 20"
if [ "$GROUP" = "w2048" ]; then
  ARMS=(
    "gru_h2048_p6|--cell gru --H 2048 --pairs 6 --lr 3e-4"
    "lstm_h2048_p6|--cell lstm --H 2048 --pairs 6 --lr 3e-4"
  )
elif [ "$GROUP" = "lr" ]; then
  # Separates width from learning rate: the w2048 arms run at 3e-4, the
  # w512 arms at 1e-3.
  ARMS=(
    "gru_h512_p6_lr3e-4|--cell gru --H 512 --pairs 6 --lr 3e-4"
    "lstm_h512_p6_lr3e-4|--cell lstm --H 512 --pairs 6 --lr 3e-4"
    "gru_h1024_p6_lr3e-4|--cell gru --H 1024 --pairs 6 --lr 3e-4"
    "lstm_h1024_p6_lr3e-4|--cell lstm --H 1024 --pairs 6 --lr 3e-4"
  )
else
  ARMS=(
    "gru_h512_p6|--cell gru --H 512 --pairs 6"
    "lstm_h512_p6|--cell lstm --H 512 --pairs 6"
    "gru_h512_mixed|--cell gru --H 512 --pairs 20 --pairs_min 2"
    "lstm_h512_mixed|--cell lstm --H 512 --pairs 20 --pairs_min 2"
    "gru_h512_cur|--cell gru --H 512 --pairs 20 --curriculum"
    "lstm_h512_cur|--cell lstm --H 512 --pairs 20 --curriculum"
  )
fi
for spec in "${ARMS[@]}"; do
  name=${spec%%|*}; args=${spec#*|}
  python -u -m analysis.idea4.recall_grid $args $COMMON --out "$OUT/$name" &
done
wait
echo "[i4push] done $(date)"
