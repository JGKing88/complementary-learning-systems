#!/bin/bash -l
#SBATCH --job-name=i4probe
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=pi_fiete,ou_bcs_normal
#SBATCH --mem=32G
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/logs/i4probe_%j.out

# Idea 4: which recurrent memories learn plain associative recall
# (analysis/idea4/recall_probe.py)? All variants share one GPU, one process each.
#   REPO=<code snapshot> sbatch hopfield_nav/run_idea4_probe.sh

set -uo pipefail
REPO=${REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/sensory-key-retrieval}
STEPS=${STEPS:-30000}
cd "$REPO"
module load miniforge/24.3.0-0
module load cuda/13.0.1
source activate cls
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=1
nvidia-smi --query-gpu=name --format=csv,noheader

VARIANTS=(
  "--cell gru --H 256"
  "--cell gru --H 256 --lr 3e-4"
  "--cell gru --H 256 --lr 3e-4 --chrono 100"
  "--cell gru --H 1024 --lr 3e-4"
  "--cell gru --H 256 --layers 2"
  "--cell gru --H 256 --dwell 4"
  "--cell gru --H 256 --key_dim 8"
  "--cell lstm --H 256"
  "--cell migru --H 256"
  "--cell fw --H 256 --pairs 6"
  "--cell fw --H 256 --pairs 30"
)
# VARIANTS_FILE: one variant per line, replacing the list above.
if [ -n "${VARIANTS_FILE:-}" ]; then mapfile -t VARIANTS < "$VARIANTS_FILE"; fi
for v in "${VARIANTS[@]}"; do
  python -u -m analysis.idea4.recall_probe $v --steps "$STEPS" &
done
wait
echo "[i4probe] done $(date)"
