#!/bin/bash -l
#SBATCH --job-name=i4recall
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=pi_fiete,ou_bcs_normal
#SBATCH --mem=48G
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/logs/i4recall_%j.out

# Idea 4 basic control: in-context associative recall with perfect keys,
# scored in grid space (analysis/idea4/recall_grid.py). One process per
# variant, all on one GPU.
#   REPO=<code snapshot> sbatch hopfield_nav/run_idea4_recall_grid.sh

set -uo pipefail
REPO=${REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/sensory-key-retrieval}
STEPS=${STEPS:-100000}
cd "$REPO"
module load miniforge/24.3.0-0
module load cuda/13.0.1
source activate cls
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=1
nvidia-smi --query-gpu=name --format=csv,noheader
for cell in gru lstm fw; do
  for pairs in 2 6 10 20; do
    python -u -m analysis.idea4.recall_grid --cell $cell --H 512 --pairs $pairs --steps "$STEPS" &
  done
done
wait
echo "[i4recall] done $(date)"
