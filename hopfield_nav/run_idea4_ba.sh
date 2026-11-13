#!/bin/bash -l
#SBATCH --job-name=i4ba
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=pi_fiete,ou_bcs_normal
#SBATCH --mem=32G
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/logs/i4ba_%j.out

# Idea 4 sanity check (analysis/idea4/ba_recall.py): Ba et al. 2016's
# associative retrieval, then one change at a time toward idea 4's format.
#   REPO=<code snapshot> sbatch hopfield_nav/run_idea4_ba.sh

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

VARIANTS=(
  "--cell lstm --H 100"
  "--cell gru --H 100"
  "--cell lstm --H 256"
  "--cell lstm --H 100 --pairs 2"
  "--cell lstm --H 100 --together"
  "--cell lstm --H 100 --values phase"
  "--cell lstm --H 100 --keys random --key_dim 26"
  "--cell gru --H 256 --pairs 2 --together --values phase --keys random --key_dim 180 --mlp_read"
)
for v in "${VARIANTS[@]}"; do
  python -u -m analysis.idea4.ba_recall $v --steps "$STEPS" &
done
wait
echo "[i4ba] done $(date)"
