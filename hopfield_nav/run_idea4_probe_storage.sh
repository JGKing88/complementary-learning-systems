#!/bin/bash -l
#SBATCH --job-name=i4store
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=pi_fiete,ou_bcs_normal
#SBATCH --mem=48G
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/logs/i4store_%j.out

# Idea 4: storage or retrieval? Fresh read-outs on the frozen state after N
# visits (analysis/idea4/probe_storage.py).
#   REPO=<code snapshot> sbatch hopfield_nav/run_idea4_probe_storage.sh

set -euo pipefail
REPO=${REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/sensory-key-retrieval}
R=/orcd/pool/003/jackking/cls_runs/idea4
cd "$REPO"
module load miniforge/24.3.0-0
module load cuda/13.0.1
source activate cls
unset CUDA_VISIBLE_DEVICES
nvidia-smi --query-gpu=name --format=csv,noheader
python -u -m analysis.idea4.probe_storage --N 6 10 20 \
    --ckpt $R/experience_h512_raw_cur_s0/memory.pt $R/experience_h1024_raw_cur_s0/memory.pt \
           $R/experience_h512_raw_s0/memory.pt
echo "[i4store] done $(date)"
