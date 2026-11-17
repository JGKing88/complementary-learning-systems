#!/bin/bash -l
#SBATCH --job-name=i4eval
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=pi_fiete,ou_bcs_normal
#SBATCH --mem=32G
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/logs/i4eval_%j.out

# Idea 4 retention over 64 held-out lifetimes (analysis/idea4/eval_many.py) for
# experience-trained runs (GRID_MLP_NAV_PLAN §7b.11).
#
#   TAGS="h2048_raw_gate95_lr3e4_lstm ..." [NS="1 2 3 6 10 20"] [NTAG=N30_100] \
#     REPO=<code snapshot> sbatch hopfield_nav/run_idea4_eval_many.sh
#
# Each tag reads $CLS_RUNS/idea4/experience_<tag>_s0/memory.pt and writes
# $CLS_RUNS/idea4/eval_many/arena_<tag>_step<k>[_<NTAG>].json. Give runs tested
# at different N different NTAGs, or the second test overwrites the first.

set -euo pipefail
REPO=${REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/sensory-key-retrieval}
cd "$REPO"
module load miniforge/24.3.0-0
module load cuda/13.0.1
source activate cls
unset CUDA_VISIBLE_DEVICES
source scripts/cls_env.sh
R=$CLS_RUNS/idea4
for tag in ${TAGS}; do
  ck=$R/experience_${tag}_s0/memory.pt
  step=$(python -c "import torch, sys; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['step'])" "$ck")
  python -u -m analysis.idea4.eval_many --ckpt "$ck" --N ${NS:-1 2 3 6 10 20} --lifetimes 64 \
      --out "$R/eval_many/arena_${tag}_step${step}${NTAG:+_$NTAG}.json"
done
echo "[i4eval] done $(date)"
