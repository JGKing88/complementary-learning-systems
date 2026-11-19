#!/bin/bash -l
#SBATCH --job-name=i4persist
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=pi_fiete,ou_bcs_normal
#SBATCH --mem=32G
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/logs/i4persist_%j.out

# Idea 4 memory on the continual protocol's schedule (analysis/idea4/persist.py,
# GRID_MLP_NAV_PLAN §7b.14): do its goals survive the revisits?
#
#   TAGS="h2048_raw_gate95_lr3e4_lstm ..." [ARENAS=6] [ITERS=40] [OUTTAG=...] \
#     REPO=<code snapshot> sbatch hopfield_nav/run_idea4_persist.sh
#
# Each tag reads $CLS_RUNS/idea4/experience_<tag>_s0/memory.pt; all results go
# to $CLS_RUNS/idea4/persist/<OUTTAG>.json.

set -euo pipefail
REPO=${REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/sensory-key-retrieval}
cd "$REPO"
module load miniforge/24.3.0-0
module load cuda/13.0.1
source activate cls
unset CUDA_VISIBLE_DEVICES
source scripts/cls_env.sh
R=$CLS_RUNS/idea4
mkdir -p "$R/persist"
CKPTS=()
for tag in ${TAGS}; do CKPTS+=("$R/experience_${tag}_s0/memory.pt"); done
python -u -m analysis.idea4.persist --ckpt "${CKPTS[@]}" --arenas "${ARENAS:-6}" \
    --iters "${ITERS:-40}" --lifetimes 64 ${EXTRA:-} \
    --out "$R/persist/${OUTTAG:-persist_a${ARENAS:-6}_i${ITERS:-40}}.json"
echo "[i4persist] done $(date)"
