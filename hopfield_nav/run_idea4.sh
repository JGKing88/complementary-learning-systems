#!/bin/bash -l
#SBATCH --job-name=idea4
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=pi_fiete,ou_bcs_normal
#SBATCH --mem=32G
#SBATCH --array=0-3
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/logs/idea4_%A_%a.out

# Idea 4 step 1 (GRID_MLP_NAV_PLAN §7b.2): train the recurrent memory on
# random-walk lifetimes and run idea 2's retention test, one array task per
# GRU width.
#
#   REPO=<code snapshot> sbatch hopfield_nav/run_idea4.sh
#   REPO=... HIDDENS="2048" TAG=long EXTRA="--T_min 20 --T_max 100" sbatch --array=0 ...
#
# Writes $CLS_RUNS/idea4/h<H>[_<TAG>]_s<SEED>/{log.json,memory.pt,eval_step*.json}.
# A rerun with the same output dir resumes from its memory.pt.

set -euo pipefail

REPO=${REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/sensory-key-retrieval}
read -r -a HS <<< "${HIDDENS:-256 512 1024 2048}"
H=${HS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEED:-0}
STEPS=${STEPS:-20000}
TAG=${TAG:-}
EXTRA=${EXTRA:-}
# SCRIPT=train_experience: goals from experience (grid phases + at-goal flag), no writes.
SCRIPT=${SCRIPT:-train_memory}

cd "$REPO"
module load miniforge/24.3.0-0
module load cuda/13.0.1
source activate cls
unset CUDA_VISIBLE_DEVICES
source scripts/cls_env.sh

OUT=$CLS_RUNS/idea4/${SCRIPT#train_}_h${H}${TAG:+_$TAG}_s${SEED}
mkdir -p "$OUT"
RESUME=()
if [ -f "$OUT/memory.pt" ]; then RESUME=(--init_ckpt "$OUT/memory.pt"); fi
echo "[idea4] hidden $H seed $SEED -> $OUT  repo $REPO  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -u -m analysis.idea4.$SCRIPT --out "$OUT" --hidden "$H" --seed "$SEED" \
    --steps "$STEPS" "${RESUME[@]}" $EXTRA
echo "[idea4] done $(date)"
