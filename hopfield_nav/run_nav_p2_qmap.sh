#!/bin/bash -l
#SBATCH --job-name=p2qmap
#SBATCH --time=0-02:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=mit_normal_gpu
#SBATCH --mem=96G
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/logs/nav_p2_qmap_%j.out

# P1: where does q fail to point at the goal? See EXPERIMENTS_NAV_P2 §5.
#
#   SEED=0 sbatch hopfield_nav/run_nav_p2_qmap.sh
#
# One job per scaffold seed. Seed varies both the env placement and the
# distractor draws, so running several and comparing them is what turns
# "between-world variance" from a caveat into a measured quantity -- phase 1's
# finding 19 came from noticing two worlds disagreed by 13x, with no way to say
# whether that was typical.
#
# 96G because the Npos=1716 encoded_Phi is ~12 GB and the build peaks well above
# it. No policy runs here; the GPU is for the batched Hopfield recalls.

set -euo pipefail

REPO=${REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/nav-tri-metric}
SEED=${SEED:-0}
ENVS=${ENVS:-32}
DRAWS=${DRAWS:-8}
CKPT=${CKPT:-/orcd/pool/003/jackking/cls_runs/agent_ckpts/navigate_navtri_w6_pers_s42_20499183/navigate_u1950.pt}
OUT=${OUT:-/orcd/pool/003/jackking/cls_runs/results/nav_p2}

module load miniforge/24.3.0-0
source activate cls
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd "$REPO"
source scripts/cls_env.sh
mkdir -p "$OUT"

python -u -m analysis.nav_p2.q_failure_map \
    --ckpt "$CKPT" --device cuda \
    --envs "$ENVS" --draws "$DRAWS" --seed "$SEED" \
    --n_distractors 0 1 2 3 4 5 6 7 8 9 10 \
    --out "$OUT/qmap_seed${SEED}.npz"
