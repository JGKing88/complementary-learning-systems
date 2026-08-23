#!/bin/bash -l
#SBATCH --job-name=navtrigate2
#SBATCH --time=0-03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=pi_evelina9
#SBATCH --mem=100G
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/logs/nav_tri_gate2_%j.out

# The magnitude-gating crossover, done properly.
#
# run_nav_tri_gating.sh clamped ||q|| to a constant, which also destroyed its
# variation within a trajectory -- and P0.8 showed that variation IS the cue
# (approaching a real goal shrinks ||q||, approaching a phantom does not). So
# that version moved the level and deleted the dynamics together, and its
# result was uninterpretable: the explore arm behaved as magnitude-gating
# predicts (+77% chasing at a stronger signal) and the nav arm behaved the
# opposite way.
#
# --q_scale multiplies ||q|| instead, preserving the shape of ||q||(t). A dose
# response here -- following that rises with the factor in BOTH regimes -- is
# evidence the policy reads the level. Flat curves mean it reads the dynamics
# or the direction and not the level at all.
#
# Swept over a range that brackets the natural levels: goal-present ||q|| is
# ~0.22-0.26 and decoy-only ~0.14-0.17, so 0.5x takes a goal signal below decoy
# strength and 2x takes a decoy signal well above goal strength.

set -euo pipefail
CKPT=${CKPT:-/orcd/pool/003/jackking/cls_runs/agent_ckpts/navigate_navtri_w6_pers_s42_20499183/navigate_u1950.pt}
OUT=${OUT:-/orcd/pool/003/jackking/cls_runs/results/nav_tri_probe}
NDIST=${NDIST:-10}

module load miniforge/24.3.0-0
source activate cls
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
cd /orcd/home/002/jackking/cls/.claude/worktrees/nav-tri-metric
source scripts/cls_env.sh
mkdir -p "$OUT"

for k in 0.5 1.0 2.0 4.0; do
    for mode in nav explore; do
        echo ""
        echo "######## mode=$mode  q_scale=${k}x ########"
        python -u -m analysis.nav_tri.behavior_probe --ckpt "$CKPT" --device cpu \
            --mode "$mode" --n_distractors "$NDIST" --trials 32 --envs 8 \
            --max_steps 200 --q_scale "$k" \
            --json "$OUT/gate2_${mode}_x${k}.json"
    done
done
