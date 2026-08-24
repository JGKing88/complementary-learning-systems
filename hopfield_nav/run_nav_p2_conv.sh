#!/bin/bash -l
#SBATCH --job-name=p2conv
#SBATCH --time=0-01:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=mit_normal_gpu
#SBATCH --mem=96G
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/logs/nav_p2_conv_%j.out

# Are the imperfect recalls non-converged, or converged onto mixtures?
# See analysis/nav_p2/recall_convergence.py -- Jack's spurious-fixed-point
# objection, tested rather than argued.

set -euo pipefail
REPO=${REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/nav-tri-metric}
CKPT=${CKPT:-/orcd/pool/003/jackking/cls_runs/agent_ckpts/navigate_navtri_w6_pers_s42_20499183/navigate_u1950.pt}

module load miniforge/24.3.0-0
source activate cls
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd "$REPO"
source scripts/cls_env.sh

PROBE=${PROBE:-convergence}
if [ "$PROBE" = dynamics ]; then
    python -u -m analysis.nav_p2.recall_dynamics \
        --ckpt "$CKPT" --device cuda \
        --n_distractors "${NDIST:-10}" --steps "${STEPS:-4000}"
else
    python -u -m analysis.nav_p2.recall_convergence \
        --ckpt "$CKPT" --device cuda \
        --envs "${ENVS:-16}" --draws "${DRAWS:-4}" \
        --n_distractors "${NDIST:-10}" --steps "${STEPS:-12}"
fi
