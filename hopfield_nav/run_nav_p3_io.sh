#!/bin/bash -l
#SBATCH --job-name=p3io
#SBATCH --time=0-06:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=ou_bcs_normal
#SBATCH --mem=128G
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/logs/nav_p3_io_%j.out

# P3 -- the ideal observer. EXPERIMENTS_NAV_P2 §7.
#
#   PROBE=gen    analysis/nav_p2/ideal_observer.py        (feature tensors)
#   PROBE=fit    analysis/nav_p2/ideal_observer_fit.py    (AUC, ablation, probes)
#   PROBE=score  analysis/nav_p2/ideal_observer_score.py  (§7.3 item 5)
#
# Inspect with `SCRIPT=hopfield_nav/run_nav_p3_io.sh bash hopfield_nav/check_variants.sh <v>`;
# never `source` it.

set -euo pipefail
REPO=${REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/nav-tri-metric}
CKPT=${CKPT:-/orcd/pool/003/jackking/cls_runs/agent_ckpts/navigate_navtri_w6_pers_s42_20499183/navigate_u1950.pt}
OUTDIR=${OUTDIR:-/orcd/pool/003/jackking/cls_runs/results/nav_p2}

module load miniforge/24.3.0-0
source activate cls
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd "$REPO"
source scripts/cls_env.sh
mkdir -p "$OUTDIR"

PROBE=${PROBE:-gen}
TAG=${TAG:-probe}
NPZ=${NPZ:-$OUTDIR/io_$TAG.npz}

if [ "$PROBE" = fit ]; then
    python -u -m analysis.nav_p2.ideal_observer_fit \
        --npz "$NPZ" \
        --probe "${FITPROBE:-billiard}" \
        --folds "${FOLDS:-6}" \
        --json_out "$OUTDIR/io_${TAG}_main.json" \
        ${DO:+--do $DO} \
        ${TARGETS:+--targets $TARGETS}
elif [ "$PROBE" = score ]; then
    python -u -m analysis.nav_p2.ideal_observer_score \
        --npz "$NPZ" \
        --policy_ckpts ${POLICY_CKPTS:?set POLICY_CKPTS} \
        --envs "${ENVS:-24}" --draws "${DRAWS:-4}" \
        --steps "${STEPS:-64}" \
        --n_distractors ${NDIST:-0 1 3 10} \
        --out "$OUTDIR/io_${TAG}_agents.npz" \
        --device cuda
else
    python -u -m analysis.nav_p2.ideal_observer \
        --ckpt "$CKPT" \
        --envs "${ENVS:-48}" --draws "${DRAWS:-8}" \
        --starts "${STARTS:-2}" --steps "${STEPS:-64}" \
        --n_distractors ${NDIST:-0 1 2 3 5 7 10} \
        --step_norm "${STEPNORM:-1.0}" \
        --seed "${SEED:-0}" \
        --out "$NPZ" --device cuda
fi
