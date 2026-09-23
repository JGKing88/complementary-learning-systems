#!/bin/bash -l
#SBATCH --job-name=xf_reeval
#SBATCH --time=0-02:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=ou_bcs_normal
#SBATCH --mem=64G
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/logs/xf_reeval_%j.out
#
# Re-score a run's checkpoint series SAMPLED (docs/EXPLORE_FIRST_PLAN.md §1):
# the trainer's explore eval is deterministic, and under the kappa cap the
# mean policy falls into orbits the sampled policy does not, so the
# deterministic series swings +-0.15 between neighbouring checkpoints. The
# checkpoint the forks start from is chosen on this series, not on that one.
#
#   RUN_DIR=$CLS_CKPTS/navigate_navp2_xf_explorer_s42_22889945 \
#   TAG=xf_explorer sbatch hopfield_nav/run_xf_reeval.sh
set -euo pipefail
REPO=${REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/explore-first}
cd "$REPO"
module load miniforge/24.3.0-0
module load cuda/13.0.1
source activate cls
unset CUDA_VISIBLE_DEVICES
source scripts/cls_env.sh
export PYTHONPATH="$REPO"
: "${RUN_DIR:?set RUN_DIR to the run directory holding navigate_u*.pt}"
TAG=${TAG:-$(basename "$RUN_DIR")}
OUT=${OUT:-$CLS_RESULTS/explore_first}
mkdir -p "$OUT"
python -u -m analysis.nav_tri.reeval_series \
    --run_dir "$RUN_DIR" --every "${EVERY:-25}" --no-deterministic \
    --seed "${SEED:-42}" --device cuda \
    --out "$OUT/${TAG}_reeval_stoch.log"
echo "ALL DONE"
