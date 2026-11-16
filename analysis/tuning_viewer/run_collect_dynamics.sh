#!/bin/bash -l
#SBATCH --job-name=dyn-collect
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=ou_bcs_normal
#SBATCH --mem=48G
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/logs/dyn_collect_%j.out

# Controller runs and Hopfield inputs for the dynamics tuning viewer.
#
#   sbatch analysis/tuning_viewer/run_collect_dynamics.sh            # full
#   SMOKE=1 sbatch --time=00:30:00 analysis/tuning_viewer/run_collect_dynamics.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
module load miniforge/24.3.0-0
source activate cls
source scripts/cls_env.sh

SMOKE=${SMOKE:-0}
if [ "$SMOKE" = "1" ]; then
  OUT=${OUT:-$CLS_RESULTS/tuning_viewer/dynamics_smoke}
  EXTRA=--smoke
else
  OUT=${OUT:-$CLS_RESULTS/tuning_viewer/dynamics}
  EXTRA=
fi
echo "[dyn-collect] $(git rev-parse HEAD) -> $OUT"
python -u -m analysis.tuning_viewer.collect_dynamics --out "$OUT" $EXTRA
