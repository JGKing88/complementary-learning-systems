#!/bin/bash -l
#SBATCH --job-name=cl-calib-disc
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --partition=pi_fiete
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jackking@mit.edu
#SBATCH --output=/home/jackking/cls/hopfield_nav/logs/slurm_cl_calib_disc_%j.out
set -uo pipefail

# =============================================================================
# P3: where to sweep each regulariser against the DISCRETE objective.
#
# Blocking for run_wave2d.sh, which refuses to start until its ranges have been
# filled in from this table. The discrete loss is a Categorical(4) cross-entropy
# near ln(4)=1.39 where the continuous suite's was a Gaussian NLL of order 10,
# so a coefficient carried across unchanged is wrong by roughly that ratio.
#
# One process per method rather than one for all five: they are independent,
# the node has the cores, and a failure in one method's sweep should not cost
# the other four.
# =============================================================================

module load miniforge/24.3.0-0
source activate cls

REPO="${CL_REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/continual-control-suite}"
cd "$REPO"
source scripts/cls_env.sh

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

CKPT="$CLS_CKPTS_RNN/pretrain_20x20_discrete/final.pt"
OUT="$CLS_RESULTS"
LOGS="$REPO/hopfield_nav/logs/wave2d"
mkdir -p "$OUT" "$LOGS"

if [[ ! -f "$CKPT" ]]; then
    echo "[calib] FATAL: discrete checkpoint missing: $CKPT" >&2
    exit 1
fi

echo "[calib] ckpt=$CKPT"
echo "[calib] started $(date -Is)"

PIDS=(); NAMES=()
for M in online_ewc si lwf clear derpp; do
    python -u -m analysis.continual.calibrate_discrete \
        --ckpt "$CKPT" --methods "$M" \
        --n_envs 3 --updates 25 --lr 1e-3 --seed 1 \
        --out "$OUT/calibrate_discrete_${M}.json" \
        > "$LOGS/calibration_${M}.txt" 2>&1 &
    PIDS+=($!); NAMES+=("$M")
done

FAILED=()
for k in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$k]}"; then FAILED+=("${NAMES[$k]}"); fi
done

echo "[calib] finished $(date -Is)"
if (( ${#FAILED[@]} )); then
    echo "[calib] FAILED: ${FAILED[*]}" >&2
fi

echo
echo "================ CALIBRATION TABLE ================"
for M in online_ewc si lwf clear derpp; do
    echo
    cat "$LOGS/calibration_${M}.txt"
done
echo "==================================================="
echo
echo "Fill the ranges into run_wave2d.sh (the decades bracketing ratio ~1e-1)"
echo "and set CALIBRATED=yes."

exit 0
