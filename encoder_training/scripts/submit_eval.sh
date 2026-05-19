#!/bin/bash -l
#SBATCH --job-name=et_eval
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --partition=pi_evelina9
#SBATCH --output=/home/jackking/cls/encoder_training/scripts/logs/slurm-%j.out

# ==========================================================================
# Edit below, then: sbatch submit_eval.sh
# Leave a variable empty ("") to fall back to the default in evaluate_nav.py.
# ==========================================================================

# CKPT=/home/jackking/cls/encoders/binary_20260420_183304/encoder_best.pt
# mid MSE

# CKPT=/home/jackking/cls/encoders/binary_20260409_083227/encoder_final.pt
# great accuracy but a little slow. not trained with MSE

# CKPT=/home/jackking/cls/encoders/run_20260421_194532/encoder_best.pt
# mid MSE

CKPT=/home/jackking/cls/encoders/run_20260422_185816/encoder_best.pt
# MSE AND actually good

CKPT=/home/jackking/cls/encoders/run_20260424_003439/encoder_best.pt

# --- Nav eval overrides ---
ENV_SIZE=20
N_TRAIN_ENVS=5
N_VAL_ENVS=5
NUM_HOPFIELDS=50
N_STARTS_PER_ENV=100
PLATFORM_RADIUS=1.0
MAX_STEPS_MULT=3
SCALE=1.0
NORMALIZE=1               # 0 or 1
RECOMPUTE_INTERVAL=1
HOPFIELD_ALPHA=0.8

# --- Meta ---
INCLUDE_TRAIN_EVAL=1      # 1 to also eval on training patches
EMIT_JSON=1               # 1 to emit a JSON line
SEED=42

# ==========================================================================

module load miniforge/24.3.0-0
module load cuda/13.0.1
source activate cls
unset CUDA_VISIBLE_DEVICES

WORKDIR=/orcd/home/002/jackking/cls
cd $WORKDIR
mkdir -p "$WORKDIR/encoder_training/logs"
STEM=$(basename "$(dirname "$CKPT")")
LOG="$WORKDIR/encoder_training/logs/eval_${STEM}.log"

# Build flag list dynamically — empty vars fall back to evaluate_nav.py defaults.
ARGS=(--ckpt "$CKPT" --seed "$SEED")

[ -n "$ENV_SIZE" ]           && ARGS+=(--env_size "$ENV_SIZE")
[ -n "$N_TRAIN_ENVS" ]       && ARGS+=(--n_train_envs "$N_TRAIN_ENVS")
[ -n "$N_VAL_ENVS" ]         && ARGS+=(--n_val_envs "$N_VAL_ENVS")
[ -n "$NUM_HOPFIELDS" ]      && ARGS+=(--num_hopfields "$NUM_HOPFIELDS")
[ -n "$N_STARTS_PER_ENV" ]   && ARGS+=(--n_starts_per_env "$N_STARTS_PER_ENV")
[ -n "$PLATFORM_RADIUS" ]    && ARGS+=(--platform_radius "$PLATFORM_RADIUS")
[ -n "$MAX_STEPS_MULT" ]     && ARGS+=(--max_steps_mult "$MAX_STEPS_MULT")
[ -n "$SCALE" ]              && ARGS+=(--scale "$SCALE")
[ -n "$NORMALIZE" ]          && ARGS+=(--normalize "$NORMALIZE")
[ -n "$RECOMPUTE_INTERVAL" ] && ARGS+=(--recompute_interval "$RECOMPUTE_INTERVAL")
[ -n "$HOPFIELD_ALPHA" ]     && ARGS+=(--hopfield_alpha "$HOPFIELD_ALPHA")

[ "$INCLUDE_TRAIN_EVAL" = "1" ] && ARGS+=(--train_eval)
[ "$EMIT_JSON" = "1" ]          && ARGS+=(--json)

echo "Host: $(hostname)" | tee "$LOG"
echo "Ckpt: $CKPT" | tee -a "$LOG"
echo "Flags: ${ARGS[@]}" | tee -a "$LOG"
echo "----- start -----" | tee -a "$LOG"

# Pre-flight: fail fast with a clear message if the checkpoint isn't there.
if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found: $CKPT" | tee -a "$LOG"
    DIR=$(dirname "$CKPT")
    if [ -d "$DIR" ]; then
        echo "  directory exists; contents:" | tee -a "$LOG"
        ls -la "$DIR" | tee -a "$LOG"
    else
        echo "  directory does not exist: $DIR" | tee -a "$LOG"
        PARENT=$(dirname "$DIR")
        STEM=$(basename "$DIR")
        if [ -d "$PARENT" ]; then
            echo "  nearby entries in $PARENT:" | tee -a "$LOG"
            ls "$PARENT" | grep -E "${STEM:0:15}|${STEM: -15}" | tee -a "$LOG" || \
                ls "$PARENT" | tail -10 | tee -a "$LOG"
        fi
    fi
    exit 1
fi

python -u -m encoder_training.evaluate_nav "${ARGS[@]}" >> "$LOG" 2>&1
RC=$?
echo "----- exit code: $RC -----" | tee -a "$LOG"

if [ "$RC" -ne 0 ]; then
    echo "===== FAILED — tail of log =====" | tee -a "$LOG"
    tail -30 "$LOG"
    exit "$RC"
fi

grep -E "Val nav:|Train nav:" "$LOG"
