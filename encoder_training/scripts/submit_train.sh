#!/bin/bash -l
#SBATCH --job-name=et_train
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --partition=pi_evelina9
#SBATCH --output=/home/jackking/cls/encoder_training/scripts/logs/slurm-%j.out

# ==========================================================================
# Edit the hyperparameters below, then: sbatch submit_train.sh
# Leave a variable empty ("") to fall back to the default in train.py.
# ==========================================================================

RUN_NAME="run_$(date +%Y%m%d_%H%M%S)"

# --- Model ------------------------------------------------------------------
ENCODER_TYPE=mlp              # mlp | cnn
LAMBDAS="11 12 13"            # space-separated
OUT_DIM=1024
HIDDEN_DIM=1024
NUM_HIDDEN_LAYERS=4
HIDDEN_CHANNELS=128           # cnn only
NUM_CONV_LAYERS=3             # cnn only
KERNEL_SIZE=5                 # cnn only

# --- Patches ----------------------------------------------------------------
# Use either NPOS_LIST (variable-size, comma-separated) OR NENV/NPOS (fixed).
# If NPOS_LIST is non-empty it overrides NENV/NPOS.
# NPOS_LIST="40,60,80,100,120,140,40,60,80,100,120,140,40,60,80,100,120,100,100,100,100,100,100"
# NPOS_LIST="50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,200,200,200,200,200"
NENV=60
NPOS=100

# Radius: use PER_ENV_RADIUS_FRAC (e.g. 0.1) OR RADIUS (fixed, in cells).
PER_ENV_RADIUS_FRAC=0.1
# RADIUS=10.0
SINGLE_ENV_BATCH=1            # 1 to enable --single_env_batch, 0 for mixed

# --- Loss -------------------------------------------------------------------
LOSS_MODE=mse_contrastive     # mse_contrastive | cka
ATTRACT_LAMBDA=2.0
REPEL_WEIGHT=5.0              # used only for mse_contrastive
UNIFORMITY_LAMBDA=0.0
UNIFORMITY_ANNEAL_EPOCHS=25

# --- Training ---------------------------------------------------------------
EPOCHS=1000
# LR=2.48e-4
LR=1e-4
BATCH_SIZE=8192
# 4096
SEED=42
FWHM_RATIO=0.25
GAIN_START=1.0
GAIN_END=5.0
SHUFFLE_INPUTS=0              # 1 to permute grid codes across positions (ablation)

# --- Nav eval ---------------------------------------------------------------
EVAL_EVERY=25
NAV_ENV_SIZE=20
NAV_N_TRAIN=5
NAV_N_VAL=5
NAV_NUM_HOPFIELDS=20
NAV_N_STARTS=100

# --- Save dir ---------------------------------------------------------------
SAVE_DIR=/home/jackking/cls/encoders

# ==========================================================================

module load miniforge/24.3.0-0
module load cuda/13.0.1
source activate cls
unset CUDA_VISIBLE_DEVICES

WORKDIR=/orcd/home/002/jackking/cls
cd $WORKDIR
mkdir -p "$WORKDIR/encoder_training/logs"
LOG="$WORKDIR/encoder_training/logs/${RUN_NAME}.log"

# Build flag list dynamically so empty vars fall back to train.py defaults.
ARGS=(--run_name "$RUN_NAME")

# Model
[ -n "$ENCODER_TYPE" ]        && ARGS+=(--encoder_type "$ENCODER_TYPE")
[ -n "$LAMBDAS" ]             && ARGS+=(--lambdas $LAMBDAS)     # unquoted: list
[ -n "$OUT_DIM" ]             && ARGS+=(--out_dim "$OUT_DIM")
[ -n "$HIDDEN_DIM" ]          && ARGS+=(--hidden_dim "$HIDDEN_DIM")
[ -n "$NUM_HIDDEN_LAYERS" ]   && ARGS+=(--num_hidden_layers "$NUM_HIDDEN_LAYERS")
[ -n "$HIDDEN_CHANNELS" ]     && ARGS+=(--hidden_channels "$HIDDEN_CHANNELS")
[ -n "$NUM_CONV_LAYERS" ]     && ARGS+=(--num_conv_layers "$NUM_CONV_LAYERS")
[ -n "$KERNEL_SIZE" ]         && ARGS+=(--kernel_size "$KERNEL_SIZE")

# Patches
if [ -n "$NPOS_LIST" ]; then
    ARGS+=(--npos_list "$NPOS_LIST")
else
    [ -n "$NENV" ] && ARGS+=(--nenv "$NENV")
    [ -n "$NPOS" ] && ARGS+=(--npos "$NPOS")
fi
[ -n "$PER_ENV_RADIUS_FRAC" ] && ARGS+=(--per_env_radius_frac "$PER_ENV_RADIUS_FRAC")
[ -n "$RADIUS" ]              && ARGS+=(--radius "$RADIUS")
[ "$SINGLE_ENV_BATCH" = "1" ] && ARGS+=(--single_env_batch)

# Loss
[ -n "$LOSS_MODE" ]                && ARGS+=(--loss_mode "$LOSS_MODE")
[ -n "$ATTRACT_LAMBDA" ]           && ARGS+=(--attract_lambda "$ATTRACT_LAMBDA")
[ -n "$REPEL_WEIGHT" ]             && ARGS+=(--repel_weight "$REPEL_WEIGHT")
[ -n "$UNIFORMITY_LAMBDA" ]        && ARGS+=(--uniformity_lambda "$UNIFORMITY_LAMBDA")
[ -n "$UNIFORMITY_ANNEAL_EPOCHS" ] && ARGS+=(--uniformity_anneal_epochs "$UNIFORMITY_ANNEAL_EPOCHS")

# Training
[ -n "$EPOCHS" ]      && ARGS+=(--epochs "$EPOCHS")
[ -n "$LR" ]          && ARGS+=(--lr "$LR")
[ -n "$BATCH_SIZE" ]  && ARGS+=(--batch_size "$BATCH_SIZE")
[ -n "$SEED" ]        && ARGS+=(--seed "$SEED")
[ -n "$FWHM_RATIO" ]  && ARGS+=(--fwhm_ratio "$FWHM_RATIO")
[ -n "$GAIN_START" ]  && ARGS+=(--gain_start "$GAIN_START")
[ -n "$GAIN_END" ]    && ARGS+=(--gain_end "$GAIN_END")
[ "$SHUFFLE_INPUTS" = "1" ] && ARGS+=(--shuffle)

# Nav eval
[ -n "$EVAL_EVERY" ]        && ARGS+=(--eval_every "$EVAL_EVERY")
[ -n "$NAV_ENV_SIZE" ]      && ARGS+=(--nav_env_size "$NAV_ENV_SIZE")
[ -n "$NAV_N_TRAIN" ]       && ARGS+=(--nav_n_train "$NAV_N_TRAIN")
[ -n "$NAV_N_VAL" ]         && ARGS+=(--nav_n_val "$NAV_N_VAL")
[ -n "$NAV_NUM_HOPFIELDS" ] && ARGS+=(--nav_num_hopfields "$NAV_NUM_HOPFIELDS")
[ -n "$NAV_N_STARTS" ]      && ARGS+=(--nav_n_starts "$NAV_N_STARTS")

[ -n "$SAVE_DIR" ] && ARGS+=(--save_dir "$SAVE_DIR")

echo "Host: $(hostname)" | tee "$LOG"
echo "Run: $RUN_NAME" | tee -a "$LOG"
echo "Log: $LOG" | tee -a "$LOG"
echo "Flags: ${ARGS[@]}" | tee -a "$LOG"
echo "----- start python -----" | tee -a "$LOG"

python -u -m encoder_training.train "${ARGS[@]}" >> "$LOG" 2>&1

RC=$?
echo "----- exit code: $RC -----" | tee -a "$LOG"
grep "Val nav:" "$LOG" | tail -15
