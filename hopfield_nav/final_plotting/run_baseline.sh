#!/bin/bash -l
#SBATCH --job-name=hnav-baseline
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jackking@mit.edu
#SBATCH --partition=pi_evelina9
#SBATCH --mem=100G
#SBATCH --output=/home/jackking/cls/hopfield_nav/logs/slurm_baseline_%j.out
set -euo pipefail

# === edit these =============================================================
RUN_NAME="baseline_regular_200steps"
LOAD_CKPT="/home/jackking/cls/checkpoint_rnn/pretrain_20x20/final.pt"                # leave empty for from-scratch sequential
N_ENVS=5
ITERS_PER_BLOCK=200
MAX_STEPS=200
SIZE=20
OBSERVATION_SIZE=60
MOVEMENT_MODE=continuous     # continuous | discrete (auto-restored from ckpt in finetune)
SEED=1
NUM_FULL_ITERS=30             # >1 → repeat protocol with seeds [SEED, SEED+1, ...] for mean ± 1σ bands
GOAL_RADIUS=0.5               # Euclidean radius around goal that counts as 'at goal'. 0.5 = snap-equality.
KEEP_PER_ITER_HISTORIES=0    # 1 to keep the intermediate ${RUN_NAME}_iter*.json files after merging

# Model / BC knobs
HIDDEN_SIZE=128
NUM_RNN_LAYERS=1
DROPOUT=0.0
LR=1e-3
MOVE_ENT_COEF=0.0
MAX_GRAD_NORM=1.0
EPOCHS=1
N_MINIBATCHES=1
BATCH_ENVS=1
STEPS_PER_ROLLOUT=

# If want more training than steps:
# EPOCHS=4
# N_MINIBATCHES=4
# BATCH_ENVS=16
# STEPS_PER_ROLLOUT=64

# Optional flags: 1=enable, 0=disable
INPUT_PREV_ACTION=0
INPUT_PREV_REWARD=0
INPUT_GRID_STATE=0
FWHM_RATIO=0.25         # smoothing for the gbook lookup; only used when INPUT_GRID_STATE=1
LAMBDAS="11 12 13"         # VectorHash module periods (space-separated). Auto-restored from ckpt in finetune.
ONLY_TRAIN_ON_REACHED=0 # per-update, drop trajectories whose rollout never reached goal

# Plotting
SMOOTH=10
SHOW_STD=0                   # 1 to overlay ±1 SEM bands when NUM_FULL_ITERS>1
# ============================================================================

module load miniforge/24.3.0-0
module load cuda/13.0.1
source activate cls
unset CUDA_VISIBLE_DEVICES

cd /home/jackking/cls

source scripts/cls_env.sh
HIST="$CLS_HISTORIES/${RUN_NAME}.json"
PLOT_PREFIX="hopfield_nav/final_plotting/final_figures/${RUN_NAME}"

EXTRA_FLAGS=()
[[ -n "$LOAD_CKPT" ]]                 && EXTRA_FLAGS+=(--load_checkpoint "$LOAD_CKPT")
[[ -n "$STEPS_PER_ROLLOUT" ]]         && EXTRA_FLAGS+=(--steps_per_rollout "$STEPS_PER_ROLLOUT")
[[ "$INPUT_PREV_ACTION" == "1" ]]     && EXTRA_FLAGS+=(--input_prev_action)
[[ "$INPUT_PREV_REWARD" == "1" ]]     && EXTRA_FLAGS+=(--input_prev_reward)
[[ "$INPUT_GRID_STATE" == "1" ]]      && EXTRA_FLAGS+=(--input_grid_state)
[[ "$ONLY_TRAIN_ON_REACHED" == "1" ]] && EXTRA_FLAGS+=(--only_train_on_reached)

# Soft warning: NUM_FULL_ITERS shares the GPU + CPUs allocated to this job.
N_CPUS="${SLURM_CPUS_PER_TASK:-$(nproc 2>/dev/null || echo 1)}"
if (( NUM_FULL_ITERS > N_CPUS )); then
    echo "WARNING: NUM_FULL_ITERS=$NUM_FULL_ITERS > available CPUs ($N_CPUS); " \
         "you may want to bump #SBATCH --cpus-per-task." >&2
fi

# Launch NUM_FULL_ITERS subruns in parallel. Each gets seed = SEED + k and writes
# its own per-iter JSON. We merge them after all finish.
PER_ITER_HISTS=()
PIDS=()
for k in $(seq 0 $((NUM_FULL_ITERS - 1))); do
    SEED_K=$((SEED + k))
    HIST_K="$CLS_HISTORIES/${RUN_NAME}_iter${k}.json"
    PER_ITER_HISTS+=("$HIST_K")
    LOG_K="hopfield_nav/logs/${RUN_NAME}_iter${k}.log"
    mkdir -p "$(dirname "$LOG_K")"

    python -u -m hopfield_nav.final_plotting.baseline \
        --out "$HIST_K" \
        --run_name "${RUN_NAME}_iter${k}" \
        --n_envs "$N_ENVS" \
        --iters_per_block "$ITERS_PER_BLOCK" \
        --max_steps "$MAX_STEPS" \
        --size "$SIZE" \
        --observation_size "$OBSERVATION_SIZE" \
        --movement_mode "$MOVEMENT_MODE" \
        --goal_radius "$GOAL_RADIUS" \
        --seed "$SEED_K" \
        --num_full_iters 1 \
        --hidden_size "$HIDDEN_SIZE" \
        --num_rnn_layers "$NUM_RNN_LAYERS" \
        --dropout "$DROPOUT" \
        --lr "$LR" \
        --move_ent_coef "$MOVE_ENT_COEF" \
        --epochs "$EPOCHS" \
        --n_minibatches "$N_MINIBATCHES" \
        --batch_envs "$BATCH_ENVS" \
        --max_grad_norm "$MAX_GRAD_NORM" \
        --fwhm_ratio "$FWHM_RATIO" \
        --lambdas $LAMBDAS \
        "${EXTRA_FLAGS[@]}" \
        > "$LOG_K" 2>&1 &
    PIDS+=($!)
    echo "[launcher] iter $k seed=$SEED_K pid=${PIDS[-1]} log=$LOG_K"
done

# Wait for all subruns; collect failures.
RC=0
for k in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$k]}"; then
        echo "[launcher] iter $k FAILED — see ${PER_ITER_HISTS[$k]%.json}.log" >&2
        RC=1
    fi
done
if (( RC != 0 )); then
    echo "[launcher] aborting: at least one iter failed" >&2
    exit "$RC"
fi
echo "[launcher] all $NUM_FULL_ITERS iters done"

# Merge per-iter histories → final HIST, with metadata.run_name = $RUN_NAME.
python -u -m hopfield_nav.final_plotting.merge_histories \
    --inputs "${PER_ITER_HISTS[@]}" \
    --out "$HIST" \
    --run_name "$RUN_NAME"

# Optionally clean up the intermediates.
if [[ "$KEEP_PER_ITER_HISTORIES" != "1" ]]; then
    rm -f "${PER_ITER_HISTS[@]}"
fi

# Render plots from the merged history.
python -u -m hopfield_nav.final_plotting.plotting \
    --history "$HIST" \
    --out_prefix "$PLOT_PREFIX" \
    --smooth "$SMOOTH" \
    --show_std "$SHOW_STD"
