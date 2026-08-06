#!/bin/bash -l
#SBATCH --job-name=hnav-agenthash
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jackking@mit.edu
#SBATCH --partition=pi_evelina9
#SBATCH --mem=100G
#SBATCH --output=/home/jackking/cls/hopfield_nav/logs/slurm_agenthash_%j.out
set -euo pipefail

# === edit these =============================================================
RUN_NAME="driven-snowflake-111"
CKPT="/home/jackking/cls/checkpoint/phase_a_only_driven-snowflake-111/phase_a_u360.pt"   # phase_a_only_glamorous-field-97/phase_a_u360.pt" # required: path to phased final.pt
#phase_a_only_glamorous-field-97/phase_a_u360.pt
#phase_a_only_glamorous-valley-101/phase_a_u380.pt
#phase_a_only_lively-surf-104/phase_a_u880.pt
#phase_a_only_driven-snowflake-111/phase_a_u360.pt
#phase_a_only_pretty-totem-110/phase_a_u540.pt

N_ENVS=5                        # empty = use ckpt's cfg.num_val_envs
ITERS_PER_BLOCK=200
MAX_STEPS=100
SEED=2000
ENV_SEED=2000                       # empty → use ckpt's cfg.seed (legacy). Set to baseline's SEED to match envs across scripts.
NUM_FULL_ITERS=100               # >1 → repeat protocol with seeds [SEED, SEED+1, ...] for mean ± 1σ bands
KEEP_PER_ITER_HISTORIES=0       # 1 to keep the intermediate ${RUN_NAME}_iter*.json files after merging

# Shared scaffold cache: build encoded_Phi ONCE before fanning out to N
# parallel subruns. With USE_MMAP=1, all subruns share the OS page cache for
# encoded_Phi instead of each holding a private ~3-6 GiB copy. The cache is
# content-addressed (key = lambdas + fwhm_ratio + Npos + encoder_path), so any
# future job with the same scaffold params reuses it instead of rebuilding.
USE_SCAFFOLD_CACHE=1
USE_MMAP=1
SCAFFOLD_CACHE_ROOT="$CLS_SCAFFOLD_CACHE"
KEEP_SCAFFOLD_CACHE=1           # default keep — the cross-job cache is meant to persist

# Optional cfg overrides
ENCODER_OVERRIDE=""

# Store-control flags: 1=enable, 0=disable
STATIC_VECTORHASH=1
LOCK_STORE_AFTER_GOAL=1
ORACLE_STORE_AT_GOAL=1
ORACLE_LOCK_STORE_NOT_AT_GOAL=1
STOCHASTIC_POLICY=0

# Online store-head training (BCE on agent.store_head during the rollout).
# Composes with the oracle flags above: oracle gates Hopfield writes; this
# trains store_head weights so eventually the head fires correctly on its own.
TRAIN_STORE=0
TRAIN_STORE_LR=1e-5
TRAIN_STORE_UPDATES_PER_ROLLOUT=1

# Override EnvConfig.goal_radius from the saved checkpoint. Empty → use saved value.
# 2026-08-05: at GOAL_RADIUS=1 an at-goal position can snap to a neighbouring
# cell. Stores fired there now write the GOAL cell's embedding rather than the
# neighbour's (EnvConfig.allow_offcell_store, default False). Pass
# --allow_offcell_store to reproduce figures generated before this date.
GOAL_RADIUS=1

# Plotting
SMOOTH=10
SHOW_STD=0                      # 1 to overlay ±1 SEM bands when NUM_FULL_ITERS>1
# ============================================================================

if [[ -z "$CKPT" ]]; then
    echo "ERROR: set CKPT at the top of $0" >&2
    exit 1
fi

module load miniforge/24.3.0-0
module load cuda/13.0.1
source activate cls
unset CUDA_VISIBLE_DEVICES

cd /home/jackking/cls

source scripts/cls_env.sh
HIST="$CLS_HISTORIES/${RUN_NAME}.json"
PLOT_PREFIX="$CLS_FIGURES/model_comparison/${RUN_NAME}"
SCAFFOLD_DIR=""   # resolved by prep_scaffold from $SCAFFOLD_CACHE_ROOT

EXTRA_FLAGS=()
[[ -n "$N_ENVS" ]]                              && EXTRA_FLAGS+=(--n_envs "$N_ENVS")
[[ -n "$ENCODER_OVERRIDE" ]]                    && EXTRA_FLAGS+=(--encoder_override "$ENCODER_OVERRIDE")
[[ "$STATIC_VECTORHASH" == "1" ]]               && EXTRA_FLAGS+=(--static_vectorhash)
[[ "$LOCK_STORE_AFTER_GOAL" == "1" ]]           && EXTRA_FLAGS+=(--lock_store_after_goal)
[[ "$ORACLE_STORE_AT_GOAL" == "1" ]]            && EXTRA_FLAGS+=(--oracle_store_at_goal)
[[ "$ORACLE_LOCK_STORE_NOT_AT_GOAL" == "1" ]]   && EXTRA_FLAGS+=(--oracle_lock_store_not_at_goal)
[[ "$STOCHASTIC_POLICY" == "1" ]]               && EXTRA_FLAGS+=(--stochastic_policy)
[[ -n "$GOAL_RADIUS" ]]                         && EXTRA_FLAGS+=(--goal_radius "$GOAL_RADIUS")
if [[ "$TRAIN_STORE" == "1" ]]; then
    EXTRA_FLAGS+=(--train_store
                  --train_store_lr "$TRAIN_STORE_LR"
                  --train_store_updates_per_rollout "$TRAIN_STORE_UPDATES_PER_ROLLOUT")
fi

# Soft warning: NUM_FULL_ITERS shares the GPU + CPUs allocated to this job.
N_CPUS="${SLURM_CPUS_PER_TASK:-$(nproc 2>/dev/null || echo 1)}"
if (( NUM_FULL_ITERS > N_CPUS )); then
    echo "WARNING: NUM_FULL_ITERS=$NUM_FULL_ITERS > available CPUs ($N_CPUS); " \
         "you may want to bump #SBATCH --cpus-per-task." >&2
fi

# Resolve / build the shared scaffold cache once (sequential), before fanning
# out. prep_scaffold is idempotent: if the content-keyed dir already exists it
# prints CACHE HIT to stderr and skips the build. Stdout is exactly one line:
# the resolved cache directory.
PER_ITER_CACHE_FLAGS=()
if [[ "$USE_SCAFFOLD_CACHE" == "1" ]]; then
    PREP_FLAGS=()
    [[ -n "$ENCODER_OVERRIDE" ]]      && PREP_FLAGS+=(--encoder_override "$ENCODER_OVERRIDE")
    [[ "$STATIC_VECTORHASH" == "1" ]] && PREP_FLAGS+=(--static_vectorhash)
    SCAFFOLD_DIR=$(python -u -m analysis.continual.prep_scaffold \
        --ckpt "$CKPT" --cache_root "$SCAFFOLD_CACHE_ROOT" "${PREP_FLAGS[@]}" \
        | tail -n 1)
    echo "[launcher] scaffold cache: $SCAFFOLD_DIR"
    PER_ITER_CACHE_FLAGS+=(--scaffold_cache "$SCAFFOLD_DIR")
    [[ "$USE_MMAP" == "1" ]] && PER_ITER_CACHE_FLAGS+=(--mmap)
fi

# Launch NUM_FULL_ITERS subruns in parallel.
PER_ITER_HISTS=()
PIDS=()
for k in $(seq 0 $((NUM_FULL_ITERS - 1))); do
    SEED_K=$((SEED + k))
    HIST_K="$CLS_HISTORIES/${RUN_NAME}_iter${k}.json"
    PER_ITER_HISTS+=("$HIST_K")
    LOG_K="hopfield_nav/logs/${RUN_NAME}_iter${k}.log"
    mkdir -p "$(dirname "$LOG_K")"

    # Per-iter env_seed offset: only meaningful when ENV_SEED is set.
    PER_ITER_ENV_FLAGS=()
    if [[ -n "$ENV_SEED" ]]; then
        PER_ITER_ENV_FLAGS+=(--env_seed $((ENV_SEED + k)))
    fi

    python -u -m analysis.continual.agenthash \
        --out "$HIST_K" \
        --run_name "${RUN_NAME}_iter${k}" \
        --ckpt "$CKPT" \
        --iters_per_block "$ITERS_PER_BLOCK" \
        --max_steps "$MAX_STEPS" \
        --seed "$SEED_K" \
        --num_full_iters 1 \
        "${EXTRA_FLAGS[@]}" \
        "${PER_ITER_ENV_FLAGS[@]}" \
        "${PER_ITER_CACHE_FLAGS[@]}" \
        > "$LOG_K" 2>&1 &
    PIDS+=($!)
    echo "[launcher] iter $k seed=$SEED_K pid=${PIDS[-1]} log=$LOG_K"
done

# Wait for all subruns; abort if any failed.
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

# Merge per-iter histories → final HIST.
python -u -m analysis.continual.merge_histories \
    --inputs "${PER_ITER_HISTS[@]}" \
    --out "$HIST" \
    --run_name "$RUN_NAME"

# Optionally clean up the intermediates.
if [[ "$KEEP_PER_ITER_HISTORIES" != "1" ]]; then
    rm -f "${PER_ITER_HISTS[@]}"
fi
if [[ "$USE_SCAFFOLD_CACHE" == "1" && "$KEEP_SCAFFOLD_CACHE" != "1" && -n "$SCAFFOLD_DIR" ]]; then
    rm -rf "$SCAFFOLD_DIR"
fi

# Render plots from the merged history.
python -u -m analysis.continual.plotting \
    --history "$HIST" \
    --out_prefix "$PLOT_PREFIX" \
    --smooth "$SMOOTH" \
    --show_std "$SHOW_STD"
