#!/bin/bash -l
#SBATCH --job-name=cl-wave0
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=200G
#SBATCH --partition=ou_bcs_normal
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jackking@mit.edu
#SBATCH --output=/home/jackking/cls/hopfield_nav/logs/slurm_cl_wave0_%j.out
set -uo pipefail

# =============================================================================
# Wave 0 of the continual-control suite -- see docs/CONTINUAL_CONTROLS_PLAN.md
# section 2. These are not continual-learning methods; they are the axes of the
# plot, and three of them have never been run.
#
#   T0.1  joint / multi-task ceiling, swept over capacity.  THE BLOCKING RUN.
#         If this comes in low, the existing forgetting figure is partly a
#         capacity figure and every Tier-2 number would be uninterpretable.
#   T0.3  BFS-oracle reachability under the eval step cap ("is the eval even
#         possible"). Pure env question, no agent -- runs in seconds.
#   T0.4  from-scratch sequential. The floor. Every recorded history is
#         mode=finetune, so the paper's own from-scratch control has no run.
#
# T0.2 (per-env experts) is deliberately deferred: its only job is to measure
# capacity interference as T0.1 - T0.2, which is not worth building unless T0.1
# lands below ~0.9. Decide after this job.
#
# `train_rnn --mode mixed` and `analysis.continual.baseline` both do
# torch.manual_seed(s); np.random.seed(s); rng=RandomState(s); rnn_world(cfg,rng)
# -- so a given --seed builds the SAME envs in both. That is what makes T0.1 a
# ceiling for exactly the envs T0.4 walks through, rather than for a similar
# world.
#
# Sizing: measured ~1000 env-steps/s on one CPU core at batch_envs=1. This
# workload is env-stepping-bound (a 128-unit GRU on a batch of 1 barely touches
# an accelerator), so the axis is CPU fan-out and every task pins itself to a
# single thread. No GPU is requested on purpose.
# =============================================================================

module load miniforge/24.3.0-0
source activate cls

REPO="${CL_REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/continual-control-suite}"
cd "$REPO"
source scripts/cls_env.sh

# One thread per task. Without this, N torch processes each spawn ~ncpu threads
# and the node spends its time in the scheduler instead of in the envs.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

OUT="${CLS_HISTORIES}/wave0"
LOGS="$REPO/hopfield_nav/logs/wave0"
mkdir -p "$OUT" "$LOGS"

# --- shared world/protocol shape, matched to the existing headline figure ----
N_ENVS=5
SIZE=20
OBS=60
MOVEMENT=continuous          # matches every agenthash headline run (plan 3.3)
MAX_STEPS=200
ITERS=200
GOAL_RADIUS=0.5

# --- the settled agent surface ----------------------------------------------
# input_prev_action is ON per the review decision. It required a bug fix first:
# both the collector and the evaluator built the channel only when a previous
# action existed, so t=0 fed a narrower input than the trunk was sized for and
# torch raised on the first forward. See tests/test_prev_action_channel.py.
PREV_ACTION="--input_prev_action"

echo "[wave0] repo=$REPO"
echo "[wave0] out=$OUT"
echo "[wave0] cpus=${SLURM_CPUS_PER_TASK:-$(nproc)}"
echo "[wave0] started $(date -Is)"

PIDS=()
NAMES=()

launch () {   # launch <name> <logfile> <cmd...>
    local name="$1"; shift
    local log="$1"; shift
    "$@" > "$log" 2>&1 &
    PIDS+=($!); NAMES+=("$name")
}

# =============================================================================
# T0.3 -- oracle reachability. Seconds, and it gates the meaning of every
# `reached` number below, so it runs first and inline.
# =============================================================================
echo "[wave0] T0.3 oracle reachability"
python -u -m analysis.continual.oracle_ceiling \
    --n_envs "$N_ENVS" --size "$SIZE" --observation_size "$OBS" \
    --movement_mode "$MOVEMENT" --max_steps "$MAX_STEPS" \
    --goal_radius "$GOAL_RADIUS" --n_trials 256 --seeds 1 2 3 4 5 \
    --out "$OUT/T0.3_oracle.json" 2>&1 | tee "$LOGS/T0.3_oracle.log"

# =============================================================================
# T0.1 -- joint / multi-task ceiling, swept over capacity.
#
# batch_envs=1 matches the sequential regime (one rollout per env per update),
# so the only thing joint training gets that the stream does not is
# simultaneity. n_updates=1000 gives it 5x the sequential budget on purpose:
# this is a ceiling, and a ceiling should not be budget-limited. The eval
# history is written every 25 updates, so the budget-matched point (update 200)
# is recoverable from the same run.
# =============================================================================
echo "[wave0] T0.1 joint ceiling: 4 widths x 2 depths x 3 seeds"
for HID in 128 256 512 1024; do
for LAY in 1 2; do
for SEED in 1 2 3; do
    TAG="T0.1_h${HID}_l${LAY}_s${SEED}"
    launch "$TAG" "$LOGS/${TAG}.log" \
        python -u -m hopfield_nav.train_rnn \
            --mode mixed \
            --n_envs "$N_ENVS" --n_updates 1000 \
            --size "$SIZE" --observation_size "$OBS" \
            --movement_mode "$MOVEMENT" --goal_radius "$GOAL_RADIUS" \
            --hidden_size "$HID" --num_rnn_layers "$LAY" \
            $PREV_ACTION \
            --lr 1e-3 --epochs 1 --n_minibatches 1 \
            --batch_envs 1 --steps_per_rollout "$MAX_STEPS" \
            --n_eval_trials 32 --eval_max_steps "$MAX_STEPS" --eval_every 25 \
            --seed "$SEED" --device cpu \
            --save_dir "$CLS_RUNS/rnn/wave0_${TAG}"
done; done; done

# =============================================================================
# T0.4 -- from-scratch sequential. The floor.
#
# Two arms so the new input channel is not confounded with the missing control:
#   noprev  reproduces the recorded configuration exactly (minus pretraining),
#           which is what makes it comparable to the existing histories;
#   prev    is the settled surface everything downstream will use.
# =============================================================================
echo "[wave0] T0.4 from-scratch sequential: 2 arms x 20 seeds"
for ARM in noprev prev; do
    FLAG=""
    [[ "$ARM" == "prev" ]] && FLAG="$PREV_ACTION"
    for SEED in $(seq 1 20); do
        TAG="T0.4_${ARM}_s${SEED}"
        launch "$TAG" "$LOGS/${TAG}.log" \
            python -u -m analysis.continual.baseline \
                --out "$OUT/${TAG}.json" --run_name "$TAG" \
                --n_envs "$N_ENVS" --iters_per_block "$ITERS" \
                --max_steps "$MAX_STEPS" --size "$SIZE" \
                --observation_size "$OBS" --movement_mode "$MOVEMENT" \
                --goal_radius "$GOAL_RADIUS" \
                --seed "$SEED" --num_full_iters 1 \
                --hidden_size 128 --num_rnn_layers 1 \
                $FLAG \
                --lr 1e-3 --epochs 1 --n_minibatches 1 \
                --batch_envs 1 --steps_per_rollout "$MAX_STEPS" \
                --max_grad_norm 1.0 --device cpu
    done
done

echo "[wave0] launched ${#PIDS[@]} tasks; waiting"

FAILED=()
for k in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$k]}"; then FAILED+=("${NAMES[$k]}"); fi
done

echo "[wave0] finished $(date -Is)"
if (( ${#FAILED[@]} )); then
    echo "[wave0] ${#FAILED[@]} FAILED: ${FAILED[*]}" >&2
else
    echo "[wave0] all ${#PIDS[@]} tasks OK"
fi

# --- summarise straight away so the job's own log answers the question -------
python -u -m analysis.continual.wave0_summary --dir "$OUT" \
    | tee "$LOGS/summary.txt"

exit 0
