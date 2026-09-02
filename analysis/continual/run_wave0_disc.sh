#!/bin/bash -l
#SBATCH --job-name=cl-wave0d
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=200G
#SBATCH --partition=ou_bcs_normal
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jackking@mit.edu
#SBATCH --output=/home/jackking/cls/hopfield_nav/logs/slurm_cl_wave0d_%j.out
set -uo pipefail

# =============================================================================
# P1 of the discrete suite: Wave 0's axes, re-run under discrete movement.
#
# These have to exist before any discrete method number means anything. "Is
# this forgetting or capacity?" is a question about an architecture in an
# action space, and it RESETS when the action space changes -- a discrete
# retention number read against the continuous joint ceiling is read against
# the wrong denominator.
#
#   T0.1  joint / multi-task ceiling, swept over capacity.  THE BLOCKING RUN.
#   T0.4  from-scratch sequential. The floor.
#
# T0.3 (oracle reachability) already ran, inline, before this job was written:
#
#     discrete    reached 1.0000  mean_steps 13.2  worst-case 38 / 200 budget
#     continuous  reached 1.0000  mean_steps 10.7  worst-case 27 / 200 budget
#
# So the action space costs 1.23x more steps -- Manhattan against Euclidean,
# less than the sqrt(2) worst case because most goals are off-diagonal -- and
# the step cap is not binding in either space, with 5x headroom. Task
# difficulty is unchanged. That is what licenses reading a discrete control
# against the continuous agenthash runs, and it is why the store does not also
# have to be re-run.
#
# T0.2 (per-env experts) is deferred exactly as it was in the continuous wave:
# its only job is to measure capacity interference as T0.1 - T0.2, which is
# not worth building unless T0.1 lands below ~0.9. Decide after this job.
#
# Sizing: env-stepping-bound at batch_envs=1, so the axis is CPU fan-out and
# every task pins itself to one thread. No GPU on purpose -- a 128-unit GRU on
# a batch of one spends its time in kernel-launch overhead. The GPU partition
# is used for the pretraining job, where batch_envs=32 makes it real.
# =============================================================================

module load miniforge/24.3.0-0
source activate cls

REPO="${CL_REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/continual-control-suite}"
cd "$REPO"
source scripts/cls_env.sh

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

OUT="${CLS_HISTORIES}/wave0d"
LOGS="$REPO/hopfield_nav/logs/wave0d"
mkdir -p "$OUT" "$LOGS"

# --- shared world/protocol shape, matched to the continuous wave -------------
N_ENVS=5
SIZE=20
OBS=60
MOVEMENT=discrete            # the whole point of this wave
MAX_STEPS=200
ITERS=200
GOAL_RADIUS=0.5

PREV_ACTION="--input_prev_action"

echo "[wave0d] repo=$REPO"
echo "[wave0d] out=$OUT"
echo "[wave0d] cpus=${SLURM_CPUS_PER_TASK:-$(nproc)}"
echo "[wave0d] started $(date -Is)"

PIDS=()
NAMES=()

launch () {   # launch <name> <logfile> <cmd...>
    local name="$1"; shift
    local log="$1"; shift
    "$@" > "$log" 2>&1 &
    PIDS+=($!); NAMES+=("$name")
}

# =============================================================================
# T0.1 -- joint / multi-task ceiling, swept over capacity.
#
# Same shape as the continuous wave so the two panels compare: batch_envs=1
# matches the sequential regime, n_updates=1000 gives 5x the sequential budget
# on purpose (a ceiling should not be budget-limited), and the eval history is
# written every 25 updates so the budget-matched point at update 200 is
# recoverable from the same run.
# =============================================================================
echo "[wave0d] T0.1 joint ceiling: 4 widths x 2 depths x 3 seeds"
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
            --save_dir "$CLS_RUNS/rnn/wave0d_${TAG}"
done; done; done

# =============================================================================
# T0.4 -- from-scratch sequential. The floor.
#
# Two arms, mirroring the continuous wave so the comparison is arm-for-arm.
# `noprev` is the bare input set; `prev` is the settled surface. Note that
# --init_log_std / --freeze_log_std, which the continuous A2 arm sweeps, are
# DEAD here: a Categorical head has no log_std. That sweep has to be replaced
# by something with meaning in this action space (move_ent_coef) when the
# method waves are built -- it must not be silently carried over as a knob
# that turns nothing, which is the failure this project keeps finding.
# =============================================================================
echo "[wave0d] T0.4 from-scratch sequential: 2 arms x 20 seeds"
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

echo "[wave0d] launched ${#PIDS[@]} tasks; waiting"

FAILED=()
for k in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$k]}"; then FAILED+=("${NAMES[$k]}"); fi
done

echo "[wave0d] finished $(date -Is)"
if (( ${#FAILED[@]} )); then
    echo "[wave0d] ${#FAILED[@]} FAILED: ${FAILED[*]}" >&2
else
    echo "[wave0d] all ${#PIDS[@]} tasks OK"
fi

python -u -m analysis.continual.wave0_summary --dir "$OUT" \
    | tee "$LOGS/summary.txt"

exit 0
