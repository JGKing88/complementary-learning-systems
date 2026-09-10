#!/bin/bash -l
#SBATCH --job-name=cl-wave0b
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=160G
#SBATCH --partition=ou_bcs_normal
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jackking@mit.edu
#SBATCH --output=/home/jackking/cls/hopfield_nav/logs/slurm_cl_wave0b_%j.out
set -uo pipefail

# =============================================================================
# T0.1, done properly. The first attempt (run_wave0.sh) was wrong and this is
# the correction.
#
# That run gave the joint ceiling 1000 updates at epochs=1, batch_envs=1, and
# came back with ~0.45-0.54 across every capacity from 128 to 1024 against an
# oracle of 1.000. Read naively that says "the network cannot hold 5 envs at
# once" -- a capacity result, and a big claim.
#
# It is not what happened. The eval curves are still climbing steeply where the
# budget ends (hidden=128 goes 0.47 -> 0.66 over its last 100 updates) and no
# capacity plateaus, which is the signature of an under-optimised run rather
# than a saturated one. The cause is arithmetic: at batch_envs=1 and epochs=1,
# 1000 updates is 1000 gradient steps -- against ~1M timesteps of data. The run
# was optimisation-starved, not capacity-limited.
#
# Two changes:
#
#   epochs 1 -> 8   Eight passes over the same five rollouts. Gradient steps go
#                   up 8x at zero extra environment cost, which is the cheap
#                   axis when the bottleneck is optimisation and the collector
#                   is what costs wall-clock.
#   1000 -> 8000    64,000 joint gradient steps against the original 1,000.
#
# Note this run deliberately does NOT respect the online regime. T0.1 is an
# *offline* reference -- the best a single network can do given all the envs at
# once -- so hobbling it with the streaming protocol's one-step-per-rollout
# rule would understate the ceiling, which is the one thing a ceiling must not
# do. n_minibatches stays 1 so every gradient step sees all five envs, which is
# what makes it joint training rather than round-robin.
#
# The capacity axis is narrowed to {128, 512} x {1, 2}: the first run already
# showed capacity is not the binding constraint in this range, so the budget is
# better spent on convergence and on an lr axis, which was never swept at all.
# =============================================================================

module load miniforge/24.3.0-0
source activate cls

REPO="${CL_REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/continual-control-suite}"
cd "$REPO"
source scripts/cls_env.sh

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

LOGS="$REPO/hopfield_nav/logs/wave0b"
mkdir -p "$LOGS"

N_ENVS=5
SIZE=20
OBS=60
MOVEMENT=continuous
MAX_STEPS=200
N_UPDATES=8000

echo "[wave0b] repo=$REPO  cpus=${SLURM_CPUS_PER_TASK:-$(nproc)}  started $(date -Is)"

PIDS=(); NAMES=()
for HID in 128 512; do
for LAY in 1 2; do
for LR in 1e-3 3e-3; do
for SEED in 1 2 3; do
    TAG="T0.1b_h${HID}_l${LAY}_lr${LR}_s${SEED}"
    python -u -m hopfield_nav.train_rnn \
        --mode mixed \
        --n_envs "$N_ENVS" --n_updates "$N_UPDATES" \
        --size "$SIZE" --observation_size "$OBS" \
        --movement_mode "$MOVEMENT" --goal_radius 0.5 \
        --hidden_size "$HID" --num_rnn_layers "$LAY" \
        --input_prev_action \
        --lr "$LR" --epochs 8 --n_minibatches 1 \
        --batch_envs 1 --steps_per_rollout "$MAX_STEPS" \
        --n_eval_trials 32 --eval_max_steps "$MAX_STEPS" --eval_every 100 \
        --seed "$SEED" --device cpu \
        --save_dir "$CLS_RUNS/rnn/wave0b_${TAG}" \
        > "$LOGS/${TAG}.log" 2>&1 &
    PIDS+=($!); NAMES+=("$TAG")
done; done; done; done

echo "[wave0b] launched ${#PIDS[@]} tasks; waiting"

FAILED=()
for k in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$k]}"; then FAILED+=("${NAMES[$k]}"); fi
done

echo "[wave0b] finished $(date -Is)"
if (( ${#FAILED[@]} )); then
    echo "[wave0b] ${#FAILED[@]} FAILED: ${FAILED[*]}" >&2
else
    echo "[wave0b] all ${#PIDS[@]} tasks OK"
fi

python -u -m analysis.continual.wave0_summary \
    --dir "$CLS_HISTORIES/wave0" --runs_root "$CLS_RUNS" \
    | tee "$LOGS/summary.txt"

exit 0
