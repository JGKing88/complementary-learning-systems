#!/bin/bash -l
#SBATCH --job-name=cl-ic-ub-gpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --gres=gpu:4
#SBATCH --partition=ou_bcs_low
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jackking@mit.edu
#SBATCH --output=/home/jackking/cls/hopfield_nav/logs/slurm_cl_ic_ub_gpu_%j.out
set -uo pipefail

# =============================================================================
# The h=1024 arms of the section 5.2 upper-bound wave, on GPUs.
#
# Split out from the CPU job for a measured reason rather than a hunch. With
# the rollout batched (n_envs 4 x batch_envs 4 rather than 16 x 1), the time
# split inverts:
#
#   h=1024, batch_envs=1    rollout 3.46s   update 1.69s    rollout dominates
#   h=1024, batch_envs=16   rollout 0.65s   update 1.62s    update dominates
#
# The rollout is sequential single-step policy calls -- launch-latency bound,
# and the regime a GPU is *worst* at, which is why moving the old
# batch_envs=1 configuration to a GPU would have been neutral at best. Once it
# is batched, 71% of the wall time is the BC update: a 200-step GRU forward and
# backward over a real batch at hidden 1024, fused by cuDNN. That is GPU-shaped.
#
# h=256 and h=512 stay on CPU in run_incontext_ub.sh, where they finish in a
# few hours and a GPU would buy little.
#
# Four GPUs, twelve processes, three per GPU -- these models are ~6M parameters
# and one H100 is nowhere near saturated by a single one of them.
# =============================================================================

module load miniforge/24.3.0-0
source activate cls

REPO="${CL_REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/continual-control-suite}"
cd "$REPO"
source scripts/cls_env.sh

# Fewer threads than the CPU job: with the heavy matmuls on the GPU, the CPU
# side is env stepping and numpy, which does not thread well.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

CKPTS="$CLS_RUNS/rnn"
OUT="$CLS_HISTORIES/incontext_ub"
LOGS="$REPO/hopfield_nav/logs/incontext_ub"
mkdir -p "$OUT" "$LOGS"

POOL=4
PARALLEL=4
UPDATES=8000
LIFETIME_UPDATES=10
EPISODE_CAP=200
SIZE=20
OBS=60
STEPS=200
SEEDS=3
H=1024
N_GPUS=4

echo "[ic-ub-gpu] repo=$REPO  gpus=$N_GPUS  started $(date -Is)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true

TRAIN_COMMON=(--mode mixed --n_envs "$POOL" --n_updates "$UPDATES"
              --size "$SIZE" --observation_size "$OBS"
              --movement_mode continuous --goal_radius 0.5
              --hidden_size "$H" --num_rnn_layers 1
              --input_prev_action --input_prev_reward
              --lr 1e-3 --epochs 4 --n_minibatches 4
              --batch_envs "$PARALLEL" --steps_per_rollout "$STEPS"
              --n_eval_trials 16 --eval_max_steps "$STEPS" --eval_every 500
              --device cuda --n_holdout_envs 16
              --resample_envs_every "$LIFETIME_UPDATES"
              --episode_max_steps "$EPISODE_CAP")

PIDS=(); NAMES=(); SLOT=0
train () {
    local tag="$1"; shift
    CUDA_VISIBLE_DEVICES=$((SLOT % N_GPUS)) \
    python -u -m hopfield_nav.train_rnn "${TRAIN_COMMON[@]}" "$@" \
        --save_dir "$CKPTS/icub_${tag}" > "$LOGS/${tag}.log" 2>&1 &
    PIDS+=($!); NAMES+=("$tag"); SLOT=$((SLOT + 1))
}

for S in $(seq 1 $SEEDS); do
    train "ceilabs_h${H}_s${S}" --seed "$S" --goal_channel abs \
        --carry_across_episodes
    train "ic_h${H}_s${S}"      --seed "$S" --carry_across_episodes
    train "ep_h${H}_s${S}"      --seed "$S"
    train "carry_h${H}_s${S}"   --seed "$S" --goal_channel abs \
        --goal_visible_episodes 1 --carry_across_episodes
done

echo "[ic-ub-gpu] launched ${#PIDS[@]} runs; waiting"
FAILED=()
for k in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$k]}"; then FAILED+=("${NAMES[$k]}"); fi
done
echo "[ic-ub-gpu] pretraining done $(date -Is)"
if (( ${#FAILED[@]} )); then
    echo "[ic-ub-gpu] ${#FAILED[@]} FAILED: ${FAILED[*]:0:20}" >&2
fi

# --- the in-context measurement --------------------------------------------
EPIDS=(); ENAMES=()
for S in $(seq 1 $SEEDS); do
    LT="$CKPTS/icub_ic_h${H}_s${S}/final.pt"
    EP="$CKPTS/icub_ep_h${H}_s${S}/final.pt"
    if [[ -f "$LT" && -f "$EP" ]]; then
        python -u -m analysis.continual.incontext \
            --out "$OUT/icub_h${H}_s${S}.json" \
            --load_checkpoint "$LT" --control_checkpoint "$EP" \
            --n_envs 8 --seed $((9000 + S)) \
            --size "$SIZE" --observation_size "$OBS" --movement_mode continuous \
            --n_lifetimes 64 --n_episodes 10 --max_steps "$STEPS" \
            --device cpu > "$LOGS/eval_h${H}_s${S}.log" 2>&1 &
        EPIDS+=($!); ENAMES+=("h${H}_s${S}")
    fi
    CK="$CKPTS/icub_carry_h${H}_s${S}/final.pt"
    if [[ -f "$CK" ]]; then
        python -u -m analysis.continual.incontext \
            --out "$OUT/carry_h${H}_s${S}.json" \
            --load_checkpoint "$CK" \
            --n_envs 8 --seed $((9000 + S)) \
            --size "$SIZE" --observation_size "$OBS" --movement_mode continuous \
            --n_lifetimes 64 --n_episodes 10 --max_steps "$STEPS" \
            --goal_visible_episodes 1 \
            --device cpu > "$LOGS/eval_carry_h${H}_s${S}.log" 2>&1 &
        EPIDS+=($!); ENAMES+=("carry_h${H}_s${S}")
    fi
done
for k in "${!EPIDS[@]}"; do
    if ! wait "${EPIDS[$k]}"; then echo "[ic-ub-gpu] eval ${ENAMES[$k]} FAILED" >&2; fi
done

echo "[ic-ub-gpu] finished $(date -Is)"
exit 0
