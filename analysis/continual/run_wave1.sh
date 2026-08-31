#!/bin/bash -l
#SBATCH --job-name=cl-wave1
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=220G
#SBATCH --partition=ou_bcs_normal
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jackking@mit.edu
#SBATCH --output=/home/jackking/cls/hopfield_nav/logs/slurm_cl_wave1_%j.out
set -uo pipefail

# =============================================================================
# Wave 1 -- docs/CONTINUAL_CONTROLS_PLAN.md section 6.
#
#   A   Tier-1 tuning of the PRETRAINED control (W1-W4, W6)
#   A2  Tier-1 tuning of the FROM-SCRATCH control (W5)
#   B   Experience Replay, buffer size x replay ratio
#   C   Online EWC, lambda over decades
#   R   a matched `none` reference so B and C are read against the right number
#
# Independent of T0.1: the joint ceiling says how to *interpret* these numbers,
# not what to run, so this does not wait on wave0b.
#
# **Why the pretrained arm is primary.** T0.4 came back with the from-scratch
# control at 0.05 retained and only ~0.55 on the env it is currently training
# on, with 71-74% of envs never reaching criterion at all. It is not merely
# forgetting; it is barely learning, so differences between methods measured on
# it would be noise on top of a broken control. The pretrained arm reaches ~0.99
# on the current env, which is where a retention difference can actually show.
#
# **Why init_log_std is only in the from-scratch arm.** `movement_log_std` is a
# Parameter, so `load_state_dict` overwrites whatever `--init_log_std` asked for
# whenever a checkpoint is loaded. Sweeping it on the pretrained arm would sweep
# a value that never takes effect -- exactly the kind of silently-inert knob
# that W5 was about in the first place.
# =============================================================================

module load miniforge/24.3.0-0
source activate cls

REPO="${CL_REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/continual-control-suite}"
cd "$REPO"
source scripts/cls_env.sh

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

OUT="$CLS_HISTORIES/wave1"
LOGS="$REPO/hopfield_nav/logs/wave1"
mkdir -p "$OUT" "$LOGS"

CKPT="/home/jackking/cls/checkpoint_rnn/pretrain_20x20/final.pt"
if [[ ! -f "$CKPT" ]]; then
    echo "[wave1] FATAL: pretrain checkpoint missing: $CKPT" >&2
    exit 1
fi

N_ENVS=5; SIZE=20; OBS=60; MOVEMENT=continuous
MAX_STEPS=200; ITERS=200; SEEDS=8

COMMON=(--n_envs "$N_ENVS" --iters_per_block "$ITERS" --max_steps "$MAX_STEPS"
        --size "$SIZE" --observation_size "$OBS" --movement_mode "$MOVEMENT"
        --goal_radius 0.5 --num_full_iters 1 --steps_per_rollout "$MAX_STEPS"
        --hidden_size 128 --num_rnn_layers 1 --max_grad_norm 1.0 --device cpu)

echo "[wave1] repo=$REPO  cpus=${SLURM_CPUS_PER_TASK:-$(nproc)}  started $(date -Is)"

PIDS=(); NAMES=()
launch () {  # launch <tag> <extra args...>
    local tag="$1"; shift
    python -u -m analysis.continual.baseline \
        --out "$OUT/${tag}.json" --run_name "$tag" --no-world_spec \
        "${COMMON[@]}" "$@" > "$LOGS/${tag}.log" 2>&1 &
    PIDS+=($!); NAMES+=("$tag")
}

# --- A: Tier-1 tuning, pretrained arm ---------------------------------------
# lr is the strongest forgetting/plasticity knob and was never swept; the
# optimizer reset is W2; epochs is the only live gradient-budget knob at
# batch_envs=1 (n_minibatches cannot split a batch of one).
for LR in 3e-4 1e-3 3e-3; do
for RST in 0 1; do
for EP in 1 4; do
for S in $(seq 1 $SEEDS); do
    F=(); [[ "$RST" == "1" ]] && F+=(--reset_optimizer_each_block)
    launch "A_lr${LR}_rst${RST}_ep${EP}_s${S}" \
        --load_checkpoint "$CKPT" --seed "$S" \
        --lr "$LR" --epochs "$EP" --n_minibatches 1 --batch_envs 1 "${F[@]}"
done; done; done; done

# --- A-batch: the W1 sensitivity condition ----------------------------------
# batch_envs=1 stays the headline regime (it is what makes the x-axis read as
# episodes consumed). This one condition measures the single real residual:
# whether the gradient noise of one autocorrelated trajectory is itself doing
# part of the forgetting.
for S in $(seq 1 $SEEDS); do
    launch "Abatch_be16_s${S}" \
        --load_checkpoint "$CKPT" --seed "$S" \
        --lr 1e-3 --epochs 1 --n_minibatches 4 --batch_envs 16
done

# --- A2: Tier-1 tuning, from-scratch arm (W5) -------------------------------
for ILS in 0.0 -1.0 -1.5; do
for LR in 1e-3 3e-3; do
for S in $(seq 1 $SEEDS); do
    launch "A2_ils${ILS}_lr${LR}_s${S}" \
        --seed "$S" --input_prev_action \
        --init_log_std "$ILS" --freeze_log_std \
        --lr "$LR" --epochs 1 --n_minibatches 1 --batch_envs 1
done; done; done

# --- R: matched reference for B and C ---------------------------------------
# Same pretrained arm, same knobs, method=none. B and C are meaningless without
# a `none` run at exactly their configuration.
for S in $(seq 1 $SEEDS); do
    launch "R_none_s${S}" \
        --load_checkpoint "$CKPT" --seed "$S" \
        --lr 1e-3 --epochs 1 --n_minibatches 1 --batch_envs 1 --method none
done

# --- B: Experience Replay ---------------------------------------------------
# buffer_size=inf is the perfect-memory bound (plan 0.1): the whole 5x200 stream
# is ~192 MB, so it is free, and it doubles as the GDumb answer. The bounded
# sizes are what put ER on the memory axis rather than at a single point.
for BUF in inf 200 50 10; do
for RB in 1 4; do
for S in $(seq 1 $SEEDS); do
    launch "B_er_buf${BUF}_rb${RB}_s${S}" \
        --load_checkpoint "$CKPT" --seed "$S" \
        --lr 1e-3 --epochs 1 --n_minibatches 1 --batch_envs 1 \
        --method er --method_args "buffer_size=${BUF},replay_batches=${RB},sampling=balanced"
done; done; done

# --- C: Online EWC ----------------------------------------------------------
# lambda over six decades. An under-tuned lambda is the standard way EWC gets
# accidentally strawmanned, and the stability/plasticity curve as it rises is
# itself a figure.
for LAM in 1 10 100 1000 10000 100000; do
for S in $(seq 1 $SEEDS); do
    launch "C_ewc_lam${LAM}_s${S}" \
        --load_checkpoint "$CKPT" --seed "$S" \
        --lr 1e-3 --epochs 1 --n_minibatches 1 --batch_envs 1 \
        --method online_ewc --method_args "lam=${LAM},gamma=1.0,fisher=true"
done; done

echo "[wave1] launched ${#PIDS[@]} tasks; waiting"

FAILED=()
for k in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$k]}"; then FAILED+=("${NAMES[$k]}"); fi
done

echo "[wave1] finished $(date -Is)"
if (( ${#FAILED[@]} )); then
    echo "[wave1] ${#FAILED[@]} FAILED: ${FAILED[*]:0:20}" >&2
else
    echo "[wave1] all ${#PIDS[@]} tasks OK"
fi

python -u -m analysis.continual.wave1_summary --dir "$OUT" \
    | tee "$LOGS/summary.txt"

exit 0
