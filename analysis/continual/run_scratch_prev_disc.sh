#!/bin/bash -l
#SBATCH --job-name=cl-scratch-pd
#SBATCH --time=23:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=220G
#SBATCH --partition=ou_bcs_normal
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jackking@mit.edu
#SBATCH --output=/home/jackking/cls/hopfield_nav/logs/slurm_cl_scratch_pd_%j.out
set -uo pipefail

# Partition: ou_bcs_normal, not pi_fiete, whose QOS caps the group at cpu=48
# against this job's 96 -- such a job never starts.

# =============================================================================
# From scratch, WITH the previous-action channel.
#
# Plan decision #1 settled `--input_prev_action` on. It has been on in exactly
# one arm of the whole suite -- A2, the from-scratch tuning sweep -- and off in
# 616 of 664 continuous runs and 424 of 472 discrete ones. Not because anyone
# overrode it: the pretraining checkpoint was built without the channel, and
# `restore_arch_from_ckpt` takes `input_prev_action` from the checkpoint
# because the two imply different input widths. Every arm that loads that
# checkpoint therefore runs without it regardless of its command line.
#
# The companion run_scratch.sh keeps the channel OFF so that pretrained-vs-
# scratch differs in exactly one variable. That is the right contrast and the
# wrong configuration: T0.4 measured the channel directly and it is worth
# +0.045 current-env in discrete, while costing 0.069 in continuous.
#
#     continuous   with 0.510 current-env   without 0.579
#     discrete     with 0.879               without 0.834
#
# So this wave asks the other question -- what the best from-scratch
# configuration actually reaches -- on the naive control and the methods worth
# the compute, rather than the full sweep. Read against run_scratch.sh
# arm-for-arm, the difference is the channel and nothing else.
# =============================================================================

module load miniforge/24.3.0-0
source activate cls

REPO="${CL_REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/continual-control-suite}"
cd "$REPO"
source scripts/cls_env.sh

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

OUT="$CLS_HISTORIES/wave1d_fsp"
LOGS="$REPO/hopfield_nav/logs/scratch_pd"
mkdir -p "$OUT" "$LOGS"

N_ENVS=5; SIZE=20; OBS=60; MOVEMENT=discrete
MAX_STEPS=200; ITERS=200; SEEDS=8

COMMON=(--n_envs "$N_ENVS" --iters_per_block "$ITERS" --max_steps "$MAX_STEPS"
        --size "$SIZE" --observation_size "$OBS" --movement_mode "$MOVEMENT"
        --goal_radius 0.5 --num_full_iters 1 --steps_per_rollout "$MAX_STEPS"
        --hidden_size 128 --num_rnn_layers 1 --max_grad_norm 1.0 --device cpu
        --no-world_spec --input_prev_action)
BASE=(--lr 1e-3 --epochs 1 --n_minibatches 1 --batch_envs 1)

echo "[scratch-pd] repo=$REPO cpus=${SLURM_CPUS_PER_TASK:-$(nproc)} started $(date -Is)"

PIDS=(); NAMES=()
launch () {
    local tag="$1"; shift
    python -u -m analysis.continual.baseline \
        --out "$OUT/${tag}.json" --run_name "$tag" \
        "${COMMON[@]}" "$@" > "$LOGS/${tag}.log" 2>&1 &
    PIDS+=($!); NAMES+=("$tag")
}

# --- A: enough of the tuning grid to know the lr is not the story ------------
for LR in 1e-3 3e-3; do
for S in $(seq 1 $SEEDS); do
    launch "A_lr${LR}_rst0_ep1_s${S}" --seed "$S" \
        --lr "$LR" --epochs 1 --n_minibatches 1 --batch_envs 1
done; done

# --- R: the matched reference -----------------------------------------------
for S in $(seq 1 $SEEDS); do
    launch "R_none_s${S}" "${BASE[@]}" --seed "$S" --method none
done

# --- B: Experience Replay, the arm that wins elsewhere ----------------------
for RB in 1 4; do
for S in $(seq 1 $SEEDS); do
    launch "B_er_bufinf_rb${RB}_s${S}" "${BASE[@]}" --seed "$S" \
        --method er --method_args "buffer_size=inf,replay_batches=${RB},sampling=balanced"
done; done

# --- C / F: the two regularisers, at their usable decades -------------------
for LAM in 10000 100000 1000000; do
for S in $(seq 1 $SEEDS); do
    launch "C_ewc_lam${LAM}_s${S}" "${BASE[@]}" --seed "$S" \
        --method online_ewc --method_args "lam=${LAM},gamma=1.0,fisher=true"
done; done
for LAM in 1000 10000 100000; do
for S in $(seq 1 $SEEDS); do
    launch "F_si_lam${LAM}_s${S}" "${BASE[@]}" --seed "$S" \
        --method si --method_args "lam=${LAM},xi=0.1"
done; done

# --- D / E: the replay-family methods ---------------------------------------
for CC in 1 10 100; do
for S in $(seq 1 $SEEDS); do
    launch "D_clear_cc${CC}_s${S}" "${BASE[@]}" --seed "$S" \
        --method clear --method_args "buffer_size=inf,replay_batches=1,sampling=balanced,clone_coef=${CC}"
done; done
for A in 0.03 0.1 0.3; do
for S in $(seq 1 $SEEDS); do
    launch "E_derpp_a${A}_s${S}" "${BASE[@]}" --seed "$S" \
        --method derpp --method_args "buffer_size=inf,replay_batches=1,sampling=balanced,alpha=${A}"
done; done

# --- M / N2: parameter isolation --------------------------------------------
for S in $(seq 1 $SEEDS); do
    launch "M_multihead_s${S}" "${BASE[@]}" --seed "$S" --arch multihead
done
for LAM in 1000 10000; do
for S in $(seq 1 $SEEDS); do
    launch "N2_xdgsi_g0.5_lam${LAM}_s${S}" "${BASE[@]}" --seed "$S" \
        --arch xdg --xdg_gating 0.5 \
        --method si --method_args "lam=${LAM},xi=0.1"
done; done

echo "[scratch-pd] launched ${#PIDS[@]} tasks; waiting"
FAILED=()
for k in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$k]}"; then FAILED+=("${NAMES[$k]}"); fi
done
echo "[scratch-pd] finished $(date -Is)"
if (( ${#FAILED[@]} )); then
    echo "[scratch-pd] ${#FAILED[@]} FAILED: ${FAILED[*]:0:20}" >&2
else
    echo "[scratch-pd] all ${#PIDS[@]} tasks OK"
fi
exit 0
