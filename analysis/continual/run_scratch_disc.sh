#!/bin/bash -l
#SBATCH --job-name=cl-scratch-d
#SBATCH --time=23:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=220G
#SBATCH --partition=ou_bcs_normal
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jackking@mit.edu
#SBATCH --output=/home/jackking/cls/hopfield_nav/logs/slurm_cl_scratch_d_%j.out
set -uo pipefail

# Partition: ou_bcs_normal, not pi_fiete, whose QOS caps the whole group at
# cpu=48 against this job's 96 -- such a job never starts, it sits at
# QOSGrpCpuLimit looking exactly like an ordinary queue wait.

# =============================================================================
# Every method, from scratch. The contrast the suite has never run.
#
# Every method arm in waves 1-3 loads the pretrained checkpoint: 600 of 664
# continuous runs. Only the controls (A2, L, L0, T0.4) start from random
# weights. So the suite measures whether a *pretrained navigator* retains
# per-environment knowledge across a stream -- not whether an agent can learn
# continual navigation from nothing. Those are different claims and only the
# first is currently supported.
#
# The reason for the choice is on the record and was defensible: T0.4 came back
# at ~0.55 on the environment it was training on with 71-74% of environments
# never reaching criterion, so from-scratch was barely learning and method
# differences on it would have been noise on a broken control. Two things have
# changed. The discrete from-scratch arm reaches 0.875 current-env, so "barely
# learning" is not true in the action space the results now favour; and the
# matched comparison shows pretraining roughly doubles retention in continuous
# and triples it in discrete, so the absolute numbers carry a component that is
# not about the method.
#
# **The configurations are character-identical to the pretrained arms with
# `--load_checkpoint` removed, and nothing else.** That is the whole design: one
# variable. In particular `--input_prev_action` stays OFF, because the
# pretrained arms inherit it off from the checkpoint via
# `restore_arch_from_ckpt` regardless of what their command line says -- so
# turning it on here would silently change a second thing.
#
# Output goes to its own directory, so arm names can be reused without
# colliding with the pretrained histories.
# =============================================================================

module load miniforge/24.3.0-0
source activate cls

REPO="${CL_REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/continual-control-suite}"
cd "$REPO"
source scripts/cls_env.sh

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

OUT="$CLS_HISTORIES/wave1d_fs"
LOGS="$REPO/hopfield_nav/logs/scratch_d"
mkdir -p "$OUT" "$LOGS"

N_ENVS=5; SIZE=20; OBS=60; MOVEMENT=discrete
MAX_STEPS=200; ITERS=200; SEEDS=8

COMMON=(--n_envs "$N_ENVS" --iters_per_block "$ITERS" --max_steps "$MAX_STEPS"
        --size "$SIZE" --observation_size "$OBS" --movement_mode "$MOVEMENT"
        --goal_radius 0.5 --num_full_iters 1 --steps_per_rollout "$MAX_STEPS"
        --hidden_size 128 --num_rnn_layers 1 --max_grad_norm 1.0 --device cpu
        --no-world_spec)
BASE=(--lr 1e-3 --epochs 1 --n_minibatches 1 --batch_envs 1)

echo "[scratch-d] repo=$REPO cpus=${SLURM_CPUS_PER_TASK:-$(nproc)} started $(date -Is)"

PIDS=(); NAMES=()
launch () {
    local tag="$1"; shift
    python -u -m analysis.continual.baseline \
        --out "$OUT/${tag}.json" --run_name "$tag" \
        "${COMMON[@]}" "$@" > "$LOGS/${tag}.log" 2>&1 &
    PIDS+=($!); NAMES+=("$tag")
}

# --- A: Tier-1 tuning -------------------------------------------------------
for LR in 3e-4 1e-3 3e-3; do
for RST in 0 1; do
for EP in 1 4; do
for S in $(seq 1 $SEEDS); do
    F=(); [[ "$RST" == "1" ]] && F+=(--reset_optimizer_each_block)
    launch "A_lr${LR}_rst${RST}_ep${EP}_s${S}" --seed "$S" \
        --lr "$LR" --epochs "$EP" --n_minibatches 1 --batch_envs 1 "${F[@]}"
done; done; done; done

# --- R: the matched reference every method is read against -------------------
for S in $(seq 1 $SEEDS); do
    launch "R_none_s${S}" "${BASE[@]}" --seed "$S" --method none
done

# --- B: Experience Replay ---------------------------------------------------
for BUF in inf 200 50 10; do
for RB in 1 4; do
for S in $(seq 1 $SEEDS); do
    launch "B_er_buf${BUF}_rb${RB}_s${S}" "${BASE[@]}" --seed "$S" \
        --method er --method_args "buffer_size=${BUF},replay_batches=${RB},sampling=balanced"
done; done; done

# --- C: online EWC ----------------------------------------------------------
for LAM in 1000 10000 100000 1000000 10000000; do
for S in $(seq 1 $SEEDS); do
    launch "C_ewc_lam${LAM}_s${S}" "${BASE[@]}" --seed "$S" \
        --method online_ewc --method_args "lam=${LAM},gamma=1.0,fisher=true"
done; done

# --- F: Synaptic Intelligence -----------------------------------------------
for LAM in 100 1000 10000 100000 1000000; do
for S in $(seq 1 $SEEDS); do
    launch "F_si_lam${LAM}_s${S}" "${BASE[@]}" --seed "$S" \
        --method si --method_args "lam=${LAM},xi=0.1"
done; done

# --- D: CLEAR ---------------------------------------------------------------
for CC in 0.1 1 10 100 1000; do
for S in $(seq 1 $SEEDS); do
    launch "D_clear_cc${CC}_s${S}" "${BASE[@]}" --seed "$S" \
        --method clear --method_args "buffer_size=inf,replay_batches=1,sampling=balanced,clone_coef=${CC}"
done; done

# --- E: DER++ ---------------------------------------------------------------
for A in 0.01 0.03 0.1 0.3 1; do
for S in $(seq 1 $SEEDS); do
    launch "E_derpp_a${A}_s${S}" "${BASE[@]}" --seed "$S" \
        --method derpp --method_args "buffer_size=inf,replay_batches=1,sampling=balanced,alpha=${A}"
done; done

# --- G: LwF -----------------------------------------------------------------
for A in 0.1 1 10 100 1000; do
for S in $(seq 1 $SEEDS); do
    launch "G_lwf_a${A}_s${S}" "${BASE[@]}" --seed "$S" \
        --method lwf --method_args "alpha=${A}"
done; done

# --- M / N / N2: parameter isolation ----------------------------------------
for S in $(seq 1 $SEEDS); do
    launch "M_multihead_s${S}" "${BASE[@]}" --seed "$S" --arch multihead
done
for G in 0.2 0.5 0.8 0.9; do
for S in $(seq 1 $SEEDS); do
    launch "N_xdg_g${G}_s${S}" "${BASE[@]}" --seed "$S" \
        --arch xdg --xdg_gating "$G"
done; done
for LAM in 1000 10000 100000; do
for S in $(seq 1 $SEEDS); do
    launch "N2_xdgsi_g0.5_lam${LAM}_s${S}" "${BASE[@]}" --seed "$S" \
        --arch xdg --xdg_gating 0.5 \
        --method si --method_args "lam=${LAM},xi=0.1"
done; done

echo "[scratch-d] launched ${#PIDS[@]} tasks; waiting"
FAILED=()
for k in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$k]}"; then FAILED+=("${NAMES[$k]}"); fi
done
echo "[scratch-d] finished $(date -Is)"
if (( ${#FAILED[@]} )); then
    echo "[scratch-d] ${#FAILED[@]} FAILED: ${FAILED[*]:0:20}" >&2
else
    echo "[scratch-d] all ${#PIDS[@]} tasks OK"
fi
exit 0
