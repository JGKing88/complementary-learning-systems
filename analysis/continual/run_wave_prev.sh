#!/bin/bash -l
#SBATCH --job-name=cl-wave-prev
#SBATCH --time=23:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=220G
#SBATCH --partition=ou_bcs_normal
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jackking@mit.edu
#SBATCH --output=/home/jackking/cls/hopfield_nav/logs/slurm_cl_wave_prev_%j.out
set -uo pipefail

# Partition: ou_bcs_normal, not pi_fiete, whose QOS caps the whole group at
# cpu=48 against this job's 96 -- such a job never starts, it sits at
# QOSGrpCpuLimit looking exactly like an ordinary queue wait.

# =============================================================================
# Every method, on a checkpoint pretrained WITH the previous-action
# channel.
#
# The published suite could not use the channel at all: its checkpoints
# were built without it, and restore_arch_from_ckpt takes
# input_prev_action from the checkpoint because the two imply different
# input widths. Plan decision #1 settled the channel on and no method arm
# has ever had it. Giving the pretrained arms the channel means
# pretraining with it, which run_pretrain_prev_*.sh does.
#
# Arms and sweeps are run_scratch.sh's, with --load_checkpoint added back,
# so this wave and that one differ by exactly the checkpoint.
#
# Whether this is worth its compute was genuinely unknown when it was
# launched. The only measurement of the channel is T0.4's, which is
# from-scratch: +0.045 current-env in discrete, -0.069 in continuous. A
# pretrained trunk has already learned the sensorimotor mapping the
# channel reports, so it plausibly gains far less -- and discrete has
# only ~4% of its headroom left, while continuous has 58%.
# =============================================================================

module load miniforge/24.3.0-0
source activate cls

REPO="${CL_REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/continual-control-suite}"
cd "$REPO"
source scripts/cls_env.sh

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

OUT="$CLS_HISTORIES/wave1_p"
LOGS="$REPO/hopfield_nav/logs/wave_prev"
mkdir -p "$OUT" "$LOGS"

N_ENVS=5; SIZE=20; OBS=60; MOVEMENT=continuous
MAX_STEPS=200; ITERS=200; SEEDS=8

COMMON=(--n_envs "$N_ENVS" --iters_per_block "$ITERS" --max_steps "$MAX_STEPS"
        --size "$SIZE" --observation_size "$OBS" --movement_mode "$MOVEMENT"
        --goal_radius 0.5 --num_full_iters 1 --steps_per_rollout "$MAX_STEPS"
        --hidden_size 128 --num_rnn_layers 1 --max_grad_norm 1.0 --device cpu
        --no-world_spec)

CKPT="$CLS_CKPTS_RNN/pretrain_20x20_prev/final.pt"
if [[ ! -f "$CKPT" ]]; then
    echo "[wave-prev] FATAL: checkpoint missing: $CKPT" >&2
    echo "[wave-prev]        run analysis/continual/run_pretrain_prev_c.sh first." >&2
    exit 1
fi
# Both properties are checked, because both are silently taken from the
# checkpoint rather than from the command line: `restore_arch_from_ckpt`
# overrides movement_mode and input_prev_action alike, since each changes the
# input width. A wrong checkpoint here would not fail -- it would produce a
# wave whose every filename claimed something the runs did not do.
read -r CK_MODE CK_PREV <<< "$(python - "$CKPT" <<'PY'
import sys, torch
a = (torch.load(sys.argv[1], map_location="cpu",
                weights_only=False).get("cfg") or {}).get("agent") or {}
print(a.get("movement_mode", "?"), bool(a.get("input_prev_action")))
PY
)"
if [[ "$CK_MODE" != "continuous" || "$CK_PREV" != "True" ]]; then
    echo "[wave-prev] FATAL: checkpoint is movement_mode=$CK_MODE prev_action=$CK_PREV;" >&2
    echo "[wave-prev]        expected continuous / True." >&2
    exit 1
fi
echo "[wave-prev] checkpoint verified: continuous, prev_action on"

BASE=(--load_checkpoint "$CKPT" --lr 1e-3 --epochs 1 --n_minibatches 1 --batch_envs 1)

echo "[wave-prev] repo=$REPO cpus=${SLURM_CPUS_PER_TASK:-$(nproc)} started $(date -Is)"

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
    launch "A_lr${LR}_rst${RST}_ep${EP}_s${S}" --load_checkpoint "$CKPT" --seed "$S" \
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
for LAM in 1 10 100 1000 10000 100000; do
for S in $(seq 1 $SEEDS); do
    launch "C_ewc_lam${LAM}_s${S}" "${BASE[@]}" --seed "$S" \
        --method online_ewc --method_args "lam=${LAM},gamma=1.0,fisher=true"
done; done

# --- F: Synaptic Intelligence -----------------------------------------------
for LAM in 0.1 1 10 100 1000 10000; do
for S in $(seq 1 $SEEDS); do
    launch "F_si_lam${LAM}_s${S}" "${BASE[@]}" --seed "$S" \
        --method si --method_args "lam=${LAM},xi=0.1"
done; done

# --- D: CLEAR ---------------------------------------------------------------
for CC in 0.01 0.1 1.0 3 10 30; do
for S in $(seq 1 $SEEDS); do
    launch "D_clear_cc${CC}_s${S}" "${BASE[@]}" --seed "$S" \
        --method clear --method_args "buffer_size=inf,replay_batches=1,sampling=balanced,clone_coef=${CC}"
done; done

# --- E: DER++ ---------------------------------------------------------------
for A in 0.1 1 10 100; do
for S in $(seq 1 $SEEDS); do
    launch "E_derpp_a${A}_s${S}" "${BASE[@]}" --seed "$S" \
        --method derpp --method_args "buffer_size=inf,replay_batches=1,sampling=balanced,alpha=${A}"
done; done

# --- G: LwF -----------------------------------------------------------------
for A in 0.1 1 10; do
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
for LAM in 100 10000 100000 1000000; do
for S in $(seq 1 $SEEDS); do
    launch "N2_xdgsi_g0.5_lam${LAM}_s${S}" "${BASE[@]}" --seed "$S" \
        --arch xdg --xdg_gating 0.5 \
        --method si --method_args "lam=${LAM},xi=0.1"
done; done

# --- J: the hypernetwork, warm-started ---------------------------------------
# base='learned', not 'none'. The no-warm-start form is what run_scratch.sh
# runs; here there IS a checkpoint, and `hnet base='none' has no base vector to
# warm-start` -- the agent says so itself and refuses, which is how the first
# version of this block failed 24 tasks rather than quietly training something
# that was not a warm start.
for B in 10000 100000 1000000; do
for S in $(seq 1 $SEEDS); do
    launch "J_hnet_b${B}_s${S}" "${BASE[@]}" --seed "$S" \
        --arch hnet --hnet_base learned --method hnet --method_args "beta=${B}"
done; done

echo "[wave-prev] launched ${#PIDS[@]} tasks; waiting"
FAILED=()
for k in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$k]}"; then FAILED+=("${NAMES[$k]}"); fi
done
echo "[wave-prev] finished $(date -Is)"
if (( ${#FAILED[@]} )); then
    echo "[wave-prev] ${#FAILED[@]} FAILED: ${FAILED[*]:0:20}" >&2
else
    echo "[wave-prev] all ${#PIDS[@]} tasks OK"
fi
exit 0
