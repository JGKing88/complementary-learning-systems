#!/bin/bash -l
#SBATCH --job-name=cl-wave1d
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=220G
#SBATCH --partition=pi_fiete
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jackking@mit.edu
#SBATCH --output=/home/jackking/cls/hopfield_nav/logs/slurm_cl_wave1d_%j.out
set -uo pipefail

# =============================================================================
# Wave 1 of the discrete suite: the tuning arms and the one method with no
# coefficient to calibrate.
#
#   A       Tier-1 tuning of the PRETRAINED control (lr x optimizer-reset x epochs)
#   A2      Tier-1 tuning of the FROM-SCRATCH control
#   Abatch  the batch_envs=16 sensitivity condition
#   R       a matched `none` reference, so B is read against the right number
#   B       Experience Replay, buffer size x replay ratio
#
# The regularisers (EWC, SI, LwF, CLEAR, DER++) are deliberately NOT here. They
# all carry a coefficient, and a coefficient calibrated against a Gaussian
# negative log-likelihood of order 10 is wrong against a Categorical
# cross-entropy of order ln(4) = 1.39. They live in run_wave2d.sh and wait for
# analysis/continual/calibrate_discrete.py to say where to sweep. Splitting the
# waves on that dependency rather than on the continuous suite's wave numbering
# is deliberate: it is the only real ordering constraint here, and the failure
# it prevents (DER++ and CLEAR, twice) is the most expensive one this project
# has had.
#
# ER is here because it has no such knob -- a buffer is a buffer at any loss
# scale -- so it can start as soon as the checkpoint exists.
#
# **A2 does not sweep init_log_std.** The continuous A2 arm sweeps it because
# `movement_log_std` is the from-scratch arm's exploration knob. A Categorical
# head has no log_std at all, so porting that sweep would sweep a flag that
# turns nothing -- exactly the silently-inert-knob failure the continuous W5
# was about, and which this project has now found three times
# (`--freeze_log_std` on train_navigate, `--input_prev_action` under a
# checkpoint restore, and the beta scale). The live analogue is
# `--move_ent_coef`, the entropy bonus on the Categorical, so that is what is
# swept.
# =============================================================================

module load miniforge/24.3.0-0
source activate cls

REPO="${CL_REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/continual-control-suite}"
cd "$REPO"
source scripts/cls_env.sh

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

OUT="$CLS_HISTORIES/wave1d"
LOGS="$REPO/hopfield_nav/logs/wave1d"
mkdir -p "$OUT" "$LOGS"

CKPT="$CLS_CKPTS_RNN/pretrain_20x20_discrete/final.pt"
if [[ ! -f "$CKPT" ]]; then
    echo "[wave1d] FATAL: discrete pretrain checkpoint missing: $CKPT" >&2
    echo "[wave1d]        run analysis/continual/run_pretrain_discrete.sh first." >&2
    exit 1
fi

# A continuous checkpoint here would not fail -- it would silently win.
# `restore_arch_from_ckpt` overrides --movement_mode with the checkpoint's, so
# the whole wave would come out continuous while every filename said otherwise.
CKPT_MODE=$(python - "$CKPT" <<'PY'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(((ck.get("cfg") or {}).get("agent") or {}).get("movement_mode", "unknown"))
PY
)
if [[ "$CKPT_MODE" != "discrete" ]]; then
    echo "[wave1d] FATAL: checkpoint movement_mode=$CKPT_MODE, expected discrete." >&2
    echo "[wave1d]        restore_arch_from_ckpt would silently make this wave $CKPT_MODE." >&2
    exit 1
fi
echo "[wave1d] checkpoint verified discrete: $CKPT"

N_ENVS=5; SIZE=20; OBS=60; MOVEMENT=discrete
MAX_STEPS=200; ITERS=200; SEEDS=8

COMMON=(--n_envs "$N_ENVS" --iters_per_block "$ITERS" --max_steps "$MAX_STEPS"
        --size "$SIZE" --observation_size "$OBS" --movement_mode "$MOVEMENT"
        --goal_radius 0.5 --num_full_iters 1 --steps_per_rollout "$MAX_STEPS"
        --hidden_size 128 --num_rnn_layers 1 --max_grad_norm 1.0 --device cpu)

echo "[wave1d] repo=$REPO  cpus=${SLURM_CPUS_PER_TASK:-$(nproc)}  started $(date -Is)"

PIDS=(); NAMES=()
launch () {  # launch <tag> <extra args...>
    local tag="$1"; shift
    python -u -m analysis.continual.baseline \
        --out "$OUT/${tag}.json" --run_name "$tag" --no-world_spec \
        "${COMMON[@]}" "$@" > "$LOGS/${tag}.log" 2>&1 &
    PIDS+=($!); NAMES+=("$tag")
}

# --- A: Tier-1 tuning, pretrained arm ---------------------------------------
# The lr that was right for a Gaussian NLL is not automatically right for a
# cross-entropy an order of magnitude smaller, so this is a genuine re-tune
# rather than a repeat. Same grid as the continuous wave so the two panels
# compare cell for cell.
for LR in 3e-4 1e-3 3e-3; do
for RST in 0 1; do
for EP in 1 4; do
for S in $(seq 1 $SEEDS); do
    F=(); [[ "$RST" == "1" ]] && F+=(--reset_optimizer_each_block)
    launch "A_lr${LR}_rst${RST}_ep${EP}_s${S}" \
        --load_checkpoint "$CKPT" --seed "$S" \
        --lr "$LR" --epochs "$EP" --n_minibatches 1 --batch_envs 1 "${F[@]}"
done; done; done; done

# --- Abatch: the batch_envs=16 sensitivity condition -------------------------
for S in $(seq 1 $SEEDS); do
    launch "Abatch_be16_s${S}" \
        --load_checkpoint "$CKPT" --seed "$S" \
        --lr 1e-3 --epochs 1 --n_minibatches 4 --batch_envs 16
done

# --- A2: Tier-1 tuning, from-scratch arm ------------------------------------
# move_ent_coef replaces init_log_std -- see the header. 0.0 is the no-bonus
# control and must stay in the sweep, or a positive result would have nothing
# to be positive against.
for ENT in 0.0 0.01 0.03; do
for LR in 1e-3 3e-3; do
for S in $(seq 1 $SEEDS); do
    launch "A2_ent${ENT}_lr${LR}_s${S}" \
        --seed "$S" --input_prev_action \
        --move_ent_coef "$ENT" \
        --lr "$LR" --epochs 1 --n_minibatches 1 --batch_envs 1
done; done; done

# --- R: matched reference for B ---------------------------------------------
for S in $(seq 1 $SEEDS); do
    launch "R_none_s${S}" \
        --load_checkpoint "$CKPT" --seed "$S" \
        --lr 1e-3 --epochs 1 --n_minibatches 1 --batch_envs 1 --method none
done

# --- B: Experience Replay ---------------------------------------------------
# buffer_size=inf is the perfect-memory bound and is free at this stream size.
for BUF in inf 200 50 10; do
for RB in 1 4; do
for S in $(seq 1 $SEEDS); do
    launch "B_er_buf${BUF}_rb${RB}_s${S}" \
        --load_checkpoint "$CKPT" --seed "$S" \
        --lr 1e-3 --epochs 1 --n_minibatches 1 --batch_envs 1 \
        --method er --method_args "buffer_size=${BUF},replay_batches=${RB},sampling=balanced"
done; done; done

echo "[wave1d] launched ${#PIDS[@]} tasks; waiting"

FAILED=()
for k in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$k]}"; then FAILED+=("${NAMES[$k]}"); fi
done

echo "[wave1d] finished $(date -Is)"
if (( ${#FAILED[@]} )); then
    echo "[wave1d] ${#FAILED[@]} FAILED: ${FAILED[*]:0:20}" >&2
else
    echo "[wave1d] all ${#PIDS[@]} tasks OK"
fi

python -u -m analysis.continual.wave1_summary --dir "$OUT" \
    | tee "$LOGS/summary.txt"

exit 0
