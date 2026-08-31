#!/bin/bash -l
#SBATCH --job-name=cl-wave2
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=220G
#SBATCH --partition=ou_bcs_normal
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jackking@mit.edu
#SBATCH --output=/home/jackking/cls/hopfield_nav/logs/slurm_cl_wave2_%j.out
set -uo pipefail

# =============================================================================
# Wave 2 -- the modern competition. docs/CONTINUAL_CONTROLS_PLAN.md section 4.2.
#
#   CLEAR   replay + distillation to one converged past self
#   DER++   replay + distillation to the target frozen at insertion time
#   SI      path-integral importance, no Fisher pass, boundary-free accumulation
#   LwF     output-space regularisation with NO buffer at all
#
# Same pretrained arm, same lr/epochs/batch_envs as Wave 1's arms B, C and R, so
# every number here is directly comparable to Wave 1's without re-running the
# reference. R (method=none) already exists from Wave 1 and is not repeated.
#
# The coefficient sweeps are deliberately over decades. Every one of these
# methods has a single strength knob, and an under-tuned knob is how a method
# gets accidentally strawmanned -- the same argument as lambda for EWC.
# =============================================================================

module load miniforge/24.3.0-0
source activate cls

REPO="${CL_REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/continual-control-suite}"
cd "$REPO"
source scripts/cls_env.sh

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

OUT="$CLS_HISTORIES/wave1"      # same directory: wave2_summary reads both
LOGS="$REPO/hopfield_nav/logs/wave2"
mkdir -p "$OUT" "$LOGS"

CKPT="/home/jackking/cls/checkpoint_rnn/pretrain_20x20/final.pt"
if [[ ! -f "$CKPT" ]]; then
    echo "[wave2] FATAL: pretrain checkpoint missing: $CKPT" >&2
    exit 1
fi

SEEDS=8
COMMON=(--n_envs 5 --iters_per_block 200 --max_steps 200 --size 20
        --observation_size 60 --movement_mode continuous --goal_radius 0.5
        --num_full_iters 1 --steps_per_rollout 200 --hidden_size 128
        --num_rnn_layers 1 --max_grad_norm 1.0 --device cpu
        --load_checkpoint "$CKPT"
        --lr 1e-3 --epochs 1 --n_minibatches 1 --batch_envs 1)

echo "[wave2] repo=$REPO  cpus=${SLURM_CPUS_PER_TASK:-$(nproc)}  started $(date -Is)"

PIDS=(); NAMES=()
launch () {
    local tag="$1"; shift
    python -u -m analysis.continual.baseline \
        --out "$OUT/${tag}.json" --run_name "$tag" --no-world_spec \
        "${COMMON[@]}" "$@" > "$LOGS/${tag}.log" 2>&1 &
    PIDS+=($!); NAMES+=("$tag")
}

# --- D: CLEAR ---------------------------------------------------------------
for CC in 0.01 0.1 1.0; do
for S in $(seq 1 $SEEDS); do
    launch "D_clear_cc${CC}_s${S}" --seed "$S" --method clear \
        --method_args "buffer_size=inf,replay_batches=1,sampling=balanced,clone_coef=${CC}"
done; done

# --- E: DER++ ---------------------------------------------------------------
for AL in 0.1 0.5 1.0; do
for S in $(seq 1 $SEEDS); do
    launch "E_derpp_a${AL}_s${S}" --seed "$S" --method derpp \
        --method_args "buffer_size=inf,replay_batches=1,sampling=balanced,alpha=${AL}"
done; done

# --- F: SI ------------------------------------------------------------------
for LAM in 0.1 1 10 100; do
for S in $(seq 1 $SEEDS); do
    launch "F_si_lam${LAM}_s${S}" --seed "$S" --method si \
        --method_args "lam=${LAM},xi=0.1"
done; done

# --- G: LwF -----------------------------------------------------------------
for AL in 0.1 1 10; do
for S in $(seq 1 $SEEDS); do
    launch "G_lwf_a${AL}_s${S}" --seed "$S" --method lwf \
        --method_args "alpha=${AL}"
done; done

echo "[wave2] launched ${#PIDS[@]} tasks; waiting"

FAILED=()
for k in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$k]}"; then FAILED+=("${NAMES[$k]}"); fi
done

echo "[wave2] finished $(date -Is)"
if (( ${#FAILED[@]} )); then
    echo "[wave2] ${#FAILED[@]} FAILED: ${FAILED[*]:0:20}" >&2
else
    echo "[wave2] all ${#PIDS[@]} tasks OK"
fi

python -u -m analysis.continual.wave1_summary --dir "$OUT" \
    | tee "$LOGS/summary.txt"

exit 0
