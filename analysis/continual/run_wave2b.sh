#!/bin/bash -l
#SBATCH --job-name=cl-wave2b
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=160G
#SBATCH --partition=ou_bcs_normal
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jackking@mit.edu
#SBATCH --output=/home/jackking/cls/hopfield_nav/logs/slurm_cl_wave2b_%j.out
set -uo pipefail

# =============================================================================
# Wave 2b -- the coefficient ranges Wave 2 got wrong.
#
# Wave 2 swept DER++ over alpha in {0.1, 0.5, 1.0} and CLEAR over clone_coef in
# {0.01, 0.1, 1.0}, taking both from their papers. Both were far too low, and
# DER++'s numbers came back bit-identical to plain ER because its distillation
# term never amounted to anything.
#
# Measured at the real configuration, as a fraction of the BC loss
# (move_loss ~ 17.8):
#
#     DER++  alpha=0.1    ratio 0.003     <- effectively off
#     DER++  alpha=1.0    ratio 0.029     <- still effectively off
#     DER++  alpha=100    ratio 2.9
#     CLEAR  cc=1.0       ratio 0.40      <- top of its sweep, still climbing
#     LwF    alpha=1.0    ratio 0.11
#
# The cause is a units mismatch. Buzzega's alpha=0.5 is calibrated against a
# cross-entropy over CIFAR logits; here the primary loss is a Gaussian NLL of
# magnitude ~18, so the same constant buys a thirtieth of the influence. A
# coefficient copied from a paper is only meaningful with that paper's loss
# scale.
#
# So the ranges here are set by *ratio to the primary loss*, spanning roughly
# 0.03 to 10, which is the same standard the plan applies to EWC's lambda.
# Both of Wave 2's coefficient sweeps were monotone-increasing to their top
# value, which is the signature of a range that stops before the method does
# anything -- exactly the strawman the plan warns about, committed by us.
#
# SI is included at higher lambda for completeness, though its Wave-2 sweep
# already turned over (lam=10 -> 0.074, lam=100 -> 0.066), so its peak is
# genuinely inside the range that was run.
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
LOGS="$REPO/hopfield_nav/logs/wave2b"
mkdir -p "$OUT" "$LOGS"

CKPT="/home/jackking/cls/checkpoint_rnn/pretrain_20x20/final.pt"
[[ -f "$CKPT" ]] || { echo "[wave2b] FATAL: missing $CKPT" >&2; exit 1; }

SEEDS=8
COMMON=(--n_envs 5 --iters_per_block 200 --max_steps 200 --size 20
        --observation_size 60 --movement_mode continuous --goal_radius 0.5
        --num_full_iters 1 --steps_per_rollout 200 --hidden_size 128
        --num_rnn_layers 1 --max_grad_norm 1.0 --device cpu
        --load_checkpoint "$CKPT" --no-world_spec
        --lr 1e-3 --epochs 1 --n_minibatches 1 --batch_envs 1)

echo "[wave2b] repo=$REPO  cpus=${SLURM_CPUS_PER_TASK:-$(nproc)}  started $(date -Is)"

PIDS=(); NAMES=()
launch () {
    local tag="$1"; shift
    python -u -m analysis.continual.baseline \
        --out "$OUT/${tag}.json" --run_name "$tag" \
        "${COMMON[@]}" "$@" > "$LOGS/${tag}.log" 2>&1 &
    PIDS+=($!); NAMES+=("$tag")
}

# DER++ across the range where the term is actually worth something.
for AL in 10 100 1000; do
for S in $(seq 1 $SEEDS); do
    launch "E_derpp_a${AL}_s${S}" --seed "$S" --method derpp \
        --method_args "buffer_size=inf,replay_batches=1,sampling=balanced,alpha=${AL}"
done; done

# CLEAR above the top of its Wave-2 sweep.
for CC in 3 10 30; do
for S in $(seq 1 $SEEDS); do
    launch "D_clear_cc${CC}_s${S}" --seed "$S" --method clear \
        --method_args "buffer_size=inf,replay_batches=1,sampling=balanced,clone_coef=${CC}"
done; done

# SI: its sweep already turned over, so this only confirms the peak is real.
for LAM in 1000; do
for S in $(seq 1 $SEEDS); do
    launch "F_si_lam${LAM}_s${S}" --seed "$S" --method si \
        --method_args "lam=${LAM},xi=0.1"
done; done

echo "[wave2b] launched ${#PIDS[@]} tasks; waiting"
FAILED=()
for k in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$k]}"; then FAILED+=("${NAMES[$k]}"); fi
done
echo "[wave2b] finished $(date -Is)"
if (( ${#FAILED[@]} )); then
    echo "[wave2b] ${#FAILED[@]} FAILED: ${FAILED[*]:0:20}" >&2
else
    echo "[wave2b] all ${#PIDS[@]} tasks OK"
fi

python -u -m analysis.continual.wave1_summary --dir "$OUT" \
    | tee "$LOGS/summary.txt"

exit 0
