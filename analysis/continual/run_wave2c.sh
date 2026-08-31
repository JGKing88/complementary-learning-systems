#!/bin/bash -l
#SBATCH --job-name=cl-wave2c
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=120G
#SBATCH --partition=ou_bcs_normal
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jackking@mit.edu
#SBATCH --output=/home/jackking/cls/hopfield_nav/logs/slurm_cl_wave2c_%j.out
set -uo pipefail

# =============================================================================
# DER++, re-run after the distillation term turned out to be a no-op.
#
# Waves 2 and 2b both returned DER++ statistics that were bit-identical across
# alpha -- first {0.1, 0.5, 1.0}, then {10, 100, 1000}. Four orders of magnitude
# with no effect is not a coefficient-scale problem, and it was not.
#
# `_dist_params` detached, and it was used for BOTH roles: storing the anchor at
# insertion time (where detaching is correct) and computing the live prediction
# in `aux_loss` (where it is fatal). The loss came out nonzero, scaled correctly
# with alpha, and had `requires_grad=False` -- so it added a *constant* to the
# objective and contributed nothing to any gradient. DER++ ran as plain ER for
# two entire waves.
#
# Every value-based test passed throughout, which is the point: "the loss is
# nonzero and moves in the right direction" is not evidence that a loss does
# anything. `test_aux_loss_is_differentiable` now covers all three distillation
# methods, and `test_derpp_gradient_scales_with_alpha` asserts that alpha
# reaches the gradient rather than only the reported value.
#
# The 48 stale DER++ histories have been deleted rather than left to be
# averaged in; they describe a method that was not running.
#
# alpha spans the ratio range measured against the BC loss (move_loss ~ 17.8),
# now that the term can actually contribute.
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
LOGS="$REPO/hopfield_nav/logs/wave2c"
mkdir -p "$OUT" "$LOGS"

CKPT="/home/jackking/cls/checkpoint_rnn/pretrain_20x20/final.pt"
[[ -f "$CKPT" ]] || { echo "[wave2c] FATAL: missing $CKPT" >&2; exit 1; }

SEEDS=8
COMMON=(--n_envs 5 --iters_per_block 200 --max_steps 200 --size 20
        --observation_size 60 --movement_mode continuous --goal_radius 0.5
        --num_full_iters 1 --steps_per_rollout 200 --hidden_size 128
        --num_rnn_layers 1 --max_grad_norm 1.0 --device cpu
        --load_checkpoint "$CKPT" --no-world_spec
        --lr 1e-3 --epochs 1 --n_minibatches 1 --batch_envs 1)

echo "[wave2c] repo=$REPO  cpus=${SLURM_CPUS_PER_TASK:-$(nproc)}  started $(date -Is)"

PIDS=(); NAMES=()
launch () {
    local tag="$1"; shift
    python -u -m analysis.continual.baseline \
        --out "$OUT/${tag}.json" --run_name "$tag" \
        "${COMMON[@]}" "$@" > "$LOGS/${tag}.log" 2>&1 &
    PIDS+=($!); NAMES+=("$tag")
}

for AL in 0.1 1 10 100; do
for S in $(seq 1 $SEEDS); do
    launch "E_derpp_a${AL}_s${S}" --seed "$S" --method derpp \
        --method_args "buffer_size=inf,replay_batches=1,sampling=balanced,alpha=${AL}"
done; done

echo "[wave2c] launched ${#PIDS[@]} tasks; waiting"
FAILED=()
for k in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$k]}"; then FAILED+=("${NAMES[$k]}"); fi
done
echo "[wave2c] finished $(date -Is)"
if (( ${#FAILED[@]} )); then
    echo "[wave2c] ${#FAILED[@]} FAILED: ${FAILED[*]:0:20}" >&2
else
    echo "[wave2c] all ${#PIDS[@]} tasks OK"
fi

python -u -m analysis.continual.wave1_summary --dir "$OUT" \
    | tee "$LOGS/summary.txt"

exit 0
