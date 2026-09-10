#!/bin/bash -l
#SBATCH --job-name=cl-wave2d
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=120G
#SBATCH --partition=ou_bcs_normal
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jackking@mit.edu
#SBATCH --output=/home/jackking/cls/hopfield_nav/logs/slurm_cl_wave2d_%j.out
set -uo pipefail

# =============================================================================
# Online EWC and SI, re-run after the block-rollout sampler was fixed.
#
# Two defects in `OnlineEWC.after_update`, both found by reading rather than by
# a failure:
#
#   * It drew from the `random` module, whose global state neither
#     `torch.manual_seed` nor `np.random.seed` touches -- so EWC runs were
#     silently non-reproducible while every other method in the suite was fine.
#   * It was not a reservoir. `j` was drawn against the *buffer size* rather
#     than the number of items seen, giving a constant acceptance of k/(k+1),
#     about 0.97. The buffer therefore held roughly the last 32 rollouts of the
#     block -- exactly what its own comment said it avoided.
#
# Mutation-checked over 400 updates with k=8: the old draw retains items whose
# mean index is 389.7 (tail-only would be 396), the fix gives 209.6 (uniform
# would be 200).
#
# So the Fisher was being estimated on the tail of each block rather than on a
# uniform sample of it. Both are defensible choices for "states the block
# visited" and the numbers may barely move -- but the run that produced them
# was not the run the code documented, and EWC is a headline method. The 48
# affected histories are deleted rather than kept.
#
# SI shares the block boundary but not the sampler, so it is included only to
# extend its lambda range, which Wave 2b showed had also stopped short.
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
LOGS="$REPO/hopfield_nav/logs/wave2d"
mkdir -p "$OUT" "$LOGS"

CKPT="/home/jackking/cls/checkpoint_rnn/pretrain_20x20/final.pt"
[[ -f "$CKPT" ]] || { echo "[wave2d] FATAL: missing $CKPT" >&2; exit 1; }

SEEDS=8
COMMON=(--n_envs 5 --iters_per_block 200 --max_steps 200 --size 20
        --observation_size 60 --movement_mode continuous --goal_radius 0.5
        --num_full_iters 1 --steps_per_rollout 200 --hidden_size 128
        --num_rnn_layers 1 --max_grad_norm 1.0 --device cpu
        --load_checkpoint "$CKPT" --no-world_spec
        --lr 1e-3 --epochs 1 --n_minibatches 1 --batch_envs 1)

echo "[wave2d] repo=$REPO  cpus=${SLURM_CPUS_PER_TASK:-$(nproc)}  started $(date -Is)"

PIDS=(); NAMES=()
launch () {
    local tag="$1"; shift
    python -u -m analysis.continual.baseline \
        --out "$OUT/${tag}.json" --run_name "$tag" \
        "${COMMON[@]}" "$@" > "$LOGS/${tag}.log" 2>&1 &
    PIDS+=($!); NAMES+=("$tag")
}

for LAM in 1 10 100 1000 10000 100000; do
for S in $(seq 1 $SEEDS); do
    launch "C_ewc_lam${LAM}_s${S}" --seed "$S" --method online_ewc \
        --method_args "lam=${LAM},gamma=1.0,fisher=true"
done; done

# SI's lambda range, extended past where Wave 2b showed it still climbing.
for LAM in 10000; do
for S in $(seq 1 $SEEDS); do
    launch "F_si_lam${LAM}_s${S}" --seed "$S" --method si \
        --method_args "lam=${LAM},xi=0.1"
done; done

echo "[wave2d] launched ${#PIDS[@]} tasks; waiting"
FAILED=()
for k in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$k]}"; then FAILED+=("${NAMES[$k]}"); fi
done
echo "[wave2d] finished $(date -Is)"
if (( ${#FAILED[@]} )); then
    echo "[wave2d] ${#FAILED[@]} FAILED: ${FAILED[*]:0:20}" >&2
else
    echo "[wave2d] all ${#PIDS[@]} tasks OK"
fi

python -u -m analysis.continual.wave1_summary --dir "$OUT" \
    | tee "$LOGS/summary.txt"

exit 0
