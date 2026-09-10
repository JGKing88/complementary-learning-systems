#!/bin/bash -l
#SBATCH --job-name=cl-wave3b
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=120G
#SBATCH --partition=ou_bcs_normal
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jackking@mit.edu
#SBATCH --output=/home/jackking/cls/hopfield_nav/logs/slurm_cl_wave3b_%j.out
set -uo pipefail

# =============================================================================
# Wave 3b -- the ranges Wave 3 stopped short of, and the combination it missed.
#
# Wave 3's headline is XdG + SI at 0.739 retained, which is the best classic
# result in the suite by a wide margin. Three reasons not to publish that number
# without these forty runs first:
#
#   1. **lambda was at the top of its range when it won.** N2 swept {100,
#      10000} and 10000 was best. That is exactly the shape of the mistake Wave
#      2b had to correct for SI, whose lambda was still climbing at the end of
#      its sweep -- and it is the same knob. Extend to 1e5 and 1e6.
#
#   2. **The best gating and the best lambda were never combined.** XdG alone
#      is better at gating 0.8 (0.425) than at 0.5 (0.386), but XdG+SI was only
#      ever run at 0.5. The best of each was never run together, so the reported
#      combination is not the method's best setting -- it is the one that
#      happened to be in the script.
#
#   3. **Gating itself was still rising.** 0.2 -> 0.177, 0.5 -> 0.386,
#      0.8 -> 0.425. Three points, monotone, ending at the largest. Add 0.9.
#
# Plus one filler on the hypernetwork, whose beta peaked at 1e6 (0.454) with 1e7
# slightly lower (0.431) -- so the maximum is bracketed, but thinly.
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
LOGS="$REPO/hopfield_nav/logs/wave3b"
mkdir -p "$OUT" "$LOGS"

CKPT="/home/jackking/cls/checkpoint_rnn/pretrain_20x20/final.pt"
[[ -f "$CKPT" ]] || { echo "[wave3b] FATAL: missing $CKPT" >&2; exit 1; }

SEEDS=8
COMMON=(--n_envs 5 --iters_per_block 200 --max_steps 200 --size 20
        --observation_size 60 --movement_mode continuous --goal_radius 0.5
        --num_full_iters 1 --steps_per_rollout 200 --hidden_size 128
        --num_rnn_layers 1 --max_grad_norm 1.0 --device cpu --no-world_spec
        --load_checkpoint "$CKPT"
        --lr 1e-3 --epochs 1 --n_minibatches 1 --batch_envs 1)

echo "[wave3b] repo=$REPO  cpus=${SLURM_CPUS_PER_TASK:-$(nproc)}  started $(date -Is)"

PIDS=(); NAMES=()
launch () {
    local tag="$1"; shift
    python -u -m analysis.continual.baseline \
        --out "$OUT/${tag}.json" --run_name "$tag" \
        "${COMMON[@]}" "$@" > "$LOGS/${tag}.log" 2>&1 &
    PIDS+=($!); NAMES+=("$tag")
}

# --- N2: SI's lambda, past where Wave 3 stopped -----------------------------
for LAM in 100000 1000000; do
for S in $(seq 1 $SEEDS); do
    launch "N2_xdgsi_g0.5_lam${LAM}_s${S}" --seed "$S" --arch xdg \
        --xdg_gating 0.5 --method si --method_args "lam=${LAM},xi=0.1"
done; done

# --- N2: the combination that was never run ---------------------------------
for S in $(seq 1 $SEEDS); do
    launch "N2_xdgsi_g0.8_lam10000_s${S}" --seed "$S" --arch xdg \
        --xdg_gating 0.8 --method si --method_args "lam=10000,xi=0.1"
done

# --- N: gating past the top of its range ------------------------------------
for S in $(seq 1 $SEEDS); do
    launch "N_xdg_g0.9_s${S}" --seed "$S" --arch xdg \
        --xdg_gating 0.9 --method none
done

# --- J: one more point around the hypernetwork's peak -----------------------
for S in $(seq 1 $SEEDS); do
    launch "J_hnet_b3000000_s${S}" --seed "$S" --arch hnet \
        --method hnet --method_args "beta=3000000,normalize=true"
done

echo "[wave3b] launched ${#PIDS[@]} tasks; waiting"
FAILED=()
for k in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$k]}"; then FAILED+=("${NAMES[$k]}"); fi
done
echo "[wave3b] finished $(date -Is)"
if (( ${#FAILED[@]} )); then
    echo "[wave3b] ${#FAILED[@]} FAILED: ${FAILED[*]:0:20}" >&2
else
    echo "[wave3b] all ${#PIDS[@]} tasks OK"
fi

python -u -m analysis.continual.wave3_summary --dir "$OUT" \
    | tee "$LOGS/summary.txt"

exit 0
