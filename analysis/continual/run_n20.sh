#!/bin/bash -l
#SBATCH --job-name=cl-n20
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --partition=ou_bcs_normal
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jackking@mit.edu
#SBATCH --output=/home/jackking/cls/hopfield_nav/logs/slurm_cl_n20_%j.out
set -uo pipefail

# =============================================================================
# The N=20 scaling panel (review decision 2).
#
# Methods look alike at five tasks and separate at twenty. Wave 1 already shows
# the beginnings of that -- ER at 0.419 and EWC at 0.149 against a 0.044
# reference -- but five envs is a short stream and the orderings there are not
# guaranteed to survive.
#
# Only configurations whose best setting is already known from Wave 1 are
# scaled. Running an unresolved sweep at 2.6 h a seed would spend the budget
# finding out things N=5 answers for a tenth of the cost.
#
#   R20   none                       the reference
#   A20   naive, tuned               lr=3e-4 + per-block Adam reset (Wave 1's best A)
#   B20   ER, replay x4              Wave 1's best ER
#   I20   ER, replay x16             the ratio axis, which dominated buffer size
#   C20   online EWC, lambda=1e4     Wave 1's best *usable* EWC -- lambda=1e5
#                                    scored higher on retention only by refusing
#                                    to learn, and scaling a degenerate setting
#                                    would just produce a degenerate curve
#
# ~9.2M env-steps a seed, about 2.6 h. 20 runs on 32 CPUs, so no
# oversubscription and they land together.
#
# One caveat this panel cannot fix: every recorded agenthash history is N=5, so
# there is no Hopfield number to compare against at N=20. The panel shows how
# the *methods* scale with stream length; the store's side of it needs a run
# that does not exist yet.
# =============================================================================

module load miniforge/24.3.0-0
source activate cls

REPO="${CL_REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/continual-control-suite}"
cd "$REPO"
source scripts/cls_env.sh

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

OUT="$CLS_HISTORIES/n20"
LOGS="$REPO/hopfield_nav/logs/n20"
mkdir -p "$OUT" "$LOGS"

CKPT="/home/jackking/cls/checkpoint_rnn/pretrain_20x20/final.pt"
[[ -f "$CKPT" ]] || { echo "[n20] FATAL: missing $CKPT" >&2; exit 1; }

SEEDS=4
COMMON=(--n_envs 20 --iters_per_block 200 --max_steps 200 --size 20
        --observation_size 60 --movement_mode continuous --goal_radius 0.5
        --num_full_iters 1 --steps_per_rollout 200 --hidden_size 128
        --num_rnn_layers 1 --max_grad_norm 1.0 --device cpu
        --load_checkpoint "$CKPT" --no-world_spec
        --epochs 1 --n_minibatches 1 --batch_envs 1)

echo "[n20] repo=$REPO  cpus=${SLURM_CPUS_PER_TASK:-$(nproc)}  started $(date -Is)"

PIDS=(); NAMES=()
launch () {
    local tag="$1"; shift
    python -u -m analysis.continual.baseline \
        --out "$OUT/${tag}.json" --run_name "$tag" \
        "${COMMON[@]}" "$@" > "$LOGS/${tag}.log" 2>&1 &
    PIDS+=($!); NAMES+=("$tag")
}

for S in $(seq 1 $SEEDS); do
    launch "R_none_s${S}"      --seed "$S" --lr 1e-3 --method none
    launch "A_tuned_s${S}"     --seed "$S" --lr 3e-4 --reset_optimizer_each_block \
                               --method none
    launch "B_er_rb4_s${S}"    --seed "$S" --lr 1e-3 --method er \
        --method_args "buffer_size=inf,replay_batches=4,sampling=balanced"
    launch "I_erhi_rb16_s${S}" --seed "$S" --lr 1e-3 --method er \
        --method_args "buffer_size=inf,replay_batches=16,sampling=balanced"
    launch "C_ewc_lam10000_s${S}" --seed "$S" --lr 1e-3 --method online_ewc \
        --method_args "lam=10000,gamma=1.0,fisher=true"
done

echo "[n20] launched ${#PIDS[@]} tasks; waiting"
FAILED=()
for k in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$k]}"; then FAILED+=("${NAMES[$k]}"); fi
done
echo "[n20] finished $(date -Is)"
if (( ${#FAILED[@]} )); then
    echo "[n20] ${#FAILED[@]} FAILED: ${FAILED[*]}" >&2
else
    echo "[n20] all ${#PIDS[@]} tasks OK"
fi

python -u -m analysis.continual.wave1_summary --dir "$OUT" \
    | tee "$LOGS/summary.txt"

exit 0
