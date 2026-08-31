#!/bin/bash -l
#SBATCH --job-name=cl-incontext
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --partition=ou_bcs_normal
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jackking@mit.edu
#SBATCH --output=/home/jackking/cls/hopfield_nav/logs/slurm_cl_incontext_%j.out
set -uo pipefail

# =============================================================================
# Plan section 5.2 -- in-context adaptation with ZERO weight updates.
#
# The only control that meets the Hopfield store on its own terms. Both models
# get an environment they have never seen and neither may learn from it: the
# store writes one Hebbian outer product, the RNN gets nothing but its own
# recurrent activity. If the RNN adapts anyway, forgetting stops being the
# interesting axis and the framing has to change.
#
# Two arms, identical except for one flag:
#
#   lifetime   --carry_across_episodes: a rollout is a LIFETIME. Reaching the
#              goal teleports the agent and KEEPS the hidden state, so
#              consecutive episodes in one env are linked only by activity.
#   episodic   the default. Reaching the goal freezes the row.
#
# The control arm is not optional. On its own, a rising success-vs-episode curve
# proves little: a policy that drifts toward the middle of the arena, or simply
# explores well, can produce one for reasons that have nothing to do with
# memory. The episodic arm is trained identically and differs only in whether
# state survived a goal-reach, so the DIFFERENCE between the curves is the part
# attributable to carrying anything.
#
# Evaluation is on held-out envs at a seed the pretraining never saw, which is
# also what guards against the arms simply memorising the 32 training envs.
# =============================================================================

module load miniforge/24.3.0-0
source activate cls

REPO="${CL_REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/continual-control-suite}"
cd "$REPO"
source scripts/cls_env.sh

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

OUT="$CLS_HISTORIES/incontext"
LOGS="$REPO/hopfield_nav/logs/incontext"
mkdir -p "$OUT" "$LOGS"

# A pool big enough that memorising it is not obviously easier than learning
# the strategy, and small enough that an update stays affordable: 32 envs x 200
# steps = 6400 env-steps per update.
POOL=32
UPDATES=2000
SIZE=20
OBS=60
STEPS=200
HID=256          # in-context memory has to live in the hidden state
SEEDS=3

echo "[incontext] repo=$REPO  cpus=${SLURM_CPUS_PER_TASK:-$(nproc)}  started $(date -Is)"

PIDS=(); NAMES=()
for ARM in lifetime episodic; do
    FLAG=""
    [[ "$ARM" == "lifetime" ]] && FLAG="--carry_across_episodes"
    for S in $(seq 1 $SEEDS); do
        TAG="pre_${ARM}_s${S}"
        python -u -m hopfield_nav.train_rnn \
            --mode mixed \
            --n_envs "$POOL" --n_updates "$UPDATES" \
            --size "$SIZE" --observation_size "$OBS" \
            --movement_mode continuous --goal_radius 0.5 \
            --hidden_size "$HID" --num_rnn_layers 1 \
            --input_prev_action --input_prev_reward \
            --lr 1e-3 --epochs 4 --n_minibatches 4 \
            --batch_envs 1 --steps_per_rollout "$STEPS" \
            --n_eval_trials 16 --eval_max_steps "$STEPS" --eval_every 250 \
            --seed "$S" --device cpu \
            $FLAG \
            --save_dir "$CLS_RUNS/rnn/incontext_${TAG}" \
            > "$LOGS/${TAG}.log" 2>&1 &
        PIDS+=($!); NAMES+=("$TAG")
    done
done

echo "[incontext] launched ${#PIDS[@]} pretraining runs; waiting"
FAILED=()
for k in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$k]}"; then FAILED+=("${NAMES[$k]}"); fi
done
if (( ${#FAILED[@]} )); then
    echo "[incontext] ${#FAILED[@]} pretraining runs FAILED: ${FAILED[*]}" >&2
fi
echo "[incontext] pretraining done $(date -Is)"

# --- the measurement --------------------------------------------------------
for S in $(seq 1 $SEEDS); do
    LT="$CLS_RUNS/rnn/incontext_pre_lifetime_s${S}/final.pt"
    EP="$CLS_RUNS/rnn/incontext_pre_episodic_s${S}/final.pt"
    if [[ ! -f "$LT" || ! -f "$EP" ]]; then
        echo "[incontext] seed $S: missing a checkpoint, skipping" >&2
        continue
    fi
    python -u -m analysis.continual.incontext \
        --out "$OUT/incontext_s${S}.json" \
        --load_checkpoint "$LT" --control_checkpoint "$EP" \
        --n_envs 8 --seed $((9000 + S)) \
        --size "$SIZE" --observation_size "$OBS" \
        --movement_mode continuous \
        --n_lifetimes 64 --n_episodes 10 --max_steps "$STEPS" \
        --device cpu 2>&1 | tee "$LOGS/eval_s${S}.log"
done

echo "[incontext] finished $(date -Is)"
exit 0
