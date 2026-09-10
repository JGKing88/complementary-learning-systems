#!/bin/bash -l
#SBATCH --job-name=cl-ic-ub
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=220G
#SBATCH --partition=ou_bcs_normal
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jackking@mit.edu
#SBATCH --output=/home/jackking/cls/hopfield_nav/logs/slurm_cl_ic_ub_%j.out
set -uo pipefail

# =============================================================================
# Section 5.2, redone with a ceiling. What is the UPPER BOUND on in-context
# performance for a model trained like this?
#
# The first attempt reported a flat success-vs-episode curve and read it as
# "activation memory does not do this job". It was withdrawn: the pretrained
# policy scored 0.80 on its fixed pool of 32 environments and 0.10 on held-out
# ones -- below the 0.21 a random walker gets -- so it had memorised the pool,
# the evaluation ran on a policy that could not navigate its test environments,
# and a flat curve was the only outcome available. A null with no ceiling under
# it and no floor beside it is not a measurement.
#
# So this wave fixes the training and, more importantly, brackets the answer.
#
# -- what changed in the training ---------------------------------------------
#   --resample_envs_every 10  The pool is redrawn at every lifetime boundary, so
#                             the run sees ~12,800 distinct environments and
#                             never the same one twice. Memorising is not an
#                             available solution, which is the only way "learn
#                             to adapt" becomes the easier option.
#   (carried hidden state)    Within a lifetime the state persists ACROSS
#                             rollouts, and `bc_rnn_update` starts its forward
#                             pass from it -- truncated BPTT over a 2000-step
#                             lifetime in 200-step windows. Without this the
#                             lifetime is the rollout, and the old run trained
#                             on 200 steps while being scored on 2000.
#   --episode_max_steps 200   An episode ends on a timeout as well as a
#                             goal-reach, matching the evaluator. Without it a
#                             row that never finds the goal spends the whole
#                             rollout in one episode and never crosses a
#                             boundary -- the common case on a fresh
#                             environment, which left the cross-episode regime
#                             almost absent from its own training data.
#   --n_holdout_envs 16       A fixed set never trained on, evaluated on the
#                             same cadence. The pool-to-holdout gap is printed
#                             every eval, so memorisation is visible while the
#                             run is happening rather than four hours later.
#
# -- the arms, which are the point --------------------------------------------
#   CEIL_REL   goal_channel=rel. The agent is handed the displacement to the
#              goal. Follow the arrow. Not a realistic bound -- it is the
#              architecture sanity check, and a policy that cannot do this
#              cannot do anything, so a low number here invalidates the whole
#              wave rather than saying something about memory.
#   CEIL_ABS   goal_channel=abs. The agent is handed the goal's coordinates and
#              must still work out where *it* is from the barcode ray-cast.
#              **This is the upper bound in-context memory could actually
#              reach**: remembering where the goal is does not tell you where
#              you are, so no amount of recurrent memory can beat a policy that
#              was simply told the answer.
#   IC         the real thing. Lifetime rollouts, hidden state carried across
#              episode boundaries, goal never observed.
#   EP         the episodic control. Identical but the state does not survive a
#              goal-reach, so IC minus EP is what carrying state is worth.
#
# Read as: CEIL_ABS is 100%. IC's position between EP and CEIL_ABS is the
# fraction of the available in-context signal this model class actually
# captures. That is a bounded, interpretable answer whichever way it falls --
# which is exactly what the withdrawn version did not have.
#
# Capacity is swept because "the state was too small" is the first objection to
# a null, and it should be answered by data rather than by a sentence.
# =============================================================================

module load miniforge/24.3.0-0
source activate cls

REPO="${CL_REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/continual-control-suite}"
cd "$REPO"
source scripts/cls_env.sh

# 24 processes on 96 allocated cores. The Wave 1-3 launchers pin this to 1
# because they run 144 processes and oversubscription is the risk; here the
# same setting left ~60 cores idle for three and a half hours.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

CKPTS="$CLS_RUNS/rnn"
OUT="$CLS_HISTORIES/incontext_ub"
LOGS="$REPO/hopfield_nav/logs/incontext_ub"
mkdir -p "$OUT" "$LOGS"

# Sixteen trajectories per gradient step either way -- the split between
# environments and parallel rollouts within one is a pure throughput knob, and
# the previous setting had it at the worst end. Measured, same 16 trajectories:
#
#   n_envs=16 batch_envs=1   rollout 2.39s + update 0.16s = 2.55s   (h=256)
#   n_envs=1  batch_envs=16  rollout 0.23s + update 0.12s = 0.35s   7.3x
#
# The rollout was 94% of the time and all of it launch overhead: sixteen
# separate 200-step Python loops doing forward passes at batch size 1. Four
# environments of four rollouts keeps four distinct goals in every gradient
# batch while cutting the sequential step count fourfold. Total distinct
# environments over training is still ~3,200, against the 32 that caused the
# memorisation failure.
POOL=4           # distinct environments pooled per update
PARALLEL=4       # parallel lifetimes within each
UPDATES=8000
# A LIFETIME is LIFETIME_UPDATES consecutive rollouts on one environment with
# the hidden state carried across them, and it ends when the environment is
# redrawn -- so this single number is both the lifetime length and the
# resampling cadence, because they are the same boundary.
#
#   lifetime = STEPS * LIFETIME_UPDATES = 200 * 10 = 2000 steps
#   evaluation = n_episodes * max_steps = 10 * 200  = 2000 steps
#
# Those two have to match. In the withdrawn version they did not: the hidden
# state was reset at the start of every rollout, so training lifetimes were 200
# steps and the evaluation measured 2000-step ones. The network was asked to
# use a horizon ten times longer than any it had ever been trained on.
LIFETIME_UPDATES=10
EPISODE_CAP=200  # matches the evaluator's per-episode cap exactly
SIZE=20
OBS=60
STEPS=200
SEEDS=3

echo "[ic-ub] repo=$REPO  cpus=${SLURM_CPUS_PER_TASK:-$(nproc)}  started $(date -Is)"

TRAIN_COMMON=(--mode mixed --n_envs "$POOL" --n_updates "$UPDATES"
              --size "$SIZE" --observation_size "$OBS"
              --movement_mode continuous --goal_radius 0.5
              --num_rnn_layers 1 --input_prev_action --input_prev_reward
              --lr 1e-3 --epochs 4 --n_minibatches 4
              --batch_envs "$PARALLEL" --steps_per_rollout "$STEPS"
              --n_eval_trials 16 --eval_max_steps "$STEPS" --eval_every 500
              --device cpu --n_holdout_envs 16
              --resample_envs_every "$LIFETIME_UPDATES"
              --episode_max_steps "$EPISODE_CAP")

PIDS=(); NAMES=()
train () {
    local tag="$1"; shift
    python -u -m hopfield_nav.train_rnn "${TRAIN_COMMON[@]}" "$@" \
        --save_dir "$CKPTS/icub_${tag}" > "$LOGS/${tag}.log" 2>&1 &
    PIDS+=($!); NAMES+=("$tag")
}

# h1024 runs in run_incontext_ub_gpu.sh instead. Once the rollout is batched
# its BC update is 71% of the wall time and is a fused 200-step GRU over a real
# batch -- which is GPU-shaped, where the batch-1 rollout never was.
for H in 256 512; do
for S in $(seq 1 $SEEDS); do
    # The upper bound: told the goal, still has to localise itself.
    train "ceilabs_h${H}_s${S}" --hidden_size "$H" --seed "$S" \
        --goal_channel abs --carry_across_episodes
    # The real in-context arm, and its episodic control.
    train "ic_h${H}_s${S}"      --hidden_size "$H" --seed "$S" \
        --carry_across_episodes
    train "ep_h${H}_s${S}"      --hidden_size "$H" --seed "$S"
done; done

# Architecture sanity: if this is not near the top, nothing else in the wave
# means anything, because the network cannot act on the goal even when handed
# it directly.
for S in $(seq 1 $SEEDS); do
    train "ceilrel_h512_s${S}" --hidden_size 512 --seed "$S" \
        --goal_channel rel --carry_across_episodes
done

# The architecture-level positive control. The goal is shown during the first
# episode of each lifetime and withheld afterwards, so the network is *trained*
# to carry it across a boundary rather than to discover it. This is what turns
# a null into a legible failure mode: if this arm succeeds, the recurrence can
# hold a goal across episodes and the in-context arm's shortfall is about
# discovering one; if it fails, the recurrence cannot carry the fact at all and
# no amount of exploration would have helped.
for H in 512; do
for S in $(seq 1 $SEEDS); do
    train "carry_h${H}_s${S}" --hidden_size "$H" --seed "$S" \
        --goal_channel abs --goal_visible_episodes 1 --carry_across_episodes
done; done

echo "[ic-ub] launched ${#PIDS[@]} pretraining runs; waiting"
FAILED=()
for k in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$k]}"; then FAILED+=("${NAMES[$k]}"); fi
done
echo "[ic-ub] pretraining done $(date -Is)"
if (( ${#FAILED[@]} )); then
    echo "[ic-ub] ${#FAILED[@]} FAILED: ${FAILED[*]:0:20}" >&2
fi

# --- the in-context measurement, on the arms that have no goal channel ------
EPIDS=(); ENAMES=()
for H in 256 512; do
for S in $(seq 1 $SEEDS); do
    LT="$CKPTS/icub_ic_h${H}_s${S}/final.pt"
    EP="$CKPTS/icub_ep_h${H}_s${S}/final.pt"
    [[ -f "$LT" && -f "$EP" ]] || { echo "[ic-ub] h${H} s${S}: missing ckpt" >&2; continue; }
    python -u -m analysis.continual.incontext \
        --out "$OUT/icub_h${H}_s${S}.json" \
        --load_checkpoint "$LT" --control_checkpoint "$EP" \
        --n_envs 8 --seed $((9000 + S)) \
        --size "$SIZE" --observation_size "$OBS" --movement_mode continuous \
        --n_lifetimes 64 --n_episodes 10 --max_steps "$STEPS" \
        --device cpu > "$LOGS/eval_h${H}_s${S}.log" 2>&1 &
    EPIDS+=($!); ENAMES+=("h${H}_s${S}")
done; done

# The carry arm is evaluated with the goal shown in episode 1 only, matching
# how it was trained -- so episodes 2..10 measure what it retained.
for H in 512; do
for S in $(seq 1 $SEEDS); do
    CK="$CKPTS/icub_carry_h${H}_s${S}/final.pt"
    [[ -f "$CK" ]] || continue
    python -u -m analysis.continual.incontext \
        --out "$OUT/carry_h${H}_s${S}.json" \
        --load_checkpoint "$CK" \
        --n_envs 8 --seed $((9000 + S)) \
        --size "$SIZE" --observation_size "$OBS" --movement_mode continuous \
        --n_lifetimes 64 --n_episodes 10 --max_steps "$STEPS" \
        --goal_visible_episodes 1 \
        --device cpu > "$LOGS/eval_carry_h${H}_s${S}.log" 2>&1 &
    EPIDS+=($!); ENAMES+=("carry_h${H}_s${S}")
done; done

for k in "${!EPIDS[@]}"; do
    if ! wait "${EPIDS[$k]}"; then echo "[ic-ub] eval ${ENAMES[$k]} FAILED" >&2; fi
done

echo "[ic-ub] finished $(date -Is)"

# The ceiling, the floor and the gate, in one table.
python -u -m analysis.continual.incontext_upper_bound \
    --logs "$LOGS" --incontext_dir "$OUT" \
    --out "$CLS_RESULTS/incontext_upper_bound.json" \
    | tee "$LOGS/summary.txt"

exit 0
