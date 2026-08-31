#!/bin/bash -l
#SBATCH --job-name=cl-ic-eval
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G
#SBATCH --partition=ou_bcs_normal
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jackking@mit.edu
#SBATCH --output=/home/jackking/cls/hopfield_nav/logs/slurm_cl_ic_eval_%j.out
set -uo pipefail

# =============================================================================
# Re-evaluate the in-context checkpoints with `memory_lift`.
#
# The first pass reported the mean success-vs-episode curve, which came back
# flat for both arms at every seed. Flat is the answer we wanted -- but a flat
# *mean* is ambiguous, because it pools lifetimes that found the goal with
# lifetimes that never did. These policies solve only ~10% of first episodes on
# held-out environments, so nine tenths of that average is measuring blind
# search, and its flatness says nothing about memory either way.
#
# `memory_lift` conditions instead: among consecutive episode pairs, the
# success rate when the previous episode FOUND the goal, minus the rate when it
# did not. An agent holding the goal in its activations must do better having
# just been there; one searching blind cannot tell the difference.
#
# The statistic is validated against a positive control before being trusted
# (tests/test_memory_lift.py): a scripted agent that homes on the goal once its
# lifetime has found it scores +0.559 with a curve rising 0.31 -> 0.82, while
# the same agent with memory disabled scores -0.090 and stays flat. Writing
# that control took two attempts -- the evaluator teleports on a goal-reach
# *before* calling the policy, so the first scripted agent never observed
# itself at the goal and never armed. It looked like the metric had missed an
# obvious memory when the fixture had simply never produced one.
#
# Training is not repeated; this reuses the six pretrained checkpoints.
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

echo "[ic-eval] started $(date -Is)"

PIDS=(); NAMES=()
for S in 1 2 3; do
    LT="$CLS_RUNS/rnn/incontext_pre_lifetime_s${S}/final.pt"
    EP="$CLS_RUNS/rnn/incontext_pre_episodic_s${S}/final.pt"
    if [[ ! -f "$LT" || ! -f "$EP" ]]; then
        echo "[ic-eval] seed $S: missing a checkpoint, skipping" >&2
        continue
    fi
    python -u -m analysis.continual.incontext \
        --out "$OUT/incontext_s${S}.json" \
        --load_checkpoint "$LT" --control_checkpoint "$EP" \
        --n_envs 8 --seed $((9000 + S)) \
        --size 20 --observation_size 60 --movement_mode continuous \
        --n_lifetimes 64 --n_episodes 10 --max_steps 200 \
        --device cpu > "$LOGS/eval2_s${S}.log" 2>&1 &
    PIDS+=($!); NAMES+=("s${S}")
done

echo "[ic-eval] launched ${#PIDS[@]} evaluations; waiting"
FAILED=()
for k in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$k]}"; then FAILED+=("${NAMES[$k]}"); fi
done
echo "[ic-eval] finished $(date -Is)"
if (( ${#FAILED[@]} )); then
    echo "[ic-eval] ${#FAILED[@]} FAILED: ${FAILED[*]}" >&2
else
    echo "[ic-eval] all ${#PIDS[@]} evaluations OK"
fi

for S in 1 2 3; do
    [[ -f "$LOGS/eval2_s${S}.log" ]] || continue
    echo "--- seed $S ---"
    grep -aE "MEAN curve|memory_lift|attributable|Activation memory|The RNN adapts" \
        "$LOGS/eval2_s${S}.log" || true
done

exit 0
