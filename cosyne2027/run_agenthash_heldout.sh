#!/bin/bash -l
#SBATCH --job-name=cosyne-hash
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=ou_bcs_normal
#SBATCH --mem=64G
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/logs/cosyne_hash_%j.out

# Agent-HaSH (d0_base u725) on the continual controls' schedule, for the
# Cosyne main-results figure. Per update: one store-allowed episode in the
# current arena (200-step cap, the controls' rollout length), then one
# held-out, no-store episode from a random start in every arena seen so far
# (100-step cap = T_eval). 5 arenas x 200 updates, seeds 1..8 with
# --env_seed = seed, the controls' seeds (each model keeps its own wall
# resolution, so the walls differ). Goal radius 0.5, the controls' success
# criterion. Sampled actions, Agent-HaSH's evaluation convention. One store
# per arena, at the goal, as in every Agent-HaSH continual run.
#
#   sbatch cosyne2027/run_agenthash_heldout.sh             # 8 seeds
#   SMOKE=1 sbatch cosyne2027/run_agenthash_heldout.sh     # 2 arenas x 3 updates

set -euo pipefail

REPO=${REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/cosyne2027-figs}
SEEDS=${SEEDS:-"1 2 3 4 5 6 7 8"}
N_ENVS=${N_ENVS:-5}
ITERS_PER_BLOCK=${ITERS_PER_BLOCK:-200}
MAX_STEPS=${MAX_STEPS:-200}
T_EVAL=${T_EVAL:-100}
GOAL_RADIUS=${GOAL_RADIUS:-0.5}
SMOKE=${SMOKE:-0}
if [ "$SMOKE" = "1" ]; then SEEDS=1; N_ENVS=2; ITERS_PER_BLOCK=3; fi

cd "$REPO"
module load miniforge/24.3.0-0
module load cuda/13.0.1
source activate cls
unset CUDA_VISIBLE_DEVICES
source scripts/cls_env.sh

# A plain if: `$([ ... ] && echo ...)` returns 1 when false, and set -e
# then ends the script silently.
SUFFIX=""
if [ "$SMOKE" = "1" ]; then SUFFIX=_smoke; fi
OUT=${OUT:-$CLS_RESULTS/cosyne2027/agenthash_heldout$SUFFIX}
mkdir -p "$OUT"
ckpt="$CLS_CKPTS/navigate_navp2_d0_base_s42_22133273/navigate_u725.pt"
scaffold=$(python -u -m analysis.continual.prep_scaffold \
    --ckpt "$ckpt" --cache_root "$CLS_SCAFFOLD_CACHE" --static_vectorhash | tail -n 1)
echo "[cosyne-hash] scaffold cache: $scaffold"

for seed in $SEEDS; do
  echo; echo "=== seed $seed  ($(date))"
  python -u -m analysis.continual.agenthash \
      --out "$OUT/agenthash_s${seed}.json" \
      --run_name "d0_base u725, controls' schedule, held-out T_eval=$T_EVAL, seed $seed" \
      --ckpt "$ckpt" --device cuda \
      --n_envs "$N_ENVS" --iters_per_block "$ITERS_PER_BLOCK" \
      --max_steps "$MAX_STEPS" --heldout_eval_steps "$T_EVAL" \
      --seed "$((3000 + seed))" --env_seed "$seed" --num_full_iters 1 \
      --static_vectorhash --scaffold_cache "$scaffold" --mmap \
      --lock_store_after_goal --oracle_store_at_goal \
      --oracle_lock_store_not_at_goal --goal_radius "$GOAL_RADIUS" \
      --stochastic_policy
done

echo; echo "=== done $(date)  outputs: $OUT"
