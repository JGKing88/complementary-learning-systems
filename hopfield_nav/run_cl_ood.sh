#!/bin/bash -l
#SBATCH --job-name=cl-ood
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=ou_bcs_normal
#SBATCH --mem=64G
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/logs/cl_ood_%j.out

# Continual-learning protocol (analysis/continual/agenthash.py, sequential
# env introduction, Hopfield never reset) for the place-OOD arms, on their own
# in-region val envs (`recorded`) and on envs minted OUTSIDE the declared
# training region (`place=ood`). Same protocol as the d0_base u725 figures in
# docs/DUAL_TRAINING.md ("Continual learning -- 2,240 revisits, zero
# failures"): 6 envs (d0_base used 5), 40 iters/block, 200-step cap, seed 3000, deterministic
# policy, autostore with one store per env (--oracle_store_at_goal
# --oracle_lock_store_not_at_goal --lock_store_after_goal; the store head was
# frozen in training, so there is no learned store policy to use).
#
#   sbatch hopfield_nav/run_cl_ood.sh                    # corner arms, u1200
#   ARMS="ood_place ood_place_rp" UPDATE=1200 sbatch hopfield_nav/run_cl_ood.sh
#
# Writes <OUT>/cl_<arm>_u<N>_<split>.json plus the plotter's three panels
# (_forgetting, _steps_to_goal, _path_to_goal .png/.pdf) and a retention
# summary per history. Reading trap, from the d0_base write-up: the plotter
# draws a FAILED trial at the max_steps ceiling, so spikes to 200 are
# timeouts, not slow successes.

set -euo pipefail

REPO=${REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/nav-ood-place}
ARMS=${ARMS:-"ood_corner ood_corner_rp"}
UPDATE=${UPDATE:-1200}
SPLITS=${SPLITS:-"recorded place=ood"}
N_ENVS=${N_ENVS:-6}          # recorded is the run's 6 base_val envs verbatim; minted splits match it
ITERS_PER_BLOCK=${ITERS_PER_BLOCK:-40}
MAX_STEPS=${MAX_STEPS:-200}
SEED=${SEED:-3000}
VAL_SEED=${VAL_SEED:-0}
SMOOTH=${SMOOTH:-10}

cd "$REPO"
module load miniforge/24.3.0-0
module load cuda/13.0.1
source activate cls
unset CUDA_VISIBLE_DEVICES
source scripts/cls_env.sh

OUT=${OUT:-$CLS_RESULTS/nav_tri_probe}
mkdir -p "$OUT"

declare -A RUN=(
  [d0_base]=navigate_navp2_d0_base_s42_22133273
  [ood_place]=navigate_navp2_ood_place_s42_22599420
  [ood_place_rp]=navigate_navp2_ood_place_rp_s42_22599421
  [ood_corner]=navigate_navp2_ood_corner_s42_22629938
  [ood_corner_rp]=navigate_navp2_ood_corner_rp_s42_22629939
)

for arm in $ARMS; do
  ckpt="$CLS_CKPTS/${RUN[$arm]}/navigate_u${UPDATE}.pt"
  if [ ! -f "$ckpt" ]; then echo "!!! MISSING $ckpt"; continue; fi
  # Content-addressed on (encoder, lambdas, Npos, fwhm): every arm here shares
  # d0_base's encoder, so this is one cache hit, not a 12 GB rebuild per arm.
  scaffold=$(python -u -m analysis.continual.prep_scaffold \
      --ckpt "$ckpt" --cache_root "$CLS_SCAFFOLD_CACHE" --static_vectorhash | tail -n 1)
  echo "[cl_ood] $arm u$UPDATE scaffold cache: $scaffold"
  for split in $SPLITS; do
    tag=$(echo "$split" | tr '=' '-' | tr ',' '_')
    prefix="$OUT/cl_${arm}_u${UPDATE}_${tag}"
    echo; echo "=== $arm u$UPDATE split=$split  ($(date))"
    python -u -m analysis.continual.agenthash \
        --out "$prefix.json" \
        --run_name "$arm u$UPDATE, $split, one store per env" \
        --ckpt "$ckpt" --device cuda \
        --n_envs "$N_ENVS" --iters_per_block "$ITERS_PER_BLOCK" \
        --max_steps "$MAX_STEPS" --seed "$SEED" --num_full_iters 1 \
        --split "$split" --val_seed "$VAL_SEED" \
        --static_vectorhash --scaffold_cache "$scaffold" --mmap \
        --lock_store_after_goal --oracle_store_at_goal \
        --oracle_lock_store_not_at_goal --goal_radius 1 \
      || { echo "!!! $arm $split agenthash FAILED"; continue; }
    python -u -m analysis.continual.plotting \
        --history "$prefix.json" --out_prefix "$prefix" \
        --smooth "$SMOOTH" --show_std 0
    echo "--- retention: $arm u$UPDATE $split"
    python -u -m analysis.continual.retention --history "$prefix.json"
  done
done

echo; echo "=== done $(date)  outputs: $OUT/cl_*_u${UPDATE}_*"
