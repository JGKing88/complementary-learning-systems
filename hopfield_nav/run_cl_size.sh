#!/bin/bash -l
#SBATCH --job-name=cl-size
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=ou_bcs_normal
#SBATCH --mem=64G
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/logs/cl_size_%j.out

# The continual-learning protocol (analysis/continual/agenthash.py: sequential
# env introduction, Hopfield never reset) on one checkpoint at an arena size
# the run never trained on -- the size-OOD axis of the protocol. Same store
# conventions as run_cl_ood.sh (one oracle store per env; the store head was
# frozen in training) and the §37.6 convention of sampling the policy
# throughout (STOCHASTIC=1, the default here) so the argmax explorer's dead
# spots do not masquerade as forgetting.
#
#   sbatch hopfield_nav/run_cl_size.sh                      # d0_base u725 at 40
#   VAL_SIZE=60 N_ENVS=8 sbatch hopfield_nav/run_cl_size.sh
#   CKPT=/path/to/navigate_u1200.pt TAG=corner_u1200 sbatch hopfield_nav/run_cl_size.sh
#
# Writes <OUT>/cl_<TAG>_<split>_size<VAL_SIZE>[_stoch].json plus the plotter's
# three panels (_forgetting, _steps_to_goal, _path_to_goal .png/.pdf) and a
# retention summary. Reading trap, from the d0_base write-up: the plotter draws
# a FAILED trial at the max_steps ceiling, so spikes to MAX_STEPS are timeouts,
# not slow successes.

set -euo pipefail

REPO=${REPO:-/orcd/home/002/jackking/cls}
CKPT=${CKPT:-/orcd/pool/003/jackking/cls_runs/agent_ckpts/navigate_navp2_d0_base_s42_22133273/navigate_u725.pt}
TAG=${TAG:-d0_base_u725}
VAL_SIZE=${VAL_SIZE:-40}
# `recorded` is refused with a size (the recorded set has the size it was
# built at); unnamed traits default to held_out, so this is the fresh draw.
SPLIT=${SPLIT:-place=held_out}
N_ENVS=${N_ENVS:-5}
ITERS_PER_BLOCK=${ITERS_PER_BLOCK:-100}
MAX_STEPS=${MAX_STEPS:-500}
SEED=${SEED:-3000}
VAL_SEED=${VAL_SEED:-0}
SMOOTH=${SMOOTH:-10}
STOCHASTIC=${STOCHASTIC:-1}

cd "$REPO"
module load miniforge/24.3.0-0
module load cuda/13.0.1
source activate cls
unset CUDA_VISIBLE_DEVICES
source scripts/cls_env.sh

OUT=${OUT:-$CLS_RESULTS/cl_size}
mkdir -p "$OUT"

if [ ! -f "$CKPT" ]; then echo "!!! MISSING $CKPT"; exit 1; fi
# Content-addressed on (encoder, lambdas, Npos, fwhm): a cache hit for any
# checkpoint sharing d0_base's encoder, not a 12 GB rebuild.
scaffold=$(python -u -m analysis.continual.prep_scaffold \
    --ckpt "$CKPT" --cache_root "$CLS_SCAFFOLD_CACHE" --static_vectorhash | tail -n 1)
echo "[cl_size] $TAG scaffold cache: $scaffold"

split_tag=$(echo "$SPLIT" | tr '=' '-' | tr ',' '_')
tag="${split_tag}_size${VAL_SIZE}"
if [ "$STOCHASTIC" = "1" ]; then tag="${tag}_stoch"; fi
prefix="$OUT/cl_${TAG}_${tag}"
echo; echo "=== $TAG split=$SPLIT size=$VAL_SIZE n_envs=$N_ENVS iters=$ITERS_PER_BLOCK max_steps=$MAX_STEPS  ($(date))"
python -u -m analysis.continual.agenthash \
    --out "$prefix.json" \
    --run_name "$TAG, $SPLIT, size $VAL_SIZE, one store per env$([ "$STOCHASTIC" = "1" ] && echo ", sampled")" \
    --ckpt "$CKPT" --device cuda \
    --n_envs "$N_ENVS" --iters_per_block "$ITERS_PER_BLOCK" \
    --max_steps "$MAX_STEPS" --seed "$SEED" --num_full_iters 1 \
    --split "$SPLIT" --val_seed "$VAL_SEED" --val_size "$VAL_SIZE" \
    --static_vectorhash --scaffold_cache "$scaffold" --mmap \
    --lock_store_after_goal --oracle_store_at_goal \
    --oracle_lock_store_not_at_goal --goal_radius 1 \
    $([ "$STOCHASTIC" = "1" ] && echo --stochastic_policy)
python -u -m analysis.continual.plotting \
    --history "$prefix.json" --out_prefix "$prefix" \
    --smooth "$SMOOTH" --show_std 0
echo "--- retention: $TAG $SPLIT size $VAL_SIZE"
python -u -m analysis.continual.retention --history "$prefix.json"

echo; echo "=== done $(date)  outputs: $prefix*"
