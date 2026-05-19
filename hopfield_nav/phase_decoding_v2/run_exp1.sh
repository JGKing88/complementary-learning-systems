#!/bin/bash -l
#SBATCH --job-name=hnav-abstract-decoding
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=pi_evelina9
#SBATCH --mem=64G
#SBATCH --output=/home/jackking/cls/hopfield_nav/logs/slurm_abstract_decoding_%j.out

# Exp 1: cross-arena parallelism + decodability bars across 4 split families.
#
# Usage:
#   CKPT=/path/to/ckpt.pt sbatch run_exp1.sh
#   CKPT=/path/to/ckpt.pt NUM_ARENAS=100 N_STARTS=100 bash run_exp1.sh
#
# Resume from already-collected trials (skips the 40-min collect phase):
#   TRIALS_DIR=/path/to/exp1_run/trials sbatch run_exp1.sh
#
# Cap LR training rows to bound memory on big LOO pools:
#   SUBSAMPLE_TRAIN=200000 sbatch run_exp1.sh
#
# Skip the LOO split family (LOO has 100 folds and dominates eval time):
#   SKIP_LOO=1 sbatch run_exp1.sh
#
# Random-init control (cfg from ckpt, but agent weights are random):
#   RANDOM_AGENT=1 RANDOM_INIT_SEED=0 sbatch run_exp1.sh
# (OUT auto-suffixes with _random_init when set so it doesn't collide with
#  the trained-agent run.)

module load miniforge/24.3.0-0 2>/dev/null || true
module load cuda/13.0.1 2>/dev/null || true
source activate cls 2>/dev/null || true
unset CUDA_VISIBLE_DEVICES

cd /home/jackking/cls

CKPT="/orcd/home/002/jackking/cls/checkpoint/phase_a_only_glamorous-field-97/phase_a_u360.pt"
ENCODER="${ENCODER:-}"
TRIALS_DIR="${TRIALS_DIR:-}"
NUM_ARENAS="${NUM_ARENAS:-100}"
N_STARTS="${N_STARTS:-100}"
MAX_STEPS="${MAX_STEPS:-100}"
N_DIST_MIN="${N_DIST_MIN:-0}"
N_DIST_MAX="${N_DIST_MAX:-5}"
N_RAND="${N_RAND:-20}"
TEST_FRAC="${TEST_FRAC:-0.2}"
SUBSAMPLE_TRAIN="${SUBSAMPLE_TRAIN:-}"
SKIP_LOO="${SKIP_LOO:-}"
RANDOM_AGENT="${RANDOM_AGENT:-}"
RANDOM_INIT_SEED="${RANDOM_INIT_SEED:-0}"
SEED="${SEED:-0}"
POLICY="${POLICY:-stochastic}"
POLICY_FLAG="--${POLICY}"

CKPT_TAG="$(basename "$(dirname "$CKPT")")_$(basename "$CKPT" .pt)"
RANDOM_SUFFIX=""
if [ -n "$RANDOM_AGENT" ] && [ "$RANDOM_AGENT" != "0" ]; then
    RANDOM_SUFFIX="_random_init_seed${RANDOM_INIT_SEED}"
fi
OUT="${OUT:-/home/jackking/cls/hopfield_nav/phase_decoding_v2/results/exp1_${CKPT_TAG}_${POLICY}${RANDOM_SUFFIX}}"
mkdir -p "$OUT"

ENCODER_FLAG=""
if [ -n "$ENCODER" ]; then ENCODER_FLAG="--encoder $ENCODER"; fi

TRIALS_FLAG=""
if [ -n "$TRIALS_DIR" ]; then TRIALS_FLAG="--trials_dir $TRIALS_DIR"; fi

SUBSAMPLE_FLAG=""
if [ -n "$SUBSAMPLE_TRAIN" ]; then
    SUBSAMPLE_FLAG="--subsample_train $SUBSAMPLE_TRAIN"
fi

SKIP_LOO_FLAG=""
if [ -n "$SKIP_LOO" ] && [ "$SKIP_LOO" != "0" ]; then
    SKIP_LOO_FLAG="--skip_loo"
fi

RANDOM_AGENT_FLAG=""
if [ -n "$RANDOM_AGENT" ] && [ "$RANDOM_AGENT" != "0" ]; then
    RANDOM_AGENT_FLAG="--random_agent --random_init_seed $RANDOM_INIT_SEED"
fi

echo "[run_exp1] ckpt=$CKPT out=$OUT num_arenas=$NUM_ARENAS n_starts=$N_STARTS policy=$POLICY_FLAG trials_dir=$TRIALS_DIR subsample=$SUBSAMPLE_TRAIN skip_loo=$SKIP_LOO random_agent=$RANDOM_AGENT random_init_seed=$RANDOM_INIT_SEED"

python -m hopfield_nav.phase_decoding_v2.exp1 \
    --ckpt "$CKPT" $ENCODER_FLAG \
    --out_dir "$OUT" \
    $TRIALS_FLAG \
    --num_arenas "$NUM_ARENAS" \
    --n_starts "$N_STARTS" \
    --max_steps "$MAX_STEPS" \
    --n_dist_min "$N_DIST_MIN" \
    --n_dist_max "$N_DIST_MAX" \
    --n_random_splits "$N_RAND" \
    --test_frac "$TEST_FRAC" \
    $SUBSAMPLE_FLAG \
    $SKIP_LOO_FLAG \
    $RANDOM_AGENT_FLAG \
    --seed "$SEED" \
    $POLICY_FLAG

