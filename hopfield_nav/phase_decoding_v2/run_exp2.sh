#!/bin/bash -l
#SBATCH --job-name=hnav-phase-v2-exp2
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=pi_evelina9
#SBATCH --mem=100G
#SBATCH --output=/home/jackking/cls/hopfield_nav/logs/slurm_phase_v2_exp2_%j.out

# Exp 2: regular PCA + trajectory PCA visualizations.
#
# Usage:
#   CKPT=/path/to/ckpt.pt TRIALS_DIR=/path/to/exp1/trials sbatch run_exp2.sh
#   CKPT=/path/to/ckpt.pt bash run_exp2.sh   # collects fresh trials

module load miniforge/24.3.0-0 2>/dev/null || true
module load cuda/13.0.1 2>/dev/null || true
source activate cls 2>/dev/null || true
unset CUDA_VISIBLE_DEVICES

cd /home/jackking/cls

CKPT="${CKPT:?CKPT is required}"
ENCODER="${ENCODER:-}"
TRIALS_DIR="${TRIALS_DIR:-}"
NUM_ARENAS="${NUM_ARENAS:-100}"
N_STARTS="${N_STARTS:-10}"
MAX_STEPS="${MAX_STEPS:-100}"
N_DIST_MIN="${N_DIST_MIN:-0}"
N_DIST_MAX="${N_DIST_MAX:-5}"
N_TRAJ="${N_TRAJ:-20}"
EP1_STEPS="${EP1_STEPS:-500}"
EP2_STEPS="${EP2_STEPS:-100}"
MLP_HIDDEN="${MLP_HIDDEN:-64}"
MLP_EPOCHS="${MLP_EPOCHS:-50}"
SEED="${SEED:-0}"
RANDOM_AGENT="${RANDOM_AGENT:-}"
RANDOM_INIT_SEED="${RANDOM_INIT_SEED:-0}"
POLICY_FLAG="${POLICY_FLAG:---stochastic}"

CKPT_TAG="$(basename "$(dirname "$CKPT")")_$(basename "$CKPT" .pt)"
RANDOM_SUFFIX=""
if [ -n "$RANDOM_AGENT" ] && [ "$RANDOM_AGENT" != "0" ]; then
    RANDOM_SUFFIX="_random_init_seed${RANDOM_INIT_SEED}"
fi
OUT="${OUT:-/home/jackking/cls/hopfield_nav/phase_decoding_v2/results/exp2_${CKPT_TAG}${RANDOM_SUFFIX}}"
mkdir -p "$OUT"

ENCODER_FLAG=""
if [ -n "$ENCODER" ]; then ENCODER_FLAG="--encoder $ENCODER"; fi

TRIALS_FLAG=""
if [ -n "$TRIALS_DIR" ]; then TRIALS_FLAG="--trials_dir $TRIALS_DIR"; fi

RANDOM_AGENT_FLAG=""
if [ -n "$RANDOM_AGENT" ] && [ "$RANDOM_AGENT" != "0" ]; then
    RANDOM_AGENT_FLAG="--random_agent --random_init_seed $RANDOM_INIT_SEED"
fi

echo "[run_exp2] ckpt=$CKPT out=$OUT num_arenas=$NUM_ARENAS n_traj=$N_TRAJ trials_dir=$TRIALS_DIR random_agent=$RANDOM_AGENT"

python -m hopfield_nav.phase_decoding_v2.exp2 \
    --ckpt "$CKPT" $ENCODER_FLAG \
    --out_dir "$OUT" \
    $TRIALS_FLAG \
    --num_arenas "$NUM_ARENAS" \
    --n_starts "$N_STARTS" \
    --max_steps "$MAX_STEPS" \
    --n_dist_min "$N_DIST_MIN" \
    --n_dist_max "$N_DIST_MAX" \
    --n_traj_per_arena "$N_TRAJ" \
    --max_steps_ep1 "$EP1_STEPS" \
    --max_steps_ep2 "$EP2_STEPS" \
    --mlp_hidden "$MLP_HIDDEN" \
    --mlp_epochs "$MLP_EPOCHS" \
    --seed "$SEED" \
    $RANDOM_AGENT_FLAG \
    $POLICY_FLAG
