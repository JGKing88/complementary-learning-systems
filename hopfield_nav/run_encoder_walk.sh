#!/bin/bash -l
#SBATCH --job-name=encwalk
#SBATCH --time=0-02:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal_gpu
#SBATCH --mem=40G
#SBATCH --requeue
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/logs/encoder_walk_%j.out

# Phase 1, encoder side (plan sec 6.4): the encoder objective on the decode's
# walks. att0.5's configuration by default. Fixed save dir by TAG so a requeued
# job resumes.
#
#   LABELS=odometry SEED=0 TAG=p1e_odo64_s0 sbatch hopfield_nav/run_encoder_walk.sh
#   EVAL_ONLY=/path/encoder_final.pt sbatch ... (readout of a pre-trained encoder on this world)

set -euo pipefail

REPO=${REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/nn-generalization-control}

LABELS=${LABELS:-odometry}              # odometry | coords
POSITIONS=${POSITIONS:-walk}            # walk | iid
RADIUS=${RADIUS:-20}
HIDDEN=${HIDDEN:-256}
LAYERS=${LAYERS:-4}
OUT_DIM=${OUT_DIM:-1024}
GAIN_START=${GAIN_START:-1.0}
GAIN_END=${GAIN_END:-100.0}
ATTRACT=${ATTRACT:-0.5}
REPEL=${REPEL:-1.0}
RATE=${RATE:-0.5}
LR=${LR:-3e-4}

N_ENVS=${N_ENVS:-64}
N_VAL_ENVS=${N_VAL_ENVS:-16}
N_SAME_ENVS=${N_SAME_ENVS:-4}
WALKERS=${WALKERS:-8}
STEPS_PER_UPDATE=${STEPS_PER_UPDATE:-64}
BUFFER_UPDATES=${BUFFER_UPDATES:-20}
BATCHES_PER_UPDATE=${BATCHES_PER_UPDATE:-8}
BATCH_ENVS=${BATCH_ENVS:-8}
PER_WALKER=${PER_WALKER:-64}
MAX_ABS=${MAX_ABS:-19}
SIZE=${SIZE:-20}
PLACE_MARGIN=${PLACE_MARGIN:-20}

N_UPDATES=${N_UPDATES:-4000}
EVAL_EVERY=${EVAL_EVERY:-50}
EVAL_PAIRS=${EVAL_PAIRS:-2048}
CKPT_EVERY=${CKPT_EVERY:-250}
SEED=${SEED:-0}
TAG=${TAG:-p1e_${LABELS}${N_ENVS}_s${SEED}}
EVAL_ONLY=${EVAL_ONLY:-}

module load miniforge/24.3.0-0
module load cuda/13.0.1
source activate cls
unset CUDA_VISIBLE_DEVICES
cd "$REPO"
source scripts/cls_env.sh
SAVE_DIR=${SAVE_DIR:-$CLS_RUNS/agent_ckpts/goal_pairs_${TAG}}

EVAL_FLAG=""
[[ -n "$EVAL_ONLY" ]] && EVAL_FLAG="--eval_only $EVAL_ONLY"

PYTHONUNBUFFERED=1 python -m hopfield_nav.train_encoder_walk \
  --labels "$LABELS" --positions "$POSITIONS" --radius "$RADIUS" --hidden_dim "$HIDDEN" --num_hidden_layers "$LAYERS" --out_dim "$OUT_DIM" \
  --gain_start "$GAIN_START" --gain_end "$GAIN_END" --attract "$ATTRACT" --repel "$REPEL" --rate "$RATE" --lr "$LR" \
  --n_envs "$N_ENVS" --n_val_envs "$N_VAL_ENVS" --n_same_envs "$N_SAME_ENVS" \
  --walkers "$WALKERS" --steps_per_update "$STEPS_PER_UPDATE" --buffer_updates "$BUFFER_UPDATES" \
  --batches_per_update "$BATCHES_PER_UPDATE" --batch_envs "$BATCH_ENVS" --per_walker "$PER_WALKER" --max_abs "$MAX_ABS" \
  --size "$SIZE" --place_margin "$PLACE_MARGIN" \
  --n_updates "$N_UPDATES" --eval_every "$EVAL_EVERY" --eval_pairs "$EVAL_PAIRS" --ckpt_every "$CKPT_EVERY" \
  --seed "$SEED" --tag "$TAG" --save_dir "$SAVE_DIR" $EVAL_FLAG ${EXTRA:-}
