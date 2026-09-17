#!/bin/bash -l
#SBATCH --job-name=decodewalk
#SBATCH --time=0-02:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal_gpu
#SBATCH --mem=32G
#SBATCH --requeue
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/logs/decode_walk_%j.out

# Phase 1 of the goal-conditioned NN control (plan sec 6.4): the displacement
# decode learned from random walks and odometry alone. Every knob is spelled
# out so a run in a comparison series can say what it ran. The save dir is
# fixed by TAG so a requeued job resumes from its last checkpoint.
#
#   MODE=grid N_ENVS=64 SEED=0 TAG=p1_grid64_s0 sbatch hopfield_nav/run_decode_walk.sh

set -euo pipefail

REPO=${REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/nn-generalization-control}

MODE=${MODE:-grid}                      # grid | regular
HIDDEN=${HIDDEN:-768}
LAYERS=${LAYERS:-5}
NONLIN=${NONLIN:-relu}

N_ENVS=${N_ENVS:-64}
N_VAL_ENVS=${N_VAL_ENVS:-16}
N_SAME_ENVS=${N_SAME_ENVS:-4}
WALKERS=${WALKERS:-8}
STEPS_PER_UPDATE=${STEPS_PER_UPDATE:-64}
BUFFER_UPDATES=${BUFFER_UPDATES:-20}
K_MAX=${K_MAX:-30}
MAX_ABS=${MAX_ABS:-19}
PAIRS_PER_UPDATE=${PAIRS_PER_UPDATE:-32768}
TARGET=${TARGET:-direction}             # direction | heading8
BALANCE=${BALANCE:-0}                   # 1: --balance_range
RANGE_WARMUP=${RANGE_WARMUP:-0}
SIZE=${SIZE:-20}
OBS=${OBS:-120}
LAMBDAS=${LAMBDAS:-"11 12 13"}
FWHM=${FWHM:-0.25}
PLACE_MARGIN=${PLACE_MARGIN:-20}
PLACE_REGION=${PLACE_REGION:-anywhere}

N_UPDATES=${N_UPDATES:-4000}
LR=${LR:-1e-3}
LR_STEP_AT=${LR_STEP_AT:-0.7}
LR_STEP_GAMMA=${LR_STEP_GAMMA:-0.1}
EVAL_EVERY=${EVAL_EVERY:-50}
EVAL_PAIRS=${EVAL_PAIRS:-2048}
CKPT_EVERY=${CKPT_EVERY:-250}
SEED=${SEED:-0}
TAG=${TAG:-p1_${MODE}${N_ENVS}_s${SEED}}
USE_WANDB=${USE_WANDB:-0}

module load miniforge/24.3.0-0
module load cuda/13.0.1
source activate cls
unset CUDA_VISIBLE_DEVICES
cd "$REPO"
source scripts/cls_env.sh

BALANCE_FLAG=--no-balance_range
[[ "$BALANCE" == "1" ]] && BALANCE_FLAG=--balance_range
WANDB_FLAG=--no-use_wandb
[[ "$USE_WANDB" == "1" ]] && WANDB_FLAG=--use_wandb
SAVE_DIR=${SAVE_DIR:-$CLS_RUNS/agent_ckpts/goal_pairs_${TAG}}

PYTHONUNBUFFERED=1 python -m hopfield_nav.train_decode_walk \
  --mode "$MODE" --hidden_size "$HIDDEN" --num_layers "$LAYERS" --nonlinearity "$NONLIN" \
  --n_envs "$N_ENVS" --n_val_envs "$N_VAL_ENVS" --n_same_envs "$N_SAME_ENVS" \
  --walkers "$WALKERS" --steps_per_update "$STEPS_PER_UPDATE" --buffer_updates "$BUFFER_UPDATES" \
  --k_max "$K_MAX" --max_abs "$MAX_ABS" --pairs_per_update "$PAIRS_PER_UPDATE" --target "$TARGET" $BALANCE_FLAG --range_warmup_updates "$RANGE_WARMUP" \
  --size "$SIZE" --observation_size "$OBS" --lambdas $LAMBDAS --fwhm_ratio "$FWHM" \
  --place_margin "$PLACE_MARGIN" --place_region "$PLACE_REGION" \
  --n_updates "$N_UPDATES" --lr "$LR" --lr_step_at "$LR_STEP_AT" --lr_step_gamma "$LR_STEP_GAMMA" \
  --eval_every "$EVAL_EVERY" --eval_pairs "$EVAL_PAIRS" --ckpt_every "$CKPT_EVERY" \
  --seed "$SEED" --tag "$TAG" --save_dir "$SAVE_DIR" $WANDB_FLAG ${EXTRA:-}
