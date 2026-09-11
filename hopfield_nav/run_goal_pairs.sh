#!/bin/bash -l
#SBATCH --job-name=goalpairs
#SBATCH --time=0-02:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal_gpu
#SBATCH --mem=32G
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/logs/goal_pairs_%j.out

# Experiment A of the goal-conditioned NN control: a memoryless network on
# i.i.d. (start, goal) pairs. See docs/NN_CONTROL_PLAN.md for the design and
# docs/NN_CONTROL_LOG.md for what has been run; this file only holds knobs.
#
#   MODE=grid MOVEMENT=continuous LAYERS=4 TAG=a1_mlp4 sbatch hopfield_nav/run_goal_pairs.sh
#
# Every knob is spelled out rather than inherited from argparse defaults so
# a run in a comparison series can say what it ran.

set -euo pipefail

REPO=${REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/nn-generalization-control}

MODE=${MODE:-grid}                      # xy | grid | regular
MOVEMENT=${MOVEMENT:-continuous}        # continuous | discrete
HIDDEN=${HIDDEN:-256}
LAYERS=${LAYERS:-2}
NONLIN=${NONLIN:-relu}
DROPOUT=${DROPOUT:-0.0}

N_ENVS=${N_ENVS:-64}
N_VAL_ENVS=${N_VAL_ENVS:-16}
N_SAME_ENVS=${N_SAME_ENVS:-8}
PAIRS_PER_ENV=${PAIRS_PER_ENV:-512}
SIZE=${SIZE:-20}
OBS=${OBS:-120}
LAMBDAS=${LAMBDAS:-"11 12 13"}
FWHM=${FWHM:-0.25}
PLACE_MARGIN=${PLACE_MARGIN:-20}
GOAL_VAL_FRAC=${GOAL_VAL_FRAC:-0.2}
REGION_VAL_FRAC=${REGION_VAL_FRAC:-0.1}
PLACE_REGION=${PLACE_REGION:-anywhere}
N_OOD_PLACE=${N_OOD_PLACE:-0}

N_UPDATES=${N_UPDATES:-2000}
LR=${LR:-1e-3}
WD=${WD:-0.0}
LR_SCHEDULE=${LR_SCHEDULE:-none}
LR_STEP_AT=${LR_STEP_AT:-0.75}
LR_STEP_GAMMA=${LR_STEP_GAMMA:-0.1}
EVAL_EVERY=${EVAL_EVERY:-100}
EVAL_PAIRS=${EVAL_PAIRS:-2048}
CKPT_EVERY=${CKPT_EVERY:-500}
SEED=${SEED:-0}
TAG=${TAG:-${MODE}_${MOVEMENT}_l${LAYERS}h${HIDDEN}}
WANDB_PROJECT=${WANDB_PROJECT:-train_goal_pairs}
USE_WANDB=${USE_WANDB:-1}

module load miniforge/24.3.0-0
module load cuda/13.0.1
source activate cls
unset CUDA_VISIBLE_DEVICES
cd "$REPO"
source scripts/cls_env.sh

WANDB_FLAG=--use_wandb
[[ "$USE_WANDB" == "0" ]] && WANDB_FLAG=--no-use_wandb

PYTHONUNBUFFERED=1 python -m hopfield_nav.train_goal_pairs \
  --mode "$MODE" --movement_mode "$MOVEMENT" \
  --hidden_size "$HIDDEN" --num_layers "$LAYERS" --nonlinearity "$NONLIN" --dropout "$DROPOUT" \
  --n_envs "$N_ENVS" --n_val_envs "$N_VAL_ENVS" --n_same_envs "$N_SAME_ENVS" \
  --pairs_per_env "$PAIRS_PER_ENV" --size "$SIZE" --observation_size "$OBS" \
  --lambdas $LAMBDAS --fwhm_ratio "$FWHM" --place_margin "$PLACE_MARGIN" \
  --goal_val_frac "$GOAL_VAL_FRAC" --region_val_frac "$REGION_VAL_FRAC" \
  --place_region "$PLACE_REGION" --n_ood_place "$N_OOD_PLACE" \
  --n_updates "$N_UPDATES" --lr "$LR" --weight_decay "$WD" --lr_schedule "$LR_SCHEDULE" --lr_step_at "$LR_STEP_AT" --lr_step_gamma "$LR_STEP_GAMMA" \
  --eval_every "$EVAL_EVERY" --eval_pairs "$EVAL_PAIRS" --ckpt_every "$CKPT_EVERY" \
  --seed "$SEED" --tag "$TAG" --wandb_project "$WANDB_PROJECT" $WANDB_FLAG
