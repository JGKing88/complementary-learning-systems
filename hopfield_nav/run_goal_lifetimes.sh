#!/bin/bash -l
#SBATCH --job-name=goallife
#SBATCH --time=0-04:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal_gpu
#SBATCH --mem=32G
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/logs/goal_lifetimes_%j.out

# Experiment B: recurrent lifetimes. See docs/NN_CONTROL_PLAN.md sec 4.
#   MODE=grid ARM=full PLACE_REGION=rect:0,0,400,400 N_OOD_PLACE=16 TAG=b1x_full_s0 \
#     sbatch hopfield_nav/run_goal_lifetimes.sh
set -euo pipefail
REPO=${REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/nn-generalization-control}

MODE=${MODE:-grid}
ARM=${ARM:-full}                        # full | rec | dist
MOVEMENT=${MOVEMENT:-continuous}
HIDDEN=${HIDDEN:-512}
LAYERS=${LAYERS:-1}
NONLIN=${NONLIN:-relu}

N_ENVS=${N_ENVS:-64}
N_VAL_ENVS=${N_VAL_ENVS:-16}
N_SAME_ENVS=${N_SAME_ENVS:-8}
SIZE=${SIZE:-20}
OBS=${OBS:-120}
LAMBDAS=${LAMBDAS:-"11 12 13"}
FWHM=${FWHM:-0.25}
PLACE_MARGIN=${PLACE_MARGIN:-20}
GOAL_VAL_FRAC=${GOAL_VAL_FRAC:-0.2}
REGION_VAL_FRAC=${REGION_VAL_FRAC:-0.1}
PLACE_REGION=${PLACE_REGION:-anywhere}
N_OOD_PLACE=${N_OOD_PLACE:-0}

BATCH_ENVS=${BATCH_ENVS:-64}
ENVS_PER_UPDATE=${ENVS_PER_UPDATE:-8}
STEPS_PER_ROLLOUT=${STEPS_PER_ROLLOUT:-64}
RESAMPLE_ENVS_EVERY=${RESAMPLE_ENVS_EVERY:-32}
EPISODE_MAX_STEPS=${EPISODE_MAX_STEPS:-60}
INIT_LOG_STD=${INIT_LOG_STD:--1.0}

N_UPDATES=${N_UPDATES:-2000}
LR=${LR:-1e-3}
EPOCHS=${EPOCHS:-4}
N_MINIBATCHES=${N_MINIBATCHES:-4}
EVAL_EVERY=${EVAL_EVERY:-100}
EVAL_PAIRS=${EVAL_PAIRS:-2048}
LIFETIME_EVERY=${LIFETIME_EVERY:-500}
N_LIFETIMES=${N_LIFETIMES:-64}
N_EVAL_EPISODES=${N_EVAL_EPISODES:-20}
N_LIFETIME_ENVS=${N_LIFETIME_ENVS:-8}
CKPT_EVERY=${CKPT_EVERY:-500}
SEED=${SEED:-0}
TAG=${TAG:-b_${MODE}_${ARM}_s${SEED}}
WANDB_PROJECT=${WANDB_PROJECT:-train_goal_lifetimes}
USE_WANDB=${USE_WANDB:-1}

module load miniforge/24.3.0-0
module load cuda/13.0.1
source activate cls
unset CUDA_VISIBLE_DEVICES
cd "$REPO"
source scripts/cls_env.sh
WANDB_FLAG=--use_wandb
[[ "$USE_WANDB" == "0" ]] && WANDB_FLAG=--no-use_wandb

PYTHONUNBUFFERED=1 python -m hopfield_nav.train_goal_lifetimes \
  --mode "$MODE" --arm "$ARM" --movement_mode "$MOVEMENT" \
  --hidden_size "$HIDDEN" --num_layers "$LAYERS" --nonlinearity "$NONLIN" \
  --n_envs "$N_ENVS" --n_val_envs "$N_VAL_ENVS" --n_same_envs "$N_SAME_ENVS" \
  --size "$SIZE" --observation_size "$OBS" --lambdas $LAMBDAS --fwhm_ratio "$FWHM" \
  --place_margin "$PLACE_MARGIN" --goal_val_frac "$GOAL_VAL_FRAC" --region_val_frac "$REGION_VAL_FRAC" \
  --place_region "$PLACE_REGION" --n_ood_place "$N_OOD_PLACE" \
  --batch_envs "$BATCH_ENVS" --envs_per_update "$ENVS_PER_UPDATE" \
  --steps_per_rollout "$STEPS_PER_ROLLOUT" --resample_envs_every "$RESAMPLE_ENVS_EVERY" \
  --episode_max_steps "$EPISODE_MAX_STEPS" --init_log_std "$INIT_LOG_STD" \
  --n_updates "$N_UPDATES" --lr "$LR" --epochs "$EPOCHS" --n_minibatches "$N_MINIBATCHES" \
  --eval_every "$EVAL_EVERY" --eval_pairs "$EVAL_PAIRS" --lifetime_every "$LIFETIME_EVERY" \
  --n_lifetimes "$N_LIFETIMES" --n_eval_episodes "$N_EVAL_EPISODES" --n_lifetime_envs "$N_LIFETIME_ENVS" \
  --ckpt_every "$CKPT_EVERY" --seed "$SEED" --tag "$TAG" --wandb_project "$WANDB_PROJECT" $WANDB_FLAG
