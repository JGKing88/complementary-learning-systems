#!/bin/bash -l
#SBATCH --job-name=hnav-repro-v35
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jackking@mit.edu
#SBATCH --partition=pi_evelina9
#SBATCH --mem=100G
#SBATCH --output=/home/jackking/cls/hopfield_nav/logs/slurm_repro_v35_%j.out

# Reproduce gentle-terrain-124 (sweep variant v18d39_size20_v35, seed 42) on the
# post-2026-08 schedule system. Target: its u380 checkpoint, 0.994 nav success /
# 22.9 mean speed on a fresh eval.
#
# The config is recovered from slurm_phase_a_sweep_13737223.out (the variant's
# EXTRA string plus the sweep script's fixed base) and cross-checked field by
# field against phase_a_u380.pt's saved config -- see
# docs/EXPERIMENTS_SCHEDULE_REPRO.md for that diff and what each knob became.
#
# Every value is spelled out rather than inherited from run_navigate.sh's
# defaults, because this run's whole purpose is to match a specific historical
# config and an inherited default would silently break that.
#
#   RUN_TAG=r1 sbatch hopfield_nav/run_repro_v35.sh
#
# Anything can still be overridden from the environment for a follow-up run:
#   RUN_TAG=r2 SEED=43 sbatch hopfield_nav/run_repro_v35.sh

JOB_LABEL=repro_v35

# --- the schedule ----------------------------------------------------------
# Was: --warmup_explore_only_updates 0 --interleave_empty_fraction 1.0
#      --interleave_empty_target 0.50 --interleave_anneal_updates 50
#      --phase_a_updates 3000
# With no warmup the old global anneal clock and the new stage-local one
# coincide, so this is the bit-exact translation (pinned by
# test_schedule.py::test_an_anneal_with_no_warmup_is_unchanged).
#
# 600 not 3000: nothing in this config reads the total. novelty_anneal is the
# only consumer and it is off, so the first 600 updates are identical either
# way, and the original died at ~u460 regardless.
SCHEDULE=${SCHEDULE:-'interleave:600,empty_frac=1.0->0.5,anneal=50'}

# --- encoder / scaffold ----------------------------------------------------
ENCODER=${ENCODER:-encoders/run_20260422_185816/encoder_best.pt}
FWHM_RATIO=${FWHM_RATIO:-0.25}
LAMBDAS=${LAMBDAS:-"11 12 13"}
NP=${NP:-400}
STATIC_VECTORHASH=${STATIC_VECTORHASH:-1}

# --- environment -----------------------------------------------------------
SIZE=${SIZE:-20}
OBSERVATION_SIZE=${OBSERVATION_SIZE:-12}
MOVEMENT_MODE=${MOVEMENT_MODE:-continuous}
GOAL_REWARD=${GOAL_REWARD:-5.0}
GOAL_RADIUS=${GOAL_RADIUS:-1.0}
TIME_PENALTY=${TIME_PENALTY:-0.05}

# --- agent -----------------------------------------------------------------
HOPFIELD_MODE=${HOPFIELD_MODE:-continuous}
HIDDEN_SIZE=${HIDDEN_SIZE:-1024}
NUM_RNN_LAYERS=${NUM_RNN_LAYERS:-1}
INIT_LOG_STD=${INIT_LOG_STD:--1.8}
FREEZE_LOG_STD=${FREEZE_LOG_STD:-1}
INPUT_PREV_REWARD=${INPUT_PREV_REWARD:-1}
INPUT_PREV_ACTION=${INPUT_PREV_ACTION:-0}
INPUT_HOPFIELD_RAW=${INPUT_HOPFIELD_RAW:-1}
INPUT_HOPFIELD_SIGNAL=${INPUT_HOPFIELD_SIGNAL:-1}
INPUT_SENSORY=${INPUT_SENSORY:-1}
INPUT_ENCODED_STATE=${INPUT_ENCODED_STATE:-0}
INPUT_GOAL_IN_MEMORY=${INPUT_GOAL_IN_MEMORY:-0}
INPUT_HOPFIELD_MULTISTEP=${INPUT_HOPFIELD_MULTISTEP:-"1 2 3"}

# --- optimization ----------------------------------------------------------
LR=${LR:-3e-4}
MOVE_ENT_COEF=${MOVE_ENT_COEF:-0.005}
PPO_CLIP_COEF=${PPO_CLIP_COEF:-0.15}

# --- reward shaping --------------------------------------------------------
NOVELTY_REWARD=${NOVELTY_REWARD:-0.3}
NOVELTY_ANNEAL=${NOVELTY_ANNEAL:-0}
NOVELTY_SCALE_REMAINING=${NOVELTY_SCALE_REMAINING:-1}
NOVELTY_SCALE_CAP=${NOVELTY_SCALE_CAP:-10}
REVISIT_PENALTY=${REVISIT_PENALTY:-0}
WALL_PENALTY=${WALL_PENALTY:-0.1}
PERSISTENCE_BONUS=${PERSISTENCE_BONUS:-0.05}

# --- explore-regime behavior -----------------------------------------------
EXPLORE_GOALS_OFF=${EXPLORE_GOALS_OFF:-1}
EPSILON_EXPLORE=${EPSILON_EXPLORE:-0.4}
EPSILON_ANNEAL_UPDATES=${EPSILON_ANNEAL_UPDATES:-200}

# --- distractors -----------------------------------------------------------
N_TRAIN_DISTRACTORS_MIN=${N_TRAIN_DISTRACTORS_MIN:-0}
N_TRAIN_DISTRACTORS_MAX=${N_TRAIN_DISTRACTORS_MAX:-10}
N_TRAIN_EMP_DISTRACTORS_MIN=${N_TRAIN_EMP_DISTRACTORS_MIN:-0}
N_TRAIN_EMP_DISTRACTORS_MAX=${N_TRAIN_EMP_DISTRACTORS_MAX:-10}

# --- rollout shape ---------------------------------------------------------
BATCH_ENVS=${BATCH_ENVS:-16}
STEPS_PER_ROLLOUT=${STEPS_PER_ROLLOUT:-400}
NUM_WORLDS=${NUM_WORLDS:-1}
ENVS_PER_WORLD=${ENVS_PER_WORLD:-80}
SEED=${SEED:-42}

# --- eval ------------------------------------------------------------------
NUM_VAL_ENVS=${NUM_VAL_ENVS:-10}
N_VAL_TRIALS=${N_VAL_TRIALS:-32}
VAL_DISTRACTORS=${VAL_DISTRACTORS:-"0 5 10"}
EVAL_EVERY=${EVAL_EVERY:-20}
# unset, i.e. follow EVAL_EVERY -- the original did, and u380 lands on it.
CKPT_EVERY=${CKPT_EVERY:-}

# --- logging ---------------------------------------------------------------
DEVICE=${DEVICE:-cuda}
USE_WANDB=${USE_WANDB:-1}
# Same project as the original, so the two sit side by side in one workspace.
WANDB_PROJECT=${WANDB_PROJECT:-hopfield-nav-phase-a-sweep}

cd /home/jackking/cls
source hopfield_nav/navigate_job.sh
