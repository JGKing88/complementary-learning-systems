#!/bin/bash -l
#SBATCH --job-name=hnav-navigate
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jackking@mit.edu
#SBATCH --partition=pi_fiete
#SBATCH --mem=100G
#SBATCH --output=/home/jackking/cls/hopfield_nav/logs/slurm_navigate_%j.out

# The composed run: explore first, then interleave the two regimes while
# shifting the mix toward following, then a short pure-follow tail.
#
#   sbatch hopfield_nav/run_navigate.sh
#   SCHEDULE='explore:400 ; exploit:200' SEED=7 sbatch hopfield_nav/run_navigate.sh
#
# Every one of train_navigate's 71 flags is set below, so this file is the whole
# control panel -- edit a value here rather than remembering an EXTRA string.
# Each is written `X=${X:-value}`, so an environment variable still wins:
#
#   WALL_PENALTY=0.2 sbatch hopfield_nav/run_navigate.sh
#
# Booleans are 1/0. Lists are space-separated strings. Leaving a value empty
# ("") drops the flag entirely, which falls back to the trainer's own default.

JOB_LABEL=navigate
SCHEDULE=${SCHEDULE:-'explore:200 ; interleave:800,empty_frac=1.0->0.5,anneal=50 ; exploit:100'}

# Start from an existing checkpoint. Empty = fresh random init.
LOAD_CKPT=${LOAD_CKPT:-}

# ---------------------------------------------------------------------------
# Everything below applies to a FRESH run only.
#
# On a resume the trainer takes the parent checkpoint's config as its base and
# lets only the flags actually passed override it, so spelling out a value here
# would overwrite the parent with this file's defaults rather than inheriting
# them. Set any of these in the environment to override the parent deliberately:
#
#   LOAD_CKPT=…/navigate_final.pt WALL_PENALTY=0.2 sbatch hopfield_nav/run_navigate.sh
# ---------------------------------------------------------------------------
if [ -z "$LOAD_CKPT" ]; then

# --- Encoder / scaffold ----------------------------------------------------
ENCODER=${ENCODER:-encoders/run_20260422_185816/encoder_best.pt}
ENCODER_GAIN=${ENCODER_GAIN:-}          # empty = read the gain out of the encoder checkpoint
FWHM_RATIO=${FWHM_RATIO:-0.25}
LAMBDAS=${LAMBDAS:-"11 12 13"}
NP=${NP:-400}
STATIC_VECTORHASH=${STATIC_VECTORHASH:-1}

# --- Environment -----------------------------------------------------------
SIZE=${SIZE:-8}
OBSERVATION_SIZE=${OBSERVATION_SIZE:-12}
MOVEMENT_MODE=${MOVEMENT_MODE:-continuous}
GOALS_ACTIVE=${GOALS_ACTIVE:-1}
GOAL_REWARD=${GOAL_REWARD:-1.0}
GOAL_RADIUS=${GOAL_RADIUS:-0.5}
TIME_PENALTY=${TIME_PENALTY:-0.01}
CONTINUOUS_NORMALIZE=${CONTINUOUS_NORMALIZE:-0}
MAX_ACTION_NORM=${MAX_ACTION_NORM:-}    # empty = uncapped
MIN_ACTION_NORM=${MIN_ACTION_NORM:-}    # empty = no floor
ALLOW_OFFCELL_STORE=${ALLOW_OFFCELL_STORE:-0}

# --- Agent -----------------------------------------------------------------
HOPFIELD_MODE=${HOPFIELD_MODE:-continuous}
HIDDEN_SIZE=${HIDDEN_SIZE:-128}
NUM_RNN_LAYERS=${NUM_RNN_LAYERS:-1}
INIT_LOG_STD=${INIT_LOG_STD:--0.5}
FREEZE_LOG_STD=${FREEZE_LOG_STD:-0}
INPUT_PREV_REWARD=${INPUT_PREV_REWARD:-1}
INPUT_PREV_ACTION=${INPUT_PREV_ACTION:-1}
INPUT_HOPFIELD_RAW=${INPUT_HOPFIELD_RAW:-1}
INPUT_HOPFIELD_SIGNAL=${INPUT_HOPFIELD_SIGNAL:-1}
INPUT_SENSORY=${INPUT_SENSORY:-1}
INPUT_ENCODED_STATE=${INPUT_ENCODED_STATE:-0}
INPUT_GOAL_IN_MEMORY=${INPUT_GOAL_IN_MEMORY:-0}
INPUT_HOPFIELD_MULTISTEP=${INPUT_HOPFIELD_MULTISTEP:-}   # e.g. "1 2 3"

# --- Optimization ----------------------------------------------------------
LR=${LR:-3e-4}
MOVE_ENT_COEF=${MOVE_ENT_COEF:-0.01}
PPO_CLIP_COEF=${PPO_CLIP_COEF:-0.2}

# --- Reward shaping --------------------------------------------------------
NOVELTY_REWARD=${NOVELTY_REWARD:-0.3}
NOVELTY_ANNEAL=${NOVELTY_ANNEAL:-0}
NOVELTY_SCALE_REMAINING=${NOVELTY_SCALE_REMAINING:-0}
NOVELTY_SCALE_CAP=${NOVELTY_SCALE_CAP:-10.0}
REVISIT_PENALTY=${REVISIT_PENALTY:-0.0}
WALL_PENALTY=${WALL_PENALTY:-0.0}
PERSISTENCE_BONUS=${PERSISTENCE_BONUS:-0.0}

# --- Explore-regime behavior -----------------------------------------------
EXPLORE_GOALS_OFF=${EXPLORE_GOALS_OFF:-0}
RANDOMIZE_GOAL_PER_ROLLOUT=${RANDOMIZE_GOAL_PER_ROLLOUT:-0}
EPSILON_EXPLORE=${EPSILON_EXPLORE:-0.0}
EPSILON_ANNEAL_UPDATES=${EPSILON_ANNEAL_UPDATES:-0}

# --- Distractors -----------------------------------------------------------
N_TRAIN_DISTRACTORS_MIN=${N_TRAIN_DISTRACTORS_MIN:-0}
N_TRAIN_DISTRACTORS_MAX=${N_TRAIN_DISTRACTORS_MAX:-0}
N_TRAIN_EMP_DISTRACTORS_MIN=${N_TRAIN_EMP_DISTRACTORS_MIN:-0}
N_TRAIN_EMP_DISTRACTORS_MAX=${N_TRAIN_EMP_DISTRACTORS_MAX:-0}
N_TRAIN_DISTRACTORS_MAX_END=${N_TRAIN_DISTRACTORS_MAX_END:-}       # empty = no curriculum
N_TRAIN_EMP_DISTRACTORS_MAX_END=${N_TRAIN_EMP_DISTRACTORS_MAX_END:-}
DISTRACTOR_CURRICULUM_UPDATES=${DISTRACTOR_CURRICULUM_UPDATES:-0}

# --- log-sigma anneal (0/0 disables) ---------------------------------------
LOG_STD_ANNEAL_START_UPDATE=${LOG_STD_ANNEAL_START_UPDATE:-0}
LOG_STD_ANNEAL_END_UPDATE=${LOG_STD_ANNEAL_END_UPDATE:-0}
LOG_STD_ANNEAL_TARGET=${LOG_STD_ANNEAL_TARGET:-}

# --- Rollout shape ---------------------------------------------------------
BATCH_ENVS=${BATCH_ENVS:-16}
STEPS_PER_ROLLOUT=${STEPS_PER_ROLLOUT:-400}
NUM_WORLDS=${NUM_WORLDS:-1}
ENVS_PER_WORLD=${ENVS_PER_WORLD:-20}
SEED=${SEED:-42}

# --- Eval ------------------------------------------------------------------
NUM_VAL_ENVS=${NUM_VAL_ENVS:-10}
N_VAL_TRIALS=${N_VAL_TRIALS:-32}
VAL_DISTRACTORS=${VAL_DISTRACTORS:-"0 5 10"}
EVAL_EVERY=${EVAL_EVERY:-50}

fi   # end fresh-run block

# --- Artifacts / logging (apply to both fresh and resumed runs) -------------
CKPT_EVERY=${CKPT_EVERY:-}              # empty = follow EVAL_EVERY
SAVE_DIR=${SAVE_DIR:-}                  # empty = $CLS_RUNS/agent_ckpts/navigate_<wandb name>
DEVICE=${DEVICE:-cuda}
USE_WANDB=${USE_WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-hopfield-nav-navigate}
EXTRA=${EXTRA:-}                        # appended last, so it beats everything above

cd /home/jackking/cls
source hopfield_nav/navigate_job.sh
