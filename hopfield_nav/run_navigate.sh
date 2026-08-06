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
SCHEDULE=${SCHEDULE:-'explore:200,eps=0.4 ; interleave:800,empty_frac=1.0->0.5,anneal=50,eps=0.2'}

# INPUTS
INPUT_PREV_REWARD=${INPUT_PREV_REWARD:-True}
INPUT_PREV_ACTION=${INPUT_PREV_ACTION:-True}
INPUT_HOPFIELD_RAW=${INPUT_HOPFIELD_RAW:-True}
INPUT_HOPFIELD_MULTISTEP=${INPUT_HOPFIELD_MULTISTEP:-"2 3"}
INPUT_SENSORY=${INPUT_SENSORY:-True}
INPUT_ENCODED_STATE=${INPUT_ENCODED_STATE:-False}
INPUT_HOPFIELD_SIGNAL=${INPUT_HOPFIELD_SIGNAL:-True}
INPUT_GOAL_IN_MEMORY=${INPUT_GOAL_IN_MEMORY:-False}

# ENV
GOAL_RADIUS=${GOAL_RADIUS:-1.0}
N_TRAIN_DISTRACTORS_MIN=${N_TRAIN_DISTRACTORS_MIN:-0}
N_TRAIN_DISTRACTORS_MAX=${N_TRAIN_DISTRACTORS_MAX:-5}
N_TRAIN_EMP_DISTRACTORS_MIN=${N_TRAIN_EMP_DISTRACTORS_MIN:-0}
N_TRAIN_EMP_DISTRACTORS_MAX=${N_TRAIN_EMP_DISTRACTORS_MAX:-5}
VAL_DISTRACTORS=${VAL_DISTRACTORS:-"0 5 10"}

# SAMPLING
BATCH_ENVS=${BATCH_ENVS:-16}
STEPS_PER_ROLLOUT=${STEPS_PER_ROLLOUT:-400}
NUM_WORLDS=${NUM_WORLDS:-1}
ENVS_PER_WORLD=${ENVS_PER_WORLD:-20}
NUM_VAL_ENVS=${NUM_VAL_ENVS:-10}
N_VAL_TRIALS=${N_VAL_TRIALS:-32}

# REWARD SHAPING
NOVELTY_REWARD=${NOVELTY_REWARD:-0.1}
WALL_PENALTY=${WALL_PENALTY:--0.1}
GOAL_REWARD=${GOAL_REWARD:-5.0}

cd /home/jackking/cls
source hopfield_nav/navigate_job.sh