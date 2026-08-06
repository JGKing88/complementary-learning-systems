#!/bin/bash -l
#SBATCH --job-name=hnav-explore
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jackking@mit.edu
#SBATCH --partition=pi_fiete
#SBATCH --mem=100G
#SBATCH --output=/home/jackking/cls/hopfield_nav/logs/slurm_explore_%j.out

# Pure exploration: every env starts with nothing useful in its Hopfield, so
# the only thing paying is coverage. No goal-following gradient at all.
#
#   sbatch hopfield_nav/run_explore.sh
#   SCHEDULE='explore:1200' SEED=7 sbatch hopfield_nav/run_explore.sh
#
# See hopfield_nav/navigate_job.sh for every environment variable this accepts.

JOB_LABEL=explore
SCHEDULE=${SCHEDULE:-'explore:600'}
NOVELTY_REWARD=${NOVELTY_REWARD:-0.0}
SIZE=${SIZE:-20}
ENVS_PER_WORLD=${ENVS_PER_WORLD:-1}
STEPS_PER_ROLLOUT=${STEPS_PER_ROLLOUT:-100}
BATCH_ENVS=${BATCH_ENVS:-1}
EVAL_EVERY=${EVAL_EVERY:-100}
CKPT_EVERY=${CKPT_EVERY:-100}
VAL_DISTRACTORS=${VAL_DISTRACTORS:-"0 5 10"}

cd /home/jackking/cls
source hopfield_nav/navigate_job.sh
