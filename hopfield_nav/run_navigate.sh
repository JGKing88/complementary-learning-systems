#!/bin/bash -l
#SBATCH --job-name=hnav-navigate
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jackking@mit.edu
#SBATCH --partition=pi_evelina9
#SBATCH --mem=100G
#SBATCH --output=/home/jackking/cls/hopfield_nav/logs/slurm_navigate_%j.out

# The composed run: explore first, then interleave the two regimes while
# shifting the mix toward following, then a short pure-follow tail.
#
#   sbatch hopfield_nav/run_navigate.sh
#   SCHEDULE='explore:400 ; exploit:200' SEED=7 sbatch hopfield_nav/run_navigate.sh
#
# The default below is the shape the old --warmup_explore_only_updates +
# --interleave_empty_target flags used to express, with a pure-follow stage on
# the end that they could not express at all.
#
# See hopfield_nav/navigate_job.sh for every environment variable this accepts.

JOB_LABEL=navigate
SCHEDULE=${SCHEDULE:-'explore:200 ; interleave:800,empty_frac=1.0->0.5,anneal=50 ; exploit:100'}

cd /home/jackking/cls
source hopfield_nav/navigate_job.sh
