#!/bin/bash -l
#SBATCH --job-name=placefield
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jackking@mit.edu
#SBATCH --partition=pi_evelina9
###SBATCH --partition=mit_normal
#SBATCH --mem=300G

module load miniforge/24.3.0-0
module load cuda/13.0.1

source activate cls
# wandb auth comes from ~/.netrc (machine api.wandb.ai). Run `wandb login`
# once if it is missing; never paste an API key into a tracked script.

# find the user name
USER_NAME=$(whoami)
unset CUDA_VISIBLE_DEVICES

MT_HOME="/home/$USER_NAME/cls"

# (notebooks/ was archived to $CLS_RUNS/archive/ in phase 6)
python -m encoder_training.sweep_cosine_width -o cosine_width_sweep.csv