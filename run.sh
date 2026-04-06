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
export WANDB_API_KEY=5aee75a09d43e7f6c9ec80e003687a8a3a820b08

# find the user name
USER_NAME=$(whoami)
unset CUDA_VISIBLE_DEVICES

MT_HOME="/home/$USER_NAME/cls"

# python notebooks/test_placefield.py
python sweep_cosine_width.py -o cosine_width_sweep.csv