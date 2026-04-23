#!/bin/bash -l
#SBATCH --job-name=et_exp
#SBATCH --time=1:30:00
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --partition=pi_evelina9
#SBATCH --output=/home/jackking/cls/encoder_training/scripts/logs/slurm-%j.out

module load miniforge/24.3.0-0
module load cuda/13.0.1
source activate cls
unset CUDA_VISIBLE_DEVICES

python -m encoder_training.experiments.capacity_scaling --wgp_rule pseudo
python -m encoder_training.experiments.capacity_scaling --wgp_rule hebbian