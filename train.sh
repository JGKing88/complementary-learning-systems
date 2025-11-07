#!/bin/bash -l
#SBATCH --job-name=cls
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:a100:1
##SBATCH --gres=gpu:RTXA6000:1
###SBATCH --gres=gpu:GEFORCERTX2080:2

#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jackking@mit.edu
##SBATCH --partition=evlab
#SBATCH --mem=100G

#SBATCH --output=/om2/user/jackking/cls/slurm_outputs/output_%j.txt

module load openmind8/gcc/12.2.0
module load openmind8/cuda/12.4
source /om2/user/jackking/miniconda3/etc/profile.d/conda.sh

# find the user name
USER_NAME=$(whoami)
unset CUDA_VISIBLE_DEVICES

MT_HOME="/om2/user/${USER_NAME}/cls"

conda activate cls
echo $(which python)

export WANDB_API_KEY='5aee75a09d43e7f6c9ec80e003687a8a3a820b08'

python notebooks/testing_dist_encoder.py 

# python train_action_classifier.py \
#     --size 40 \
#     --speed 1 \
#     --observation_size 128 \
#     --seed 0 \
#     --num_samples 3000 \
#     --val_fraction 0.1 \
#     --hidden_size 256 \
#     --num_model_layers 2 \
#     --batch_size 256 \
#     --n_epochs 2000 \
#     --lr 1e-3 \
#     --vectorhash \
#     --Np 1600 \
#     --lambdas 11 12 \
#     --input_type g_idx \
#     --use_wandb \
#     --wandb_project cls_action_classifier

# python train.py \
#     --size 8 \
#     --encoder_dim 128 \
#     --num_encoder_layers 0 \
#     --model_class MLP \
#     --hidden_size 256 \
#     --num_model_layers 2 \
#     --speed 1 \
#     --n_epochs 3000 \
#     --num_envs 8 \
#     --num_val_envs 8 \
#     --batch_episodes 512 \
#     --val_batch_episodes 128 \
#     --vectorhash \
#     --lambdas 11 12 13 \
#     --input_type g_hot \
#     --use_wandb \
#     --lr 1e-3 \
#     --max_envs_per_epoch 8 \
#     --input_addendum next_best \
#     --train_method supervised \
#     --ppo_clip 0.2 \
#     --ppo_vf_coef 0.5 \
#     --ppo_ent_coef 0.04 \
#     --ppo_epochs 4 \
#     --use_preconv_codebook
