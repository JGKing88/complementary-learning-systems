#!/bin/bash -l
#SBATCH --job-name=cls
#SBATCH --time=0-06:00:00
###SBATCH --gres=gpu:a100:1
#SBATCH --gres=gpu:1
###SBATCH --gres=gpu:GEFORCERTX2080:2

#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jackking@mit.edu
###SBATCH --partition=pi_evelina9
#SBATCH --partition=mit_normal_gpu
#SBATCH --mem=100G

module load miniforge/24.3.0-0
module load cuda/13.0.1

source activate cls
export WANDB_API_KEY=5aee75a09d43e7f6c9ec80e003687a8a3a820b08

# find the user name
USER_NAME=$(whoami)
unset CUDA_VISIBLE_DEVICES

MT_HOME="/home/$USER_NAME/cls"

python notebooks/train_dist_encoder.py

# prrint examples

# python train.py \
#     --num_train_worlds 1 \
#     --size 8 \
#     --encoder_dim 64 \
#     --num_encoder_layers 1 \
#     --model_class MLP \
#     --hidden_size 128 \
#     --num_model_layers 1 \
#     --speed 1 \
#     --n_epochs 5 \
#     --num_envs 2 \
#     --num_val_envs 1 \
#     --vectorhash \
#     --input_type hopfield_onehot \
#     --encoder_weights goated_cnn_encoder.pt \
#     --lr 1e-3 \
#     --max_envs_per_epoch 8 \
#     --Np 1600 \
#     --lambdas 11 12 13 \
#     --input_addendum hopfield \
#     --train_method supervised \
#     --use_preconv_codebook \

# python train.py \
#     --num_train_worlds 8 \
#     --size 8 \
#     --encoder_dim 128 \
#     --num_encoder_layers 1 \
#     --model_class MLP \
#     --hidden_size 256 \
#     --num_model_layers 2 \
#     --speed 1 \
#     --n_epochs 3000 \
#     --num_envs 3 \
#     --num_val_envs 4 \
#     --vectorhash \
#     --input_type encoded_g \
#     --use_wandb \
#     --wandb_project cls \
#     --lr 1e-3 \
#     --max_envs_per_epoch 8 \
#     --Np 1600 \
#     --lambdas 11 12 \
#     --input_addendum next_best \
#     --train_method supervised \
#     --use_preconv_codebook \
#     --encoder_weights my_encoder.pt

# python train.py \
#     --num_train_worlds 8 \
#     --size 8 \
#     --encoder_dim 128 \
#     --num_encoder_layers 1 \
#     --model_class MLP \
#     --hidden_size 256 \
#     --num_model_layers 2 \
#     --speed 1 \
#     --n_epochs 3000 \
#     --num_envs 3 \
#     --num_val_envs 4 \
#     --vectorhash \
#     --input_type encoded_g \
#     --use_wandb \
#     --wandb_project cls \
#     --lr 1e-3 \
#     --max_envs_per_epoch 8 \
#     --Np 1600 \
#     --lambdas 11 12 \
#     --input_addendum hopfield \
#     --train_method supervised \
#     --use_preconv_codebook \
#     --encoder_weights my_encoder.pt

# python train.py \
#     --num_train_worlds 1 \
#     --size 8 \
#     --encoder_dim 128 \
#     --num_encoder_layers 1 \
#     --model_class MLP \
#     --hidden_size 256 \
#     --num_model_layers 2 \
#     --speed 1 \
#     --n_epochs 3000 \
#     --num_envs 3 \
#     --num_val_envs 4 \
#     --vectorhash \
#     --input_type encoded_g \
#     --use_wandb \
#     --wandb_project cls \
#     --lr 1e-3 \
#     --max_envs_per_epoch 8 \
#     --Np 1600 \
#     --lambdas 11 12 \
#     --input_addendum next_best \
#     --train_method supervised \
#     --use_preconv_codebook \
#     --encoder_weights my_encoder.pt

# python train.py \
#     --num_train_worlds 1 \
#     --size 8 \
#     --encoder_dim 128 \
#     --num_encoder_layers 1 \
#     --model_class MLP \
#     --hidden_size 256 \
#     --num_model_layers 2 \
#     --speed 1 \
#     --n_epochs 3000 \
#     --num_envs 2 \
#     --num_val_envs 4 \
#     --vectorhash \
#     --input_type encoded_g \
#     --use_wandb \
#     --wandb_project cls \
#     --lr 1e-3 \
#     --max_envs_per_epoch 8 \
#     --Np 1600 \
#     --lambdas 11 12 \
#     --input_addendum hopfield \
#     --train_method supervised \
#     --use_preconv_codebook \
#     --encoder_weights my_encoder.pt

# python train.py \
#     --size 8 \
#     --n_epochs 3000 \
#     --num_envs 2 \
#     --num_val_envs 1 \
#     --vectorhash \
#     --input_type encoded_g \
#     --encoder_weights my_encoder.pt \
#     --use_wandb \
#     --wandb_project cls \
#     --lr 1e-3 \
#     --max_envs_per_epoch 8 \
#     --Np 1600 \
#     --lambdas 11 12 \
#     --input_addendum hopfield \
#     --hopfield_alpha 0.9 \
#     --hopfield_steps 1 \
#     --train_method supervised \
#     --use_preconv_codebook \

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
