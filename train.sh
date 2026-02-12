#!/bin/bash -l
#SBATCH --job-name=cls
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:a100:1
####SBATCH --gres=gpu:1
###SBATCH --gres=gpu:GEFORCERTX2080:2

#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jackking@mit.edu
#SBATCH --partition=pi_fiete
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

# python train_action_classifier.py \
#     --model_type mlp \
#     --vectorhash \
#     --input_type encoded_g \
#     --max_steps 1 4 \
#     --fwhm_ratio 0.25 \
#     --size 8 \
#     --num_train_envs 20 \
#     --num_val_new_envs 2 \
#     --Np 4000 \
#     --lambdas 11 12 13 \
#     --Npos 60 \
#     --observation_size 512 \
#     --thresh 0.5 \
#     --val_fraction 0.2 \
#     --dropout 0.2 \
#     --hidden_size 512 \
#     --num_model_layers 2 \
#     --batch_size 256 \
#     --n_epochs 20000 \
#     --lr 1e-6 \
#     --encoder_weights goated_cnn_encoder.pt \
#     --use_wandb \
#     --wandb_project cls_action_classifier \
#     --warmup_epochs 0 --scheduler cosine --scheduler_T_max 20000 \
    #--use_displacement
    # --use_displacement
        # --regenerate_envs_per_epoch
    # --shared_vectorhash \
    # --full_scaffold

# python train_action_classifier.py \
#     --model_type mlp \
#     --vectorhash \
#     --input_type encoded_g \
#     --max_steps 1 4 \
#     --fwhm_ratio 0.25 \
#     --lambdas 11 12 13 \
#     --Npos 300 \
#     --thresh 0.5 \
#     --val_fraction 0.2 \
#     --dropout 0.2 \
#     --hidden_size 512 \
#     --num_model_layers 2 \
#     --batch_size 2048 \
#     --n_epochs 5000 \
#     --lr 2e-5 \
#     --encoder_weights goated_cnn_encoder.pt \
#     --use_wandb \
#     --wandb_project cls_action_classifier \
#     --global_space \
#     --save_every 1000 \
#     --save_dir action_classifiers
    #--use_displacement

# python train_action_classifier.py \
#     --model_type mlp \
#     --vectorhash \
#     --input_type encoded_g \
#     --max_steps 1 4 \
#     --fwhm_ratio 0.25 \
#     --size 15 \
#     --lambdas 11 12 13 \
#     --Npos 300 \
#     --thresh 0.5 \
#     --val_fraction 0.2 \
#     --dropout 0.2 \
#     --hidden_size 64 \
#     --num_model_layers 2 \
#     --batch_size 128 \
#     --n_epochs 20000 \
#     --lr 5e-6 \
#     --encoder_weights separated_encoder.pt \
#     --use_wandb \
#     --wandb_project cls_action_classifier \
#     --warmup_epochs 0 --scheduler cosine --scheduler_T_max 20000 \
#     --global_space \
#     --use_displacement

# python train_action_classifier.py \
#     --vectorhash \
#     --input_type g_hot \
#     --fwhm_ratio 0.25 \
#     --size 10 \
#     --num_train_envs 10 \
#     --Np 3200 \
#     --lambdas 11 12 \
#     --thresh 0.3 \
#     --val_fraction 0.2 \
#     --dropout 0.2 \
#     --hidden_size 64 \
#     --num_model_layers 2 \
#     --batch_size 256 \
#     --n_epochs 10000 \
#     --lr 1e-3 \
#     --use_wandb \
#     --encoder_weights my_encoder.pt \
#     --wandb_project cls_action_classifier


# python train.py \
#     --num_train_worlds 1 \
#     --size 8 \
#     --encoder_dim 64 \
#     --num_encoder_layers 1 \
#     --model_class MLP \
#     --hidden_size 128 \
#     --num_model_layers 1 \
#     --speed 1 \
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
#     --train_method supervised \
#     --use_preconv_codebook \
#     --fwhm_ratio 0.25 \
#     --expert_actions \
#     --batch_episodes 64

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
