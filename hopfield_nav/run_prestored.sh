#!/bin/bash -l
#SBATCH --job-name=hnav-noemb
#SBATCH --time=0-02:00:00
###SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jackking@mit.edu
#SBATCH --partition=pi_fiete
#SBATCH --mem=32G
#SBATCH --output=slurm_hnav_noemb_%j.out

module load miniforge/24.3.0-0
module load cuda/13.0.1

source activate cls
export WANDB_API_KEY=5aee75a09d43e7f6c9ec80e003687a8a3a820b08
unset CUDA_VISIBLE_DEVICES

cd /home/jackking/cls

python -m hopfield_nav.train \
    --encoder_checkpoint encoders/confused-sweep-160/encoder_final.pt \
    --encoder_gain 3.0 \
    --fwhm_ratio 0.25 \
    --size 8 \
    --observation_size 512 \
    --time_penalty 0.01 \
    --movement_mode discrete \
    --lambdas 11 12 13 \
    --Np 1600 \
    --Npos 200 \
    --hopfield_beta 3.0 \
    --hopfield_alpha 0.8 \
    --hopfield_steps 1 \
    --hopfield_init pre_stored \
    --no-agent_can_store \
    --hidden_size 128 \
    --num_rnn_layers 1 \
    --hopfield_mode discrete \
    --no-input_encoded_state \
    --num_worlds 1 \
    --envs_per_world 2 \
    --val_envs_per_world 1 \
    --batch_envs 16 \
    --steps_per_rollout 32 \
    --n_updates 500 \
    --lr 3e-4 \
    --eval_every 25 \
    --save_every 100 \
    --seed 42 \
    --device cpu
