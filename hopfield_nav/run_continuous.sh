#!/bin/bash -l
#SBATCH --job-name=hnav-cont
#SBATCH --time=0-02:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jackking@mit.edu
#SBATCH --partition=mit_normal_gpu
#SBATCH --mem=64G
#SBATCH --output=/home/jackking/cls/hopfield_nav/logs/slurm_hnav_cont_%j.out

module load miniforge/24.3.0-0
module load cuda/13.0.1

source activate cls
# wandb auth comes from ~/.netrc (machine api.wandb.ai). Run `wandb login`
# once if it is missing; never paste an API key into a tracked script.
unset CUDA_VISIBLE_DEVICES

cd /home/jackking/cls

python -m hopfield_nav.train \
    --encoder_checkpoint encoders/run_20260422_185816/encoder_best.pt \
    --fwhm_ratio 0.25 \
    --size 8 \
    --observation_size 512 \
    --time_penalty 0.01 \
    --movement_mode continuous \
    --lambdas 11 12 13 \
    --Np 400 \
    --static-vectorhash \
    --hopfield_alpha 0.8 \
    --hopfield_steps 1 \
    --hopfield_init empty \
    --agent_can_store \
    --store_cost 0.0 \
    --hidden_size 128 \
    --num_rnn_layers 1 \
    --hopfield_mode continuous \
    --no-input_encoded_state \
    --num_worlds 1 \
    --envs_per_world 20 \
    --num_val_envs 10 \
    --batch_envs 16 \
    --steps_per_rollout 200 \
    --auto_nav_warmup 0 \
    --explore_steps 100 \
    --n_updates 400 \
    --lr 3e-4 \
    --eval_every 25 \
    --save_every 100 \
    --seed 42 \
    --use_wandb \
    --device cuda