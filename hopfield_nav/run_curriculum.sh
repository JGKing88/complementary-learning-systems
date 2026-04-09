#!/bin/bash -l
#SBATCH --job-name=hnav-curriculum
#SBATCH --time=0-04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jackking@mit.edu
#SBATCH --partition=pi_fiete
#SBATCH --mem=16G
#SBATCH --output=slurm_hnav_curriculum_%j.out

module load miniforge/24.3.0-0
module load cuda/13.0.1

source activate cls
export WANDB_API_KEY=5aee75a09d43e7f6c9ec80e003687a8a3a820b08
unset CUDA_VISIBLE_DEVICES

cd /home/jackking/cls

# ---------------------------------------------------------------
# Phase 1: Train pre-stored discrete baseline (to get checkpoint)
# ---------------------------------------------------------------
echo "=== PHASE 1: Pre-stored discrete baseline ==="
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
    --eval_every 50 \
    --save_every 500 \
    --save_dir checkpoints/prestored_discrete \
    --seed 42 \
    --device cpu

echo ""

# ---------------------------------------------------------------
# Phase 2: Curriculum fine-tune — discrete
# Load nav-competent agent, train with empty Hopfield + store
# Key changes: init_mode=empty, agent_can_store, longer rollouts,
#              lower LR for fine-tuning
# ---------------------------------------------------------------
echo "=== PHASE 2: Curriculum fine-tune (discrete) ==="
python -m hopfield_nav.train \
    --encoder_checkpoint encoders/confused-sweep-160/encoder_final.pt \
    --encoder_gain 3.0 \
    --load_checkpoint checkpoints/prestored_discrete/hopfield_nav_update500.pt \
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
    --hopfield_init empty \
    --agent_can_store \
    --hidden_size 128 \
    --num_rnn_layers 1 \
    --hopfield_mode discrete \
    --no-input_encoded_state \
    --num_worlds 1 \
    --envs_per_world 2 \
    --val_envs_per_world 1 \
    --batch_envs 16 \
    --steps_per_rollout 128 \
    --n_updates 1000 \
    --lr 1e-4 \
    --eval_every 50 \
    --save_every 500 \
    --save_dir checkpoints/curriculum_discrete \
    --seed 42 \
    --device cpu

echo ""

# ---------------------------------------------------------------
# Phase 3: Curriculum fine-tune — continuous
# Uses existing continuous checkpoint from prior pre-stored run
# ---------------------------------------------------------------
echo "=== PHASE 3: Curriculum fine-tune (continuous) ==="
python -m hopfield_nav.train \
    --encoder_checkpoint encoders/confused-sweep-160/encoder_final.pt \
    --encoder_gain 3.0 \
    --load_checkpoint checkpoints/hopfield_nav_update500.pt \
    --fwhm_ratio 0.25 \
    --size 8 \
    --observation_size 512 \
    --time_penalty 0.01 \
    --movement_mode continuous \
    --lambdas 11 12 13 \
    --Np 1600 \
    --Npos 200 \
    --hopfield_beta 3.0 \
    --hopfield_alpha 0.8 \
    --hopfield_steps 1 \
    --hopfield_init empty \
    --agent_can_store \
    --hidden_size 128 \
    --num_rnn_layers 1 \
    --hopfield_mode continuous \
    --no-input_encoded_state \
    --num_worlds 1 \
    --envs_per_world 2 \
    --val_envs_per_world 1 \
    --batch_envs 16 \
    --steps_per_rollout 128 \
    --n_updates 1000 \
    --lr 1e-4 \
    --eval_every 50 \
    --save_every 500 \
    --save_dir checkpoints/curriculum_continuous \
    --seed 42 \
    --device cpu
