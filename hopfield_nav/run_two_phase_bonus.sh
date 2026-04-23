#!/bin/bash -l
#SBATCH --job-name=hnav-2ph-bonus
#SBATCH --time=0-06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jackking@mit.edu
#SBATCH --partition=pi_fiete
#SBATCH --mem=16G
#SBATCH --output=slurm_hnav_2ph_bonus_%j.out

module load miniforge/24.3.0-0
module load cuda/13.0.1

source activate cls
export WANDB_API_KEY=5aee75a09d43e7f6c9ec80e003687a8a3a820b08
unset CUDA_VISIBLE_DEVICES

cd /home/jackking/cls

# Two-phase + auxiliary store reward
# store_bonus: direct reward for storing at goal
# store_cost: penalty for storing elsewhere
# Combined: agent gets +bonus at goal, -cost everywhere else

# Sweep: bonus only, and bonus+cost combo
for BONUS in 0.5 1.0; do
    for COST in 0.0 0.1; do
        echo "=== explore=64, bonus=${BONUS}, cost=${COST} ==="
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
            --store_cost ${COST} \
            --store_bonus ${BONUS} \
            --hidden_size 128 \
            --num_rnn_layers 1 \
            --hopfield_mode continuous \
            --no-input_encoded_state \
            --num_worlds 1 \
            --envs_per_world 2 \
            --num_val_envs 1 \
            --batch_envs 16 \
            --steps_per_rollout 128 \
            --explore_steps 64 \
            --n_updates 1000 \
            --lr 1e-4 \
            --eval_every 50 \
            --save_every 500 \
            --save_dir checkpoints/twophase_bonus${BONUS}_cost${COST} \
            --seed 42 \
            --device cpu
        echo ""
    done
done
