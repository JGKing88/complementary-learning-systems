#!/bin/bash -l
#SBATCH --job-name=hnav-sweep
#SBATCH --time=0-08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jackking@mit.edu
#SBATCH --partition=pi_fiete
#SBATCH --mem=16G
#SBATCH --output=slurm_hnav_sweep_%j.out

module load miniforge/24.3.0-0
module load cuda/13.0.1

source activate cls
unset CUDA_VISIBLE_DEVICES

cd /home/jackking/cls

# ==========================================================================
# Store-learning sweep — all continuous, curriculum from pre-stored checkpoint
#
# New in this sweep:
#   - All 3 evals wired up (nav, goal_discovery, exploration)
#   - store_bonus: direct reward for storing at goal
#   - Two-phase: explore_steps limits when stores can happen
#
# Axes:
#   A. Structure:  single-phase vs two-phase (explore=64)
#   B. Bonus:      0.0, 0.5, 1.0
#   C. Cost:       0.0, 0.1
#
# 7 configs total, ~15 min each = ~2 hours
# ==========================================================================

COMMON="--encoder_checkpoint encoders/confused-sweep-160/encoder_final.pt \
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
    --num_val_envs 1 \
    --batch_envs 16 \
    --steps_per_rollout 128 \
    --n_updates 1000 \
    --lr 1e-4 \
    --eval_every 50 \
    --save_every 0 \
    --seed 42 \
    --device cpu"

# ------------------------------------------------------------------
# 1. Pre-stored baseline (no store learning, just get new eval metrics)
# ------------------------------------------------------------------
echo "=== 1: pre_stored baseline ==="
python -m hopfield_nav.train \
    --encoder_checkpoint encoders/confused-sweep-160/encoder_final.pt \
    --encoder_gain 3.0 \
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
    --hopfield_init pre_stored \
    --no-agent_can_store \
    --hidden_size 128 \
    --num_rnn_layers 1 \
    --hopfield_mode continuous \
    --no-input_encoded_state \
    --num_worlds 1 \
    --envs_per_world 2 \
    --num_val_envs 1 \
    --batch_envs 16 \
    --steps_per_rollout 32 \
    --n_updates 200 \
    --lr 3e-4 \
    --eval_every 50 \
    --save_every 0 \
    --seed 42 \
    --device cpu
echo ""

# ------------------------------------------------------------------
# 2. Two-phase, bonus only (can the bonus signal teach store timing?)
# ------------------------------------------------------------------
echo "=== 2: two-phase, bonus=0.5, cost=0.0 ==="
python -m hopfield_nav.train $COMMON \
    --explore_steps 64 \
    --store_bonus 0.5 \
    --store_cost 0.0 \
    --save_dir checkpoints/sweep_2ph_b0.5_c0.0
echo ""

echo "=== 3: two-phase, bonus=1.0, cost=0.0 ==="
python -m hopfield_nav.train $COMMON \
    --explore_steps 64 \
    --store_bonus 1.0 \
    --store_cost 0.0 \
    --save_dir checkpoints/sweep_2ph_b1.0_c0.0
echo ""

# ------------------------------------------------------------------
# 3. Two-phase, bonus + cost (carrot + stick)
# ------------------------------------------------------------------
echo "=== 4: two-phase, bonus=0.5, cost=0.1 ==="
python -m hopfield_nav.train $COMMON \
    --explore_steps 64 \
    --store_bonus 0.5 \
    --store_cost 0.1 \
    --save_dir checkpoints/sweep_2ph_b0.5_c0.1
echo ""

echo "=== 5: two-phase, bonus=1.0, cost=0.1 ==="
python -m hopfield_nav.train $COMMON \
    --explore_steps 64 \
    --store_bonus 1.0 \
    --store_cost 0.1 \
    --save_dir checkpoints/sweep_2ph_b1.0_c0.1
echo ""

# ------------------------------------------------------------------
# 4. Single-phase with bonus (does bonus work without two-phase?)
# ------------------------------------------------------------------
echo "=== 6: single-phase, bonus=1.0, cost=0.0 ==="
python -m hopfield_nav.train $COMMON \
    --store_bonus 1.0 \
    --store_cost 0.0 \
    --save_dir checkpoints/sweep_1ph_b1.0_c0.0
echo ""

echo "=== 7: single-phase, bonus=1.0, cost=0.1 ==="
python -m hopfield_nav.train $COMMON \
    --store_bonus 1.0 \
    --store_cost 0.1 \
    --save_dir checkpoints/sweep_1ph_b1.0_c0.1
echo ""
