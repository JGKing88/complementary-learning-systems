#!/bin/bash -l
#SBATCH --job-name=hnav-pretrain-baseline-rnn
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jackking@mit.edu
#SBATCH --partition=mit_normal_gpu
#SBATCH --mem=80G
#SBATCH --output=/home/jackking/cls/hopfield_nav/logs/slurm_pretrain_baseline_rnn_%j.out

module load miniforge/24.3.0-0
module load cuda/13.0.1
source activate cls
# wandb auth comes from ~/.netrc (machine api.wandb.ai). Run `wandb login`
# once if it is missing; never paste an API key into a tracked script.
unset CUDA_VISIBLE_DEVICES

cd /home/jackking/cls

python -m hopfield_nav.train_rnn --mode mixed --n_envs 32 --n_updates 1000 \
--save_dir checkpoint_rnn/pretrain_20x20_w_gridstate --batch_envs 32 --steps_per_rollout 100 \
--size 20 --observation_size 60 --movement_mode continuous --hidden_size 128 --num_rnn_layers 1 \
--input_grid_state \
--lambdas 11 12 13 \