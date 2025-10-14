#!/bin/bash
#SBATCH --job-name=cls
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:a100:1
##SBATCH --gres=gpu:RTXA6000:1
###SBATCH --gres=gpu:GEFORCERTX2080:2

#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jackking@mit.edu
#SBATCH --partition=evlab
#SBATCH --mem=50G

#SBATCH --output=/om2/user/jackking/cls/slurm_outputs/output_%j.txt

source ~/.bashrc

###source /om2/vast/evlab/jackking/.bashrc

module load openmind8/cuda/11.7
# find the user name
USER_NAME=$(whoami)
unset CUDA_VISIBLE_DEVICES

MT_HOME="/om2/user/${USER_NAME}/cls"

conda activate cls
echo $(which python)

export WANDB_API_KEY='5aee75a09d43e7f6c9ec80e003687a8a3a820b08'

python train.py \
    --size 10 \
    --speed 1 \
    --n_epochs 1000 \
    --num_envs 32 \
    --num_val_envs 4 \
    --batch_episodes 2048 \
    --val_batch_episodes 128 \
    --vectorhash \
    --input_type g_idx \
    --use_wandb \
    --lr 1e-3 \
    --max_envs_per_epoch 8 \
    --input_addendum goal \
    --train_method supervised \
    --ppo_clip 0.2 \
    --ppo_vf_coef 0.5 \
    --ppo_ent_coef 0.04 \
    --ppo_epochs 4 \
    --encoder_dim 128 \
    --num_encoder_layers 2 \
    --num_rnn_layers 1 \
    --use_preconv_codebook
