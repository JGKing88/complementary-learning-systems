#!/bin/bash -l
#SBATCH --job-name=enc-sweep
#SBATCH --time=2-00:00:00
###SBATCH --gres=gpu:a100:1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jackking@mit.edu
#SBATCH --partition=pi_fiete
#SBATCH --mem=100G

module load miniforge/24.3.0-0
module load cuda/13.0.1

source activate cls
export WANDB_API_KEY=5aee75a09d43e7f6c9ec80e003687a8a3a820b08

unset CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1

# Sweep ID is passed as first argument: sbatch sweep.sh <SWEEP_ID>
SWEEP_ID=${1:?Usage: sbatch sweep.sh <SWEEP_ID>}

wandb agent "generalization-bounds/dist-encoder/$SWEEP_ID"


## Example of running a sweep
# WANDB_API_KEY=5aee75a09d43e7f6c9ec80e003687a8a3a820b08 wandb sweep sweep_encoder.yaml --project dist-encoder                                                                                                                                    
# for i in $(seq 6); do sbatch sweep.sh <SWEEP_ID>; done
# current CNN sweep: xzd090lc

# WANDB_API_KEY=5aee75a09d43e7f6c9ec80e003687a8a3a820b08 wandb sweep sweep_encoder_mlp.yaml --project dist-encoder
# current MLP sweep: 7nclsvfg

# WANDB_API_KEY=5aee75a09d43e7f6c9ec80e003687a8a3a820b08 wandb sweep sweep_encoder_rhc.yaml --project dist-encoder
# current RHC sweep: tg741fme