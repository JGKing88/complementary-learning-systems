#!/bin/bash -l
#SBATCH -J jupyter
#SBATCH --time=1-00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
###SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:1
###SBATCH --gres=gpu:1
###SBATCH --gres=gpu:GEFORCERTX2080:1
#SBATCH --mem 100G
###SBATCH --partition=evlab
#SBATCH -o jupyter.out

source ~/.bashrc

module load openmind8/cuda/12.4
conda activate cls

unset XDG_RUNTIME_DIR

PORT=8091

jupyter lab --ip=0.0.0.0 --port=${PORT} --no-browser --NotebookApp.allow_origin='*' --NotebookApp.port_retries=0
