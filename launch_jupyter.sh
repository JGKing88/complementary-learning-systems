#!/bin/bash -l
#SBATCH -J jupyter
#SBATCH --time=0-12:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
###SBATCH --cpus-per-task=8
###SBATCH --gres=gpu:a100:1
#SBATCH --gres=gpu:1
###SBATCH --gres=gpu:GEFORCERTX2080:1
#SBATCH --mem 200G
#SBATCH --partition=pi_evelina9
###BATCH --partition=mit_normal
#SBATCH -o jupyter.out

module load miniforge/24.3.0-0
module load cuda/13.0.1

source activate cls

unset XDG_RUNTIME_DIR

PORT=8091

jupyter lab --ip=0.0.0.0 --port=${PORT} --no-browser --NotebookApp.allow_origin='*' --NotebookApp.port_retries=0