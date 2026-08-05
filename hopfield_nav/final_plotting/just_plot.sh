#!/bin/bash -l
#SBATCH --job-name=hnav-plot
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jackking@mit.edu
#SBATCH --partition=mit_normal
#SBATCH --mem=10G
#SBATCH --output=/home/jackking/cls/hopfield_nav/logs/slurm_plot_sequential_%j.out
set -euo pipefail

# === edit these =============================================================
RUN_NAME="lively-surf-104"

# Plotting
SMOOTH=10
SHOW_STD=false
# ============================================================================

module load miniforge/24.3.0-0
module load cuda/13.0.1
source activate cls
unset CUDA_VISIBLE_DEVICES

cd /home/jackking/cls

source scripts/cls_env.sh
HIST="$CLS_HISTORIES/${RUN_NAME}.json"
PLOT_PREFIX="$CLS_FIGURES/model_comparison/${RUN_NAME}"

python -u -m hopfield_nav.final_plotting.plotting \
    --history "$HIST" \
    --out_prefix "$PLOT_PREFIX" \
    --smooth "$SMOOTH" \
    --show_std "$SHOW_STD"
