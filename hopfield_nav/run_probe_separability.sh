#!/bin/bash -l
#SBATCH --job-name=hnav-probe-sep
#SBATCH --time=0-01:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal_gpu
#SBATCH --mem=150G
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/logs/probe_sep_%j.out

# Measure whether "the recalled pattern belongs to my env" is readable off the
# policy's own input channels. See hopfield_nav/diagnostics/hopfield_separability.py.
#
#   sbatch hopfield_nav/run_probe_separability.sh

REPO=${REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/navigate-explore-exploit}
ENCODER=${ENCODER:-/orcd/pool/003/jackking/cls_runs/sweeps/ur_loss2_repel_low/029_repel_weight=2_per_env_radius_frac=0.1_seed=44/encoder_best.pt}
OUT=${OUT:-/orcd/pool/003/jackking/cls_runs/results/hopfield_separability.json}

module load miniforge/24.3.0-0
module load cuda/13.0.1
source activate cls
unset CUDA_VISIBLE_DEVICES

cd "$REPO"
source scripts/cls_env.sh

python -u -m hopfield_nav.diagnostics.hopfield_separability \
    --encoder "$ENCODER" \
    --size 20 --observation_size 60 --wall_resolution 4 --goal_radius 1.0 \
    --lambdas 11 12 13 --Np 400 --fwhm_ratio 0.25 \
    --steps 1 2 3 --n_envs 4 --n_dist 10 --seed 0 \
    --device cuda --output_json "$OUT"
