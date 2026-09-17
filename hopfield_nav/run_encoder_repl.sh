#!/bin/bash -l
#SBATCH --job-name=encrepl
#SBATCH --time=0-03:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_preemptable
#SBATCH --mem=40G
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/logs/encoder_repl_%j.out

# The encoder package's own trainer, att0.5's configuration verbatim, on NENV
# patches of NPOS x NPOS (plan sec 6.4: the replication check for the
# encoder-vs-decode comparison). Output under agent_ckpts/<RUN_NAME>.

set -euo pipefail
REPO=${REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/nn-generalization-control}
NENV=${NENV:-25}
NPOS=${NPOS:-50}
EPOCHS=${EPOCHS:-1000}
SEED=${SEED:-43}
RUN_NAME=${RUN_NAME:-enc_repl_att05_n${NENV}_p${NPOS}_s${SEED}}

module load miniforge/24.3.0-0
module load cuda/13.0.1
source activate cls
unset CUDA_VISIBLE_DEVICES
cd "$REPO"
source scripts/cls_env.sh

PYTHONUNBUFFERED=1 python -m encoder_training.train \
  --nenv "$NENV" --npos "$NPOS" --radius 20 --per_env_radius_frac 0 \
  --attract_lambda 0.5 --repel_weight 1.0 --exclude_cross_env_pairs --rate_lambda 0.5 --rate_eps 1.0 \
  --out_dim 1024 --hidden_dim 256 --num_hidden_layers 4 --gain_start 1.0 --gain_end 100.0 \
  --lr 3e-4 --weight_decay 1e-4 --epochs "$EPOCHS" --batch_size 4096 --seed "$SEED" \
  --lazy_codes --eval_every 0 --no_unique_radius \
  --save_dir "$CLS_RUNS/agent_ckpts" --run_name "$RUN_NAME" ${EXTRA:-}
