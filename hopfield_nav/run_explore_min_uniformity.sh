#!/bin/bash -l
#SBATCH --job-name=hxm-unif
#SBATCH --time=0-01:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal_gpu
#SBATCH --mem=100G
#SBATCH --output=/home/jackking/cls/hopfield_nav/logs/slurm_hxm_unif_%j.out

# Spatial breakdown of the coverage the verdict pass scored.
#
# mean_coverage counts cells without locating them, and this lineage has a
# known failure -- the v36 perimeter-orbit basin -- where a policy scores
# respectably by circling the edge. Four checkpoints spanning the wave's whole
# range are run so the question "is low coverage just perimeter-hugging?" gets
# an answer rather than an assumption.

set -euo pipefail

CKPT_ROOT=${CKPT_ROOT:-/orcd/pool/003/jackking/cls_runs/agent_ckpts}
RUNS=${RUNS:-"e16:navigate_explore_min_e16_20110837:1000 \
e8:navigate_explore_min_e8_20110835:1000 \
c2s42:navigate_explore_min_c2s42_20110699:1000 \
e2s42:navigate_explore_min_e2s42_20110698:1000"}

module load miniforge/24.3.0-0
module load cuda/13.0.1
source activate cls
unset CUDA_VISIBLE_DEVICES
cd "${REPO_DIR:-/home/jackking/cls}"
echo "    repo: $PWD @ $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
source scripts/cls_env.sh

for spec in $RUNS; do
    label=${spec%%:*}; rest=${spec#*:}
    dir=${rest%%:*}; upd=${rest##*:}
    ckpt="$CKPT_ROOT/$dir/navigate_u${upd}.pt"
    if [ ! -f "$ckpt" ]; then
        echo "SKIP $label: no $ckpt" >&2
        continue
    fi
    python -u -m hopfield_nav.explore_min_uniformity \
        --ckpt "$ckpt" --device cuda --label "$label" \
        --num-val-envs 10 --num_trials 32 --max_steps 400 \
        --n_distractors 0 10
done
