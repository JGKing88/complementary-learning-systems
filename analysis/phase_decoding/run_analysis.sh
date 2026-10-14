#!/bin/bash -l
#SBATCH --job-name=pa-analysis
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --partition=ou_bcs_normal
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/results/phase_abstraction/logs/analysis_%x_%j.out
# Abstraction metrics for one run of configs/abstraction_v3.json.
#
#   ONLY=d0_A sbatch --time=04:00:00 --job-name=pa-an-d0_A analysis/phase_decoding/run_analysis.sh
set -euo pipefail
module load miniforge/24.3.0-0 2>/dev/null || true
source activate cls 2>/dev/null || true
cd "$SLURM_SUBMIT_DIR"
CONFIG="${CONFIG:-analysis/phase_decoding/configs/abstraction_v3.json}"
OUT="${OUT:-/orcd/pool/003/jackking/cls_runs/results/phase_abstraction/analysis_v3}"
python -m analysis.phase_decoding.run_analysis --config "$CONFIG" --out "$OUT" \
    --only "${ONLY:?}"
echo ANALYSIS_DONE
