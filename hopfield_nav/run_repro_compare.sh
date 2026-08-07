#!/bin/bash -l
#SBATCH --job-name=hnav-repro-cmp
#SBATCH --time=0-02:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=pi_evelina9
#SBATCH --mem=100G
#SBATCH --output=/home/jackking/cls/hopfield_nav/logs/slurm_repro_cmp_%j.out

# Evaluate the reproduction against gentle-terrain-124 under ONE protocol, in
# ONE job, so nothing about the comparison is confounded by when or where it
# ran. Both checkpoints get identical flags.
#
# The reference run's own numbers are the control: if the original u380 does not
# come back at srD 0.994 / msD 22.9, this protocol is not the one that produced
# the target and no verdict from it means anything. See
# docs/EXPERIMENTS_SCHEDULE_REPRO.md.
#
#   NEW_CKPT=$CLS_RUNS/agent_ckpts/navigate_<run>/navigate_u380.pt \
#       sbatch hopfield_nav/run_repro_compare.sh

set -euo pipefail

OLD_CKPT=${OLD_CKPT:-/orcd/pool/003/jackking/cls_runs/agent_ckpts/phase_a_only_gentle-terrain-124/phase_a_u380.pt}
if [ -z "${NEW_CKPT:-}" ]; then
    echo "ERROR: NEW_CKPT is required" >&2
    exit 1
fi

# Matches the header of the reference log: 10 val envs, 32 trials/bucket,
# max_steps 400, distractor levels 0/5/10.
NUM_VAL_ENVS=${NUM_VAL_ENVS:-10}
NUM_TRIALS=${NUM_TRIALS:-32}
MAX_STEPS=${MAX_STEPS:-400}
DISTRACTORS=${DISTRACTORS:-"0 5 10"}

module load miniforge/24.3.0-0
module load cuda/13.0.1
source activate cls
unset CUDA_VISIBLE_DEVICES
cd /home/jackking/cls
source scripts/cls_env.sh

OUT=${OUT:-$CLS_RESULTS/repro_v35_compare_$SLURM_JOB_ID}
mkdir -p "$OUT"

for pair in "reference:$OLD_CKPT" "reproduction:$NEW_CKPT"; do
    label=${pair%%:*}; ckpt=${pair#*:}
    echo "=== $label  $ckpt ==="
    python -u -m hopfield_nav.eval_all \
        --ckpt "$ckpt" --device cuda \
        --num-val-envs "$NUM_VAL_ENVS" \
        --num_trials "$NUM_TRIALS" --max_steps "$MAX_STEPS" \
        --n_distractors $DISTRACTORS \
        --no-nav-stoch --skip-realistic --repeat-trials 0 \
        --output-json "$OUT/$label.json"
done

echo
echo "=== srD / msD, and the two averages that define the target ==="
python - "$OUT" <<'PY'
import json, sys, pathlib
out = pathlib.Path(sys.argv[1])
print(f"{'run':<14} {'n_dist':>7} {'srD':>7} {'msD':>7}")
for label in ("reference", "reproduction"):
    f = out / f"{label}.json"
    if not f.exists():
        print(f"{label:<14} (no json)"); continue
    nav = json.load(open(f))["nav_det"]
    srs, mss = [], []
    for d in sorted(nav, key=int):
        r = nav[d]
        print(f"{label:<14} {d:>7} {r['success_rate']:>7.3f} {r['mean_steps']:>7.1f}")
        srs.append(r["success_rate"]); mss.append(r["mean_steps"])
    print(f"{label:<14} {'MEAN':>7} {sum(srs)/len(srs):>7.3f} {sum(mss)/len(mss):>7.1f}")
print("\ntarget (reference u380, recorded 2026-05-18): srD 0.994  msD 22.9")
PY
echo "JSON in $OUT"
