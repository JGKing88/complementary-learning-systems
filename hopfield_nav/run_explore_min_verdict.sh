#!/bin/bash -l
#SBATCH --job-name=hxm-verdict
#SBATCH --time=0-03:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal_gpu
#SBATCH --mem=100G
#SBATCH --output=/home/jackking/cls/hopfield_nav/logs/slurm_hxm_verdict_%j.out

# The verdict pass for the explore-min wave: every finished checkpoint scored
# under the SAME protocol v35's 0.53 / 0.51 / 0.48 was measured with, in one
# job, so nothing about the comparison is confounded by when or where it ran.
#
# In-training evals used 4 val envs x 16 trials at n_dist {0,10} -- deliberately
# cheap, for monitoring. Those numbers do not settle anything; this does.
#
#   sbatch hopfield_nav/run_explore_min_verdict.sh
#
# Note the cost asymmetry: nav_det and disc run on a policy that was never
# trained to reach or store a goal, so they find nothing and burn the full
# 400-step budget on every trial. They are kept anyway -- "an explore
# specialist cannot navigate" is a claim worth having a number for rather than
# assuming, and it is the only measurement of what the specialization cost.

set -euo pipefail

CKPT_ROOT=${CKPT_ROOT:-/orcd/pool/003/jackking/cls_runs/agent_ckpts}
UPDATE=${UPDATE:-300}
RUNS=${RUNS:-"s1:navigate_explore_min_s1_19847282 s2:navigate_explore_min_s2_19847283 d3:navigate_explore_min_d3_19847566"}

module load miniforge/24.3.0-0
module load cuda/13.0.1
source activate cls
unset CUDA_VISIBLE_DEVICES
cd "${REPO_DIR:-/home/jackking/cls}"   # see run_explore_min.sh; sbatch exports it
source scripts/cls_env.sh

OUT=${OUT:-$CLS_RESULTS/explore_min_verdict_$SLURM_JOB_ID}
mkdir -p "$OUT"

for pair in $RUNS; do
    label=${pair%%:*}; dir=${pair#*:}
    ckpt="$CKPT_ROOT/$dir/navigate_u${UPDATE}.pt"
    if [ ! -f "$ckpt" ]; then
        echo "SKIP $label: no $ckpt" >&2
        continue
    fi
    echo "=== $label  $ckpt ==="
    python -u -m hopfield_nav.eval_all \
        --ckpt "$ckpt" --device cuda \
        --num-val-envs 10 --num_trials 32 --max_steps 400 \
        --n_distractors 0 5 10 \
        --no-nav-stoch --skip-realistic --repeat-trials 0 \
        --output-json "$OUT/$label.json"
done

echo
echo "=== exploration under the v35 protocol (10 envs x 32 trials, 400 steps) ==="
python - "$OUT" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
hdr = f"{'run':<6} {'n_dist':>6} {'cov':>7} {'union':>7} {'redund':>7} {'findR':>7}"
print(hdr); print("-" * len(hdr))
for f in sorted(out.glob("*.json")):
    d = json.load(open(f))
    expl = d.get("expl") or d.get("exploration") or {}
    covs = []
    for k in sorted(expl, key=int):
        r = expl[k]
        covs.append(r["mean_coverage"])
        print(f"{f.stem:<6} {k:>6} {r['mean_coverage']:>7.3f} "
              f"{r['union_coverage']:>7.3f} {r['redundancy']:>7.4f} "
              f"{r['goal_find_rate']:>7.3f}")
    if covs:
        print(f"{f.stem:<6} {'MEAN':>6} {sum(covs)/len(covs):>7.3f}")
        print(f"{f.stem:<6} {'0->10':>6} {covs[0]-covs[-1]:>7.3f}   "
              f"(distractor gap; v35 was 0.050)")
    print()
print("v35 reference, same protocol: cov 0.53 / 0.51 / 0.48  (mean 0.507)")
PY
echo "JSON in $OUT"
