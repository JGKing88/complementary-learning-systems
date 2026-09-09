#!/usr/bin/env bash
# Is the coverage floor a coverage floor, or a Hebbian cross-talk floor?
#
# Sec 10.14 concluded "coverage buys the ability to hold MANY goals apart, not
# the ability to point at one", from dead-goal rates at K=20 of 0.08 / 0.25 /
# 0.42 down the ladder, and Sec 10.15 put the floor between 2.5% and 1.25%.
#
# But job 22343138 found that at 10% coverage the K=20 limit is largely a
# STORAGE RULE limit: swapping the Hebbian outer product for the projection rule
# takes reach at K=20 from 0.911 to 0.986, at no cost to basin, acc45 or |err|.
# If the same holds lower down, then what Sec 10.14 measured as a capacity
# ceiling is substantially the cross-talk the Hebbian rule propagates, and the
# 2.5% floor is not where coverage runs out -- it is where cross-talk wins.
#
# Both rules run again here rather than reusing the ladder's published numbers,
# because the whole value of the 10% run was having a control on the identical
# config; a cross-run comparison is what went wrong earlier in this campaign.
#
# Two rungs x two rules x two seeds. Gain is 100 for both rungs (from the ladder
# manifest), so no --encoder_gain override is needed.
#
#SBATCH --job-name=probe_projlow
#SBATCH --output=/home/jackking/.claude/jobs/d05f5770/tmp/projlow_%A_%a.out
#SBATCH --partition=ou_bcs_normal
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --array=0-7
set -euo pipefail
cd /orcd/home/002/jackking/cls/.claude/worktrees/encoder-hopfield-eval-spec

PY=/home/jackking/.conda/envs/cls/bin/python
S=/orcd/pool/003/jackking/cls_runs/sweeps
OUT=/home/jackking/.claude/jobs/d05f5770/tmp/probe_projlow

i=$SLURM_ARRAY_TASK_ID
rung=$(( i / 4 ))            # 0 = 2.5%, 1 = 1.25%
rule_i=$(( (i / 2) % 2 ))    # 0 = hebb, 1 = proj
seed=$(( 42 + i % 2 ))

if (( rung == 0 )); then
    GROUP="2.5%"
    CK=$(ls -d "$S"/w58_cov2.5/*_q_a1_seed=$seed)/encoder_final.pt
else
    GROUP="1.25%"
    CK=$(ls -d "$S"/w60_cov1.25/*_sm35x_a2_seed=$seed)/encoder_final.pt
fi

if (( rule_i == 0 )); then
    RULE=hebb
else
    RULE=proj
fi

"$PY" -m analysis.hopfield_probe.run \
    --ckpt "$CK" --label "$GROUP · $RULE · s$seed" \
    --storage_rule "$RULE" \
    --n_worlds 8 --n_envs_per_world 20 --k 1 3 5 10 20 \
    --steps 1 2 3 5 10 15 --env_size 20 --Npos 1716 \
    --n_alias 5000 --n_cont_samples 60000 --n_cont_annulus 20000 \
    --seed 0 --device cpu --fwhm_fallback 0.25 \
    --out "$OUT/t$i"

echo "DONE task $i  rung=$GROUP rule=$RULE seed=$seed  -> $OUT/t$i"
