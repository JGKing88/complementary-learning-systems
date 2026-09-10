#!/usr/bin/env bash
# The full suite on the production encoder, Hebbian storage against the
# projection rule.
#
# Sec 7.1: an exact fixed point does not require a binary code, it requires a
# recurrence whose fixed set contains the code. `proj` sets W to the orthogonal
# projector onto span(Z), so `W z_i = z_i` exactly -- cos_self 1.0000 at s=1 and
# s=15 against Hebbian's 0.998 and 0.845 -- while keeping every property that
# made the Hebbian rule attractive: one-shot, online (`graded_fixedpoint_check.py
# --incremental`), D^2 synapses, no stored patterns.
#
# What is untested and this run measures: whether the exact fixed point costs
# anything. `proj` is idempotent, so it has NO error correction -- one step
# projects the cue onto span(Z) and further steps do nothing. That projection is
# the least-squares blend of the stored patterns, not the nearest one. It could
# beat Hebbian (it deconvolves the pattern overlaps rather than propagating
# them) or lose to it (no contraction toward a memory). The basin, which asks
# for an argmax over 12,853 cells rather than over K patterns, is where that
# shows.
#
# Both arms run here rather than comparing against the published production
# numbers, because a control on the identical config is worth more than a
# cross-run comparison -- and the `hebb` arm doubles as a check that this
# harness still reproduces reach 0.987 / basin 27.0 / acc45 0.995.
#
# The discriminating column is K=20, not K=5. At K=5 on the 10% encoder
# production already has reach 0.987, basin 27, dead goals 0.00, and there is
# nothing for a better storage rule to fix. Cross-talk between stored patterns
# only bites at K=20 (dead goals 0.08).
#
#SBATCH --job-name=probe_proj
#SBATCH --output=/home/jackking/.claude/jobs/d05f5770/tmp/proj_%A_%a.out
#SBATCH --partition=ou_bcs_normal
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --array=0-3
set -euo pipefail
cd /orcd/home/002/jackking/cls/.claude/worktrees/encoder-hopfield-eval-spec

PY=/home/jackking/.conda/envs/cls/bin/python
S=/orcd/pool/003/jackking/cls_runs/sweeps
OUT=/home/jackking/.claude/jobs/d05f5770/tmp/probe_proj

i=$SLURM_ARRAY_TASK_ID
rule_i=$(( i / 2 ))
seed=$(( 42 + i % 2 ))
CK=$(ls -d "$S"/w52_attract_fwhm/*_att0.5_seed=$seed)/encoder_final.pt

if (( rule_i == 0 )); then
    RULE=hebb
else
    RULE=proj
fi

"$PY" -m analysis.hopfield_probe.run \
    --ckpt "$CK" --label "$RULE · att0.5 · s$seed" \
    --storage_rule "$RULE" \
    --n_worlds 8 --n_envs_per_world 20 --k 1 3 5 10 20 \
    --steps 1 2 3 5 10 15 --env_size 20 --Npos 1716 \
    --n_alias 5000 --n_cont_samples 60000 --n_cont_annulus 20000 \
    --seed 0 --device cpu --fwhm_fallback 0.25 \
    --out "$OUT/t$i"

echo "DONE task $i  rule=$RULE seed=$seed  -> $OUT/t$i"
