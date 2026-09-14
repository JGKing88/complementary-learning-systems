#!/bin/bash -l
#SBATCH --job-name=seprobe
#SBATCH --time=0-04:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=ou_bcs_normal
#SBATCH --mem=64G
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/logs/se_probe_%j.out
#
# The sample-efficiency VERDICT: score candidate checkpoints against d0_base
# u725 with exactly the wave-1 pipeline (run_wave1_final.sh), in ONE process so
# envs, starts and memory contents match -- docs/EXPERIMENTS_SAMPLE_EFF.md §1.
#
#   CKPTS="/path/a/navigate_u400.pt /path/b/navigate_u600.pt" \
#   LABELS="se_b8_lr1_u400 se_b4_lr1_u600" \
#   TAG=s1_round1 sbatch hopfield_nav/run_se_probe.sh
#
# d0_base u725 is ALWAYS appended (label d0_base_u725); REFS=1 adds p20_e u700
# / p19_kcap u800 as the specialist references, as in wave 1 (off by default:
# they are not part of the pass/fail and cost a third of the probe). The
# probe's collapsed-tail number depends on the --ckpt list length (DUAL_TRAINING
# §9.0.1 caveat 5), so compare within one run of this script, never across two.
set -euo pipefail
REPO=${REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/sample-eff}
cd "$REPO"
module load miniforge/24.3.0-0
module load cuda/13.0.1
source activate cls
unset CUDA_VISIBLE_DEVICES
source scripts/cls_env.sh
export PYTHONPATH="$REPO"
PY=python
CKD=$CLS_CKPTS
OUT=${OUT:-$CLS_RESULTS/nav_tri_probe}
mkdir -p "$OUT"
DEVICE=${DEVICE:-cuda}
TAG=${TAG:-se}
TRIALS=${TRIALS:-24}       # x 6 envs = 144 explore trials, as wave 1
NAV_TRIALS=${NAV_TRIALS:-32}

BASE=$CKD/navigate_navp2_d0_base_s42_22133273/navigate_u725.pt
SPEC=$CKD/navigate_navp2_p20_e_s42_21695407/navigate_u700.pt
EXPLOIT=$CKD/navigate_navp2_p19_kcap_s42_21656252/navigate_u800.pt

[ -n "${CKPTS:-}" ] || { echo "CKPTS is required" >&2; exit 1; }
[ -n "${LABELS:-}" ] || { echo "LABELS is required" >&2; exit 1; }
ALL="$CKPTS $BASE"; ALLL="$LABELS d0_base_u725"
REFS=${REFS:-0}
if [ "$REFS" = 1 ]; then
  ESPEC="$SPEC"; ESPECL="p20_e_u700"; NREF="$EXPLOIT"
else
  ESPEC=""; ESPECL=""; NREF=""
fi
echo "candidates: $LABELS  (+ d0_base_u725$( [ "$REFS" = 1 ] && echo ', p20_e_u700, p19_kcap_u800'))"
for c in $CKPTS; do
  # Cost of what is being scored, from the checkpoint itself.
  $PY - "$c" <<'EOF'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(f"  {sys.argv[1]}: u{ck.get('update')} cum_episodes={ck.get('cum_episodes')} "
      f"cum_env_steps={ck.get('cum_env_steps')}")
EOF
done

for ND in 0 10; do
  J="$OUT/se_${TAG}_d${ND}.json"
  echo "################ explore  n_dist=$ND ################"
  $PY -u -m analysis.nav_tri.explore_traj \
      --ckpt $ALL $ESPEC --labels $ALLL $ESPECL \
      --envs 6 --trials "$TRIALS" --n_distractors "$ND" \
      --max_steps 200 --split place=held_out \
      --seed 42 --device "$DEVICE" --no-deterministic --json "$J"
  echo "---- swept + billiard efficiency ----"
  $PY -u -m analysis.nav_tri.swept_from_traj --json "$J" --radius 1.0 --vs_billiard
  echo "---- collapsed tail, split by chase_q ----"
  $PY -u -m analysis.nav_tri.tail_report "$J"
done

echo "################ EXPLOIT half ################"
$PY -u -m analysis.nav_tri.behavior_probe \
    --ckpt $ALL $NREF \
    --mode nav --n_distractors 0 5 10 \
    --split place=held_out --trials "$NAV_TRIALS" --envs 6 \
    --max_steps 200 --seed 42 --device "$DEVICE" \
    --json "$OUT/se_${TAG}_nav.json"
echo "ALL DONE"
