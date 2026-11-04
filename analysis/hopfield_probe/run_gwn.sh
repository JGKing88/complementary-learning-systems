#!/usr/bin/env bash
# Sec 20: the grid wave network (gwn_encoder.py).
#
#   build   one GPU job: gwn_check step0, gwn_l1_learn (step 1), gwn_check
#           step2, step3 -- kernel metrics, saving the encoders the probe uses
#   probe   CPU array: run.py (run_ideal.sh settings, hebb) on the saved
#           encoders; the main network in 4 regions, the rest whole + corner.
#           Compare with ideal r=16 at ideal_encoder/probe/t12-t15.
#
#   bash analysis/hopfield_probe/run_gwn.sh build
#   bash analysis/hopfield_probe/run_gwn.sh step1      # step 1 alone (GWN_DEP=afterany:JOB to chain)
#   bash analysis/hopfield_probe/run_gwn.sh probe [ARRAY]     # default all (0-37)
#   bash analysis/hopfield_probe/run_gwn.sh extra      # step3b + step2b
#   bash analysis/hopfield_probe/run_gwn.sh local      # Sec 20.1: gains from one small environment
#   bash analysis/hopfield_probe/run_gwn.sh l1online [ARRAY]  # Sec 21: GPU array, one task per
#                                                        # condition (gwn_l1_online.CONDS), 4 at a time
#   bash analysis/hopfield_probe/run_gwn.sh l1twin     # Sec 21: learned L1 vs its exact twin (gwn_l1_twin)
#   bash analysis/hopfield_probe/run_gwn.sh l1lms      # Sec 21: route 1c learning rate (gwn_l1_lms)
#   bash analysis/hopfield_probe/run_gwn.sh l1noisefix # Sec 21: noise-bias correction (gwn_l1_noisefix)
#   bash analysis/hopfield_probe/run_gwn.sh l1decay    # Sec 21: 1c with a falling rate (gwn_l1_decay)
#   bash analysis/hopfield_probe/run_gwn.sh l1lockin2  # Sec 21: lock-in turn fixes (gwn_l1_lockin2)
#   bash analysis/hopfield_probe/run_gwn.sh l1mult     # Sec 21: uneven units per pattern (gwn_l1_mult)
#   bash analysis/hopfield_probe/run_gwn.sh l1multienv # Sec 21: 6 boxes / one 40-box, 1M steps (gwn_l1_multienv)
#   bash analysis/hopfield_probe/run_gwn.sh l1l3       # Sec 21.1: L1 and L3 from local rules (gwn_l1l3_local)
#   bash analysis/hopfield_probe/run_gwn.sh l1noiseslow # Sec 21: slow rates + bias correction under noise
#
# Runs from THIS worktree: do not delete it while jobs are live.
set -euo pipefail

WT=${GWN_WT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}
export GWN_WT=$WT
PY=/home/jackking/.conda/envs/cls/bin/python
OUT=${OUT:-/orcd/pool/003/jackking/cls_runs/results/hopfield_probe/gwn}
E=$OUT/encoders
MODE=${1:-build}
ARG=${2:-}

# encoder | label | regions (w = whole, c = corner, all = 4)
PROBES=(
    "exact_400k_gauss|exact 400k gauss|all"
    "exact_all_gauss|exact all-triples gauss|wc"
    "exact_400k_cut|exact 400k cut|wc"
    "exact_400k_rational|exact 400k rational|wc"
    "exact_400k_linear|exact 400k linear|wc"
    "learned_pca_fwhm0.25|PCA L1 bump 0.25|wc"
    "learned_move100k_fwhm0.25|movement L1 bump 0.25|wc"
    "learned_pca_fwhm0.1|PCA L1 bump 0.10|wc"
    "learned_move100k_fwhm0.1|movement L1 bump 0.10|wc"
    "exact_400k_coef0.1|exact 400k coef error 0.1|wc"
    # appended after the first array (tasks 22-25): from `extra`
    "exact_branch17k_gauss|branch design 17k neurons|wc"
    "exact_400k_noise0.01_fwhm0.25|gains learned at cell noise 0.01, bump 0.25|wc"
    # Sec 21 (tasks 26-31): L1 learned in one 14-cell box, bump 0.10
    "../l1_online/encoders/side14_fwhm0.1_1c|one box, online shift rule 1c|wc"
    "../l1_online/encoders/side14_fwhm0.1_1b|one box, shift estimate 1b|wc"
    "../l1_online/encoders/side14_fwhm0.1_noise0.01_1b|one box, 1b at cell noise 0.01|wc"
    # Sec 21.1 (tasks 32-37): L3 by the depression rule, all learned in one 14-cell box
    "../l1_online/encoders/local_exact_dep|exact L1, depression L3|wc"
    "../l1_online/encoders/local_1c_dep|online shift-rule L1, depression L3|wc"
    "../l1_online/encoders/local_2c_dep|lock-in L1 (all units), depression L3|wc"
)
TASKS=()
for p in "${PROBES[@]}"; do
    IFS='|' read -r enc lab regs <<< "$p"
    if [[ $regs == all ]]; then
        for r in "|whole" "0 0 500|corner" "608 608 500|centre" "1216 1216 500|opposite"; do
            TASKS+=("$enc|$lab|$r"); done
    else
        TASKS+=("$enc|$lab||whole" "$enc|$lab|0 0 500|corner"); fi
done

if [[ -z "${GWN_IN_JOB:-}" ]]; then
    mkdir -p "$OUT/logs"
    if [[ $MODE == l1twin || $MODE == l1lms || $MODE == l1noisefix || $MODE == l1decay || $MODE == l1lockin2 || $MODE == l1mult || $MODE == l1multienv || $MODE == l1l3 || $MODE == l1noiseslow ]]; then
        GWN_IN_JOB=1 exec sbatch ${GWN_DEP:+--dependency=$GWN_DEP} --job-name=gwn_$MODE \
            --partition=pi_evelina9,pi_fiete,ou_bcs_normal,mit_normal_gpu --gres=gpu:1 --exclude=node3804 \
            --time=${GWN_TIME:-4:00:00} --cpus-per-task=2 --mem=${GWN_MEM:-32G} \
            --output="$OUT/logs/${MODE}_%j.out" "$0" $MODE
    fi
    if [[ $MODE == l1online ]]; then
        GWN_IN_JOB=1 exec sbatch ${GWN_DEP:+--dependency=$GWN_DEP} --job-name=gwn_l1online \
            --partition=pi_evelina9,pi_fiete,ou_bcs_normal,mit_normal_gpu --gres=gpu:1 --exclude=node3804 \
            --time=${GWN_TIME:-3:00:00} --cpus-per-task=2 --mem=${GWN_MEM:-32G} --array="${ARG:-0-15}%4" \
            --output="$OUT/logs/l1online_%A_%a.out" "$0" l1online
    fi
    if [[ $MODE == build || $MODE == step1 || $MODE == extra || $MODE == local ]]; then
        GWN_IN_JOB=1 exec sbatch ${GWN_DEP:+--dependency=$GWN_DEP} --job-name=gwn_$MODE --partition=pi_evelina9,pi_fiete,ou_bcs_normal,mit_normal_gpu \
            --gres=gpu:1 --exclude=node3804 --time=${GWN_TIME:-5:00:00} --cpus-per-task=2 \
            --mem=${GWN_MEM:-32G} --output="$OUT/logs/${MODE}_%j.out" "$0" $MODE
    else
        GWN_IN_JOB=1 exec sbatch --job-name=gwn_probe --partition=pi_evelina9,pi_fiete,ou_bcs_normal,mit_normal \
            ${GWN_DEP:+--dependency=$GWN_DEP} --time=${GWN_PTIME:-4:00:00} --cpus-per-task=16 --mem=32G --array="${ARG:-0-$(( ${#TASKS[@]} - 1 ))}" \
            --output="$OUT/logs/probe_%A_%a.out" "$0" probe
    fi
fi

cd "$WT"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}

if [[ $MODE == l1twin ]]; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    "$PY" -m analysis.hopfield_probe.gwn_l1_twin --out "$OUT/l1_online"; echo "DONE l1twin"
    exit 0
fi
if [[ $MODE == l1lms ]]; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    "$PY" -m analysis.hopfield_probe.gwn_l1_lms --out "$OUT/l1_online"; echo "DONE l1lms"
    exit 0
fi
if [[ $MODE == l1noiseslow ]]; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    "$PY" -m analysis.hopfield_probe.gwn_l1_noise_slow --out "$OUT/l1_online"; echo "DONE l1noiseslow"
    exit 0
fi
if [[ $MODE == l1l3 ]]; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    "$PY" -m analysis.hopfield_probe.gwn_l1l3_local --out "$OUT/l1_online"; echo "DONE l1l3"
    exit 0
fi
if [[ $MODE == l1multienv ]]; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    "$PY" -m analysis.hopfield_probe.gwn_l1_multienv --out "$OUT/l1_online"; echo "DONE l1multienv"
    exit 0
fi
if [[ $MODE == l1mult ]]; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    "$PY" -m analysis.hopfield_probe.gwn_l1_mult --out "$OUT/l1_online"; echo "DONE l1mult"
    exit 0
fi
if [[ $MODE == l1lockin2 ]]; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    "$PY" -m analysis.hopfield_probe.gwn_l1_lockin2 --out "$OUT/l1_online"; echo "DONE l1lockin2"
    exit 0
fi
if [[ $MODE == l1decay ]]; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    "$PY" -m analysis.hopfield_probe.gwn_l1_decay --out "$OUT/l1_online"; echo "DONE l1decay"
    exit 0
fi
if [[ $MODE == l1noisefix ]]; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    "$PY" -m analysis.hopfield_probe.gwn_l1_noisefix --out "$OUT/l1_online"; echo "DONE l1noisefix"
    exit 0
fi
if [[ $MODE == l1online ]]; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    "$PY" -m analysis.hopfield_probe.gwn_l1_online run --task "$SLURM_ARRAY_TASK_ID" --out "$OUT/l1_online"
    echo "DONE l1online $SLURM_ARRAY_TASK_ID"
    exit 0
fi
if [[ $MODE == local ]]; then
    "$PY" -m analysis.hopfield_probe.gwn_local_gain --out "$OUT"; echo "DONE local"
    exit 0
fi
if [[ $MODE == extra ]]; then
    "$PY" -m analysis.hopfield_probe.gwn_check extra --out "$OUT"; echo "DONE extra"
    exit 0
fi
if [[ $MODE == step1 ]]; then
    "$PY" -m analysis.hopfield_probe.gwn_l1_learn --out "$OUT"; echo "DONE step1"
    exit 0
fi
if [[ $MODE == build ]]; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    "$PY" -m analysis.hopfield_probe.gwn_check step0 --out "$OUT"; echo "DONE step0"
    "$PY" -m analysis.hopfield_probe.gwn_l1_learn --out "$OUT"; echo "DONE step1"
    "$PY" -m analysis.hopfield_probe.gwn_check step2 --out "$OUT"; echo "DONE step2"
    "$PY" -m analysis.hopfield_probe.gwn_check step3 --out "$OUT"; echo "DONE step3"
    exit 0
fi

i=$SLURM_ARRAY_TASK_ID
IFS='|' read -r enc lab region rlabel <<< "${TASKS[$i]}"
REG=(); [[ -n "$region" ]] && REG=(--world_region $region)
"$PY" -m analysis.hopfield_probe.run \
    --ckpt "gwn:$E/$enc.pt" --label "$lab | $rlabel" ${REG[@]+"${REG[@]}"} \
    --n_worlds 8 --n_envs_per_world 20 --k 1 3 5 10 20 \
    --steps 1 2 3 5 10 15 --env_size 20 --Npos 1716 \
    --n_alias 5000 --n_cont_samples 60000 --n_cont_annulus 20000 \
    --seed 0 --device cpu --fwhm_fallback 0.25 --storage_rule hebb \
    --out "$OUT/probe/t$i"
echo "DONE probe $i  $lab  region=$rlabel -> $OUT/probe/t$i"
