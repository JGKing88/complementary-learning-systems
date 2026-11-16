#!/usr/bin/env bash
# Non-negative ideal code: relu(Q z) (ideal_encoder.py relu_mix=1), the ideal's units remixed by a
# fixed orthogonal Walsh-Hadamard rotation and rectified, against the plain ideal through the
# identical probe (run_ideal.sh settings, Hebbian storage).
#
#   plain ideal r=16          reference, rerun here so every row shares one commit
#   relu_mix r=16             non-negative activity, production memory
#   relu_mix r=16 center      the code's mean vector removed (= mean-subtracted storage and cue
#                             in this linear memory); activity itself stays non-negative
#   relu_mix r=17.9 (center)  r raised so the peak's half-width matches the ideal's 19.4 cells
#
# SET=b (OUT relu_mix_b): one global inhibition constant (center_global=1) in place of the
# per-unit mean, and a sparser code with 10% of units active (active=0.1), with and without the
# constant, at r = 16 and at r = 23.2 (the sparse code's half-width matched to the ideal's 19.4).
#
# Index = 2 * encoder + region (whole, corner). Throttled to 4 tasks at a time.
#
#   bash analysis/hopfield_probe/run_relu_mix.sh [ARRAY]          # default 0-9
#   SET=b bash analysis/hopfield_probe/run_relu_mix.sh [ARRAY]
#   SET=c bash analysis/hopfield_probe/run_relu_mix.sh [ARRAY]    # slow waves removed (relu_band)
#   SET=d bash analysis/hopfield_probe/run_relu_mix.sh [ARRAY]    # trained relu_rank encoders
#   SET=e bash analysis/hopfield_probe/run_relu_mix.sh [ARRAY]    # w70 homeostatic + signed control
#   SET=f bash analysis/hopfield_probe/run_relu_mix.sh [ARRAY]    # w70 relu codes + global inhibition
#   SET=g bash analysis/hopfield_probe/run_relu_mix.sh [ARRAY]    # w71 radius 24, raw and inhibited
#   SET=h bash analysis/hopfield_probe/run_relu_mix.sh [ARRAY]    # att0.5 sparsified after training
#   SET=i bash analysis/hopfield_probe/run_relu_mix.sh [ARRAY]    # w72 2048 units, raw and inhibited
#   SET=j bash analysis/hopfield_probe/run_relu_mix.sh [ARRAY]    # att0.5 with a plain ReLU after training
#   SET=k bash analysis/hopfield_probe/run_relu_mix.sh [ARRAY]    # SET=j on seeds 42/44 + tanh-with-inhibition control
#   python -m analysis.hopfield_probe.ideal_summary "$OUT"        # the table
#
# Runs from THIS worktree: do not delete it while jobs are live.
set -euo pipefail

WT=${RELU_WT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}
export RELU_WT=$WT
PY=/home/jackking/.conda/envs/cls/bin/python
SET=${SET:-a}
export SET
B="n_freq=512,seed=0,gain=100"

if [[ "$SET" == k ]]; then
# Is SET=j's gain over att0.5 (retrieval, basin, dead goals) from the ReLU or from the inhibition?
# "tanh + inhibition" is att0.5 with one constant subtracted. It is also a sigmoid-rate encoder:
# sigmoid(2x) = (1 + tanh x) / 2, so a sigmoid output plus inhibition is this row, halved. Plus
# SET=j's relu(output) + inhibition on two more seeds.
OUT=${OUT:-/orcd/pool/003/jackking/cls_runs/results/hopfield_probe/att05_rectified_seeds}
S=/orcd/pool/003/jackking/cls_runs/sweeps/w52_attract_fwhm
ENCS=("inhib:$S/001_att0.5_seed=43/encoder_final.pt|att0.5 s43 tanh + inhibition")
for x in "000 42" "002 44"; do
    read -r n sd <<< "$x"
    A="$S/${n}_att0.5_seed=$sd/encoder_final.pt"
    ENCS+=("$A|att0.5 s$sd tanh" "inhib:$A|att0.5 s$sd tanh + inhibition"
           "inhib:rectify:out:$A|att0.5 s$sd relu(output) + inhibition")
done
elif [[ "$SET" == j ]]; then
# att0.5 s43 with a plain ReLU added after training (harness rectify: prefix): on its output
# (nearly binary, ~half on) and on its graded pre-tanh values; raw and with global inhibition.
OUT=${OUT:-/orcd/pool/003/jackking/cls_runs/results/hopfield_probe/att05_rectified}
A=/orcd/pool/003/jackking/cls_runs/sweeps/w52_attract_fwhm/001_att0.5_seed=43/encoder_final.pt
ENCS=(
    "rectify:out:$A|att0.5 s43 relu(output)"
    "inhib:rectify:out:$A|att0.5 s43 relu(output) + inhibition"
    "rectify:pre:$A|att0.5 s43 relu(pre-tanh)"
    "inhib:rectify:pre:$A|att0.5 s43 relu(pre-tanh) + inhibition"
)
elif [[ "$SET" == h ]]; then
# att0.5 s43 made sparse and non-negative after training (harness sparsify: prefix), 20% and 10%,
# raw and with global inhibition. No retraining.
OUT=${OUT:-/orcd/pool/003/jackking/cls_runs/results/hopfield_probe/att05_sparsified}
A=/orcd/pool/003/jackking/cls_runs/sweeps/w52_attract_fwhm/001_att0.5_seed=43/encoder_final.pt
ENCS=(
    "sparsify:0.2:$A|att0.5 s43 sparsified 20%"
    "inhib:sparsify:0.2:$A|att0.5 s43 sparsified 20% + inhibition"
    "sparsify:0.1:$A|att0.5 s43 sparsified 10%"
    "inhib:sparsify:0.1:$A|att0.5 s43 sparsified 10% + inhibition"
)
elif [[ "$SET" == i ]]; then
# w72: the non-negative 10% and 20% homeostatic codes with 2048 units, raw and inhibited.
OUT=${OUT:-/orcd/pool/003/jackking/cls_runs/results/hopfield_probe/relu_2048}
W=/orcd/pool/003/jackking/cls_runs/sweeps/w72_rank_homeo_2048
ENCS=()
for r in "000_att0.5_rank_homeo2048_10_seed=43|relu10 2048 s43" "001_att0.5_rank_homeo2048_10_seed=44|relu10 2048 s44" \
         "002_att0.5_rank_homeo2048_20_seed=43|relu20 2048 s43" "003_att0.5_rank_homeo2048_20_seed=44|relu20 2048 s44"; do
    IFS='|' read -r d lab <<< "$r"
    ENCS+=("$W/$d/encoder_final.pt|$lab" "inhib:$W/$d/encoder_final.pt|$lab + inhibition")
done
elif [[ "$SET" == g ]]; then
# w71: the non-negative 10% and 20% homeostatic codes trained with attract radius 24 (wider peak),
# raw and with global inhibition.
OUT=${OUT:-/orcd/pool/003/jackking/cls_runs/results/hopfield_probe/relu_r24}
W=/orcd/pool/003/jackking/cls_runs/sweeps/w71_rank_homeo_r24
ENCS=()
for r in "000_att0.5_rank_homeo10_seed=43|relu10 r24 s43" "001_att0.5_rank_homeo10_seed=44|relu10 r24 s44" \
         "002_att0.5_rank_homeo20_seed=43|relu20 r24 s43" "003_att0.5_rank_homeo20_seed=44|relu20 r24 s44"; do
    IFS='|' read -r d lab <<< "$r"
    ENCS+=("$W/$d/encoder_final.pt|$lab" "inhib:$W/$d/encoder_final.pt|$lab + inhibition")
done
elif [[ "$SET" == f ]]; then
# Global inhibition in the memory, no retraining: the w70 non-negative encoders (and w69's 20%)
# with one constant subtracted from every unit (harness inhib: prefix). Compare with SET=e.
OUT=${OUT:-/orcd/pool/003/jackking/cls_runs/results/hopfield_probe/relu_inhib}
W=/orcd/pool/003/jackking/cls_runs/sweeps/w70_rank_homeo_slow
ENCS=(
    "inhib:$W/000_att0.5_rank_homeo10_seed=43/encoder_final.pt|relu10 homeo s43 + inhibition"
    "inhib:$W/001_att0.5_rank_homeo10_seed=44/encoder_final.pt|relu10 homeo s44 + inhibition"
    "inhib:$W/002_att0.5_rank_homeo20_seed=43/encoder_final.pt|relu20 homeo s43 + inhibition"
    "inhib:$W/003_att0.5_rank_homeo20_seed=44/encoder_final.pt|relu20 homeo s44 + inhibition"
)
elif [[ "$SET" == e ]]; then
# w70_rank_homeo_slow (old loss, within-env; slow delayed homeostasis): even unit usage at 10% and 20% non-negative, and the signed
# 10% control (sparsity without non-negativity), beside att0.5 s43 rerun at this commit.
OUT=${OUT:-/orcd/pool/003/jackking/cls_runs/results/hopfield_probe/relu_homeo}
W=/orcd/pool/003/jackking/cls_runs/sweeps/w70_rank_homeo_slow
ENCS=(
    "/orcd/pool/003/jackking/cls_runs/sweeps/w52_attract_fwhm/001_att0.5_seed=43/encoder_final.pt|att0.5 s43 tanh"
    "$W/000_att0.5_rank_homeo10_seed=43/encoder_final.pt|relu10 homeo s43"
    "$W/001_att0.5_rank_homeo10_seed=44/encoder_final.pt|relu10 homeo s44"
    "$W/002_att0.5_rank_homeo20_seed=43/encoder_final.pt|relu20 homeo s43"
    "$W/003_att0.5_rank_homeo20_seed=44/encoder_final.pt|relu20 homeo s44"
    "$W/004_att0.5_signed_homeo10_seed=43/encoder_final.pt|signed10 homeo s43"
    "$W/005_att0.5_signed_homeo10_seed=44/encoder_final.pt|signed10 homeo s44"
    "${W/w70_rank_homeo_slow/w69_rank_homeo}/002_att0.5_rank_homeo20_seed=43/encoder_final.pt|relu20 fast-homeo s43"
    "${W/w70_rank_homeo_slow/w69_rank_homeo}/003_att0.5_rank_homeo20_seed=44/encoder_final.pt|relu20 fast-homeo s44"
)
elif [[ "$SET" == d ]]; then
# Trained encoders with the sparse ideal's output form (branch worktree-relu-sparse-encoder,
# wave w67_relu_rank10: w52 att0.5 loss and data, output = the sparse ideal's sorted profile by
# rank). Compare with SET=b's "relu10% r=23.2" (the sparse ideal, no inhibition).
OUT=${OUT:-/orcd/pool/003/jackking/cls_runs/results/hopfield_probe/relu_trained}
W=/orcd/pool/003/jackking/cls_runs/sweeps/w67_relu_rank10
ENCS=(
    "$W/000_att0.5_relu_rank10_seed=43/encoder_final.pt|trained relu_rank s43"
    "$W/001_att0.5_relu_rank10_seed=44/encoder_final.pt|trained relu_rank s44"
    "${W/w67_relu_rank10/w68_relu_rank10_xenv}/000_att0.5_relu_rank10_xenv_seed=43/encoder_final.pt|trained relu_rank xenv s43"
    "${W/w67_relu_rank10/w68_relu_rank10_xenv}/001_att0.5_relu_rank10_xenv_seed=44/encoder_final.pt|trained relu_rank xenv s44"
)
elif [[ "$SET" == c ]]; then
# Remove slow waves: draws with |n| < nmin rejected and replaced from the same Gaussian (r = 16),
# encoder relu(Q z) with one global inhibition constant. Controls: no waves removed, r lowered to
# give the same half-width as nmin = 10, 25, 50 (16, 12, 7 cells), to separate "no slow waves"
# from "narrower peak". The nmin = 0, r = 16 row is SET=b's "relu r=16 global inh".
OUT=${OUT:-/orcd/pool/003/jackking/cls_runs/results/hopfield_probe/relu_band}
G="relu_mix=1,center_global=1"
ENCS=(
    "ideal:r=16,$B,$G,nmin=5|relu inh r=16 no |n|<5"
    "ideal:r=16,$B,$G,nmin=10|relu inh r=16 no |n|<10"
    "ideal:r=16,$B,$G,nmin=17|relu inh r=16 no |n|<17"
    "ideal:r=16,$B,$G,nmin=25|relu inh r=16 no |n|<25"
    "ideal:r=16,$B,$G,nmin=34|relu inh r=16 no |n|<34"
    "ideal:r=16,$B,$G,nmin=50|relu inh r=16 no |n|<50"
    "ideal:r=14.2,$B,$G|relu inh r=14.2 (width of no<10)"
    "ideal:r=10.7,$B,$G|relu inh r=10.7 (width of no<25)"
    "ideal:r=6.2,$B,$G|relu inh r=6.2 (width of no<50)"
)
elif [[ "$SET" == b ]]; then
OUT=${OUT:-/orcd/pool/003/jackking/cls_runs/results/hopfield_probe/relu_mix_b}
ENCS=(
    "ideal:r=16,$B,relu_mix=1,center_global=1|relu r=16 global inh"
    "ideal:r=16,$B,relu_mix=1,active=0.1|relu10% r=16"
    "ideal:r=16,$B,relu_mix=1,active=0.1,center_global=1|relu10% r=16 global inh"
    "ideal:r=23.2,$B,relu_mix=1,active=0.1|relu10% r=23.2"
    "ideal:r=23.2,$B,relu_mix=1,active=0.1,center_global=1|relu10% r=23.2 global inh"
)
else
OUT=${OUT:-/orcd/pool/003/jackking/cls_runs/results/hopfield_probe/relu_mix}
ENCS=(
    "ideal:r=16,$B|ideal r=16"
    "ideal:r=16,$B,relu_mix=1|relu_mix r=16"
    "ideal:r=16,$B,relu_mix=1,center=1|relu_mix r=16 center"
    "ideal:r=17.9,$B,relu_mix=1|relu_mix r=17.9"
    "ideal:r=17.9,$B,relu_mix=1,center=1|relu_mix r=17.9 center"
)
fi
REGIONS=("|whole" "0 0 500|corner")

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    mkdir -p "$OUT/logs"
    exec sbatch --job-name=relu_mix_probe --partition=pi_evelina9,ou_bcs_normal \
        --time=4:00:00 --cpus-per-task=16 --mem=32G \
        --array="${1:-0-$(( 2 * ${#ENCS[@]} - 1 ))}%4" \
        --output="$OUT/logs/probe_%A_%a.out" "$0"
fi

cd "$WT"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
i=$SLURM_ARRAY_TASK_ID
echo "host $(hostname)  commit $(git rev-parse --short HEAD)"
IFS='|' read -r CK elabel <<< "${ENCS[$(( i / 2 ))]}"
IFS='|' read -r region rlabel <<< "${REGIONS[$(( i % 2 ))]}"
REG=()
[[ -n "$region" ]] && REG=(--world_region $region)

"$PY" -m analysis.hopfield_probe.run \
    --ckpt "$CK" --label "$elabel | $rlabel" ${REG[@]+"${REG[@]}"} \
    --n_worlds 8 --n_envs_per_world 20 --k 1 3 5 10 20 \
    --steps 1 2 3 5 10 15 --env_size 20 --Npos 1716 \
    --n_alias 5000 --n_cont_samples 60000 --n_cont_annulus 20000 \
    --seed 0 --device cpu --fwhm_fallback 0.25 \
    --storage_rule hebb \
    --out "$OUT/probe/t$i"

echo "DONE probe $i  $elabel  region=$rlabel -> $OUT/probe/t$i"
