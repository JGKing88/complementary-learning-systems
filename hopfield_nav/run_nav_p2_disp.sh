#!/bin/bash -l
#SBATCH --job-name=p2disp
#SBATCH --time=0-04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=ou_bcs_normal
#SBATCH --mem=96G
#SBATCH --output=/orcd/pool/003/jackking/cls_runs/logs/nav_p2_disp_%j.out

# P2 -- is relative displacement decodable from two sensory cones, in a NEW
# env? See docs/EXPERIMENTS_NAV_P2.md section 6 and
# analysis/nav_p2/displacement_decodability.py.
#
# No encoder, no scaffold, no GPU: this is a property of the sensor, so it is
# a CPU job. `--cpus-per-task 16` is for the BLAS in the 3648-feature bilinear
# ridge, which is the only expensive arm.
#
#   PROBE=main    sbatch hopfield_nav/run_nav_p2_disp.sh # all framings, res 4
#   PROBE=res     sbatch hopfield_nav/run_nav_p2_disp.sh # wall_resolution sweep
#   PROBE=sens    sbatch hopfield_nav/run_nav_p2_disp.sh # range sensor
#   PROBE=turn    sbatch hopfield_nav/run_nav_p2_disp.sh # walk persistence
#   PROBE=egoturn sbatch hopfield_nav/run_nav_p2_disp.sh # persistence x ego framing
#   PROBE=best    sbatch hopfield_nav/run_nav_p2_disp.sh # most favourable case
#   PROBE=adapt   sbatch hopfield_nav/run_nav_p2_disp.sh # k-shot in a new env
#
# `egoturn` and `best` are the two that decide the question: `derot_ego` is the
# only framing in which it is well posed, and every row there must be read
# against `side-only LEAK` and against the arm's own shuffled control.

set -euo pipefail
REPO=${REPO:-/orcd/home/002/jackking/cls/.claude/worktrees/nav-tri-metric}
OUT=${OUT:-/orcd/pool/003/jackking/cls_runs/results/nav_p2}

module load miniforge/24.3.0-0
source activate cls
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
cd "$REPO"
source scripts/cls_env.sh
mkdir -p "$OUT"

PROBE=${PROBE:-main}
TRAIN_ENVS=${TRAIN_ENVS:-64}
TEST_ENVS=${TEST_ENVS:-48}

if [ "$PROBE" = res ]; then
    # Does the +/-1 code's spatial granularity control how much geometry
    # survives the hash? wall_resolution is a launcher knob (default 4).
    for R in 1 2 4 8; do
        python -u -m analysis.nav_p2.displacement_decodability \
            --resolution "$R" \
            --train_envs "$TRAIN_ENVS" --test_envs "$TEST_ENVS" \
            --framings fixed free derot \
            --features spec xcorr bilin \
            --json "$OUT/disp_res${R}.json"
    done
elif [ "$PROBE" = sens ]; then
    # The range sensor the env does not have -- an upper bound on what
    # restructuring the sensory input could buy.
    python -u -m analysis.nav_p2.displacement_decodability \
        --sensors dist code \
        --train_envs "$TRAIN_ENVS" --test_envs "$TEST_ENVS" \
        --framings fixed free derot \
        --features spec xcorr bilin --mlp --mlp_features raw bilin \
        --json "$OUT/disp_sensor.json"
elif [ "$PROBE" = egoturn ]; then
    # The properly-posed realistic framing, across walk persistence.
    # `derot_ego` aligns the two cones by dpsi AND asks for the answer in view
    # 1's frame -- `derot` asks for a world-frame answer without ever being
    # given psi1, which is not a well-posed question. Read every row against
    # `side-only LEAK`: under a persistent walk the egocentric displacement is
    # nearly always straight ahead, so heading alone scores well and even the
    # constant predictor lands near 20 deg.
    for T in uniform 45 20 10; do
        if [ "$T" = uniform ]; then TARG=""; else TARG="--turn_sd_deg $T"; fi
        # shellcheck disable=SC2086
        python -u -m analysis.nav_p2.displacement_decodability \
            $TARG --resolution 1 \
            --train_envs "$TRAIN_ENVS" --test_envs "$TEST_ENVS" \
            --framings fixed free derot derot_ego ego \
            --features xcorr bilin --no_inenv \
            --json "$OUT/disp_egoturn_${T}.json"
    done
elif [ "$PROBE" = turn ]; then
    # How straight does the walk have to be before two 120-deg cones overlap
    # enough to decode from? PERSISTENCE_BONUS pays the policy to go straight,
    # so the uniform-turn reference is the pessimistic end, not the answer.
    for T in 90 45 20 10; do
        python -u -m analysis.nav_p2.displacement_decodability \
            --turn_sd_deg "$T" --resolution 1 \
            --train_envs "$TRAIN_ENVS" --test_envs "$TEST_ENVS" \
            --framings fixed free derot \
            --features xcorr bilin --no_inenv \
            --json "$OUT/disp_turn${T}.json"
    done
elif [ "$PROBE" = best ]; then
    # The most favourable configuration the sensor allows: the coarsest wall
    # code (where the resolution sweep says the geometry survives best), twice
    # the training envs, and the MLP. The decision rule should be judged
    # against this, not against the default.
    python -u -m analysis.nav_p2.displacement_decodability \
        --resolution 1 --train_envs 128 --test_envs 48 \
        ${TURN:+--turn_sd_deg $TURN} \
        --framings fixed derot_ego ego \
        --features xcorr bilin --mlp --mlp_features raw bilin \
        --mlp_epochs 60 --no_inenv \
        --json "$OUT/disp_best${TURN:+_turn$TURN}.json"
elif [ "$PROBE" = adapt ]; then
    python -u -m analysis.nav_p2.displacement_adaptation \
        --train_envs 48 --test_envs "${TEST_ENVS}" \
        --k 0 4 16 64 256 --framings fixed free \
        --sensors code dist --locality_res 1 2 4 8 \
        --json "$OUT/disp_adapt.json"
else
    python -u -m analysis.nav_p2.displacement_decodability \
        --train_envs "$TRAIN_ENVS" --test_envs "$TEST_ENVS" \
        --framings fixed free derot ego world \
        --features spec xcorr bilin --mlp --mlp_features raw bilin \
        --json "$OUT/disp_main.json"
fi
