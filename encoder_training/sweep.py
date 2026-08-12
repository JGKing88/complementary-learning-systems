"""Encoder-training sweep.

Edit the BASE / GRID / EVAL / SLURM dicts below, then:
    python -m encoder_training.sweep [sweep_name]

For each point in the Cartesian product of GRID, submits one SLURM job that
trains the encoder (writing encoder_best.pt into the run dir) and then runs
a comprehensive nav eval, saving result.json per run.

Plot aggregated results with:
    python -m analysis.encoder_sweep <sweep_dir>
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from datetime import datetime

from cls_paths import REPO_ROOT, sweeps_dir

# ---------------------------------------------------------------------------
# Base training config — matches flags in encoder_training.train
# Any value below can be overridden per-run by putting it in GRID.
# ---------------------------------------------------------------------------
BASE = dict(
    # Model. hidden_dim is capped at 512; out_dim is independent of it and is
    # the stronger lever anyway (corr with r_min +0.286 vs +0.227 across the
    # 407-encoder audit), and hidden=512 has never been paired with out_dim
    # above 256, so this corner is unexplored.
    encoder_type="mlp",               # mlp | cnn
    lambdas=[11, 12, 13],
    out_dim=1024,
    hidden_dim=512,
    num_hidden_layers=4,              # 4 beat 3 in the audit (max 16 vs 6)
    hidden_channels=128,              # cnn only
    num_conv_layers=3,                # cnn only
    kernel_size=5,                    # cnn only
    # Patches. 60 x 100 is the winner's layout; coverage correlated +0.445
    # with r_min while max patch size correlated -0.123, so prefer many small
    # patches over few large ones.
    # Geometry, and the epochs that keep its step count at 60,000 -- see the
    # STEP COUNT block below GRID before changing either.
    nenv=4,
    npos=400,
    # Explicit patch sizes; overrides nenv/npos. Putting this in GRID is how
    # patch count and patch size get varied *together* (the Cartesian product
    # cannot couple two keys), which is what holding coverage fixed requires.
    npos_list="",                     # "" → use nenv/npos above
    # Fixed 10-cell "near" rather than a fraction of the patch side. With a
    # fraction, growing the patch would grow the near-radius too and the two
    # effects could not be told apart; 10 cells is what 60x100 at frac=0.1
    # used, so the True baseline (r_min 2) is directly comparable.
    per_env_radius_frac=0.0,          # 0 → use the fixed `radius` below
    radius=10.0,
    # THE finding of the audit. With single_env_batch=True every batch comes
    # from one environment, so the loss never sees a cross-environment pair,
    # nothing pushes distant places apart, and the alias ceiling sits at 0.98.
    # At False the repulsion term engages and it drops to 0.79. At fixed seed
    # and otherwise identical config that boolean alone is r_min 16 vs 2.
    single_env_batch=True,
    # Loss
    loss_mode="mse_contrastive",      # mse_contrastive | cka
    attract_lambda=2.0,
    # 1.0, not the old default 5.0: in ur_loss_20260811 raising repel lowered
    # the alias ceiling only slightly (0.903 -> 0.884 from 1 to 40) while
    # shrinking the decay width much faster (39 -> 26), so median r_min fell
    # 15 -> 10. Over-repelling flattens the neighbourhood being preserved.
    repel_weight=1.0,
    uniformity_lambda=0.0,
    uniformity_anneal_epochs=25,
    # Training (the winner's values)
    epochs=15000,                     # nenv=4 -> 4 steps/epoch -> 60,000 steps
    lr=1e-4,
    batch_size=8192,
    seed=42,
    fwhm_ratio=0.25,
    gain_start=1.0,
    gain_end=5.0,
    shuffle=False,
    # Hopfield nav eval: OFF. This sweep is scored on the unique radius alone,
    # which needs no Hopfield and is measured over the whole arena rather than
    # inside 20-cell patches. With eval_every=0 the radius also selects
    # encoder_best.pt.
    eval_every=0,
    nav_env_size=20,
    nav_n_train=5,
    nav_n_val=5,
    nav_num_hopfields=20,
    nav_n_starts=100,
    # Unique-radius eval (~15 s per call at lambdas 11,12,13)
    ur_every=1500,                    # ~10 radius evals, matched to `epochs`
    ur_n_refs=20,
    ur_border=100,
    ur_seed=0,                        # fixed: every run scored at the same spots
)

# ---------------------------------------------------------------------------
# GRID — keys must appear in BASE. Sweeps are the Cartesian product.
# Leave empty to launch a single run with BASE values.
# ---------------------------------------------------------------------------
# The radius is where the local decay curve crosses the far-field ceiling, so
# the two axes below are the two halves of that: repel_weight sets how hard
# cross-environment pairs are pushed apart (the ceiling), per_env_radius_frac
# sets what counts as "near" (the decay width). Three seeds because the audit
# could not separate a real effect from run-to-run spread at n=1.
# ur_loss2: push repel BELOW 1.0, where ur_loss_20260811's trend pointed, and
# fill the 0.10-0.20 radius_frac gap. The best cell there (repel 1.0,
# radius_frac 0.10 -> 18) sat on the grid edge, so the optimum may be outside.
# ur_seb_B: can single_env_batch=True be rescued by UNIFORMITY?
#
# uniformity_loss is logsumexp(-t*||zi-zj||^2) over the batch -- a repulsion
# that never asks which environment a pair came from, so unlike the far-pair
# term it does not need mixed batches to bite. That makes it the natural
# substitute for the cross-environment repulsion True removes.
#
# Note the high-dimensional fact that makes this promising: 3M codes spread
# near-uniformly in 1024 dimensions have a maximum pairwise cosine of ~0.16
# (measured), against the 0.988 alias ceiling True currently produces. Spread
# means near-orthogonality here, not overlap -- 3M points cannot cover a
# 1024-sphere.
#
# The range is wide because uniformity_lambda has never been anything but 0 in
# any run in the archive.
GRID: dict[str, list] = {
    "repel_weight": [1.0, 5.0],
    "seed": [42, 43],
}

# ur_seb_B (running): uniformity in the True regime.
_GRID_seb_B: dict[str, list] = {
    "uniformity_lambda": [0.0, 0.1, 0.5, 2.0, 8.0],
    "seed": [42, 43],
}

# ---------------------------------------------------------------------------
# STEP COUNT IS NOT COMPARABLE ACROSS GEOMETRIES -- read before touching nenv.
#
# single_env_batch_iterator yields exactly ONE batch per env, so with
# single_env_batch=True the optimizer takes `nenv` steps per epoch. At 1000
# epochs that is 60,000 steps for nenv=60 but only 1,000 for nenv=1. A patch-
# size sweep at fixed `epochs` therefore compares a fully trained model against
# one that took 1/60th as many steps, and the first attempt at ur_seb_A did
# exactly that and had to be cancelled.
#
# `epochs` is set per launch instead, to hold total steps at 60,000:
#     nenv=60 (100x100)  epochs  1000   ur_every  100    coverage 20.4%
#     nenv=15 (200x200)  epochs  4000   ur_every  400    coverage 20.4%
#     nenv= 4 (400x400)  epochs 15000   ur_every 1500     coverage 21.7%
# Geometry is fixed within a launch, so nenv/npos go in BASE and the tags stay
# readable -- no npos_list needed.
#
# Coverage is held near 20% because it correlated +0.445 with r_min. Note the
# remaining tension: coverage = nenv * npos^2 and steps/epoch = nenv, so
# holding coverage while varying npos forces nenv to move, which is why epochs
# has to be rescaled rather than left alone.
#
# For the mixed-batch (False) case the iterator is a DataLoader over all
# points, giving floor(600000/8192) = 73 batches/epoch -- so ur_seb_control
# compared 73,000 steps against 60,000, a 22% gap, far too small to account for
# its 2-vs-18 result. That comparison stands.

# ur_seb_A (running): can single_env_batch=True be rescued by patch GEOMETRY?
#
# Under True the only repulsion left is between far pairs *within* one patch,
# so the distance over which the code is pushed apart is bounded by the patch
# side. At 100 cells that is far smaller than the 1716-cell arena and the alias
# ceiling pins at 0.988. Growing the patch should extend that reach; in the
# limit of a single arena-sized env, True and False coincide by construction.
#
# Coverage is held near 20% (the archive's best) so this isolates granularity
# from coverage, which the audit found correlated +0.445 with r_min:
#     4 x 400  = 21.7%     2 x 600 = 24.5%     1 x 800 = 21.7%
# Known already, so not re-run: 60 x 100 -> r_min 2 (ur_seb_control),
# 15 x 200 -> r_min 2 (archive run_20260422_142233).
#
# repel_weight is included because within-patch repulsion is now the *only*
# repulsion, so its optimum need not match the False regime's (where it wanted
# to go down).
_GRID_seb_A: dict[str, list] = {
    "npos_list": ["400,400,400,400", "600,600", "800"],
    "repel_weight": [1.0, 5.0],
    "seed": [42, 43],
    # --- the False-regime axes, held fixed here ---
    # "repel_weight": [0.25, 0.5, 1.0, 2.0],
    # "per_env_radius_frac": [0.10, 0.15, 0.20],
    # --- done ---
    # ur_loss_20260811: repel [1,5,15,40] x frac [0.05,0.1,0.2] -- repel down,
    #   not up; ridge runs diagonally; best 20 at repel 5 / frac 0.2 / seed 43
    # ur_seb_control: single_env_batch [True,False] x 3 seeds -- True pins at
    #   r_min 2 and alias 0.988 on every seed; False reaches 18
    # --- not yet explored ---
    # "uniformity_lambda": [0.0, 0.1, 0.5, 2.0],   # the True-regime hypothesis
    # "out_dim": [256, 512, 1024],
    # "attract_lambda": [1.0, 2.0, 5.0],
}

# ---------------------------------------------------------------------------
# Post-train Hopfield nav eval. UNUSED by the current sweep, which scores on
# the unique radius only -- kept so an eval pass can be restored by putting
# the evaluate_nav block back into the sbatch template below.
# ---------------------------------------------------------------------------
EVAL = dict(
    env_size=20,
    n_train_envs=5,
    n_val_envs=5,
    num_hopfields=50,
    n_starts_per_env=100,
    platform_radius=1.0,
    max_steps_mult=3,
    scale=1.0,
    normalize=1,
    recompute_interval=1,
    hopfield_alpha=0.8,
    seed=42,
    include_train_eval=True,
)

# ---------------------------------------------------------------------------
# SLURM resources
# ---------------------------------------------------------------------------
SLURM = dict(
    # Three partitions, all PreemptMode=OFF; slurm takes whichever frees first.
    # NOT ou_bcs_low: it is preemptible and killed a smoke run mid-training.
    # Short analysis jobs survive there, multi-hour training does not.
    #   mit_normal_gpu  67 nodes, MaxTime 6h
    #   ou_bcs_normal   21 nodes, MaxTime 1d
    #   pi_fiete         1 node,  MaxTime 7d
    # The 6h limit is what makes the 67-node partition eligible, and a run is
    # ~45 min (2.5 s/epoch x 1000, plus ten 15 s radius evals).
    partition="mit_normal_gpu,ou_bcs_normal,pi_fiete",
    time="6:00:00",
    mem="64G",
    gres="gpu:a100:1",
    cpus_per_task=4,
)

SWEEP_BASE_DIR = str(sweeps_dir())
WORKDIR = str(REPO_ROOT)

# `store_true` flags on train.py — these take no value, just the flag name.
_BOOL_FLAGS = {"single_env_batch", "shuffle"}


def _fmt(v) -> str:
    if isinstance(v, list):
        return " ".join(str(x) for x in v)
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _build_train_flags(cfg: dict) -> str:
    parts = []
    for k, v in cfg.items():
        if k in _BOOL_FLAGS:
            if v:
                parts.append(f"--{k}")
        elif v == "":
            continue          # empty string → omit, fall back to train.py default
        else:
            parts.append(f"--{k} {_fmt(v)}")
    return " ".join(parts)


def _build_eval_flags(cfg: dict) -> str:
    parts = []
    for k, v in cfg.items():
        if k == "include_train_eval":
            continue
        if isinstance(v, bool):
            parts.append(f"--{k} {int(v)}")
        else:
            parts.append(f"--{k} {_fmt(v)}")
    return " ".join(parts)


def _tag(keys: list[str], combo: tuple) -> str:
    parts = []
    for k, v in zip(keys, combo):
        if isinstance(v, list):
            vs = "-".join(str(x) for x in v)
        elif isinstance(v, float):
            vs = f"{v:g}"
        else:
            vs = str(v)
        parts.append(f"{k}={vs}")
    return "_".join(parts)


def main():
    # A real parser, because this reads argv[1] as the sweep *name* and then
    # submits one GPU job per grid point. Without it, `python -m
    # encoder_training.sweep --help` -- which is exactly what
    # `scripts/check_entry_points.py` runs against every entry point -- named
    # the sweep "--help" and submitted the whole grid. That gate had queued
    # ~150 stray jobs before anyone noticed, and each one trains an encoder for
    # 12 h on an A100. `--help` must not launch anything.
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sweep_name", nargs="?", default=None,
                   help="Directory name under the sweeps root. "
                        "Default: sweep_<timestamp>.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be submitted and exit without "
                        "calling sbatch or creating run directories.")
    args = p.parse_args()

    # Validate grid keys
    unknown = [k for k in GRID if k not in BASE]
    if unknown:
        sys.exit(f"GRID keys not in BASE: {unknown}")

    sweep_name = args.sweep_name or \
        datetime.now().strftime("sweep_%Y%m%d_%H%M%S")
    sweep_dir = os.path.join(SWEEP_BASE_DIR, sweep_name)

    keys = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys])) or [()]
    print(f"Sweep {sweep_name}: {len(combos)} runs  →  {sweep_dir}")
    for k in keys:
        print(f"  {k}: {GRID[k]}")
    if args.dry_run:
        for i, combo in enumerate(combos):
            tag = _tag(keys, combo) if keys else "run"
            print(f"  [{i:3d}] {f'{i:03d}_{tag}' if tag != 'run' else f'{i:03d}'}")
        print("--dry-run: nothing submitted, nothing written")
        return

    os.makedirs(os.path.join(sweep_dir, "slurm"), exist_ok=True)

    train_time = SLURM["time"]

    for i, combo in enumerate(combos):
        cfg = {**BASE, **dict(zip(keys, combo))}
        tag = _tag(keys, combo) if keys else "run"
        run_name = f"{i:03d}_{tag}" if tag != "run" else f"{i:03d}"
        run_dir = os.path.join(sweep_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)

        # train.py will write to {save_dir}/{run_name}/encoder_*.pt
        train_flags = _build_train_flags(
            {**cfg, "save_dir": sweep_dir, "run_name": run_name})

        # Record sweep metadata alongside the run so the plot script can read it.
        meta = {"index": i, "run_name": run_name,
                "grid": dict(zip(keys, combo))}
        with open(os.path.join(run_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)

        sbatch = f"""#!/bin/bash -l
#SBATCH --job-name=sw_{sweep_name}_{i}
#SBATCH --time={train_time}
#SBATCH --cpus-per-task={SLURM["cpus_per_task"]}
#SBATCH --ntasks=1
#SBATCH --gres={SLURM["gres"]}
#SBATCH --mem={SLURM["mem"]}
#SBATCH --partition={SLURM["partition"]}
#SBATCH --output={sweep_dir}/slurm/slurm-%j_{i:03d}.out

module load miniforge/24.3.0-0
module load cuda/13.0.1
source activate cls
unset CUDA_VISIBLE_DEVICES
cd {WORKDIR}

echo "=== train: {run_name} ==="
python -u -m encoder_training.train {train_flags}
RC=$?
if [ $RC -ne 0 ]; then echo "TRAIN FAILED (rc=$RC)"; exit $RC; fi

# No Hopfield nav eval: this sweep is scored on the unique radius, which
# train.py already logged during training and stored in both checkpoints.
echo "=== done ==="
grep 'Unique radius' {sweep_dir}/slurm/slurm-*_{i:03d}.out | tail -3
"""
        r = subprocess.run(["sbatch"], input=sbatch, text=True,
                           capture_output=True)
        msg = r.stdout.strip() or r.stderr.strip()
        print(f"  [{i:3d}] {run_name}: {msg}")
        if r.returncode != 0:
            sys.exit(r.returncode)


if __name__ == "__main__":
    main()
