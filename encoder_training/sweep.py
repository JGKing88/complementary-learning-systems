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
    # Model
    encoder_type="mlp",               # mlp | cnn
    lambdas=[11, 12, 13],
    out_dim=512,
    hidden_dim=1024,
    num_hidden_layers=4,
    hidden_channels=128,              # cnn only
    num_conv_layers=3,                # cnn only
    kernel_size=5,                    # cnn only
    # Patches
    nenv=40,
    npos=100,
    # npos_list="40,60,80,100,120",   # optional; overrides nenv/npos
    per_env_radius_frac=0.1,
    radius=10.0,
    single_env_batch=True,
    # Loss
    loss_mode="mse_contrastive",      # mse_contrastive | cka
    attract_lambda=2.0,
    repel_weight=5.0,
    uniformity_lambda=0.0,
    uniformity_anneal_epochs=25,
    # Training
    epochs=400,
    lr=2e-4,
    batch_size=4096,
    seed=42,
    fwhm_ratio=0.25,
    gain_start=1.0,
    gain_end=5.0,
    shuffle=False,
    # Nav eval during training (lightweight)
    eval_every=25,
    nav_env_size=20,
    nav_n_train=5,
    nav_n_val=5,
    nav_num_hopfields=20,
    nav_n_starts=100,
)

# ---------------------------------------------------------------------------
# GRID — keys must appear in BASE. Sweeps are the Cartesian product.
# Leave empty to launch a single run with BASE values.
# ---------------------------------------------------------------------------
GRID: dict[str, list] = {
    # "nenv": [20],
    # "npos": [150, 200, 250],
    "per_env_radius_frac": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
    # "out_dim": [512, 1024],
    # "attract_lambda": [1.0, 2.0, 4.0],
    # "lambdas": [[9, 10, 11], [11, 12, 13], [13, 14, 15]],
}

# ---------------------------------------------------------------------------
# Comprehensive post-train eval (flags for encoder_training.evaluate_nav)
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
    partition="pi_fiete",
    time="12:00:00",
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
    # Validate grid keys
    unknown = [k for k in GRID if k not in BASE]
    if unknown:
        sys.exit(f"GRID keys not in BASE: {unknown}")

    sweep_name = sys.argv[1] if len(sys.argv) > 1 else \
        datetime.now().strftime("sweep_%Y%m%d_%H%M%S")
    sweep_dir = os.path.join(SWEEP_BASE_DIR, sweep_name)
    os.makedirs(os.path.join(sweep_dir, "slurm"), exist_ok=True)

    keys = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys])) or [()]
    print(f"Sweep {sweep_name}: {len(combos)} runs  →  {sweep_dir}")
    for k in keys:
        print(f"  {k}: {GRID[k]}")

    train_time = SLURM["time"]
    include_train_flag = "--train_eval" if EVAL.get("include_train_eval") else ""
    eval_flags = _build_eval_flags(EVAL)

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

CKPT={run_dir}/encoder_best.pt
[ -f "$CKPT" ] || CKPT={run_dir}/encoder_final.pt

echo "=== eval: $CKPT ==="
python -u -m encoder_training.evaluate_nav \\
    --ckpt "$CKPT" {eval_flags} {include_train_flag} --json \\
    > {run_dir}/eval.log 2>&1
RC=$?
if [ $RC -ne 0 ]; then echo "EVAL FAILED (rc=$RC)"; tail -30 {run_dir}/eval.log; exit $RC; fi

grep '^JSON:' {run_dir}/eval.log | sed 's/^JSON: //' > {run_dir}/result.json
echo "=== done: $(grep -o '\"accuracy\": [0-9.]*' {run_dir}/result.json | head -1) ==="
"""
        r = subprocess.run(["sbatch"], input=sbatch, text=True,
                           capture_output=True)
        msg = r.stdout.strip() or r.stderr.strip()
        print(f"  [{i:3d}] {run_name}: {msg}")
        if r.returncode != 0:
            sys.exit(r.returncode)


if __name__ == "__main__":
    main()
