#!/usr/bin/env python3
"""Sweeps for the ``exclude_cross_env_pairs=True`` campaign.

    python -m encoder_training.sweep_ecp <wave> [--dry-run] [--name NAME]
    python -m encoder_training.sweep_ecp --list

Why this exists rather than more edits to ``encoder_training.sweep``:

1. **Named grid values.** The mandated axis here is a *mix* of patch sizes, and
   ``--npos_list`` for a 93-patch mix is 400 characters. ``sweep``'s tag builder
   would put all of it in the run directory name. Here a grid value may be a
   ``{label: value}`` dict and the label is what reaches the name.
2. **Step-matched epochs.** With mixed batches the optimizer takes
   ``floor(N_points / batch_size)`` steps per epoch, so a geometry that changes
   coverage changes the step count at fixed ``epochs``. ``ur_seb_A`` had to be
   cancelled for exactly this. ``epochs`` is derived from a target step count
   instead of being a grid value.
3. Every wave that has ever been launched stays in this file, so a result in
   the log can be traced to the grid that produced it.

Partition is ``ou_bcs_normal`` only, by instruction.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys

from cls_paths import REPO_ROOT, sweeps_dir

ARENA = 1716            # prod(11, 12, 13)
TARGET_STEPS = 73_000   # what the 60x100 mixed-batch reference runs took

# ---------------------------------------------------------------------------
# Patch-size mixes. Every size <= 200, by instruction. Big patches are listed
# first because `sample_nonoverlapping_patches` places them in order and a large
# square is what fails to find room once the arena is speckled with small ones.
#
# Coverage is held near 20% across the mixes so the comparison is granularity,
# not area -- the audit found coverage correlated +0.445 with r_min, which would
# otherwise swamp the axis being tested.
# ---------------------------------------------------------------------------
def _mix(*pairs: tuple[int, int]) -> str:
    """(size, count) pairs → the comma string ``--npos_list`` wants."""
    out: list[int] = []
    for size, count in pairs:
        out += [size] * count
    return ",".join(str(s) for s in out)


SIZE_MIXES: dict[str, str] = {
    # --- uniform controls -------------------------------------------------
    "u100": _mix((100, 60)),                    # 600k, 20.4% -- the known baseline
    "u200": _mix((200, 15)),                    # 600k, 20.4% -- largest allowed
    # --- mixes ------------------------------------------------------------
    "mix2": _mix((200, 9), (100, 24)),          # 600k, 20.4%,  33 envs
    "mix5": _mix((200, 3), (140, 6), (100, 12),
                 (70, 24), (50, 48)),           # 595k, 20.2%,  93 envs
    "mixbig": _mix((200, 9), (140, 6), (100, 6),
                   (70, 8), (50, 12)),          # 607k, 20.6%,  41 envs
    "mixwide": _mix((200, 6), (150, 7), (100, 11),
                    (60, 18), (30, 36)),        # 605k, 20.5%,  78 envs
    # --- coverage variants of the best mix (used in later waves) ----------
    # Rejection sampling was measured to place these at seeds 42/43; it starts
    # failing around 65%, so ~53% is the usable ceiling for sizes <= 200.
    "mix2_lo": _mix((200, 4), (100, 12)),       # 280k,  9.5%,  16 envs
    "mix2_hi": _mix((200, 15), (100, 40)),      # 1.00M, 34.0%, 55 envs
    "mix3_45": _mix((200, 20), (150, 20), (100, 30)),   # 1.55M, 52.6%, 70 envs
    "mixsmall": _mix((200, 12), (100, 40), (50, 200)),  # 1.38M, 46.9%, 252 envs
}


def mix_points(spec: str) -> int:
    return sum(int(s) ** 2 for s in spec.split(","))


# ---------------------------------------------------------------------------
# BASE — the reference config for this campaign. `exclude_cross_env_pairs` is
# True in every wave: that is the question.
# ---------------------------------------------------------------------------
BASE = dict(
    encoder_type="mlp",
    lambdas=[11, 12, 13],
    out_dim=1024,
    hidden_dim=512,
    num_hidden_layers=4,
    npos_list=SIZE_MIXES["mix2"],
    per_env_radius_frac=0.0,          # 0 → the fixed `radius` below
    radius=10.0,
    single_env_batch=False,           # mixed batches; only the PAIRS are withheld
    loss_mode="mse_contrastive",      # cka is excluded by instruction
    attract_lambda=2.0,
    repel_weight=1.0,
    uniformity_lambda=0.0,
    uniformity_anneal_epochs=25,
    uniformity_t=2.0,
    uniformity_scope="all",
    var_lambda=0.0,
    cov_lambda=0.0,
    var_gamma=1.0,
    graded_sigma=0.0,
    input_far_tau=-1.0,
    exclude_cross_env_pairs=True,     # THE constraint
    epochs=1000,                      # overwritten by the step-match below
    lr=1e-4,
    weight_decay=1e-4,
    batch_size=8192,
    seed=42,
    fwhm_ratio=0.25,
    gain_start=1.0,
    gain_end=5.0,
    shuffle=False,
    lazy_codes=True,                  # ~1 GB host, so runs can share a node
    eval_every=0,                     # no Hopfield nav eval; radius selects best
    ur_every=100,                     # rescaled with epochs
    ur_n_refs=20,
    ur_border=100,
    ur_seed=0,
)

# ---------------------------------------------------------------------------
# WAVES. A value may be a plain list, or a dict {label: value} when the value
# is too long to put in a directory name.
# ---------------------------------------------------------------------------
WAVES: dict[str, dict] = {
    # W1 -- the mandated axis. Does a mix of patch sizes beat a uniform set at
    # matched coverage, and does the near-radius want to scale with the patch?
    #
    # `radius_mode` is the interesting half. With a FIXED radius every patch
    # teaches the same notion of "near" and the sizes vary only how far the
    # within-patch repulsion reaches. With a FRACTIONAL radius the patches
    # disagree about what "near" means, and a translation-invariant code cannot
    # satisfy them all -- so the encoder is pushed toward depending on absolute
    # position, which is the thing that is missing. Two different mechanisms,
    # opposite predictions, one grid.
    "w1_geometry": {
        "npos_list": {k: SIZE_MIXES[k] for k in
                      ("u100", "u200", "mix2", "mix5", "mixbig")},
        "per_env_radius_frac": [0.0, 0.1],
        "seed": [42, 43],
    },
    # W2 -- the two levers on the radius, at a fixed geometry. Every arm is
    # env-blind, so every arm is legal under the flag.
    #
    # THE DECAY (`graded_sigma`). r_min is where the decay curve crosses the
    # alias ceiling. The binary target asks for a *plateau* at cosine 1 inside
    # the radius and the radius test is a strictly-decreasing one, so what the
    # metric currently reads is the residual slope the network failed to
    # flatten. Naming the slope outright is untried in any wave so far.
    #
    # THE CEILING. The measured deficit is rank: withholding the cross-env
    # pairs takes the code from ~202 effective dimensions to 24-59 of 1024,
    # with every coordinate still alive, and r_min tracks that number across
    # every encoder measured. Three terms aim at it, differing in how they can
    # misfire:
    #   uniformity : the only one that acts on an individual collapsed pair,
    #                but logsumexp is dominated by the smallest distance and
    #                those are the pairs `attract` is holding at cosine 1
    #   var/cov    : pair-free, so it cannot fight `attract` -- but it asks for
    #                per-coordinate variance, and the collapse is not in the
    #                coordinates, it is in the spectrum
    #   rate       : pair-free and spectral, so on the diagnosis it is the one
    #                aimed at the actual deficit
    "w2_spread": {
        "arm": {
            "none":      dict(),
            "graded10":  dict(graded_sigma=10.0),
            "graded25":  dict(graded_sigma=25.0),
            "graded50":  dict(graded_sigma=50.0),
            "unif0.1":   dict(uniformity_lambda=0.1),
            "unif1":     dict(uniformity_lambda=1.0),
            "vicreg":    dict(var_lambda=1.0, cov_lambda=0.1),
            "rate0.3":   dict(rate_lambda=0.3),
            "rate3":     dict(rate_lambda=3.0),
        },
        "seed": [42, 43],
    },
    # W3 -- the near radius, as a *rank* knob rather than a locality knob.
    #
    # The measured effective dimensionality goes with how many distinguishable
    # places a patch contains, which is (side / radius)^2:
    #     60x100, radius 10 ->  100 places -> 24-59 dims
    #     15x200, radius 10 ->  400 places -> 117 dims
    #     the unconstrained regime, ~1900  -> 202 dims
    # Sublinear, but the direction is unambiguous, and shrinking the radius is
    # the one way to raise that count that costs nothing and asks nothing about
    # environments. So the prediction is that under this constraint the radius
    # wants to be far smaller than the 10 the unconstrained regime settled on --
    # the reverse of §2.2c, because a different quantity is binding.
    #
    # Against it: r_min is roughly how far the trained notion of "near"
    # generalizes, so shrinking the radius narrows the decay, and the radius is
    # the crossing of the decay and the ceiling. The two effects pull opposite
    # ways and the bracket is wide enough to find the turn.
    # radius=10 is bit-for-bit w1's mix2/per_env_radius_frac=0 cell (checked by
    # diffing the two meta.json), so those two runs were cancelled rather than
    # re-run; take that row from w1.
    "w3_radius": {
        "radius": [2.0, 3.0, 5.0, 10.0, 20.0, 40.0],
        "seed": [42, 43],
    },
    # W4 -- coverage, the one geometry axis the audit liked (+0.445 with r_min)
    # that no wave here has moved. Two mechanisms point the same way: more of
    # the arena seen is less of it extrapolated to, and more places is more
    # rank. Step-matched, so the extra points buy epochs' worth of gradient
    # rather than more of them.
    "w4_coverage": {
        "npos_list": {k: SIZE_MIXES[k] for k in
                      ("mix2_lo", "mix2", "mix2_hi", "mix3_45", "mixsmall")},
        "seed": [42, 43, 44],
    },
    # W5 -- rank from the input side and from capacity.
    #
    # The raw smoothed code has a participation ratio of 42.7 out of its 434
    # dimensions, so the *input* is itself low-rank and no linear map can beat
    # that. Every dimension past the 43rd in a trained code is a nonlinear
    # conjunction of module phases that the network had to build -- which is
    # why reaching 202 is hard, and why two knobs that have never been moved
    # here are worth a wave:
    #
    #   fwhm_ratio  smoothing is what correlates neighbouring phases, so a
    #               sharper bump raises the input's own rank. It also narrows
    #               the near-field, which is the opposing effect.
    #   hidden_dim  the conjunctions have to fit somewhere; 512 has never been
    #               raised in any wave, and out_dim is not the binding limit
    #               (the collapsed codes use 24-59 of 1024).
    "w5_input_rank": {
        "fwhm_ratio": [0.1, 0.25, 0.5],
        "hidden_dim": [512, 1024],
        "seed": [42, 43],
    },
}


def _flatten(cfg: dict) -> dict:
    """Expand an ``arm`` dict of overrides into the config it stands for."""
    arm = cfg.pop("arm", None)
    if isinstance(arm, dict):
        cfg.update(arm)
    return cfg


_BOOL_FLAGS = {"single_env_batch", "shuffle", "exclude_cross_env_pairs",
               "lazy_codes"}

# MEASURED, against the obvious guess. Four runs sharing one A100 each ran
# exactly 4x slower (5.0 epochs/min against 20), so packing buys nothing: the
# step is bandwidth-bound on the 8192^2 pair masks, not launch-bound, and
# bandwidth is the shared resource. RUNS_PER_JOB stays 1.
#
# The lever that does work on a full partition is backfill. A run is ~50 min and
# needs ~2 GB of host memory with lazy codes, so asking for 1.5 h and 16 GB lets
# the scheduler drop a job into a gap that a 12 h / 80 G request could never fit.
SLURM = dict(
    partition="ou_bcs_normal",
    time="1:30:00",
    mem="16G",
    gres="gpu:1",
    cpus_per_task=2,
)
RUNS_PER_JOB = 1


def _fmt(v) -> str:
    if isinstance(v, list):
        return " ".join(str(x) for x in v)
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _train_flags(cfg: dict) -> str:
    parts = []
    for k, v in cfg.items():
        if k.startswith("_"):
            continue          # bookkeeping for meta.json, not a train.py flag
        if k in _BOOL_FLAGS:
            if v:
                parts.append(f"--{k}")
        elif v == "":
            continue
        else:
            parts.append(f"--{k} {_fmt(v)}")
    return " ".join(parts)


def _labelled(values):
    """Normalise a grid axis to a list of (label, value)."""
    if isinstance(values, dict):
        return list(values.items())
    out = []
    for v in values:
        lab = f"{v:g}" if isinstance(v, float) else str(v)
        out.append((lab, v))
    return out


def build_runs(wave: dict) -> list[tuple[str, dict]]:
    keys = list(wave.keys())
    axes = [_labelled(wave[k]) for k in keys]
    runs = []
    for i, combo in enumerate(itertools.product(*axes)):
        cfg = dict(BASE)
        labels, label_map = [], {}
        for k, (lab, val) in zip(keys, combo):
            if k == "arm":
                cfg["arm"] = val
            else:
                cfg[k] = val
            labels.append(f"{k}={lab}" if k != "arm" else lab)
            label_map[k] = lab
        cfg = _flatten(cfg)

        # Step-match: mixed batches take floor(N / batch_size) steps an epoch,
        # so a geometry that moves coverage moves the step count. Hold steps,
        # not epochs. ur_every keeps ~10 radius evals per run either way.
        n_pts = mix_points(cfg["npos_list"])
        steps_per_epoch = max(1, n_pts // cfg["batch_size"])
        cfg["epochs"] = max(100, round(TARGET_STEPS / steps_per_epoch / 50) * 50)
        cfg["ur_every"] = max(10, cfg["epochs"] // 10)
        cfg["_labels"] = label_map
        runs.append((f"{i:03d}_" + "_".join(labels), cfg))
    return runs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("wave", nargs="?", help=f"one of: {', '.join(WAVES)}")
    p.add_argument("--name", default=None, help="sweep dir name (default: wave)")
    p.add_argument("--list", action="store_true", help="show waves and mixes")
    p.add_argument("--runs-per-job", type=int, default=RUNS_PER_JOB,
                   help="training runs sharing one GPU (throughput lever: the "
                        "partition is GPU-limited and a run uses ~6%% of one)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.list or not args.wave:
        print("size mixes (all sizes <= 200):")
        for k, v in SIZE_MIXES.items():
            n = mix_points(v)
            sizes = sorted({int(s) for s in v.split(",")}, reverse=True)
            print(f"  {k:<10} {len(v.split(',')):>3} envs  sizes {sizes}  "
                  f"{n / 1e3:>6.0f}k pts  {n / ARENA ** 2:>5.1%} coverage  "
                  f"{n // BASE['batch_size']:>3} steps/epoch")
        print("\nwaves:")
        for k, w in WAVES.items():
            print(f"  {k:<14} {len(build_runs(w)):>3} runs   axes: {list(w)}")
        return

    if args.wave not in WAVES:
        sys.exit(f"unknown wave {args.wave!r}; have {list(WAVES)}")

    runs = build_runs(WAVES[args.wave])
    sweep_name = args.name or args.wave
    sweep_dir = os.path.join(str(sweeps_dir()), sweep_name)
    print(f"Wave {args.wave}: {len(runs)} runs  →  {sweep_dir}")
    for name, cfg in runs:
        print(f"  {name:<58} epochs={cfg['epochs']:<6} "
              f"envs={len(cfg['npos_list'].split(','))}")
    if args.dry_run:
        print("--dry-run: nothing submitted")
        return

    os.makedirs(os.path.join(sweep_dir, "slurm"), exist_ok=True)
    for i, (run_name, cfg) in enumerate(runs):
        run_dir = os.path.join(sweep_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "meta.json"), "w") as f:
            json.dump({"index": i, "run_name": run_name, "wave": args.wave,
                       # The grid labels, not only the resolved values: a patch
                       # mix resolves to a 93-entry string that no grouped table
                       # can display.
                       "labels": cfg.get("_labels", {}),
                       "config": {k: v for k, v in cfg.items()
                                  if k != "_labels"}}, f, indent=2, default=str)

    per_job = max(1, args.runs_per_job)
    groups = [runs[k:k + per_job] for k in range(0, len(runs), per_job)]
    for g, group in enumerate(groups):
        # Runs share the GPU. Each writes its own log; the job's own stdout only
        # carries the launcher's bookkeeping, so a crashed run is still
        # attributable to its run directory rather than to a merged stream.
        body = []
        for run_name, cfg in group:
            flags = _train_flags({**cfg, "save_dir": sweep_dir,
                                  "run_name": run_name})
            log = f"{sweep_dir}/{run_name}/train.log"
            body.append(f'echo "=== launch {run_name} ==="\n'
                        f"python -u -m encoder_training.train {flags} "
                        f"> {log} 2>&1 &")
        launches = "\n".join(body)
        names = " ".join(n for n, _ in group)
        sbatch = f"""#!/bin/bash -l
#SBATCH --job-name={sweep_name}_g{g}
#SBATCH --time={SLURM["time"]}
#SBATCH --cpus-per-task={SLURM["cpus_per_task"]}
#SBATCH --ntasks=1
#SBATCH --gres={SLURM["gres"]}
#SBATCH --mem={SLURM["mem"]}
#SBATCH --partition={SLURM["partition"]}
#SBATCH --output={sweep_dir}/slurm/slurm-%j_g{g:03d}.out

module load miniforge/24.3.0-0
module load cuda/13.0.1
source activate cls
unset CUDA_VISIBLE_DEVICES
cd {REPO_ROOT}

{launches}
wait
echo "=== group {g} done: {names} ==="
for r in {names}; do
  echo "--- $r"; tail -2 {sweep_dir}/$r/train.log
done
"""
        r = subprocess.run(["sbatch"], input=sbatch, text=True,
                           capture_output=True)
        msg = r.stdout.strip() or r.stderr.strip()
        print(f"  [g{g:2d}] {len(group)} runs: {msg}")
        if r.returncode != 0:
            sys.exit(r.returncode)


if __name__ == "__main__":
    main()
