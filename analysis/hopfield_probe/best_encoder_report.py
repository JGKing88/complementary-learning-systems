"""Assemble the final answer: which att0.5 seed, its config, and its eval.

Pulls everything from the archive and the checkpoint rather than from notes, so
the numbers in the write-up are the ones on disk.
"""
from __future__ import annotations

import glob
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0]))
sys.path.insert(0, "/orcd/home/002/jackking/cls/.claude/worktrees/"
                   "encoder-hopfield-eval-spec")

import numpy as np
import torch

ARCH = "/orcd/pool/003/jackking/cls_runs/results/hopfield_probe/20260827"
W52 = "/orcd/pool/003/jackking/cls_runs/sweeps/w52_attract_fwhm"
SEEDS = (42, 43, 44, 45)
DRAWS = [("probe seed 0", "attlow_g100"), ("probe seed 1", "att0.5_ps1"),
         ("probe seed 2", "att0.5_ps2")]


def load(arm_dir, label):
    for f in glob.glob(f"{ARCH}/{arm_dir}/*.json"):
        if "manifest" in f:
            continue
        r = json.load(open(f))
        if (r.get("header", {}).get("label")
                or os.path.basename(f)[:-5]) == label:
            return r
    return None


def mo(d, key):
    v = d.get(key)
    return v.get("mean") if isinstance(v, dict) else v


print("=" * 74)
print("CONTINUOUS REACH, K=5, s=1, beta = gain = 100")
print("=" * 74)
print(f"{'':14s}" + "".join(f"{'s' + str(s):>9s}" for s in SEEDS)
      + f"{'median':>10s}")
print("-" * 62)
per_seed = {s: [] for s in SEEDS}
for name, d in DRAWS:
    row = []
    for s in SEEDS:
        r = load(d, f"att0.5-s{s}")
        v = (mo(r["test_d"]["k"]["5"]["1"]["continuous"]["scalars"],
                "reach_rate") if r else None)
        row.append(v)
        if v is not None:
            per_seed[s].append(v)
    cells = "".join(f"{v:9.3f}" if v is not None else f"{'-':>9s}"
                    for v in row)
    ok = [v for v in row if v is not None]
    print(f"{name:14s}{cells}{np.median(ok):10.3f}")
print("-" * 62)
means = {s: float(np.mean(v)) for s, v in per_seed.items() if v}
print(f"{'mean of 3':14s}"
      + "".join(f"{means[s]:9.3f}" for s in SEEDS)
      + f"{np.mean(list(means.values())):10.3f}")
best = max(means, key=means.get)
print(f"\nbest seed: {best}  (mean {means[best]:.3f} across three scaffolds)")

print("\n" + "=" * 74)
print(f"FULL EVAL, seed {best}, probe seed 0")
print("=" * 74)
r = load("attlow_g100", f"att0.5-s{best}")
bc = r["test_bc"]["k"]["5"]["per_step"]
ta = r["test_a"]["k"]["5"]["per_step"]["1"]
td = r["test_d"]["k"]["5"]["1"]
print(f"  mean |angle error|        {mo(bc['1']['grid']['scalars'], 'abs_err_mean'):8.2f} deg")
print(f"  acc within 45 deg         {mo(bc['1']['grid']['scalars'], 'acc45'):8.3f}")
print(f"  acc within 45, continuous {mo(bc['1']['continuous']['scalars'], 'acc45'):8.3f}")
print(f"  exact retrieval           {mo(ta['scalars'], 'exact_frac'):8.3f}")
print(f"  basin radius (r_exact_95) {mo(ta['scalars'], 'r_exact_95'):8.2f} cells")
print(f"  reach, discrete           {mo(td['discrete']['scalars'], 'reach_rate'):8.3f}")
print(f"  reach, continuous         {mo(td['continuous']['scalars'], 'reach_rate'):8.3f}")
print(f"  mean steps to goal        {mo(td['continuous']['scalars'], 'mean_steps'):8.2f}")
print(f"  acc45 at s=15             {mo(bc['15']['grid']['scalars'], 'acc45'):8.3f}")

print("\n  load curve (dead-goal fraction, constant env subset):")
for K in ("1", "3", "5", "10", "20"):
    kd = r["test_d"]["k"].get(K)
    if not kd:
        continue
    v = kd["1"]["continuous"]["scalars"]["reach_rate"]["values"]
    m = min(int(K), r["config"].get("n_score_envs", 5))
    sel = [x for i, x in enumerate(v) if (i % m) in (0, 1, 2)]
    print(f"    K={K:<3s} {np.mean([x < 0.5 for x in sel]):.2f}")

print("\n" + "=" * 74)
print("TRAINING CONFIG")
print("=" * 74)
d = sorted(glob.glob(f"{W52}/*_att0.5_seed={best}"))[0]
print(f"  {d}/encoder_final.pt\n")
meta = json.load(open(os.path.join(d, "meta.json")))
cfg = meta.get("config", meta)
keys = ["encoder_type", "lambdas", "out_dim", "hidden_dim",
        "num_hidden_layers", "npos_list", "radius", "per_env_radius_frac",
        "attract_lambda", "repel_weight", "rate_lambda", "rate_eps",
        "loss_mode", "exclude_cross_env_pairs", "single_env_batch",
        "lazy_codes", "lr", "batch_size", "weight_decay", "epochs",
        "fwhm_ratio", "gain_start", "gain_end", "seed"]
for k in keys:
    if k in cfg:
        v = cfg[k]
        if k == "npos_list" and isinstance(v, str) and len(v) > 40:
            n = len(v.split())
            v = f"{n} patches ({v.split()[0]} cells each)"
        print(f"  {k:24s} {v}")

ck = torch.load(os.path.join(d, "encoder_final.pt"), map_location="cpu",
                weights_only=False)
sd = ck.get("model_state_dict") or ck.get("state_dict") or {}
n = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
print(f"\n  parameters               {n / 1e6:.3f}M")
print(f"  stored gain              {ck.get('gain')}")
ur = ck.get("unique_radius")
if isinstance(ur, dict):
    print(f"  r_min (20-ref, stored)   {ur.get('r_min')}")
if "sizes" in ck:
    s = ck["sizes"]
    print(f"  coverage                 "
          f"{sum(x * x for x in s) / 1716 ** 2:.1%}  ({len(s)} patches)")
