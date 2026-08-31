"""Placement-check candidate 5% and 2.5% patch mixes at every seed a wave uses.

`sample_nonoverlapping_patches` is rejection sampling and its success depends on
the seed, so `data.py` says a mix must be checked at every seed before a wave is
launched -- a mix that places at 42 and fails at 45 wastes a quarter of the runs.

The incumbent at 10% is sm50: 118 patches of 50 cells. Two ways to halve
coverage, and they are not equivalent under the Sec 10.11 account -- fewer
patches gives the env-blind spread term less of the arena, while smaller patches
shortens the separations every pairwise term can see.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0]))
sys.path.insert(0, "/orcd/home/002/jackking/cls/.claude/worktrees/"
                   "encoder-hopfield-eval-spec")

import torch

from encoder_training.data import sample_nonoverlapping_patches

ARENA = 1716
SEEDS = (42, 43, 44, 45)

CANDS = {
    # --- ~5% ---------------------------------------------------------------
    "sm50_half":  [50] * 59,      # same size, half the count
    "sm35":       [35] * 120,     # same count, smaller
    "sm70_lo":    [70] * 30,      # ~30 envs, the Sec 6.6 rule of thumb
    "sm100_lo":   [100] * 15,
    # --- ~2.5% -------------------------------------------------------------
    "sm50_q":     [50] * 30,
    "sm25":       [25] * 118,     # same count, quarter area
    "sm70_q":     [70] * 15,
    "sm100_q":    [100] * 7,
}

print(f"{'mix':12s}{'n':>5s}{'size':>6s}{'coverage':>10s}   placement by seed")
print("-" * 62)
for name, sizes in CANDS.items():
    cov = sum(s * s for s in sizes) / ARENA ** 2
    marks = []
    for sd in SEEDS:
        torch.manual_seed(sd)
        try:
            y0s, _x0s, _s = sample_nonoverlapping_patches(
                ARENA, ARENA, sizes)
            marks.append("ok" if len(y0s) == len(sizes) else "SHORT")
        except RuntimeError as exc:
            marks.append(str(exc).split()[-1])
    print(f"{name:12s}{len(sizes):5d}{sizes[0]:6d}{cov:9.2%}   "
          + "  ".join(f"{sd}:{m}" for sd, m in zip(SEEDS, marks)))
