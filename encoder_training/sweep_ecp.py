#!/usr/bin/env python3
"""Sweeps for the ``exclude_cross_env_pairs=True`` campaign.

READ A CELL ONLY WHEN IT HAS FINISHED. Three separate conclusions in this
campaign were drawn from mid-run evaluations and all three were wrong, in the
same direction each time:

    vicreg, rate3     r_median 0 at epoch 100  ->  r_min 5-7 at epoch 1000
    graded, seed 43   r_min 0 at epoch 212     ->  r_min 11 at epoch 1000
    radius 40         alias 1.000 at ep 230    ->  alias 0.986, r_median 12.5

The bias is systematic, not bad luck. A large near radius or a strong spread
term starts slower, so any comparison at a fixed early epoch penalises exactly
the arms that need the most training, and reads their slow start as collapse.
``collect_ur --ckpt final`` is the honest view; ``peek``-style mid-run reads are
for spotting crashes and nothing else.


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
    # Big-heavy three-scale mix. mix5 said many small patches are actively bad
    # -- 93 patches with a tail down to 50 cells scored r_min 2 at a 0.989
    # ceiling, because a 50-cell patch's repulsion reaches 70 cells and the far
    # field never hears about it. This keeps the size mixing the brief asks for
    # while putting 71% of the area in 200-cell patches.
    "mixtop": _mix((200, 12), (150, 6), (100, 6)),      # 675k, 22.9%, 24 envs
    # The same shape at more coverage -- same three sizes, same ~71% of the
    # area at 200 cells, only more of the arena covered. All placed at seeds
    # 42/43/44; rejection sampling holds to 61% for this shape.
    "mixtop_hi":  _mix((200, 20), (150, 10), (100, 10)),   # 1.13M, 38.2%, 40 envs
    "mixtop_max": _mix((200, 26), (150, 14), (100, 14)),   # 1.50M, 50.8%, 54 envs
    "mixtop_xl":  _mix((200, 32), (150, 16), (100, 16)),   # 1.80M, 61.1%, 64 envs
    # 70% is where rejection sampling gives out for this shape: 77.8% failed at
    # both seeds, as did a 73% variant with a heavier small tail.
    "mixtop_xxl": _mix((200, 37), (150, 18), (100, 18)),   # 2.06M, 70.1%, 73 envs
    "mix2_lo": _mix((200, 4), (100, 12)),       # 280k,  9.5%,  16 envs
    # --- the ~10% coverage set for §5. All placement-checked at seeds
    # 42/43/44/45. Step-matching gives these ~2000 epochs, 2.4x anything in §4
    # at the same 73k steps, so check the wall clock on the first cell.
    "lo_big":    _mix((200, 7)),                          #  9.5%,  7 envs
    "lo_mixtop": _mix((200, 5), (150, 3), (100, 3)),      # 10.1%, 11 envs
    "lo_many":   _mix((100, 29)),                         #  9.9%, 29 envs
    "lo_tail":   _mix((200, 3), (150, 4), (100, 6), (70, 8)),   # 10.5%, 21 envs
    # --- the 20-100 cell set at ~10% coverage (§6). All placed at seeds 42/43
    # in well under a second. Steps/epoch ~35 and epochs ~2100, i.e. the same
    # wall clock as the lo_* set. What differs is reach: a 50-cell patch's
    # diagonal is 71 cells and a 20-cell patch's is 28, against lo_mixtop's 283
    # -- and the incumbent's decay50 is 34.5, so below about 50 cells the loss
    # never sees a pair as far apart as the decay it is being asked to produce.
    "sm100": _mix((100, 29)),                             #  9.8%,  29 envs
    "sm70":  _mix((70, 60)),                              # 10.0%,  60 envs
    "sm50":  _mix((50, 118)),                             # 10.0%, 118 envs
    "sm30":  _mix((30, 327)),                             # 10.0%, 327 envs
    "sm20":  _mix((20, 736)),                             # 10.0%, 736 envs
    "smmix": _mix((100, 15), (70, 15), (50, 20), (30, 25)),   # 10.1%, 75 envs
    "smlo":  _mix((100, 8), (70, 20), (50, 30), (30, 45)),    # 10.0%, 103 envs
    # Mix that stays above the ~50-cell reach floor §6.3 found, unlike smmix,
    # whose 30-cell tail sits under it. 62 envs, max hole 340 against sm100's
    # 475 -- the point being more of the arena's extent for the same cells.
    "smmid": _mix((100, 12), (70, 20), (50, 30)),         # 10.0%,  62 envs
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
    rate_lambda=0.0,
    rate_eps=0.5,
    # Both stay 0 in every wave from here. graded_sigma is out of scope (it
    # fits a target kernel, which is the family loss_mode=cka was excluded
    # for); input_far_tau is the labelled loophole.
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
    # W6 and W7 -- RETIRED, CANCELLED BEFORE THEY RAN. Both are built on
    # graded_sigma, which is out of scope: fitting the full pairwise kernel to
    # a target kernel is the family loss_mode=cka was excluded for. Kept so the
    # log says what was planned and why it stopped; do not relaunch.
    #
    # What survives the exclusion is the *decomposition* they were designed
    # around, since §4.4b's law does not care where the decay comes from. The
    # legal way to widen it is the near radius, which is the original binary
    # contrastive knob and was already swept in §2.2c -- just never above 0.2 of
    # the patch side. w1 shows it working: at 15x200, going from a 10-cell
    # radius to a 20-cell one took decay50 from 22 to 35 and r_min from 4.5 to
    # 6.5 with the alias ceiling unchanged. That is the same mechanism
    # graded_sigma used, reached through a knob the brief allows. See w3 and w8.
    #
    # W6 -- follow the decay. W2 put graded_sigma=50 at r_min 14 against a
    # baseline of 5, with the alias ceiling *unmoved* (0.955 against 0.954) and
    # a participation ratio of 26 -- lower than the baseline's 104. So the win
    # is the decay crossing the ceiling later, exactly the §4.2 mechanism, and
    # it is not rank: graded_sigma=10 reached a participation ratio of 272,
    # higher than the unconstrained regime's 202, and still scored r_min 3.
    #
    # Two things to establish. Where sigma turns: the bracket stopped at its own
    # top end, and the patch is only 200 wide, so a sigma that large leaves
    # almost no within-patch pair asking for separation and the ceiling should
    # eventually give. And whether coverage compounds: the ref-vs-patch
    # diagnostic put corr(distance to nearest patch, radius) at -0.47 on the
    # 200-patch encoder, so the references holding r_min down are the ones the
    # patches never reached, which is the one thing coverage fixes.
    "w6_graded_wide": {
        "arm": {
            "s50":        dict(graded_sigma=50.0),
            "s75":        dict(graded_sigma=75.0),
            "s100":       dict(graded_sigma=100.0),
            "s150":       dict(graded_sigma=150.0),
            "s75_cov45":  dict(graded_sigma=75.0, npos_list=SIZE_MIXES["mix3_45"]),
            "s150_cov45": dict(graded_sigma=150.0, npos_list=SIZE_MIXES["mix3_45"]),
        },
        "seed": [42, 43],
    },
    # W7 -- the two factors of the law at once.
    #
    #     r_pred = decay50 * sqrt(ln(1/C) / ln 2)
    #
    # graded_sigma owns decay50 and leaves C alone (§4.4); the rank terms own C
    # and, on the evidence of w2, leave the decay alone -- rate_lambda=0.3 had
    # the ceiling at 0.943 by epoch 114 where the binary baseline was at 0.988.
    # Neither reaches 21 by itself at these sigmas. Together the law says they
    # should: sigma=75 gives decay50 ~88, and 88 * sqrt(ln(1/0.90)/ln 2) = 34.
    #
    # If the product does not follow the law here, the law is where the mistake
    # is, and that is worth as much as the encoder.
    #
    # STRENGTH IS THE WHOLE THING, and the first submission of this wave had it
    # wrong. Matched at epoch 100 against the binary baseline (r_min 2, ceiling
    # 0.988):
    #     rate_lambda=0.3   r_min 2, ceiling 0.944   <- ceiling bought for free
    #     vicreg 1.0/0.1    r_min 0, ceiling 0.915
    #     rate_lambda=3     r_min 0, ceiling 0.896
    # r_median 0 means the profile is not even locally monotone, so the strong
    # settings are §3's failure mode exactly. Being pair-free is not enough to
    # keep a spread term off the neighbourhood -- it only raises the strength at
    # which the damage starts. So the grid brackets 0.1 to 1.0 rather than
    # sitting on 1.0, and vicreg is dropped at the strength that killed it.
    "w7_decay_x_ceiling": {
        "arm": {
            "s50_rate0.3":  dict(graded_sigma=50.0, rate_lambda=0.3),
            "s75_rate0.1":  dict(graded_sigma=75.0, rate_lambda=0.1),
            "s75_rate0.3":  dict(graded_sigma=75.0, rate_lambda=0.3),
            "s75_rate1":    dict(graded_sigma=75.0, rate_lambda=1.0),
            "s75_rate0.3_cov45": dict(graded_sigma=75.0, rate_lambda=0.3,
                                      npos_list=SIZE_MIXES["mix3_45"]),
        },
        "seed": [42, 43],
    },
    # W8 -- the two factors of §4.4b's law, both through knobs the brief allows.
    # This is w7's question with the excluded knob swapped out.
    #
    #   the decay   <- the near radius. w1: at 15x200, 10 -> 20 cells took
    #                  decay50 22 -> 35 and r_min 4.5 -> 6.5, ceiling unmoved.
    #                  w3 brackets it alone; here it is crossed.
    #   the ceiling <- rate_lambda. w2: 0.3 took the ceiling from the baseline's
    #                  0.946 to 0.907, the lowest reached in this regime, with
    #                  decay50 unchanged and r_min 4.5 -> 9.
    #
    # Radii are set as a fraction of the patch side rather than in cells, so a
    # 200 patch and a 100 patch each get a radius in proportion -- the fixed-vs-
    # fractional axis in w1 came out in favour of fractional at the larger
    # patches (r_min 6.5 against 4.5), which is the mixed-size case that matters
    # here. 0.2 of a 200 patch is 40 cells, the top of w3's bracket.
    "w8_rate_x_radius": {
        # No bare f0.1 arm: that cell is w1's mix2/per_env_radius_frac=0.1 and
        # is already running there.
        "arm": {
            "f0.2":            dict(per_env_radius_frac=0.20),
            "f0.3":            dict(per_env_radius_frac=0.30),
            "f0.1_rate0.3":    dict(per_env_radius_frac=0.10, rate_lambda=0.3),
            "f0.2_rate0.3":    dict(per_env_radius_frac=0.20, rate_lambda=0.3),
            "f0.3_rate0.3":    dict(per_env_radius_frac=0.30, rate_lambda=0.3),
            "f0.2_rate0.3_big": dict(per_env_radius_frac=0.20, rate_lambda=0.3,
                                     npos_list=SIZE_MIXES["mixbig"]),
        },
        "seed": [42, 43],
    },
    # W9 -- the combination, with both axes corrected by what w3 and w8 found.
    #
    # THE RADIUS TURNS EARLIER THAN w8 ASSUMED. Its 0.2 arm collapsed outright,
    # alias ceiling 0.999-1.000 and r_min 0, and w3's fixed-40 did the same:
    # once the radius is that large almost every within-patch pair counts as
    # near, nothing is left to repel, and the code goes uniform. w1 puts the
    # working value at 0.1 of the side, so this brackets 0.10 to 0.15 rather
    # than 0.2 to 0.3.
    #
    # THE MIX HAS TO BE BIG-HEAVY. mix5 (93 patches, tail to 50 cells) scored
    # r_min 2, worse than uniform 200; mix2 ties uniform 200 and is noisier.
    # `mixtop` keeps three sizes but puts 71% of the area at 200.
    #
    # rate_lambda crossed in because it owns the other factor of the law and is
    # the only thing that has moved it: 0.946 -> 0.907 at r_min 4.5 -> 7/9.
    "w9_best_combo": {
        "arm": {
            "top_f0.10":        dict(npos_list=SIZE_MIXES["mixtop"],
                                     per_env_radius_frac=0.10),
            "top_f0.15":        dict(npos_list=SIZE_MIXES["mixtop"],
                                     per_env_radius_frac=0.15),
            "top_f0.10_rate0.3": dict(npos_list=SIZE_MIXES["mixtop"],
                                      per_env_radius_frac=0.10, rate_lambda=0.3),
            "top_f0.15_rate0.3": dict(npos_list=SIZE_MIXES["mixtop"],
                                      per_env_radius_frac=0.15, rate_lambda=0.3),
            # Uniform 200 with the same treatment: the reference that says
            # whether mixing sizes is buying anything at all.
            "u200_f0.10_rate0.3": dict(npos_list=SIZE_MIXES["u200"],
                                       per_env_radius_frac=0.10, rate_lambda=0.3),
        },
        "seed": [42, 43],
    },
    # W10 -- how much rate can the decay carry?
    #
    # w2 finished with the ceiling half of the problem SOLVED and the decay half
    # untouched. rate_lambda=3 reaches an alias ceiling of 0.843, which is what
    # encoders that keep their cross-environment pairs reach (0.844-0.864) -- so
    # a term that never asks about environments can close that gap completely.
    # What it cannot do is hold the decay: decay50 14.5 against 38-40, and that
    # is the whole remaining deficit.
    #
    # The two ingredients exist separately. At the ceiling rate3 reaches, r_min
    # ~20 needs res90 ~16; rate3 has 5, but u200 at 0.1 of the side has 14. So
    # the question is where along rate_lambda the decay starts being spent, and
    # whether a larger radius pays for it. 0.3 left the decay alone (22.75
    # against a baseline 21) and 3 halved it; this brackets between.
    "w10_rate_strength": {
        "arm": {
            "top_f0.10_rate1": dict(npos_list=SIZE_MIXES["mixtop"],
                                    per_env_radius_frac=0.10, rate_lambda=1.0),
            "top_f0.15_rate1": dict(npos_list=SIZE_MIXES["mixtop"],
                                    per_env_radius_frac=0.15, rate_lambda=1.0),
            "top_f0.15_rate3": dict(npos_list=SIZE_MIXES["mixtop"],
                                    per_env_radius_frac=0.15, rate_lambda=3.0),
        },
        "seed": [42, 43],
    },
    # W11 -- the radius fraction is still climbing at the top of w9's bracket.
    #
    # On `mixtop` with rate_lambda=0.3: 0.10 -> r_min 16, 16; 0.15 -> 19, 22.
    # The second of those matches the best encoder ever trained *with* the
    # cross-environment pairs (21) on every column -- alias 0.813 against 0.814,
    # decay50 42 against 37.5. So the bracket has to go further up, and where it
    # turns is the answer to how good this regime gets.
    #
    # Two reasons to expect a turn not far above: without the rate term 0.15
    # already scores *worse* than 0.10 (6.5 against 8.5), so the rate term is
    # what makes the larger radius survivable; and on `mix2` the 0.3 fraction
    # fell to 6.5 even with rate. The third seed goes on the leaders, because
    # §2.6 puts config effects at about twice seed noise and two seeds have been
    # carrying every claim here.
    "w11_radius_top": {
        "arm": {
            "top_f0.20_rate0.3": dict(npos_list=SIZE_MIXES["mixtop"],
                                      per_env_radius_frac=0.20, rate_lambda=0.3),
            "top_f0.25_rate0.3": dict(npos_list=SIZE_MIXES["mixtop"],
                                      per_env_radius_frac=0.25, rate_lambda=0.3),
            "top_f0.15_rate0.5": dict(npos_list=SIZE_MIXES["mixtop"],
                                      per_env_radius_frac=0.15, rate_lambda=0.5),
            "top_f0.20_rate0.5": dict(npos_list=SIZE_MIXES["mixtop"],
                                      per_env_radius_frac=0.20, rate_lambda=0.5),
        },
        "seed": [42, 43],
    },
    # W12 -- a third seed on the leaders, and the same two settings on a mix
    # with more coverage. Nothing new is varied: this is the confirmation pass.
    "w12_confirm": {
        "arm": {
            "top_f0.15_rate0.3": dict(npos_list=SIZE_MIXES["mixtop"],
                                      per_env_radius_frac=0.15, rate_lambda=0.3),
            "top_f0.10_rate0.3": dict(npos_list=SIZE_MIXES["mixtop"],
                                      per_env_radius_frac=0.10, rate_lambda=0.3),
        },
        "seed": [44, 45],
    },
    # W17 -- §5 step 1. How much of §4 survives a cut to ~10% coverage?
    #
    # §4's answer used 50.8% of the arena. The coverage sweep (§4.6b) was
    # monotone and left decay50 flat, so the prediction is that 10% costs the
    # ceiling and only the ceiling -- worth about a factor 0.5 by §4.4b's law,
    # i.e. r_min around 10-13. Coming in far below that means something other
    # than the ceiling broke, which is the more interesting outcome.
    #
    # lo_mixtop keeps §4's winning shape (three sizes, ~2/3 of area at 200) at a
    # tenth of the coverage; lo_big and lo_many bracket it with uniform sets at
    # either end. Loss settings are §4's winner untouched, so this isolates
    # coverage. See docs/EXPERIMENTS_UNIQUE_RADIUS.md §5.4 for the rest.
    #
    # All five §5.3 geometries run here rather than the three the plan asked
    # for: the queue was empty and the cap is 16 GPUs, so 10 runs cost the same
    # wall clock as 6 and step 2 then tunes on a geometry that was measured
    # rather than assumed. mix2_lo is §5.3's `lo_mix2` (4x200+12x100).
    "w17_lowcov_anchor": {
        "arm": {
            "lo_mixtop": dict(npos_list=SIZE_MIXES["lo_mixtop"],
                              per_env_radius_frac=0.15, rate_lambda=0.3),
            "lo_big":    dict(npos_list=SIZE_MIXES["lo_big"],
                              per_env_radius_frac=0.15, rate_lambda=0.3),
            "lo_many":   dict(npos_list=SIZE_MIXES["lo_many"],
                              per_env_radius_frac=0.15, rate_lambda=0.3),
            "lo_mix2":   dict(npos_list=SIZE_MIXES["mix2_lo"],
                              per_env_radius_frac=0.15, rate_lambda=0.3),
            "lo_tail":   dict(npos_list=SIZE_MIXES["lo_tail"],
                              per_env_radius_frac=0.15, rate_lambda=0.3),
        },
        "seed": [42, 43],
    },
    # W18 -- §5 step 3, brought forward. Does *where* the patches sit matter?
    #
    # Ran early because w17 left 6 of the 16 GPU slots idle and the random
    # counterparts are already w17's 000/001 and 004/005 -- same loss settings,
    # same seeds, so these four cells are a paired A/B for one flag.
    #
    # The mechanism is measured, not assumed. Max distance from an arena point
    # to the nearest patch, over seeds 42-45:
    #
    #     mix          random   stratified
    #     lo_mixtop      839       461
    #     lo_many        475       279
    #     mixtop_max     192       188      <- 50.8%, §4's winner
    #
    # §4.6 put that distance at -0.47 with a reference's radius, so there is
    # room for it to matter at 10% and none at 50%. Prediction on the record:
    # stratified wins on r_min and on the alias ceiling, and leaves decay50
    # alone -- decay50 was flat across the entire coverage sweep.
    "w18_placement": {
        "arm": {
            "lo_mixtop_strat": dict(npos_list=SIZE_MIXES["lo_mixtop"],
                                    per_env_radius_frac=0.15, rate_lambda=0.3,
                                    patch_placement="stratified"),
            "lo_many_strat":   dict(npos_list=SIZE_MIXES["lo_many"],
                                    per_env_radius_frac=0.15, rate_lambda=0.3,
                                    patch_placement="stratified"),
        },
        "seed": [42, 43],
    },
    # W19 -- §5 step 2. Re-tune the two factors of the law at 10% coverage.
    #
    # Written after w17's first cells, which is why it is not the grid §5.4
    # planned. Both factors of §4.4b's law collapsed at 10%, not just the
    # ceiling the plan predicted (§5.6c): alias 0.671 -> 0.96-0.99, and decay50
    # 42 -> 23-37 when the coverage sweep above 23% had left it flat. So both
    # need re-tuning, and the bracket has to widen upward on both.
    #
    # Swept as separate axes rather than a cross, because §4.5d measured the
    # factors as composing: the radius fraction moves res90, the spread term
    # moves the ceiling, and the joint cell was predicted by the product.
    #
    # THE RUNS ARE ALLOCATED BY LEVERAGE, WHICH IS NOT EVEN. The ceiling enters
    # the law as sqrt(ln(1/C)), which is brutally sensitive as C approaches 1.
    # From the measured 10% cell (res90 14.5, C 0.985, r_min 5.5):
    #
    #     fix the ceiling alone, res90 untouched     fix the decay alone, C untouched
    #       C 0.95 -> r_min 10.1                       res90 25 -> r_min  9.5
    #       C 0.90 -> r_min 14.5                       res90 40 -> r_min 15.1
    #       C 0.80 -> r_min 21.1                       res90 50 -> r_min 18.9
    #       C 0.671 (§4's) -> r_min 28.2               (res90 50 means decay50 129)
    #
    # The ceiling alone can recover the entire §4 result. The decay cannot get
    # halfway there even at values no run has ever produced. So the spread term
    # gets six arms and the radius two.
    #
    #   rate 1/3/10  with cross-env pairs excluded, a far-apart pair is
    #                separated by the within-env repel if it shares a patch and
    #                by the env-blind spread term if not. 50.8% -> 10% removes
    #                most of what the first can reach.
    #   unif 1/3     the user's hypothesis, and §4.4 measured it beating rate on
    #                the ceiling at a *narrow* radius (0.863 vs 0.919) while
    #                losing at the wide one. This is a ceiling-limited regime
    #                now, which is the half of that trade where it won.
    #   f 0.25/0.40  §4.5e peaked at 0.15 at 22.9%; 0.40 is 80 cells on a 200
    #                patch, deliberately past where §4.5b expects it to break.
    #   fwhm 0.5     never moved in any wave (w5 was cancelled before it ran),
    #                and the one knob acting on how smoothly the *input* varies
    #                in space, which is what the far field is extrapolating from.
    #
    # Set `npos_list` from whichever geometry w17 picks before launching.
    "w19_lowcov_loss": {
        "arm": {
            **{f"rate{r:g}": dict(npos_list=SIZE_MIXES["lo_mixtop"],
                                  per_env_radius_frac=0.15, rate_lambda=r)
               for r in (1.0, 3.0, 10.0)},
            **{f"unif{u:g}": dict(npos_list=SIZE_MIXES["lo_mixtop"],
                                  per_env_radius_frac=0.15, uniformity_lambda=u)
               for u in (1.0, 3.0)},
            "unif1_rate0.3": dict(npos_list=SIZE_MIXES["lo_mixtop"],
                                  per_env_radius_frac=0.15,
                                  uniformity_lambda=1.0, rate_lambda=0.3),
            **{f"f{f:g}": dict(npos_list=SIZE_MIXES["lo_mixtop"],
                               per_env_radius_frac=f, rate_lambda=0.3)
               for f in (0.25, 0.40)},
            "fwhm0.5": dict(npos_list=SIZE_MIXES["lo_mixtop"],
                            per_env_radius_frac=0.15, rate_lambda=0.3,
                            fwhm_ratio=0.5),
        },
        "seed": [42, 43],
    },
    # W20 -- §5 step 5. Confirmation at four fresh seeds.
    #
    # Steps 1-3 left three candidates inside each other's noise: the w17
    # baseline (r_min 6.5, spread 3), w18's stratified placement (7.5, spread
    # 1) and w19's rate1 (7.0, spread 4). Two seeds cannot separate those --
    # §4.9 and §4.6b both had two-seed orderings that four seeds reversed -- so
    # this runs seeds 44-47, which no cell in §5 has used, and pools them with
    # the 42/43 cells already on disk for a six-seed read.
    #
    # `strat_rate1` is the untested corner: the two ingredients that each gained
    # about a unit came from different waves and have never been combined.
    "w20_lowcov_confirm": {
        "arm": {
            "strat_rate0.3": dict(npos_list=SIZE_MIXES["lo_mixtop"],
                                  per_env_radius_frac=0.15, rate_lambda=0.3,
                                  patch_placement="stratified"),
            "rand_rate1":    dict(npos_list=SIZE_MIXES["lo_mixtop"],
                                  per_env_radius_frac=0.15, rate_lambda=1.0),
            "strat_rate1":   dict(npos_list=SIZE_MIXES["lo_mixtop"],
                                  per_env_radius_frac=0.15, rate_lambda=1.0,
                                  patch_placement="stratified"),
        },
        "seed": [44, 45, 46, 47],
    },
    # W21 -- OUT OF BRIEF. DIAGNOSTIC ONLY. NOT AN ANSWER TO THE QUESTION.
    #
    # These runs let the spread term evaluate positions drawn from the whole
    # arena, so the encoder is no longer a 10%-coverage encoder and none of
    # these cells may be quoted as one. They exist to size a single number.
    #
    # §5.6j measured where the aliases that set r_min actually live: at 10%
    # coverage only 4.5% of them fall inside a training patch against 10.1%
    # coverage, so 95.5% sit in arena that no term in the loss ever evaluates --
    # every spread term here is computed on batch encodings and the batch is
    # training points only. That makes r_min ~7 look structural rather than
    # mistuned, and steps 1-3 all returning 5-8 from unrelated directions is
    # consistent with it.
    #
    # "Looks structural" is not a measurement. This is: give the spread term the
    # far field and nothing else -- no pair supervision, no attract, no repel,
    # the pair terms still see only the 10% -- and read how much of the gap to
    # §4's 28.5 comes back. A large jump says the wall is the loss's blindness.
    # A small one says 10% of the arena genuinely lacks the information, and the
    # answer to the brief is close to final.
    "w21_arena_spread": {
        "arm": {
            "frac0.5": dict(npos_list=SIZE_MIXES["lo_mixtop"],
                            per_env_radius_frac=0.15, rate_lambda=0.3,
                            spread_arena_frac=0.5),
            "frac2":   dict(npos_list=SIZE_MIXES["lo_mixtop"],
                            per_env_radius_frac=0.15, rate_lambda=0.3,
                            spread_arena_frac=2.0),
        },
        "seed": [42, 43],
    },
    # W22 -- the baseline at the same seed count as everything it is compared to.
    #
    # w20 put strat_rate0.3 and rand_rate1 on six seeds each and both collapsed
    # to the two-seed baseline. But that baseline is still n=2, and comparing a
    # six-seed median against a two-seed one is the mistake this campaign has
    # now made four times. These are seeds 44-47 of the plain §4 config at 10%
    # coverage, so the headline comparison is six against six.
    "w22_base_seeds": {
        "arm": {
            "rand_rate0.3": dict(npos_list=SIZE_MIXES["lo_mixtop"],
                                 per_env_radius_frac=0.15, rate_lambda=0.3),
        },
        "seed": [44, 45, 46, 47],
    },
    # W23 -- §5 step 4, the last item of the plan. Overfitting.
    #
    # The prior is weak: encoder_best and encoder_final are near-identical in
    # every w17 cell (5/5, 9/8, 6/6, 6/6) and the best epochs scatter 820-1680,
    # so there is no early peak to find. And §5.6l says the binding factor is
    # res90, which is set by pairwise structure rather than by capacity.
    #
    # Run anyway, because §5.4 listed it and an argued-away step is not a tested
    # one. At four seeds rather than the planned two: §5.6k showed the seed
    # spread here is 3-5 units, so a two-seed read could not distinguish a real
    # effect from noise either way.
    #
    # It is the one regime where overfitting is a priori plausible -- 297k
    # training points, 1.5M parameters and ~73,000 steps means each point is
    # seen ~2000 times against ~380 at 50.8% coverage -- and weight_decay has
    # sat at 1e-4 untouched through both campaigns.
    "w23_weight_decay": {
        "arm": {
            f"wd{w:g}": dict(npos_list=SIZE_MIXES["lo_mixtop"],
                             per_env_radius_frac=0.15, rate_lambda=0.3,
                             weight_decay=w)
            for w in (1e-3, 1e-2)
        },
        "seed": [44, 45, 46, 47],
    },
    # W24 -- how small can out_dim go?
    #
    # It has been 1024 since before §1 and has never been swept here. The code
    # does not use it: the participation ratio at the last epoch is 108-112 on
    # every one of the six baseline seeds, out of 1024 available, and §4.1b
    # already found rank tracking the radius without causing it. So the question
    # is where the over-provisioning stops being free, not whether it exists.
    #
    # Seeds 44-47 deliberately, because w22_base_seeds ran the same config at
    # out_dim 1024 on exactly those seeds -- that wave IS the control arm and no
    # extra runs are needed for it. Four seeds because §5.6k measured the spread
    # here at 3-5 units, which two seeds cannot see past.
    #
    # 434 is the input dimension (11^2 + 12^2 + 13^2), so 512 is the last arm
    # that is not a compression of the input and 32 is well under the measured
    # participation ratio.
    "w24_out_dim": {
        "arm": {
            f"od{d}": dict(npos_list=SIZE_MIXES["lo_mixtop"],
                           per_env_radius_frac=0.15, rate_lambda=0.3,
                           out_dim=d)
            for d in (512, 256, 128, 64, 32)
        },
        "seed": [44, 45, 46, 47],
    },
    # W25 -- how small can hidden_dim go, on top of the out_dim cut?
    #
    # Run at out_dim=64 rather than 1024, because w24 found 1024 -> 64 free
    # (r_min 7.0 -> 6.0, inside a spread of 2-3) and the user's question is the
    # compounded one. The confound that buys is real and cheap to resolve: if an
    # arm degrades here it could be the width or the interaction with the
    # narrower head, so the survivor gets re-checked at out_dim=1024 before any
    # claim is made about width alone.
    #
    # hidden_dim has been 512 with 4 layers throughout both campaigns -- about
    # 1.04M parameters, of which the three 512x512 blocks are 786k. 32 is below
    # the 108-112 participation ratio the code actually occupies, so it should
    # bind even if nothing above it does.
    "w25_hidden_dim": {
        "arm": {
            f"hd{h}": dict(npos_list=SIZE_MIXES["lo_mixtop"],
                           per_env_radius_frac=0.15, rate_lambda=0.3,
                           out_dim=64, hidden_dim=h)
            for h in (256, 128, 64, 32)
        },
        "seed": [44, 45, 46, 47],
    },
    # W26 -- the first §5 lead with a mechanism pointing up, plus the confound.
    #
    # w25 found hd128 with decay50 47.5 and res90 18.0 -- both better than §4's
    # 50.8% COVERAGE winner (42.25, 17.0) and the best in either campaign. That
    # matters because §5.6l identified the decay as the binding factor at 10%
    # coverage while every knob tried until now moved the ceiling instead.
    #
    # Narrowing the net and adding a spread term therefore push opposite
    # factors: hd128's ceiling is *worse* than the baseline's (0.984 vs 0.970),
    # which is the whole reason its r_min stops at 7. If rate can take that to
    # ~0.90 while the narrow width holds decay near 40, the law gives r_min ~15
    # rather than ~7. That is a real prediction and these arms test it.
    #
    # hd128_od1024 is the confound w25 owed: its hidden_dim was cut on an
    # already-cut head, so the decay gain could be width or interaction. This
    # separates them.
    "w26_narrow_spread": {
        "arm": {
            "hd128_rate1": dict(npos_list=SIZE_MIXES["lo_mixtop"],
                                per_env_radius_frac=0.15, rate_lambda=1.0,
                                out_dim=64, hidden_dim=128),
            "hd128_rate3": dict(npos_list=SIZE_MIXES["lo_mixtop"],
                                per_env_radius_frac=0.15, rate_lambda=3.0,
                                out_dim=64, hidden_dim=128),
            "hd128_od1024": dict(npos_list=SIZE_MIXES["lo_mixtop"],
                                 per_env_radius_frac=0.15, rate_lambda=0.3,
                                 out_dim=1024, hidden_dim=128),
        },
        "seed": [44, 45, 46, 47],
    },
    # W27 -- hidden_dim on its own, at the full out_dim=1024 head.
    #
    # w25 swept hidden_dim at out_dim=64 and the confound it flagged turned out
    # to matter enormously. hd128 scored r_min 0 at all four seeds at 100
    # references with the cut head, and 8.5 (median of 5, 7, 11, 10) with the
    # full one -- the same width, opposite verdicts. So the width sweep the
    # question actually asked for is this one, not w25.
    #
    # hd128_od1024 is already done in w26; these are the rest of the axis.
    "w27_hidden_dim_full": {
        "arm": {
            f"hd{h}_od1024": dict(npos_list=SIZE_MIXES["lo_mixtop"],
                                  per_env_radius_frac=0.15, rate_lambda=0.3,
                                  out_dim=1024, hidden_dim=h)
            for h in (256, 64, 32)
        },
        "seed": [44, 45, 46, 47],
    },
    # W28 -- four more seeds on the one §5 config that beats §4's, and on the
    # baseline it has to beat, so the comparison stays eight against eight.
    #
    # At 100 references and two draws, hd128_od1024 reads 8.0 and 6.0 against the
    # baseline's 5.0 and 5.5, with r_median 22.4 against 13.4 and decay50 47
    # against 34.6 -- the decay being the factor §5.6l identified as binding, and
    # 47 being higher than §4's 50.8% COVERAGE winner manages.
    #
    # The thing to pin down is the tail. Seven of its eight (seed x draw) cells
    # are 5-10; one is 0, from a single blown reference with p25 still 15. The
    # baseline has no zero in eight. Whether that is a rare accident or a
    # standing risk decides whether this is the new answer or a curiosity, and
    # four seeds cannot tell. Seeds 48-51 have been used by nothing.
    "w28_narrow_seeds": {
        "arm": {
            "hd128_od1024": dict(npos_list=SIZE_MIXES["lo_mixtop"],
                                 per_env_radius_frac=0.15, rate_lambda=0.3,
                                 out_dim=1024, hidden_dim=128),
            "base": dict(npos_list=SIZE_MIXES["lo_mixtop"],
                         per_env_radius_frac=0.15, rate_lambda=0.3),
        },
        "seed": [48, 49, 50, 51],
    },
    # W29 -- hd256 at the same eight seeds as the other two, since it now leads.
    #
    # At 20 references over seeds 44-51, hd128_od1024 is {5,7,11,10,7,0,8,0} --
    # median 7.0 with two zeros -- against the baseline's {5,7,7,7,6,4,5,4},
    # median 5.5 with none. hd256_od1024 is {7,8,8,9} at four seeds: the best
    # median in §5 and no zero, but it has had half the exposure of the arms it
    # is being compared with, and this campaign has reversed a four-seed reading
    # more than once. Seeds 48-51 make it eight against eight against eight.
    "w29_hd256_seeds": {
        "arm": {
            "hd256_od1024": dict(npos_list=SIZE_MIXES["lo_mixtop"],
                                 per_env_radius_frac=0.15, rate_lambda=0.3,
                                 out_dim=1024, hidden_dim=256),
        },
        "seed": [48, 49, 50, 51],
    },
    # W30 -- both cuts at once, at the points each is free alone.
    #
    # 100 references, two draws, put out_dim free to 256 (5.5 against the
    # baseline's 5.25) and hidden_dim better at 256 than at its 512 default
    # (7.0/7.5 against 5.0/5.5). The combined config is the actual answer to
    # "lower out_dim, then lower hidden_dim", and it cannot be inferred from the
    # two axes: hd128 at out_dim=64 scored 0 at every seed while each of those
    # cuts was tolerable on its own. So it gets measured.
    #
    # od256 also drops the parameter count that matters -- with hidden_dim 256
    # the head is 256x256 rather than 256x1024.
    "w30_both_cuts": {
        "arm": {
            "hd256_od256": dict(npos_list=SIZE_MIXES["lo_mixtop"],
                                per_env_radius_frac=0.15, rate_lambda=0.3,
                                out_dim=256, hidden_dim=256),
            "hd256_od512": dict(npos_list=SIZE_MIXES["lo_mixtop"],
                                per_env_radius_frac=0.15, rate_lambda=0.3,
                                out_dim=512, hidden_dim=256),
        },
        "seed": [44, 45, 46, 47],
    },
    # W31 -- uniformity at the winning geometry, at its own preferred radius.
    #
    # The campaign settled on rate_lambda, and the grounds are narrower than
    # that makes it sound. Uniformity beat rate on the ceiling at a 10-cell
    # radius (0.863 against 0.919, r_min 9 against 8) and lost at 0.15*side
    # (12.5 against 17.5). Every uniformity arm ever run sits at one of those
    # two settings -- absolute radius 10 on mix2, or frac 0.15 on mixtop and
    # lo_mixtop. It has never been run at mixtop_max, the geometry the best
    # encoder in the campaign uses, and never at frac 0.10 anywhere.
    #
    # That matters because frac 0.15 was itself chosen with rate (§4.5e). Tuning
    # the radius for one spread term and then comparing terms at that radius is
    # not a fair comparison, and the direction of the bias is known: uniformity
    # wants the radius narrower than rate does. This runs the joint cell.
    #
    # f0.10_rate0.3 is the control -- the narrower radius with the incumbent
    # term -- so a win for unif1 at f0.10 cannot be the radius alone.
    "w31_unif_at_best": {
        "arm": {
            "f0.10_unif1": dict(npos_list=SIZE_MIXES["mixtop_max"],
                                per_env_radius_frac=0.10, uniformity_lambda=1.0),
            "f0.15_unif1": dict(npos_list=SIZE_MIXES["mixtop_max"],
                                per_env_radius_frac=0.15, uniformity_lambda=1.0),
            "f0.10_rate0.3": dict(npos_list=SIZE_MIXES["mixtop_max"],
                                  per_env_radius_frac=0.10, rate_lambda=0.3),
        },
        "seed": [42, 43, 44, 45],
    },
    # W32 -- §6 step 1. Small environments (20-100 cells) at the same ~10%
    # coverage, on the hd256/od1024 architecture §5.8 settled on.
    #
    # Geometry is crossed with the near radius rather than swept at the
    # incumbent's frac 0.15, because frac 0.15 was tuned on 100-200 cell patches
    # and applying it here would confound patch size with a badly scaled radius.
    # That is the same mistake the uniformity-vs-rate comparison made (§0.4
    # note, w31), and it is avoidable for the cost of one extra axis.
    #
    # The hypothesis the cross tests: §4.5b put the *absolute* radius optimum at
    # ~20 cells and §4.5e put the *fractional* one at 0.15 of a 100-200 cell
    # side, which is 15-30 cells -- the same window. If the invariant is
    # absolute, then a fixed radius should beat the fraction by more the smaller
    # the patches get, and 20-cell patches cannot reach it at all (their whole
    # diagonal is 28).
    #
    # per_env_radius_frac=0 makes `radius` absolute for every env.
    "w32_small_geom": {
        "arm": {
            **{f"{g}_f0.15": dict(npos_list=SIZE_MIXES[g],
                                  per_env_radius_frac=0.15, rate_lambda=0.3,
                                  out_dim=1024, hidden_dim=256)
               for g in ("sm100", "sm50", "sm20", "smmix")},
            **{f"{g}_r20": dict(npos_list=SIZE_MIXES[g],
                                per_env_radius_frac=0.0, radius=20.0,
                                rate_lambda=0.3,
                                out_dim=1024, hidden_dim=256)
               for g in ("sm100", "sm50", "sm20", "smmix")},
        },
        "seed": [42, 43],
    },
    # W33 -- §6 step 2. Push the absolute radius, which w32 showed was the
    # binding knob and not patch size.
    #
    # w32, 20 references, two seeds, hd256/od1024 throughout:
    #
    #     geometry   frac 0.15   abs radius 20
    #     sm100        7.0          8.5
    #     sm50         3.5          9.0
    #     sm20         1.0          3.0
    #     smmix        5.0           -
    #
    # sm50 nearly tripled and both sm50 and sm100 at radius 20 beat the
    # lo_mixtop incumbent (7.5-8.0). The fractional radius was starving the
    # small geometries: frac 0.15 is 7.5 cells on a 50-cell patch, against the
    # ~20 §4.5b measured as the absolute optimum. Reading a size verdict off
    # frac 0.15 would have concluded "small patches fail", which is the wrong
    # answer to the question that was asked.
    #
    # 20 was the edge of w32's grid, not an optimum. §4.5b found 40 failing on
    # 200-cell patches, but these patches are smaller and the ceiling on a
    # useful radius is the patch diagonal -- 71 cells at sm50, 141 at sm100 --
    # so the two geometries should turn over at different places. sm20 is
    # dropped: its diagonal is 28 and radius 20 already covers most of it.
    "w33_radius_push": {
        "arm": {
            f"{g}_r{r}": dict(npos_list=SIZE_MIXES[g],
                              per_env_radius_frac=0.0, radius=float(r),
                              rate_lambda=0.3, out_dim=1024, hidden_dim=256)
            for g in ("sm100", "sm50") for r in (25, 30, 40)
        },
        "seed": [42, 43],
    },
    # W34 -- §6 step 3. Six seeds on the three arms that tie at the top.
    #
    # The w33 radius curve, two seeds per cell:
    #
    #     radius     frac 0.15    20     25     30     40
    #     sm50          3.5      9.0    8.5    6.5    5.0
    #     sm100         7.0      8.5    7.5    5.5    5.0
    #
    # 20 peaks both, which is where §4.5b put the absolute optimum on 200-cell
    # patches -- so the near-radius optimum is ~20 cells and does not scale with
    # patch size, and `per_env_radius_frac` was the wrong parameterisation the
    # whole time.
    #
    # sm50_r20 (9.0), sm50_r25 (8.5) and sm100_r20 (8.5) are inside each other's
    # spread at two seeds, and this campaign has reversed two-seed orderings
    # repeatedly. Seeds 44-47 take all three to six, pooled with 42/43.
    "w34_small_confirm": {
        "arm": {
            "sm50_r20":  dict(npos_list=SIZE_MIXES["sm50"],
                              per_env_radius_frac=0.0, radius=20.0,
                              rate_lambda=0.3, out_dim=1024, hidden_dim=256),
            "sm50_r25":  dict(npos_list=SIZE_MIXES["sm50"],
                              per_env_radius_frac=0.0, radius=25.0,
                              rate_lambda=0.3, out_dim=1024, hidden_dim=256),
            "sm100_r20": dict(npos_list=SIZE_MIXES["sm100"],
                              per_env_radius_frac=0.0, radius=20.0,
                              rate_lambda=0.3, out_dim=1024, hidden_dim=256),
        },
        "seed": [44, 45, 46, 47],
    },
    # W35 -- §6 step 4a. More, smaller environments, staying above the floor.
    #
    # At fixed coverage, more and smaller patches cover more of the arena's
    # *extent*. Measured, max distance from an arena point to the nearest patch
    # over seeds 42-45:
    #
    #     lo_mixtop (11 envs)  839      <- the §5 config
    #     sm100     (29 envs)  475      <- the current best
    #     sm70      (60 envs)  353
    #     smmid     (62 envs)  340
    #     sm50     (118 envs)  248
    #
    # sm100 already halved lo_mixtop's holes, which is plausibly part of §6.4's
    # consistency gain. But sm50 has the smallest holes and scores worst of the
    # three, so hole size does not decide it alone -- reach competes, and §6.3
    # put the floor near 50 cells where a patch's diagonal stops covering the
    # decay being asked for.
    #
    # sm70 is what the two constraints leave: 25% smaller holes than sm100 and
    # a 99-cell diagonal on every patch. smmid tests whether mixing adds
    # anything on top, with its smallest patch at 50 rather than smmix's 30 --
    # smmix scored 7.5 and its 30-cell tail was under the floor.
    #
    # Note the law does not predict a gain here: sm70 should land near res90 11
    # and ceiling 0.94, i.e. r_pred ~8.4 against sm100's 8.2. If it wins it will
    # be on consistency across references, the §6.4 mechanism the law is blind
    # to, which is exactly what smaller holes should buy.
    "w35_small_spread": {
        "arm": {
            "sm70_r20":  dict(npos_list=SIZE_MIXES["sm70"],
                              per_env_radius_frac=0.0, radius=20.0,
                              rate_lambda=0.3, out_dim=1024, hidden_dim=256),
            "smmid_r20": dict(npos_list=SIZE_MIXES["smmid"],
                              per_env_radius_frac=0.0, radius=20.0,
                              rate_lambda=0.3, out_dim=1024, hidden_dim=256),
        },
        "seed": [42, 43, 44, 45],
    },
    # W36 -- §6 step 4b. The two loss knobs never swept in this regime.
    #
    # repel_weight has been 1.0 since §2.2 found its optimum at 33-60 envs, and
    # has not been touched through §5 or §6. The pair sets are means, so the
    # attract:repel balance does not shift with set size -- but what the attract
    # term is being *asked* for does: an absolute radius of 20 on a 100-cell
    # patch makes 12.6% of within-env pairs attract pairs, against 7% for
    # lo_mixtop's radius 30 on 200 cells. Nearly twice the code is being pulled
    # to cosine 1, which is a reason the 2:1 balance need not still be right.
    #
    # rate_lambda 0.3 was tuned at 11 envs. sm100's ceiling (0.952) is worse
    # than sm50's (0.933) because 29 environments make a less diverse batch than
    # 118, so the spread term has more to do here than it did there. §5.6h found
    # rate>0.3 trading ceiling for decay at worse than par, but that was at 11
    # envs and is worth one arm at 29.
    "w36_sm100_loss": {
        "arm": {
            "repel2": dict(npos_list=SIZE_MIXES["sm100"],
                           per_env_radius_frac=0.0, radius=20.0,
                           rate_lambda=0.3, repel_weight=2.0,
                           out_dim=1024, hidden_dim=256),
            "repel4": dict(npos_list=SIZE_MIXES["sm100"],
                           per_env_radius_frac=0.0, radius=20.0,
                           rate_lambda=0.3, repel_weight=4.0,
                           out_dim=1024, hidden_dim=256),
            "rate1":  dict(npos_list=SIZE_MIXES["sm100"],
                           per_env_radius_frac=0.0, radius=20.0,
                           rate_lambda=1.0, out_dim=1024, hidden_dim=256),
        },
        "seed": [42, 43],
    },
    # W37 -- the control §6.1's headline is missing, and it may overturn it.
    #
    # §6.1 compared small-env configs at absolute radius 20 against the §5.8
    # incumbent at frac 0.15. §6.2 then established that radius 20 beats the
    # fraction at every patch size tested. So the comparison confounds geometry
    # with radius, and "small environments beat big ones" may be nothing more
    # than "radius 20 beats frac 0.15". lo_mixtop has never been run at radius
    # 20.
    #
    # w35 is what forces the issue. §6.4 explained sm100's win as more
    # environments giving a more diverse batch and a more consistent code, and
    # w35 falsifies that: at matched radius 20, r_min tracks patch reach and not
    # env count, in the opposite direction to the explanation --
    #
    #     sm100   29 envs, holes 475, diagonal 141   9.0
    #     sm70    60 envs, holes 353, diagonal  99   7.5
    #     sm50   118 envs, holes 248, diagonal  71   7.5
    #
    # If bigger patches are simply better at a fixed radius, lo_mixtop at radius
    # 20 should beat sm100, and §6's headline is a radius result wearing a
    # geometry costume. Six seeds, matching sm100_r20's exposure.
    "w37_lo_mixtop_r20": {
        "arm": {
            "lo_mixtop_r20": dict(npos_list=SIZE_MIXES["lo_mixtop"],
                                  per_env_radius_frac=0.0, radius=20.0,
                                  rate_lambda=0.3, out_dim=1024, hidden_dim=256),
            "lo_big_r20": dict(npos_list=SIZE_MIXES["lo_big"],
                               per_env_radius_frac=0.0, radius=20.0,
                               rate_lambda=0.3, out_dim=1024, hidden_dim=256),
        },
        "seed": [42, 43, 44, 45, 46, 47],
    },
    # W38 -- separate patch size from environment count, by varying coverage.
    #
    # §6.5 found an interior optimum at 100-cell patches / 29 environments and
    # could not tell which. At fixed coverage the two are exactly
    # anti-correlated: n = c*A/s^2. The way out is matched-coverage rows at
    # SEVERAL coverages, because the two hypotheses then name different winners
    # everywhere except at 10%, where they coincide -- which is why the 10% row
    # alone was uninformative.
    #
    #     coverage    if SIZE drives it    if COUNT drives it
    #       5%         (100, 15)            (70, 30)
    #      10%         (100, 29)            (100, 29)     <- the coincidence
    #      20%         (100, 59)            (140, 30)
    #
    # Coverage is matched to within 0.1pp inside each row, which matters because
    # coverage is the strongest lever in the campaign (§4.6b) and a 0.7pp gap
    # would bias the contrast it is meant to isolate.
    #
    # Radius is absolute 20 everywhere. §6.2 is what makes this experiment
    # possible: the radius optimum does not scale with patch size, so sizes can
    # be compared at one radius without reintroducing the fraction confound that
    # §6.1 fell into.
    #
    # The 10% row already exists -- sm50, sm70, sm100 from w32/w34/w35 -- so
    # only 140x15 is added there. Four seeds; the 5% row runs 4300 epochs at the
    # same 73k steps, so it wants the longer time limit.
    "w38_size_vs_count": {
        "arm": {
            name: dict(npos_list=_mix((size, count)),
                       per_env_radius_frac=0.0, radius=20.0,
                       rate_lambda=0.3, out_dim=1024, hidden_dim=256)
            for name, (size, count) in {
                # ~5%: size hypothesis picks s100, count hypothesis picks s70
                "c05_s50":  (50, 59),
                "c05_s70":  (70, 30),
                "c05_s100": (100, 15),
                # ~10%: completes the row; the other three already exist
                "c10_s140": (140, 15),
                # ~20%: size hypothesis picks s100, count hypothesis picks s140
                "c20_s70":  (70, 120),
                "c20_s100": (100, 59),
                "c20_s140": (140, 30),
            }.items()
        },
        "seed": [42, 43, 44, 45],
    },
    # W39 -- test §6.6's mechanism: is ~30 envs about within-env pair supply?
    #
    # §6.6 found the optimum tracks environment count (~30) rather than patch
    # size, and offered a candidate mechanism it did not test: under the
    # constraint only within-env pairs are repellable, and a batch of B points
    # over n envs holds about B^2/(2n) of them. Few envs means more usable
    # pairs; many envs means more distinct locations. ~30 would be the balance.
    #
    # batch_size moves pair supply without touching geometry, which is what
    # makes it the right probe. Pairs per step, B^2/(2n):
    #
    #                B=4096    B=8192    B=16384
    #   sm100 n=29    289k     1.16M      4.63M
    #   sm50  n=118    71k      284k      1.14M
    #
    # Two cells are matched on pair supply across different geometries, and
    # that is the sharp prediction:
    #
    #   sm100 @ 4096  (289k) should score like sm50 @ 8192  (284k) -> ~7.5
    #   sm50  @ 16384 (1.14M) should score like sm100 @ 8192 (1.16M) -> ~9.0
    #
    # If instead each geometry keeps its own score across batch sizes, pair
    # supply is not the mechanism and §6.6's candidate is dead.
    #
    # The interaction is the robust read even if the point predictions miss:
    # both arms get the same batch change, so the "bigger batch trains better"
    # confound is common to both and cancels. sm50 is pair-starved and sm100 is
    # not, so sm50 should gain more.
    #
    # B=16384 costs ~4x per step (the pair matrix is O(B^2)) at the same 73k
    # steps, so those cells want a long limit.
    "w39_batch_pairs": {
        "arm": {
            f"{g}_b{b}": dict(npos_list=SIZE_MIXES[g],
                              per_env_radius_frac=0.0, radius=20.0,
                              rate_lambda=0.3, out_dim=1024, hidden_dim=256,
                              batch_size=b)
            for g in ("sm100", "sm50") for b in (4096, 16384)
        },
        "seed": [42, 43, 44, 45],
    },
    # W40 -- how good can it get with a final gain of 100?
    #
    # Asked for directly, and it is well posed now that §0.2 has measured what
    # the gain actually does. At gain 5 the tanh never leaves its linear region
    # -- median |g*net(x)| = 0.053, no coordinate past 0.8 -- so the ramp is
    # nearly inert and the code is continuous. At gain 100 the same
    # pre-activations give |g*net(x)| ~ 1.06 at the median and ~6.7 at p99, so
    # the code genuinely saturates and becomes quasi-binary. This is the first
    # setting in the campaign where the output nonlinearity does anything.
    #
    # Two things could go wrong and both are worth seeing. Saturation kills the
    # local gradient (tanh' = 1 - tanh^2), while the gain multiplies it, so the
    # product may either stall or blow up. And a binary code changes what cosine
    # means: for +-1 codes cos = 1 - 2*hamming/D, so the loss is now shaping a
    # Hamming geometry rather than a continuous one.
    #
    # 20 and 50 are included so the answer is a curve rather than a point --
    # whether 100 sits on a trend or off a cliff is most of what is worth
    # knowing. Baseline gain 5 already has six seeds on this exact config.
    "w40_gain": {
        "arm": {
            f"gain{g:g}": dict(npos_list=SIZE_MIXES["sm100"],
                               per_env_radius_frac=0.0, radius=20.0,
                               rate_lambda=0.3, out_dim=1024, hidden_dim=256,
                               gain_end=float(g))
            for g in (20, 50, 100)
        },
        "seed": [42, 43, 44, 45],
    },
    # W41 -- how good can gain 100 get, once its knobs are re-tuned?
    #
    # w40 gave the curve on the §6 best config: gain 5 -> 9.0, 20 -> 7.5,
    # 50 -> 7.0, 100 -> 6.0. Monotone and shallow, no cliff.
    #
    # Measured saturation says where the cost comes from. Fraction of
    # coordinates past |tanh| 0.95: gain 5 -> 0.000, gain 20 -> 0.425,
    # gain 100 -> 0.522. Binarisation arrives between 5 and 20, which is exactly
    # where the drop happens; past that more gain costs little. So the price is
    # being binary, not the size of the gain, and re-tuning has to help the
    # binary code rather than avoid saturation.
    #
    # The network partly self-compensates -- at gain 100 it shrinks |net(x)| to
    # 0.0195 to hold |g*net| near 2 -- and rank survives (pr ~120), so this is
    # not collapse.
    #
    #   sm50_b4096  transfer the w39 leader. sm50 at batch 4096 scored 10.0 at
    #               gain 5, the best 10%-coverage cell in the campaign, and the
    #               gain interaction with geometry is untested.
    #   lr          the gradient through a saturated tanh is (1-tanh^2)*g, so
    #               52% of units get almost none while the rest get ~g times
    #               more. That bimodality is an argument for trying both
    #               directions, not just smaller.
    #   rate1       cosine on a binary code is 1 - 2*hamming/D, so the spread
    #               term is shaping a Hamming geometry now; its 0.3 optimum was
    #               tuned on a continuous one.
    "w41_gain100_tune": {
        "arm": {
            "sm50_b4096": dict(npos_list=SIZE_MIXES["sm50"], batch_size=4096,
                               per_env_radius_frac=0.0, radius=20.0,
                               rate_lambda=0.3, out_dim=1024, hidden_dim=256,
                               gain_end=100.0),
            "lr3e-5": dict(npos_list=SIZE_MIXES["sm100"], lr=3e-5,
                           per_env_radius_frac=0.0, radius=20.0,
                           rate_lambda=0.3, out_dim=1024, hidden_dim=256,
                           gain_end=100.0),
            "lr3e-4": dict(npos_list=SIZE_MIXES["sm100"], lr=3e-4,
                           per_env_radius_frac=0.0, radius=20.0,
                           rate_lambda=0.3, out_dim=1024, hidden_dim=256,
                           gain_end=100.0),
            "rate1": dict(npos_list=SIZE_MIXES["sm100"],
                          per_env_radius_frac=0.0, radius=20.0,
                          rate_lambda=1.0, out_dim=1024, hidden_dim=256,
                          gain_end=100.0),
        },
        "seed": [42, 43, 44, 45],
    },
    # W42 -- push the two things w39/w41 found, and combine them.
    #
    # w41: at gain 100, lr 3e-4 recovers r_min from 6.0 to 8.0, within a unit of
    # gain 5's 9.0, while lr 3e-5 gives 5.0. Higher wins, which is what the
    # saturation argument predicts -- a saturated tanh passes almost no gradient
    # (1 - tanh^2 -> 0) so the units that are stuck need a bigger step. 3e-4 was
    # the edge of that grid, so 1e-3 is the obvious next point.
    #
    # w39: sm50 at batch 4096 scores 10.0, the best 10%-coverage cell in the
    # campaign. The two winners have never been combined.
    #
    # b2048 extends the batch trend downward. For sm50 the batch moves the alias
    # ceiling monotonically -- 0.885 / 0.919 / 0.939 at 4096 / 8192 / 16384 --
    # with res90 pinned at 10.0, and sm100 shows no batch effect at all. Neither
    # pair supply (§6.6's candidate, falsified by these same runs) nor
    # overfitting (no early-peak: best-epoch fractions 0.70-0.90, best-final
    # gaps ~0) explains that, so the trend is worth mapping before theorising.
    "w42_push": {
        "arm": {
            "g100_lr1e-3": dict(npos_list=SIZE_MIXES["sm100"], lr=1e-3,
                                per_env_radius_frac=0.0, radius=20.0,
                                rate_lambda=0.3, out_dim=1024, hidden_dim=256,
                                gain_end=100.0),
            "g100_sm50b4096_lr3e-4": dict(npos_list=SIZE_MIXES["sm50"],
                                          batch_size=4096, lr=3e-4,
                                          per_env_radius_frac=0.0, radius=20.0,
                                          rate_lambda=0.3, out_dim=1024,
                                          hidden_dim=256, gain_end=100.0),
            "sm50_b2048": dict(npos_list=SIZE_MIXES["sm50"], batch_size=2048,
                               per_env_radius_frac=0.0, radius=20.0,
                               rate_lambda=0.3, out_dim=1024, hidden_dim=256),
            "sm100_b2048": dict(npos_list=SIZE_MIXES["sm100"], batch_size=2048,
                                per_env_radius_frac=0.0, radius=20.0,
                                rate_lambda=0.3, out_dim=1024, hidden_dim=256),
        },
        "seed": [42, 43, 44, 45],
    },
    # W43 -- §6.9. Does the radius optimum track the input's correlation length?
    #
    # §6.2 says the near radius wants ~20 cells and does not scale with patch
    # size, established across sizes 50-200, coverages 10-22.9%, and both
    # parameterisations. But every one of those runs had fwhm_ratio=0.25, and
    # that is the knob setting the input's own spatial scale: FWHM per module is
    # fwhm_ratio * lambda, so ~3 cells at lambda 11-13. Radius 20 is about 6x
    # that. A scale defined as a multiple of the input's scale should move when
    # the input's scale moves, so "20 absolute" is almost certainly conditional.
    #
    # The one fwhm test on record (§5.6i, 0.5 -> null) was run at frac 0.15 --
    # the radius tuned for fwhm 0.25 -- which is exactly the confound §6.1 fell
    # into and §0.3 warns about. This crosses them properly.
    #
    # Prediction, recorded before running: if the optimum tracks FWHM, then
    #
    #     fwhm 0.125 (FWHM ~1.5)  ->  best radius ~10
    #     fwhm 0.25  (FWHM ~3)    ->  best radius ~20   [already measured]
    #     fwhm 0.5   (FWHM ~6)    ->  best radius ~40
    #
    # If instead 20 wins at every fwhm, the radius is set by something else --
    # the arena, the lambdas, or the decay the metric is asking for -- and the
    # §0.3 claim can be stated unconditionally.
    #
    # fwhm 0.25 is not re-run: w32/w33 already have sm100 at radius 15, 20, 25,
    # 30 and 40, peaking at 20.
    "w43_fwhm_x_radius": {
        "arm": {
            f"fwhm{f:g}_r{r}": dict(npos_list=SIZE_MIXES["sm100"],
                                    per_env_radius_frac=0.0, radius=float(r),
                                    rate_lambda=0.3, out_dim=1024,
                                    hidden_dim=256, fwhm_ratio=f)
            for f in (0.125, 0.5) for r in (10, 20, 40)
        },
        "seed": [42, 43, 44, 45],
    },
    # W44 -- §8.1. Train the equivariant encoder's amplitudes on patches.
    #
    # The structural half is already settled: an exactly equivariant code gives
    # r_min = r_median with spread 0.0 across 40 references, and analytic
    # amplitudes reach r_min 18 with no training at all. What is open is the
    # other half -- can the campaign's contrastive loss FIND good amplitudes
    # when it only ever sees pairs inside a patch, offsets up to s*sqrt(2)?
    #
    # Sampling is the §6.7 best: sm50 (118 x 50 cells, ~10% coverage), absolute
    # radius 20, batch 4096, exclude_cross_env_pairs. Nothing here is out of
    # brief -- the model class changed, not the data.
    #
    # out_dim, hidden_dim and gain do not apply: the character table fixes the
    # output size, and any pointwise nonlinearity would break equivariance. The
    # only capacity knob is how many characters are admitted, and p_max=2 gives
    # 729 learnable amplitudes against the MLP's 571,904.
    #
    # lr is swept because 1e-4 was tuned for a 572k-parameter network and this
    # has 729 parameters in log-amplitude space, where the loss is close to a
    # kernel fit.
    #
    # Prediction on rate_lambda, recorded before running: it should HURT here.
    # The characters are near-orthogonal across positions, so the batch
    # covariance is roughly diag(a_k^2) and the coding rate is maximised by
    # flat amplitudes -- which is a delta-like kernel, exactly what the attract
    # term is trying to widen. rate 0 should beat rate 0.3.
    "w44_equivariant": {
        "arm": {
            f"lr{lr:g}_rate{r:g}": dict(
                npos_list=SIZE_MIXES["sm50"], batch_size=4096,
                per_env_radius_frac=0.0, radius=20.0,
                encoder_type="equivariant", char_p_max=2, char_m_max=120,
                lr=lr, rate_lambda=r)
            for lr in (1e-3, 1e-2) for r in (0.0, 0.3)
        },
        "seed": [42, 43, 44, 45],
    },
    # W13 -- coverage, on the winning config rather than on the bare baseline.
    #
    # This replaces w4, which swept coverage over mixes chosen before any of the
    # loss work and would have measured it against an encoder scoring 4. The
    # axis is worth keeping because §4.6 gives it a mechanism aimed at exactly
    # what `r_min` reports: the metric is the worst of 20 references, the worst
    # ones are those no training patch got near (corr -0.47 between a
    # reference's distance to the nearest patch and its radius), and coverage is
    # the only thing that fixes those.
    #
    # Same three sizes and the same ~71% of area at 200 cells throughout, so
    # coverage is the only thing moving. Step-matched, so the extra points buy
    # fewer epochs of more-varied gradient rather than more gradient.
    "w13_coverage_top": {
        "arm": {
            "cov38": dict(npos_list=SIZE_MIXES["mixtop_hi"],
                          per_env_radius_frac=0.15, rate_lambda=0.3),
            "cov51": dict(npos_list=SIZE_MIXES["mixtop_max"],
                          per_env_radius_frac=0.15, rate_lambda=0.3),
            "cov61": dict(npos_list=SIZE_MIXES["mixtop_xl"],
                          per_env_radius_frac=0.15, rate_lambda=0.3),
        },
        "seed": [42, 43],
    },
    # W14 -- uniformity in the winning configuration.
    #
    # The headline (r_min 19/22) uses `rate_lambda` for the ceiling, but that
    # was chosen when uniformity still looked like §3 said it was. It is not:
    # at lambda 1.0 it reaches an alias ceiling of 0.876 against rate0.3's
    # 0.919, with the decay held at 20.5, and scores r_min 8 -- level with the
    # coding rate and with a *lower* ceiling. Since the ceiling is the factor
    # that matters here, uniformity may go further in the same slot, and it is
    # the term the brief singled out. Same geometry and radius as the winner so
    # only the spread term differs.
    "w14_unif_combo": {
        "arm": {
            "top_f0.15_unif1":   dict(npos_list=SIZE_MIXES["mixtop"],
                                      per_env_radius_frac=0.15,
                                      uniformity_lambda=1.0),
            "top_f0.15_unif3":   dict(npos_list=SIZE_MIXES["mixtop"],
                                      per_env_radius_frac=0.15,
                                      uniformity_lambda=3.0),
            "top_f0.15_unif1_rate0.3": dict(npos_list=SIZE_MIXES["mixtop"],
                                            per_env_radius_frac=0.15,
                                            uniformity_lambda=1.0,
                                            rate_lambda=0.3),
        },
        "seed": [42, 43],
    },
    # W15 -- the top of the coverage axis, plus the seeds the claim needs.
    #
    # Coverage turned out to be the strongest lever in the campaign: at a fixed
    # geometry shape and the winning radius and rate, 22.9% -> 38.2% -> 50.8%
    # takes r_min from a 14-22 band to 22-27, with the alias ceiling falling to
    # 0.68 -- below the 0.814 of the best encoder trained *with* the
    # cross-environment pairs. It is also the axis §4.6 predicted, from the
    # -0.47 correlation between a reference's distance to the nearest patch and
    # its radius.
    #
    # COVERAGE CEILING, corrected after launch: **61.1%**, not 70%. The 70% mix
    # places at seeds 42 and 43 and fails at 44 and 45 -- identically with
    # twenty times the rejection budget, because above ~60% the constraint is
    # geometric rather than one of attempts: once the 200-cell squares are
    # scattered there is no 150-cell gap left anywhere. The reachable coverage
    # therefore depends on the seed, and this wave was launched having checked
    # only two of the four it uses. Its 70% arm survives at seeds 42/43 and is
    # reported as a 2-seed cell; 61.1% is the last step with four.
    #
    # Seeds 44/45 go on the leaders because two seeds were not enough for the
    # last headline (§4.9) and will not be enough for this one.
    "w15_coverage_top2": {
        "arm": {
            "cov70": dict(npos_list=SIZE_MIXES["mixtop_xxl"],
                          per_env_radius_frac=0.15, rate_lambda=0.3),
        },
        "seed": [42, 43, 44, 45],
    },
    "w16_coverage_seeds": {
        "arm": {
            "cov51": dict(npos_list=SIZE_MIXES["mixtop_max"],
                          per_env_radius_frac=0.15, rate_lambda=0.3),
            "cov61": dict(npos_list=SIZE_MIXES["mixtop_xl"],
                          per_env_radius_frac=0.15, rate_lambda=0.3),
        },
        "seed": [44, 45],
    },
    # W45 -- how good can gain 100 get? Knobs picked for where saturation
    # changes what they DO, not for being unswept.
    #
    # Leader after §6.8: sm50 x 118 at batch 4096, lr 3e-4, r_min 6.5 at 100
    # references against gain 5's 8.5. Only lr and geometry have been tuned
    # there. Every arm below is a single-knob delta from that leader.
    #
    # The mechanism to beat. At gain 100, 52% of coordinates sit past |tanh|
    # 0.95, so the map x -> z is close to piecewise constant: it changes only
    # where some coordinate's pre-activation crosses zero. The monotone test is
    # STRICT (`sims > beyond` in monotone_radius_per_direction), so a ray dies
    # the first time two samples land on the same value. A staircase code fails
    # that test on flat treads, with no alias involved. Three arms attack the
    # tread width and two ask whether the §6 optima moved.
    #
    #   od256 / od2048  the tread count. Cosine on a binarised code takes D+1
    #                   values, so grain ~ 2/D. §5.8a found out_dim free to cut
    #                   1024 -> 256 at gain 5, where the code is continuous and
    #                   D buys nothing. If grain is what gain 100 costs, the
    #                   SAME cut should now hurt and the doubling should pay --
    #                   a differential prediction, not a rerun.
    #   wd1e-2          weight decay was a null at gain 5 (§5.6m) for a reason
    #                   that stops applying: L2 normalisation removes scale, so
    #                   shrinking the weights of a LINEAR tanh does nothing.
    #                   Under saturation scale sets how deep into the flat
    #                   region each unit sits, so wd becomes a live knob on the
    #                   fraction binarised. The net already self-compensates in
    #                   this direction (|net| 0.075 -> 0.0195 from gain 20 to
    #                   100); this asks whether pushing further helps.
    #   lr5e-4          3e-4 -> 8.0 and 1e-3 -> 5.5 (with a zero) at 100 refs.
    #                   The optimum is inside that bracket and untested.
    #   b8192           §6.7: the batch optimum is interior and moves with
    #                   geometry -- 4096 for sm50, 8192+ for sm100. Gain 100
    #                   changes the loss surface (half the units pass almost no
    #                   gradient), so the optimum has no reason to stay put.
    #   fwhm0.5         §6.9 measured 9.5 against 9.0 at gain 5 -- a tie. A
    #                   smoother input means neighbouring positions have more
    #                   overlapping codes, which is exactly what a staircase
    #                   needs to break a tread, so the tie may not survive here.
    "w45_g100_knobs": {
        "arm": {
            name: {**dict(npos_list=SIZE_MIXES["sm50"], batch_size=4096,
                          lr=3e-4, per_env_radius_frac=0.0, radius=20.0,
                          rate_lambda=0.3, out_dim=1024, hidden_dim=256,
                          gain_end=100.0), **over}
            for name, over in (
                ("od256",   dict(out_dim=256)),
                ("od2048",  dict(out_dim=2048)),
                ("wd1e-2",  dict(weight_decay=1e-2)),
                ("lr5e-4",  dict(lr=5e-4)),
                ("b8192",   dict(batch_size=8192)),
                ("fwhm0.5", dict(fwhm_ratio=0.5)),
            )
        },
        "seed": [42, 43, 44, 45],
    },
    # W46 -- the spread term, because the diagnostics say it is the only thing
    # that can move r_min.
    #
    # Two measurements redirected this wave.
    #
    # 1. Nothing local is broken. Over ~15,000 failing rays per checkpoint, the
    #    fraction whose beating sample lies within 40 cells is 0.0% -- at gain
    #    100 and at gain 5. The near profile is monotone in every direction far
    #    past where r_min dies. So attract_lambda, fwhm smoothing and out_dim
    #    grain cannot be the lever, and od256 duly scored a null.
    #
    # 2. The ray that sets r_min dies against a far peak at d ~787-796 or
    #    ~923-929, every reference, both gains. Those are 792 = 6*132 and
    #    924 = 7*132. Computed directly on the INPUT code, its far field has a
    #    degenerate top family at cos 0.952-0.953:
    #
    #        143 = 11*13     module 12 off by -1
    #        780 = 5*156     module 11 off by -1
    #        792 = 6*132     module 13 off by -1
    #        924 = 7*132     module 13 off by +1
    #        936 = 6*156     module 11 off by +1
    #       1573 = 11*143    module 12 off by +1
    #
    #    i.e. exactly the offsets where two modules realign and the third is one
    #    cell out -- the closest the code comes to repeating short of its full
    #    1716 period. §7.3 named 132/143/156; the ones that actually bite are
    #    the higher multiples that bring the third module to +-1.
    #
    # The consequence is the point. Patches are <= 50 cells, so the largest
    # separation any pair term can see is ~71 -- every one of those offsets is
    # unreachable by attract OR repel, and the constraint makes that worse by
    # masking cross-env pairs. The far ceiling is set by the ONE env-blind term
    # in the loss. Encoders reach 0.879 against the input code's own 0.953, so
    # the spread term is already doing this work; the question is whether it is
    # being asked hard enough. rate_lambda has only ever been 0.3 (tuned on a
    # continuous code at gain 5) and 1.0 (at the untuned lr, a null).
    #
    #   rate0.1/1/3  how hard. Up should lower the ceiling until it starts
    #                costing res90, which is the other factor of §4.4b's law.
    #   eps0.25/1.0  WHICH directions. rate_eps has never been swept at all. In
    #                coding_rate_loss the gram is I + (D/(B*eps^2)) z'z, so eps
    #                sets the resolution at which a direction counts as already
    #                spread: smaller eps keeps pushing directions that larger
    #                eps has stopped caring about. lambda changes the strength,
    #                eps changes the target set.
    #
    #   wd1e-2/3e-2  w45 confirmed the prediction that weight decay stops being
    #                inert once the tanh saturates: 9/9/9/7, median 9.0 against
    #                the leader's 8.0, with the ceiling down to 0.861. It is the
    #                best single knob found at gain 100 so far. wd1e-2 here is
    #                the COMBINATION with lr 5e-4 (w45 ran it at 3e-4), and
    #                wd3e-2 asks whether the knob has more in it.
    #
    # Base is w45's lr 5e-4, which read 8.5 against the leader's 8.0 with a
    # better ceiling (0.859 vs 0.879) -- consistent with the mechanism above.
    # Every arm that helped so far helped BY lowering the ceiling, which is the
    # mechanism predicting itself.
    "w46_g100_spread": {
        "arm": {
            name: {**dict(npos_list=SIZE_MIXES["sm50"], batch_size=4096,
                          lr=5e-4, per_env_radius_frac=0.0, radius=20.0,
                          rate_lambda=0.3, out_dim=1024, hidden_dim=256,
                          gain_end=100.0), **over}
            for name, over in (
                ("rate1",    dict(rate_lambda=1.0)),
                ("rate3",    dict(rate_lambda=3.0)),
                ("eps0.25",  dict(rate_eps=0.25)),
                ("eps1.0",   dict(rate_eps=1.0)),
                ("wd1e-2",   dict(weight_decay=1e-2)),
                ("wd3e-2",   dict(weight_decay=3e-2)),
            )
        },
        "seed": [42, 43, 44, 45],
    },
    # W47 -- the ceiling is only buyable by things that are NOT loss terms.
    #
    # w46 priced the spread term along its own axis, at gain 100, 20 refs:
    #
    #   rate_lambda   0.3      1.0      3.0
    #   alias         0.879    0.839    0.780
    #   r_median      13.0     8.5      4.0
    #   r_min         8.0      5.0      1.0
    #
    # It buys the ceiling monotonically and exactly as designed. Through
    # §4.4b's law the ceiling factor gains 39% over that range while r_min
    # loses 87%, so the near field pays about three times what the ceiling
    # returns. rate_eps 0.25 lands on the same curve from the other knob
    # (0.844 / 10.0 / 6.0). The spread term is at or past its optimum at 0.3.
    #
    # There is a reason a loss term cannot do better. Patches are <= 50 cells,
    # so no pair term sees a separation beyond ~71, while the aliases that set
    # r_min sit at 792 and 924. A spread term can only reach the far field
    # by squeezing the whole representation, and it pays for that in the near
    # field it CAN see. The two knobs that lowered the ceiling without that
    # bill are not loss terms at all:
    #
    #   out_dim   256 / 1024 / 2048  ->  alias 0.911 / 0.879 / 0.848,
    #             r_min 8.0 / 8.0 / 9.0. Monotone in the ceiling across the
    #             whole range, though §5.8a found the same axis free at gain 5.
    #             More room to push aliases apart, not finer grain -- the grain
    #             hypothesis was measured and falsified (quantum 0.00009 at
    #             both gains; allowing ties changes r_min by exactly zero).
    #   wd1e-2    alias 0.861 at r_median 14.5, the best ceiling-per-near-field
    #             trade in either wave. NOT de-saturation: measured, it moves
    #             frac_sat 0.467 -> 0.459 and |g*net| 2.71 -> 2.66, i.e. not at
    #             all. The self-compensation of §6.8 is a strong attractor.
    #
    #   od4096          does the out_dim trend continue or turn over? It has to
    #                   turn over eventually -- the ceiling cannot go below what
    #                   the input code supports -- and where it does is the
    #                   useful number.
    #   od2048_wd       the two non-loss levers together. §5.8d's warning is
    #                   live: `both cuts together` cost a unit when each was
    #                   free alone, so a combination is a test, not a freebie.
    #   od2048_wd_lr    all three, including w45's lr 5e-4.
    "w47_g100_capacity": {
        "arm": {
            name: {**dict(npos_list=SIZE_MIXES["sm50"], batch_size=4096,
                          lr=3e-4, per_env_radius_frac=0.0, radius=20.0,
                          rate_lambda=0.3, out_dim=1024, hidden_dim=256,
                          gain_end=100.0), **over}
            for name, over in (
                ("od4096",       dict(out_dim=4096)),
                ("od2048_wd",    dict(out_dim=2048, weight_decay=1e-2)),
                ("od2048_wd_lr", dict(out_dim=2048, weight_decay=1e-2,
                                      lr=5e-4)),
            )
        },
        "seed": [42, 43, 44, 45],
    },
    # W48 -- is the spread term worth anything at all at gain 100?
    #
    # w46 was built to ask whether the spread term was being asked hard ENOUGH.
    # Strengthening it (rate_lambda 1/3, rate_eps 0.25) is a clear loss, which
    # was the expected half. The unexpected half is the other end: rate_eps 1.0
    # -- which flattens the log-det and so pushes LESS -- read r_min 11.0 at
    # r_median 16.5, the best gain-100 cell of the campaign, and left the
    # ceiling where it was (0.883 against 0.879).
    #
    #   rate_eps        0.25     0.5      1.0
    #   alias           0.844    0.879    0.883
    #   r_median        10.0     13.0     16.5
    #   r_min            6.0      8.0     11.0
    #
    # If weakening costs no ceiling and returns near field, the term is not
    # buying its keep at gain 100, and the incumbent 0.3/0.5 is on the wrong
    # side of its own optimum. rate0 is the limit of that and settles it.
    #
    # Caveat on record: eps1.0 is ONE seed. This wave is sized to replicate it
    # (its own 4 seeds finish in w46) while testing the limit in parallel,
    # rather than waiting a run-length to start.
    #
    #   rate0        no spread term at all. §4 established 0.3 > 0 at gain 5,
    #                on a continuous code; that is exactly the kind of transfer
    #                §6.10 has now broken three times.
    #   eps2.0       continues the weakening axis past 1.0. Distinct from rate0
    #                because the term still shapes, just at a resolution where
    #                almost every direction already counts as spread.
    #   eps1_wd      eps1.0 with the other two non-loss winners. If the spread
    #   eps1_od2048  term was suppressing the near field, these should compound
    #                rather than collide -- they act on different factors of
    #                §4.4b's law.
    "w48_g100_nospread": {
        "arm": {
            name: {**dict(npos_list=SIZE_MIXES["sm50"], batch_size=4096,
                          lr=3e-4, per_env_radius_frac=0.0, radius=20.0,
                          rate_lambda=0.3, out_dim=1024, hidden_dim=256,
                          gain_end=100.0), **over}
            for name, over in (
                ("rate0",       dict(rate_lambda=0.0)),
                ("eps2.0",      dict(rate_eps=2.0)),
                ("eps1_wd",     dict(rate_eps=1.0, weight_decay=1e-2)),
                ("eps1_od2048", dict(rate_eps=1.0, out_dim=2048)),
            )
        },
        "seed": [42, 43, 44, 45],
    },
    # W49 -- resolve the knee. w48 bracketed the optimum on both sides:
    #
    #   rate_eps      0.25    0.5     1.0     2.0     none (rate0)
    #   alias         0.869   0.879   0.880   0.946   0.990
    #   r_min          5.5     8.0    10.5     6.0     2.0
    #
    # The ceiling is FLAT from 0.25 to 1.0 and breaks upward past it. So the
    # spread term's only real job -- keeping the far field off 1.0 -- is
    # satisfied anywhere in that range, and inside it r_min is set purely by
    # res90, which improves as the term is weakened. The optimum is where the
    # ceiling is about to let go, and 1.0 is the last measured point before it
    # does. 0.5, the value every §4-§6 headline used, sits a factor of two
    # inside the safe margin and pays ~2.5 units of radius for the caution.
    #
    #   eps0.7 / eps1.4  bisect the peak. 1.4 is the more informative of the
    #                    two: if the ceiling is still ~0.88 there, the knee is
    #                    sharper than the 1.0-to-2.0 gap can resolve and the
    #                    optimum is further out than measured.
    #   eps1_rate0.5     eps and rate_lambda are not the same knob -- eps sets
    #   eps1_rate0.15    the resolution at which a direction counts as spread
    #                    (inside the log-det), rate_lambda the weight (outside
    #                    it). At eps 1.0 the ceiling has margin again, so
    #                    trading a little of it back via the weight may buy
    #                    res90 that eps alone cannot.
    "w49_g100_knee": {
        "arm": {
            name: {**dict(npos_list=SIZE_MIXES["sm50"], batch_size=4096,
                          lr=3e-4, per_env_radius_frac=0.0, radius=20.0,
                          rate_lambda=0.3, out_dim=1024, hidden_dim=256,
                          gain_end=100.0), **over}
            for name, over in (
                ("eps0.7",       dict(rate_eps=0.7)),
                ("eps1.4",       dict(rate_eps=1.4)),
                ("eps1_rate0.5", dict(rate_eps=1.0, rate_lambda=0.5)),
                ("eps1_rate.15", dict(rate_eps=1.0, rate_lambda=0.15)),
            )
        },
        "seed": [42, 43, 44, 45],
    },
    # W50 -- the control that decides how w45-w49 get written down.
    #
    # At 100 references, two draws, the retuned gain-100 config reaches r_min
    # 9.0 with a floor of 7 against the §6.8 leader's 6.5 / 5, and §6.7's best
    # GAIN 5 config scores 8.5 / 7. Read naively that says gain 100 is free.
    #
    # It is not a fair comparison. Every §4-§6 config, the gain-5 one included,
    # was tuned at rate_eps 0.5 -- the knob was never swept at any gain. If
    # gain 5 also gains ~2.5 units from eps 1.0, the gap §6.8 measured is
    # intact and all w45-w49 did was move both ends of it.
    #
    # So: the winning spread setting, at gain 5, same geometry, same seeds.
    # Two arms because eps and rate_lambda were tuned jointly at gain 100 and
    # the joint setting may not be the one that transfers.
    "w50_g5_control": {
        "arm": {
            name: {**dict(npos_list=SIZE_MIXES["sm50"], batch_size=4096,
                          lr=1e-4, per_env_radius_frac=0.0, radius=20.0,
                          rate_lambda=0.3, out_dim=1024, hidden_dim=256,
                          gain_end=5.0), **over}
            for name, over in (
                ("g5_eps1",       dict(rate_eps=1.0)),
                ("g5_eps1_rate.5", dict(rate_eps=1.0, rate_lambda=0.5)),
            )
        },
        "seed": [42, 43, 44, 45],
    },
    # W51 -- was 73,000 steps ever enough?
    #
    # TARGET_STEPS has been 73,000 since §1 and has never been varied in the
    # entire campaign. Every wave step-MATCHES to it, which makes arms
    # comparable and makes the level itself invisible. Three things say it may
    # be short:
    #
    #   * best-epoch fractions run 0.70-0.90 with best-final gaps ~0 (§6.7), so
    #     nothing peaks early and decays -- the runs end while still improving.
    #   * lr 3e-4 beats 1e-4 at gain 100 (§6.8) and 5e-4 is level with it. A
    #     model that wants a bigger step is usually a model that has not taken
    #     enough of them.
    #   * §4.4b's law over-predicts every gain-100 arm by 2-3 cells, one-signed.
    #     Under-training would look exactly like that: res90 and the ceiling are
    #     both reached before the worst DIRECTION is cleaned up.
    #
    # This is the one wave that deliberately breaks the step match, so it is
    # not comparable to anything else in §4-§6 on equal compute -- which is the
    # point. If 3x buys nothing, 73,000 is settled for the whole campaign and
    # that is worth knowing on its own.
    "w51_steps": {
        "arm": {
            name: {**dict(npos_list=SIZE_MIXES["sm50"], batch_size=4096,
                          lr=3e-4, per_env_radius_frac=0.0, radius=20.0,
                          rate_lambda=0.5, rate_eps=1.0, out_dim=1024,
                          hidden_dim=256, gain_end=100.0), **over}
            for name, over in (
                ("steps2x", dict(_step_scale=2.0)),
                ("steps3x", dict(_step_scale=3.0)),
            )
        },
        "seed": [42, 43, 44, 45],
    },
    # W52 -- the attract term asks for a plateau; the metric measures a slope.
    #
    # `mse_attract_repel` pulls EVERY pair inside radius 20 to cosine 1 and
    # pushes everything else to 0. That target is a step function, and `r_min`
    # is the length over which cosine STRICTLY DECREASES. A model that fitted
    # the attract term perfectly would be flat out to 20 and score r_min 0.
    # The measured profile is nothing like it -- res90 10, decay50 26 -- so the
    # encoder scores well precisely by FAILING to fit the target it is given.
    #
    # The graded target that would fix this properly is `graded_sigma`, which
    # is out of scope (it fits a target kernel, the family cka was excluded
    # for). But the balance between the plateau demand and the push away from
    # it is `attract_lambda`, and that has been 2.0 since §1 and has NEVER been
    # swept -- not in one arm of §1-§6. It was inherited, not chosen, and never
    # chosen for THIS metric.
    #
    # Prediction: weakening attract raises res90 and r_min, until it is too
    # weak to hold the near field together at all -- the same knee shape the
    # spread term turned out to have (§6.10g). 4.0 is included because an
    # untested axis deserves both directions.
    #
    # fwhm is here because it sets the INPUT code's own far ceiling, measured
    # directly: 0.9533 at fwhm 0.25 and 0.9878 at 0.5, since a wider bump means
    # a one-cell module offset costs less. Less smoothing lowers the ceiling
    # the encoder has to beat and costs res90 -- the same trade again. §6.9
    # priced it at the OLD spread setting and found 0.5 a tie and 0.125 bad;
    # eps 1.0 has since bought res90 back, which may be exactly the room a
    # smaller fwhm needs.
    "w52_attract_fwhm": {
        "arm": {
            name: {**dict(npos_list=SIZE_MIXES["sm50"], batch_size=4096,
                          lr=3e-4, per_env_radius_frac=0.0, radius=20.0,
                          rate_lambda=0.5, rate_eps=1.0, out_dim=1024,
                          hidden_dim=256, gain_end=100.0), **over}
            for name, over in (
                ("att0.5",   dict(attract_lambda=0.5)),
                ("att1",     dict(attract_lambda=1.0)),
                ("att4",     dict(attract_lambda=4.0)),
                ("fwhm0.15", dict(fwhm_ratio=0.15)),
                ("fwhm0.5",  dict(fwhm_ratio=0.5)),
            )
        },
        "seed": [42, 43, 44, 45],
    },
    # W53 -- find attract_lambda's knee, and test whether it transfers.
    #
    # w52: 0.5 / 1.0 / 2.0 / 4.0 -> r_min 5.5 / 9.0 / 11.0 / 11.5 and r_median
    # 9.0 / 13.0 / 16.8 / 17.9. Monotone THROUGH the incumbent, and 4.0 is the
    # tightest arm measured (11 11 12 12). So 2.0 -- the value in every
    # headline config since §1, never swept in any arm -- is below its own
    # optimum. The gap between 2.0 and 4.0 is small, so the knee is near, and
    # an unlocated knee is half a result.
    #
    # The prediction that motivated w52 was WRONG in its direction: the attract
    # term asks for a plateau, so weakening it was supposed to let a decreasing
    # profile emerge. Weakening it collapses the near field instead (res90 with
    # it), because at res90 10 against radius 20 the network never approaches
    # the plateau -- attract is what HOLDS THE NEAR FIELD UP. It is another
    # res90 knob, not a shape knob, which is what every knob in §6.10 turned
    # out to be.
    #
    #   att8 / att16   where does holding the near field up start costing the
    #                  ceiling? Every other axis in this campaign has a knee.
    #   g5_att4        the transfer test. rate_eps 1.0 won at gain 100 and LOST
    #                  three units at gain 5 (§6.10i), so "it won at gain 100"
    #                  is not evidence about anywhere else. attract_lambda sits
    #                  in every §1-§6 config, so this one matters far beyond
    #                  the gain-100 brief.
    "w53_attract_knee": {
        "arm": {
            "att8":  dict(npos_list=SIZE_MIXES["sm50"], batch_size=4096,
                          lr=3e-4, per_env_radius_frac=0.0, radius=20.0,
                          rate_lambda=0.5, rate_eps=1.0, out_dim=1024,
                          hidden_dim=256, gain_end=100.0, attract_lambda=8.0),
            "att16": dict(npos_list=SIZE_MIXES["sm50"], batch_size=4096,
                          lr=3e-4, per_env_radius_frac=0.0, radius=20.0,
                          rate_lambda=0.5, rate_eps=1.0, out_dim=1024,
                          hidden_dim=256, gain_end=100.0, attract_lambda=16.0),
            # gain 5, at ITS optimum spread setting, with attract moved
            "g5_att4": dict(npos_list=SIZE_MIXES["sm50"], batch_size=4096,
                            lr=1e-4, per_env_radius_frac=0.0, radius=20.0,
                            rate_lambda=0.3, out_dim=1024, hidden_dim=256,
                            gain_end=5.0, attract_lambda=4.0),
        },
        "seed": [42, 43, 44, 45],
    },
    # W54 -- how far up does attract go, and is it really the RATIO?
    #
    # attract_lambda, never swept in §1-§6, is monotone from 0.5 to 16 with the
    # alias ceiling FLAT across the whole 32x range:
    #
    #   attract   0.5    1.0    2.0    4.0    8.0    16.0
    #   r_min     5.5    9.0   11.0   11.5   11.0   12.0
    #   r_median  9.2   12.8   16.8   17.8   17.0   19.5
    #   alias    .874   .877   .879   .871   .885   .879
    #
    # Unlike every other knob in this campaign it is not trading against the
    # far field. §6.10 explains why it can be free: the repel term only ever
    # sees within-patch pairs (<= ~71 cells) because the constraint masks the
    # cross-env ones, while the aliases that set r_min sit at 792 and 924 and
    # belong entirely to the spread term. So repulsion is doing local work that
    # may not be worth its weight, and raising attract with repel_weight fixed
    # at 1.0 is mostly a way of REMOVING it.
    #
    #   att32 / att64  where it saturates. r_median 19.5 is already the highest
    #                  in the campaign and the ceiling has not moved.
    #   rep0.25        the mechanism test. attract 2.0 with repel 0.25 is the
    #                  same 8:1 ratio as att8 with repel 1.0. If it scores like
    #                  att8, the ratio is what matters and the finding is
    #                  "repulsion is over-weighted". If it does not, the
    #                  absolute scale matters too -- which it might, since both
    #                  weights are relative to the spread term.
    "w54_attract_far": {
        "arm": {
            name: {**dict(npos_list=SIZE_MIXES["sm50"], batch_size=4096,
                          lr=3e-4, per_env_radius_frac=0.0, radius=20.0,
                          rate_lambda=0.5, rate_eps=1.0, out_dim=1024,
                          hidden_dim=256, gain_end=100.0), **over}
            for name, over in (
                ("att32",   dict(attract_lambda=32.0)),
                ("att64",   dict(attract_lambda=64.0)),
                ("rep0.25", dict(attract_lambda=2.0, repel_weight=0.25)),
            )
        },
        "seed": [42, 43, 44, 45],
    },
    # W55 -- the first wave selected on the NAVIGATION objective rather than on
    # `r_min`. See docs/EXPERIMENTS_HOPFIELD_PROBE.md Sec 10.
    #
    # What changed: continuous reach is set by the rate of FAR-FIELD pairs above
    # cosine ~0.25, because a goal dies when one co-stored competitor crosses
    # that line (Sec 10.3), and the competitors sit ~370 cells away. `r_min` is
    # `res90 * sqrt(ln(1/C)/ln(1/0.9))`, so it prices res90 and the ceiling as
    # comparable factors; navigation reads the ceiling almost alone, since the
    # Gram-Schmidt basis only needs one-cell neighbours. res90 is a floor at
    # roughly 8 cells, not a quantity to maximise.
    #
    # That reverses the campaign's last three waves. On the alias rate the
    # attract axis is monotone the WRONG way from the incumbent 2.0 -- 2.0 gives
    # 0.0088 and 8/16/32/64 give 0.0123/0.0170/0.0269/0.0610 -- so w52-w54
    # walked uphill for three waves. Level 6 (`w49 eps1_rate0.5`, attract 2.0)
    # probes at continuous reach 0.931 against level 7's 0.806 (Sec 10.8).
    #
    # Every arm here is level 6 with ONE thing moved, so the wave reads directly
    # against `w49_g100_knee/*_eps1_rate0.5`.
    #
    #   g300_eps1    Is training at gain 300 better than training at 100 and
    #                raising gain at inference? The latter gives (far 0.0057,
    #                res90 8); this is the trained version of the same operating
    #                point, spread left alone -- the transfer test.
    #   g300_eps2    Sec 6.10i: a high gain does part of the spread term's job,
    #                so the term wants relaxing as gain rises (`rate_eps` 1.0 is
    #                right at gain 100 and *hurts* at gain 5). At gain 300 it
    #   g300_rate.25 should want relaxing further. Two arms because eps and
    #                rate_lambda were tuned jointly and may not transfer apart.
    #   rep0.1       repel_weight has only ever been sampled at 0.25, 1.0, 2.0
    #                and 4.0, and the helpful direction is down -- rep0.25
    #                already beats the level-7 headline on the alias rate
    #                (0.0106 against 0.0170).
    #   att0.25      The attract axis is covered from 0.5 to 64 across w52-w54,
    #                all sharing this exact base. Below 0.5 is the only untested
    #                part, and it is the direction the alias rate wants. Sec
    #                6.11 warns that weakening attract collapses res90, so this
    #                arm is expected to fail the floor -- it is here to locate
    #                the edge, not to win.
    #   sm30         327 patches of 30 cells at the same 10% coverage. The
    #                env-blind spread term is the only term with any far-field
    #                purchase (Sec 5.6j) and it sees only batch encodings, so
    #                more patches means it samples more of the arena. Sec 6.3
    #                rejected small patches because a 30-cell patch cannot
    #                supply pairs as far apart as the decay `r_min` wants -- an
    #                argument about res90, which navigation prices very
    #                differently. This is the legal version of the arena-spread
    #                diagnostic (Sec 10.6 item 4), whose ceiling is the best
    #                ever measured.
    "w55_nav_objective": {
        "arm": {
            name: {**dict(npos_list=SIZE_MIXES["sm50"], batch_size=4096,
                          lr=3e-4, per_env_radius_frac=0.0, radius=20.0,
                          rate_lambda=0.5, rate_eps=1.0, out_dim=1024,
                          hidden_dim=256, gain_end=100.0), **over}
            for name, over in (
                ("g300_eps1",    dict(gain_end=300.0)),
                ("g300_eps2",    dict(gain_end=300.0, rate_eps=2.0)),
                ("g300_rate.25", dict(gain_end=300.0, rate_lambda=0.25)),
                ("rep0.1",       dict(repel_weight=0.1)),
                ("att0.25",      dict(attract_lambda=0.25)),
                ("sm30",         dict(npos_list=SIZE_MIXES["sm30"])),
            )
        },
        "seed": [42, 43, 44, 45],
    },
    # W56 -- combinations, and the design target is now a NUMBER.
    #
    # Every arm in w45-w55 is Level 6 with exactly one knob moved. That was the
    # right way to find the axes and it is the wrong way to finish: the two that
    # work are independent mechanisms and have never been combined.
    #
    # What w55 and the probe established (docs/EXPERIMENTS_HOPFIELD_PROBE.md
    # Sec 10.9-10.10):
    #
    #   * reach peaks at res90 ~7 and falls off both sides. attract_lambda 0.5
    #     lands there (cont 0.987); 0.25 overshoots to res90 5 and gives 0.974
    #     with |err| 14.4 deg against 7.8 -- retrieval saturates and the
    #     direction readout degrades, the same signature as too much gain.
    #   * sm30 (327 patches of 30 cells, same 10% coverage) is the only arm that
    #     improves on Level 6 on BOTH axes at once: alias 0.0060 at res90 8
    #     against 0.0082 at res90 10. Different mechanism from attract -- the
    #     env-blind spread term sees only batch encodings, so more patches means
    #     more of the arena reaches the one term with far-field purchase.
    #   * training at gain 300 is much worse than training at 100 (alias 0.0140
    #     against 0.0056 at the same operating point), and repel_weight down is
    #     a null. Both are closed.
    #
    # So: combine attract and patch count, and use attract as the trim that
    # lands the pair at res90 ~7. sm30 alone sits at 8, attract 0.5 alone at 7,
    # so the combination will overshoot and the higher-attract variants are the
    # ones expected to win -- which is why 0.75 and 1.0 are here rather than
    # more aggressive settings.
    #
    #   a0.5_sm30 / a0.75_sm30 / a1_sm30   the combination, three trims
    #   a0.75                              fills the untested 0.5-1.0 gap alone
    #   sm20                               736 patches of 20 cells; pushes the
    #                                      count axis past sm30. Sec 6.3 called
    #                                      20 cells unusable, on a res90
    #                                      argument -- res90 is a floor near 5
    #                                      for navigation, not a maximand.
    #   a0.5_rate1                         attract 0.5 with the spread term
    #                                      doubled. rate0 has alias 0.2059
    #                                      against 0.004-0.06 for everything
    #                                      else, so the spread term does nearly
    #                                      all far-field suppression and its
    #                                      strength has never been tuned at low
    #                                      attract.
    "w56_nav_combos": {
        "arm": {
            name: {**dict(npos_list=SIZE_MIXES["sm50"], batch_size=4096,
                          lr=3e-4, per_env_radius_frac=0.0, radius=20.0,
                          rate_lambda=0.5, rate_eps=1.0, out_dim=1024,
                          hidden_dim=256, gain_end=100.0), **over}
            for name, over in (
                ("a0.5_sm30",  dict(attract_lambda=0.5,
                                    npos_list=SIZE_MIXES["sm30"])),
                ("a0.75_sm30", dict(attract_lambda=0.75,
                                    npos_list=SIZE_MIXES["sm30"])),
                ("a1_sm30",    dict(attract_lambda=1.0,
                                    npos_list=SIZE_MIXES["sm30"])),
                ("a0.75",      dict(attract_lambda=0.75)),
                ("sm20",       dict(npos_list=SIZE_MIXES["sm20"])),
                ("a0.5_rate1", dict(attract_lambda=0.5, rate_lambda=1.0)),
            )
        },
        "seed": [42, 43, 44, 45],
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
        # `_step_scale` deliberately BREAKS the step match, for the one wave
        # that asks whether 73,000 steps was ever enough. Underscore-prefixed,
        # so it never reaches the CLI.
        n_pts = mix_points(cfg["npos_list"])
        steps_per_epoch = max(1, n_pts // cfg["batch_size"])
        target = TARGET_STEPS * float(cfg.pop("_step_scale", 1.0))
        cfg["epochs"] = max(100, round(target / steps_per_epoch / 50) * 50)
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
    p.add_argument("--time", default=SLURM["time"],
                   help="wall clock. The default suits an arm at ~17 "
                        "epochs/min; uniformity runs at ~11 and needs more. A "
                        "running job's limit cannot be raised, so it has to be "
                        "right at submission.")
    p.add_argument("--only", default=None,
                   help="resubmit just the runs whose name contains this. For "
                        "relaunching the few cells of a wave that died or were "
                        "fixed, without disturbing the ones still running.")
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
    if args.only:
        runs = [(n, c) for n, c in runs if args.only in n]
        if not runs:
            sys.exit(f"--only {args.only!r} matched no run in {args.wave}")
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
    for _pos, group in enumerate(groups):
        # Name the job after the first run's own index, not its position in the
        # submitted list -- under --only those differ, and a job called g0 that
        # is really run 008 makes every squeue-to-directory mapping wrong.
        g = int(group[0][0].split("_", 1)[0])
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
#SBATCH --time={args.time}
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
