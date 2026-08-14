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
