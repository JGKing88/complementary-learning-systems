"""Does the trunk recall each held-out arena's goal, without views or policy?

    python -m analysis.idea3.synth_val_arenas --ckpt <pt> [--world <world.json>]

Synthetic lifetimes (``training/synth_decode.py``) over a run's own held-out
arenas (their scaffold places and goals, from ``world.json``), in the
protocol's order. No views and no policy: the walker is the generator's. Per
arena, the displacement head's direction error at the first step of a trial
in an arena found before (recall from a fresh start) and over all steps where
the goal was known. If an arena fails here, the memory confuses it; if it
succeeds here but fails in the continual protocol, the policy or the views are
to blame.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch

from hopfield_nav.evaluation.checkpoint_io import cfg_from_checkpoint, load_agent
from hopfield_nav.training.synth_decode import Synth
from hopfield_nav.world.spec import WorldSpec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--world", default=None)
    p.add_argument("--iters", type=int, nargs="+", default=[2, 10])
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--cap", type=int, default=50, help="steps per trial")
    p.add_argument("--prefind", choices=["pursue", "random"], default="pursue",
                   help="walker before an arena's goal is found: heads for it "
                        "after a short wander (pursue, leaks the goal) or keeps "
                        "random-walking until it touches it by chance (random)")
    p.add_argument("--order", type=int, nargs="+", default=None,
                   help="introduce the held-out arenas in this order (indices into "
                        "base_val); results stay keyed by arena, so a reversed order "
                        "separates an arena's own difficulty from its position")
    a = p.parse_args()
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = cfg_from_checkpoint(ck["config"])
    agent = load_agent(cfg, ck["agent_state_dict"], 1024, device)
    world = a.world or os.path.join(os.path.dirname(a.ckpt), "world.json")
    specs = WorldSpec.read(world).split.base_val
    if a.order is not None:
        if sorted(a.order) != list(range(len(specs))):
            p.error(f"--order must be a permutation of 0..{len(specs) - 1}")
        specs = [specs[i] for i in a.order]
    ids = list(a.order) if a.order is not None else list(range(len(specs)))
    offs = np.array([s.offset for s in specs])
    goals = np.array([s.goal for s in specs])
    synth = Synth(cfg, device)
    torch.manual_seed(0)
    res = {}
    for it in a.iters:
        with torch.no_grad():
            obs, tgt, msk, kn, fst, alive, arena = synth.lifetime(
                a.batch, len(specs), it, offsets=offs, goals=goals, return_arena=True,
                cap=a.cap, prefind=a.prefind)
            _, _, _, _, feats = agent(obs, return_features=True)
            pred = agent.goal_disp_head(feats)
        per = {}
        for j in range(len(specs)):
            row = {}
            for name, m in (("trial_start", (fst > 0) & (arena == j)),
                            ("known", (msk > 0) & (arena == j))):
                y, pr = tgt[m] * 10, pred[m] * 10
                far = y.norm(dim=-1) >= 2
                if far.sum() == 0:
                    continue
                ang = torch.atan2(pr[far, 1], pr[far, 0]) - torch.atan2(y[far, 1], y[far, 0])
                err = (torch.remainder(ang + math.pi, 2 * math.pi) - math.pi).abs() * 180 / math.pi
                row[name + "_median_deg"] = float(err.median())
                row[name + "_n"] = int(far.sum())
            per[f"arena{ids[j]}"] = dict(row, position=j)
        res[f"iters{it}"] = per
        # Printed by arena, with the position it was introduced at.
        print(f"iters {it}: " + " | ".join(
            f"a{i}@{per[f'arena{i}']['position']} "
            f"start {per[f'arena{i}'].get('trial_start_median_deg', float('nan')):.0f} "
            f"known {per[f'arena{i}'].get('known_median_deg', float('nan')):.0f}"
            for i in sorted(ids)), flush=True)
    tag = "" if a.order is None else "_order" + "".join(str(i) for i in a.order)
    if a.cap != 50 or a.prefind != "pursue":
        tag += f"_cap{a.cap}_{a.prefind}"
    out = os.path.splitext(a.ckpt)[0] + f"_synth_val{tag}.json"
    with open(out, "w") as f:
        json.dump({"ckpt": a.ckpt, "offsets": offs.tolist(), "goals": goals.tolist(), **res},
                  f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
