"""Does `q` point OUT of the arena when the recall is a phantom?

Jack, 2026-09-07: instead of gating on ||q||, the agent could notice that `q` is
steering it into a wall and conclude it is in the explore regime. That would be
a STRUCTURAL discriminator rather than a statistical one -- distractors are
drawn (rollout/distractors.py) from grid positions OUTSIDE the test env's
footprint, so a phantom target is by construction not in the box, while a real
goal always is. Unlike ||q||, it would not degrade as distractors are added,
because adding distractors does not move the walls.

Two things could sink it and neither is settled by argument:

  1. The scaffold is a periodic grid code. A distractor at a distant grid
     position may ALIAS to a direction that points back inside, in which case
     the phantom field is not outward-pointing and there is no cue.
  2. The recall under ten distractors is a BLEND, not one pattern, so the
     resulting direction is some average that may point anywhere.

Measured with no policy and no rollouts, so nothing here depends on what any
agent learned. For each memory draw the field q(x) is evaluated at every cell,
once with the goal stored and once with only distractors:

  outwardness  cos(q, outward normal) at PERIMETER cells. Directly answers
               "is q pushing me into the wall". Scale-free -- no need to
               extrapolate a target distance from ||q||.
  edge_end     fraction of flows from all cells that terminate on the
               perimeter, integrating the agent's own update rule.
  AUC          outwardness as a goal-present/goal-absent discriminator, in the
               same units as the P3 probe results (||q|| alone scores 0.881,
               chart_frac 0.974-0.988). This is the number that decides whether
               the cue is worth building.

The goal-near-wall split is the control that matters: a REAL goal one cell from
the west wall also produces wallward q, so the cue must survive that case or it
would fire on legitimate navigation.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from hopfield import Hopfield
from hopfield_nav.evaluation.checkpoint_io import (
    build_eval_world, cfg_from_checkpoint, eval_env_set,
)
from hopfield_nav.encoder_io import load_encoder
from hopfield_nav.rollout.distractors import goal_encoding, sample_distractors
from hopfield_nav.world import generate as gen
from analysis.nav_tri.readout_field import field_over_cells, integrate

# What the run found, kept here so the module states its own result:
#   per-CELL AUC 0.78-0.81 -- WORSE than ||q||'s 0.881 as a one-glance cue.
#   per-MEMORY AUC 1.000/1.000/0.995 at d=1/5/10, FLAT where ||q|| falls
#     0.96 -> 0.62. The value is robustness to distractors, not raw power.
#   The phantom does NOT point outward (+0.003 at d=1, +0.077 at d=10 --
#     isotropic). The signal is that a STORED GOAL pulls inward (-0.69).
#     So phantom flows pile on the perimeter because nothing pulls them off
#     it, not because they are driven into it.


def perimeter_normals(size):
    """(cells, 2) positions and their outward unit normals, perimeter only."""
    pos, nrm = [], []
    for i in range(size):
        for j in range(size):
            nx = (-1 if i == 0 else 0) + (1 if i == size - 1 else 0)
            ny = (-1 if j == 0 else 0) + (1 if j == size - 1 else 0)
            if nx or ny:
                v = np.array([nx, ny], dtype=np.float64)
                pos.append([i, j])
                nrm.append(v / np.linalg.norm(v))
    return np.asarray(pos), np.asarray(nrm)


def auc(pos_scores, neg_scores):
    """P(a random positive scores above a random negative), ties counted half.

    MIDRANKS, not `argsort(argsort(x))`. The latter breaks ties by array
    position, so a fully tied input -- no signal at all -- scores 1.0 or 0.0
    depending on concatenation order rather than 0.5. The cosines this module
    actually compares are continuous and never tie exactly, so the reported
    numbers are unaffected either way; the midranks are here so a future caller
    on a discrete score does not read a spurious perfect separation.
    """
    p, n = np.asarray(pos_scores, float), np.asarray(neg_scores, float)
    if not len(p) or not len(n):
        return float("nan")
    allv = np.concatenate([p, n])
    order = np.argsort(allv, kind="mergesort")
    s = allv[order]
    r = np.empty(len(allv), float)
    i = 0
    while i < len(s):                       # average the ranks within each tie
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((r[:len(p)].sum() - len(p) * (len(p) + 1) / 2)
                 / (len(p) * len(n)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="config + eval world only")
    ap.add_argument("--n_distractors", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--split", default="place=held_out")
    ap.add_argument("--val_seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    # Mirrors readout_field.main: cfg_from_checkpoint takes the saved config
    # DICT, not a path, and the env set has to come from the recorded world or
    # the field is mapped over scaffold patches the run never trained on.
    device = torch.device(a.device)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = cfg_from_checkpoint(ck["config"])
    cfg.num_val_envs = a.envs
    encoder, enc_cfg, gain = load_encoder(
        cfg.encoder_checkpoint, str(device), getattr(cfg, "encoder_gain", None))
    if cfg.hopfield.beta is None:
        cfg.hopfield.beta = float(gain)
    embed_dim = enc_cfg.out_dim
    torch.manual_seed(0)
    np.random.seed(0)
    es = eval_env_set(cfg, encoder, str(device), ckpt_path=a.ckpt,
                      levels=gen.parse_levels(a.split), val_seed=a.val_seed,
                      n_envs=cfg.num_val_envs)
    env_list, vh, offsets = es["envs"], es["field"], es["offsets"]
    envs = list(zip(env_list, offsets))[: a.envs]
    print("envs=%d trials=%d embed_dim=%d gain=%g beta=%g"
          % (len(envs), a.trials, embed_dim, gain, cfg.hopfield.beta))

    pcells, pnorm = None, None
    rows = []
    for nd in a.n_distractors:
        rng = np.random.RandomState(a.seed)
        rec = {"present": [], "absent": [],
               "cells_present": [], "cells_absent": [],
               "present_wall": [], "present_open": [],
               "edge_present": [], "edge_absent": [],
               "reach_present": [], "reach_absent": []}
        for env, offset in envs:
            size = env.size
            goal = env.goal_location
            radius = float(cfg.env.goal_radius)
            if pcells is None:
                pcells, pnorm = perimeter_normals(size)
            near_wall = min(goal[0], goal[1],
                            size - 1 - goal[0], size - 1 - goal[1]) <= 2
            for _b in range(a.trials):
                dis = (sample_distractors(vh, offset, size, nd, rng)
                       if nd > 0 else [])
                gp = [goal_encoding(vh, offset, goal)] + list(dis)
                for label, pats in (("present", gp), ("absent", list(dis))):
                    if not pats:
                        continue
                    hop = Hopfield(embed_dim, beta=cfg.hopfield.beta,
                                   device=str(a.device))
                    for pat in pats:
                        hop.input_memory(torch.from_numpy(pat).float())
                    fld = field_over_cells(vh, hop, size, offset, a.device)

                    # --- outwardness at the perimeter -------------------
                    v = fld[pcells[:, 0], pcells[:, 1]]
                    nv = np.linalg.norm(v, axis=-1)
                    ok = nv > 1e-9
                    cosn = np.full(len(v), np.nan)
                    cosn[ok] = (v[ok] * pnorm[ok]).sum(-1) / nv[ok]
                    m = float(np.nanmean(cosn))
                    rec[label].append(m)
                    # PER-CELL, the "one glance" bound: the policy sees q at
                    # the cell it is standing on, not the perimeter average.
                    rec["cells_" + label].append(cosn[np.isfinite(cosn)])
                    if label == "present":
                        rec["present_wall" if near_wall
                            else "present_open"].append(m)

                    # --- where the flow ends ----------------------------
                    ends, reached, _ = integrate(fld, size, goal, radius)
                    on_edge = ((ends[:, 0] <= 0.5) | (ends[:, 0] >= size - 1.5)
                               | (ends[:, 1] <= 0.5)
                               | (ends[:, 1] >= size - 1.5))
                    rec["edge_" + label].append(float(on_edge.mean()))
                    rec["reach_" + label].append(float(reached.mean()))

        row = {"n_dist": nd,
               "outward_present": float(np.mean(rec["present"])),
               "outward_absent": float(np.mean(rec["absent"])),
               "outward_present_sd": float(np.std(rec["present"])),
               "outward_absent_sd": float(np.std(rec["absent"])),
               "auc_outward": auc(rec["absent"], rec["present"]),
               "auc_outward_percell": auc(
                   np.concatenate(rec["cells_absent"]),
                   np.concatenate(rec["cells_present"])),
               "edge_present": float(np.mean(rec["edge_present"])),
               "edge_absent": float(np.mean(rec["edge_absent"])),
               "reach_present": float(np.mean(rec["reach_present"])),
               "reach_absent": float(np.mean(rec["reach_absent"])),
               "present_wall": (float(np.mean(rec["present_wall"]))
                                if rec["present_wall"] else float("nan")),
               "present_open": (float(np.mean(rec["present_open"]))
                                if rec["present_open"] else float("nan")),
               "n": len(rec["present"])}
        rows.append(row)
        print("  n_dist=%-3d n=%d" % (nd, row["n"]))
        print("    outwardness  present %+.4f (sd %.4f)   absent %+.4f (sd %.4f)"
              % (row["outward_present"], row["outward_present_sd"],
                 row["outward_absent"], row["outward_absent_sd"]))
        print("    AUC per-memory (mean over perimeter)  %.4f" % row["auc_outward"])
        print("    AUC PER-CELL  (one glance, honest)    %.4f"
              % row["auc_outward_percell"])
        print("    flow ends on edge   present %.3f   absent %.3f"
              % (row["edge_present"], row["edge_absent"]))
        print("    flow reaches goal   present %.3f   absent %.3f"
              % (row["reach_present"], row["reach_absent"]))
        print("    present split: goal near wall %+.4f | goal in open %+.4f"
              % (row["present_wall"], row["present_open"]))

    print()
    print("  READ: the cue is real if AUC is high AND the present/absent gap")
    print("  survives the goal-near-wall split. AUC ~0.5, or a wall-goal")
    print("  number as outward as the phantoms, falsifies it.")
    print("  Reference from P3: ||q|| alone scores AUC 0.881, chart_frac 0.974.")
    if a.json:
        json.dump({"rows": rows, "ckpt": a.ckpt}, open(a.json, "w"), indent=1)
        print("  wrote", a.json)


if __name__ == "__main__":
    main()
