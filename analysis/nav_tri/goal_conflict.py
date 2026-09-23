"""Move the goal on the arena a model memorised: does it follow `q` or its map?

`EXPERIMENTS_TASK_FAITHFUL.md` §11.3. §11 established that a fixed-goal model
knows where its training goal is (it reaches it with an empty memory, or
steers clear of it). §11.2 showed the deviation from `q` on a HELD-OUT arena
does not point at those cells -- so the map is not a competing vector field
there. This asks the question where the map is certainly active: on the
model's OWN arena, with the goal RELOCATED.

The Hopfield is loaded with the relocated cell and the env pays at the
relocated cell, so `q` points at the new goal and the memorised cell is a
distractor the weights supply. Then:

  reach / steps / follow_q / align_true   toward the NEW goal
  old_visit_rate                          trials passing within 1 of the OLD
                                          (trained) goal on the way
  null_visit_rate                         the same for random cells matched on
                                          distance from the start distribution

A pure `q`-follower reaches the new goal and visits the old cell no more than
the null. A blend detours through the old cell. The redraw model, which
memorised nothing, is the control: it should look like a `q`-follower on the
same arenas with the same relocations.

    python -m analysis.nav_tri.goal_conflict \\
        --run_dir $CLS_CKPTS/navigate_navp2_task3_k2_h128_nv_s42_22866166 \\
        --update 2500 --device cuda
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from hopfield_nav.encoder_io import load_encoder
from hopfield_nav.evaluation.checkpoint_io import (
    cfg_from_checkpoint, eval_world_from_spec, load_agent, world_spec_for)
from hopfield_nav.evaluation.metrics import random_start
from hopfield_nav.rollout.distractors import goal_encoding, sample_distractors
from hopfield import Hopfield

from .behavior_probe import rollout

EPS = 1e-8


def _cos(a, b):
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    ok = (na > 1e-6) & (nb > 1e-6)
    out = np.zeros_like(na)
    np.divide((a * b).sum(-1), np.maximum(na * nb, EPS), out=out)
    return out, ok


def _visited(cells, target, radius=1.0):
    """(B,) bool: did each trajectory pass within `radius` of `target`?"""
    d = np.linalg.norm(cells - np.asarray(target, dtype=np.float64)[None, None, :],
                       axis=-1)
    return (d <= radius).any(axis=0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--update", type=int, required=True)
    p.add_argument("--trials", type=int, default=32)
    p.add_argument("--n_relocations", type=int, default=4)
    p.add_argument("--min_move", type=float, default=10.0,
                   help="relocated goal must be this far from the trained one")
    p.add_argument("--n_distractors", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--n_null", type=int, default=40)
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    device = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available())
                          else "cpu")
    blob = torch.load(os.path.join(a.run_dir, f"navigate_u{a.update}.pt"),
                      map_location="cpu", weights_only=False)
    cfg = cfg_from_checkpoint(blob["config"])
    cfg.device = str(device)
    spec = world_spec_for(a.run_dir)
    encoder, enc_cfg, gain = load_encoder(cfg.encoder_checkpoint, str(device),
                                          cfg.encoder_gain)
    cfg.encoder_gain = gain
    agent = load_agent(cfg, blob["agent_state_dict"], enc_cfg.out_dim, device)
    envs, vh, offsets = eval_world_from_spec(spec, cfg, encoder, str(device),
                                             which="train")
    embed_dim = enc_cfg.out_dim
    rng = np.random.RandomState(a.seed)

    print(f"run={os.path.basename(a.run_dir)} u{a.update}  OWN arenas, goal "
          f"RELOCATED  n_dist={a.n_distractors}  "
          f"{'deterministic' if a.deterministic else 'sampled'}  "
          f"trials={a.trials} x {a.n_relocations} relocations")
    print(f"{'arena':>5}  {'old':>8}  {'new':>8}  {'reach':>6}  {'steps':>6}  "
          f"{'follow_q':>8}  {'align':>6}  {'old_visit':>9}  {'null_visit':>10}")

    agg = {k: [] for k in ("reach", "steps", "follow", "align", "oldv", "nullv")}
    for i, env in enumerate(envs):
        old = tuple(int(v) for v in env.goal_location)
        size = int(env.size)
        for _ in range(a.n_relocations):
            while True:
                new = (int(rng.randint(size)), int(rng.randint(size)))
                if np.linalg.norm(np.array(new) - np.array(old)) >= a.min_move:
                    break
            env.set_goal(new)
            hops, starts = [], []
            for _ in range(a.trials):
                h = Hopfield(embed_dim, beta=cfg.hopfield.beta, device=str(device))
                pats = [goal_encoding(vh, offsets[i], new)]
                if a.n_distractors:
                    pats += sample_distractors(vh, offsets[i], size,
                                               a.n_distractors, rng)
                    rng.shuffle(pats)
                for pat in pats:
                    h.input_memory(torch.from_numpy(pat).float())
                hops.append(h)
                starts.append(random_start(size, new, rng))
            rec = rollout(agent=agent, env=env, env_offset=offsets[i],
                          vectorhash=vh, hopfields=hops, cfg=cfg, device=device,
                          starts=starts, max_steps=a.max_steps,
                          ends_on_arrival=True, goal_in_memory=True,
                          deterministic=a.deterministic)
            stg = rec["steps_to_goal"]
            reach = float((stg > 0).mean())
            steps = float(stg[stg > 0].mean()) if (stg > 0).any() else float("nan")
            live = rec["alive"].astype(bool)
            f, okf = _cos(rec["action"], rec["q"])
            to_new = np.asarray(new, dtype=np.float64)[None, None, :] - rec["pos_f"]
            al, oka = _cos(rec["action"], to_new)
            follow = float(f[okf & live].mean())
            align = float(al[oka & live].mean())
            cells = rec["pos_f"]
            oldv = float(_visited(cells, old).mean())
            nulls = []
            for _ in range(a.n_null):
                c = (int(rng.randint(size)), int(rng.randint(size)))
                if (np.linalg.norm(np.array(c) - np.array(new)) < 2.0
                        or np.linalg.norm(np.array(c) - np.array(old)) < 2.0):
                    continue
                nulls.append(float(_visited(cells, c).mean()))
            nullv = float(np.mean(nulls)) if nulls else float("nan")
            print(f"{i:>5}  {str(old):>8}  {str(new):>8}  {reach:>6.3f}  "
                  f"{steps:>6.1f}  {follow:>8.3f}  {align:>6.3f}  "
                  f"{oldv:>9.3f}  {nullv:>10.3f}")
            for k, v in (("reach", reach), ("steps", steps), ("follow", follow),
                         ("align", align), ("oldv", oldv), ("nullv", nullv)):
                agg[k].append(v)

    print(f"\n  reach {np.nanmean(agg['reach']):.3f}   steps "
          f"{np.nanmean(agg['steps']):.1f}   follow_q "
          f"{np.nanmean(agg['follow']):.3f}   align(new) "
          f"{np.nanmean(agg['align']):.3f}")
    ov, nv = np.nanmean(agg["oldv"]), np.nanmean(agg["nullv"])
    print(f"  visits the OLD (trained) goal on the way: {ov:.3f}   "
          f"random cells: {nv:.3f}   ratio {ov / max(nv, 1e-9):.2f}")


if __name__ == "__main__":
    main()
