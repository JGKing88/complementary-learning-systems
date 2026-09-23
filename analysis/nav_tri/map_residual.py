"""Does a memoriser's deviation from `q` point at the cells it memorised?

`EXPERIMENTS_TASK_FAITHFUL.md` §11.2. On its own arenas a fixed-goal model's
memorised cell and the recalled direction agree exactly, so training cannot
separate "follow q" from "walk to cell C" and the learned policy is free to be
any blend of the two. On a held-out arena they disagree: `q` points at the real
goal, the map points at C. The blend predicts that what is left of the action
after removing its `q` component still points toward C.

Per step, with the goal PRE-STORED (the exploit condition):

    r      = a - (a·q̂) q̂          the part of the action `q` does not explain
    m⊥     = m̂ - (m̂·q̂) q̂          the part of "toward C" `q` does not explain
    score  = mean cos(r, m⊥)       > 0 means the deviation leans toward C

Scored for each memorised cell and for `--n_null` random cells, so the
question is whether the memorised cells sit in the upper tail of a null drawn
in the same geometry -- which controls for walls, for the goal's own position,
and for any global bias in the policy. Run the memoriser and a model that
never memorised (goals redrawn) through the same cells: the control should
score at the null median.

    python -m analysis.nav_tri.map_residual \\
        --run_dir $CLS_CKPTS/navigate_navp2_task3_k2_h128_nv_s42_22866166 \\
        --update 2500 --cells 17,0 0,13 0,15 --device cuda
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from hopfield_nav.encoder_io import load_encoder
from hopfield_nav.evaluation.checkpoint_io import (
    cfg_from_checkpoint, eval_world_from_spec, load_agent, world_spec_for)
from hopfield_nav.rollout.distractors import goal_encoding, sample_distractors
from hopfield import Hopfield

from .behavior_probe import rollout
from hopfield_nav.evaluation.metrics import random_start

EPS = 1e-8


def _unit(v, eps=EPS):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps), n[..., 0]


def _score_cell(act, q, pos_f, live, cell, min_resid=0.05, min_perp=0.10):
    """mean cos(r, m_perp) over live steps; also the raw cos(a, m_hat)."""
    qh, qn = _unit(q)
    ok = live & (qn > 1e-4)
    to_c = np.asarray(cell, dtype=np.float64)[None, None, :] - pos_f
    mh, mn = _unit(to_c)
    ok = ok & (mn > 1.5)                      # direction is meaningless on top of it

    a_par = (act * qh).sum(-1, keepdims=True) * qh
    r = act - a_par
    m_par = (mh * qh).sum(-1, keepdims=True) * qh
    mp = mh - m_par

    rh, rn = _unit(r)
    mph, mpn = _unit(mp)
    ok_r = ok & (rn > min_resid) & (mpn > min_perp)
    resid = (rh * mph).sum(-1)
    raw = (_unit(act)[0] * mh).sum(-1)
    return (float(resid[ok_r].mean()) if ok_r.any() else float("nan"),
            float(raw[ok].mean()) if ok.any() else float("nan"),
            int(ok_r.sum()))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--update", type=int, required=True)
    p.add_argument("--cells", nargs="+", required=True,
                   help="memorised cells as x,y (the run's TRAIN goals)")
    p.add_argument("--trials", type=int, default=32)
    p.add_argument("--n_distractors", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--n_null", type=int, default=120)
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction,
                   default=True, help="the exploit-probe convention")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    cells = [tuple(int(t) for t in c.split(",")) for c in a.cells]
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
                                             which="base_val")
    embed_dim = enc_cfg.out_dim
    rng = np.random.RandomState(a.seed)

    print(f"run={os.path.basename(a.run_dir)} u{a.update}  held-out arenas  "
          f"goal PRE-STORED  n_dist={a.n_distractors}  "
          f"{'deterministic' if a.deterministic else 'sampled'}  "
          f"trials={a.trials}")
    print(f"memorised cells: {cells}")

    recs = []
    for i, env in enumerate(envs):
        goal = tuple(env.goal_location)
        hops, starts = [], []
        for _ in range(a.trials):
            h = Hopfield(embed_dim, beta=cfg.hopfield.beta, device=str(device))
            pats = [goal_encoding(vh, offsets[i], goal)]
            if a.n_distractors:
                pats += sample_distractors(vh, offsets[i], int(env.size),
                                           a.n_distractors, rng)
                rng.shuffle(pats)
            for pat in pats:
                h.input_memory(torch.from_numpy(pat).float())
            hops.append(h)
            starts.append(random_start(int(env.size), goal, rng))
        rec = rollout(agent=agent, env=env, env_offset=offsets[i],
                      vectorhash=vh, hopfields=hops, cfg=cfg, device=device,
                      starts=starts, max_steps=a.max_steps,
                      ends_on_arrival=True, goal_in_memory=True,
                      deterministic=a.deterministic)
        recs.append((goal, rec))

    def pooled(cell):
        rs, raws, ns = [], [], 0
        for goal, rec in recs:
            r, w, n = _score_cell(rec["action"], rec["q"], rec["pos_f"],
                                  rec["alive"].astype(bool), cell)
            if n and r == r:
                rs.append(r * n); raws.append(w * n); ns += n
        return (sum(rs) / ns if ns else float("nan"),
                sum(raws) / ns if ns else float("nan"), ns)

    print(f"\n{'cell':>10}  {'resid cos(r, m_perp)':>21}  {'raw cos(a, m)':>14}  {'n steps':>8}")
    mem = []
    for c in cells:
        r, w, n = pooled(c)
        mem.append(r)
        print(f"{str(c):>10}  {r:>21.4f}  {w:>14.4f}  {n:>8}")
    mem_mean = float(np.nanmean(mem))

    size = int(envs[0].size)
    null = []
    tried = set(cells) | {tuple(g) for g, _ in recs}
    while len(null) < a.n_null:
        c = (int(rng.randint(size)), int(rng.randint(size)))
        if c in tried:
            continue
        tried.add(c)
        r, _, n = pooled(c)
        if n and r == r:
            null.append(r)
    null = np.array(null)
    pct = float((null < mem_mean).mean() * 100.0)
    print(f"\n  memorised mean          {mem_mean:+.4f}")
    print(f"  null ({len(null)} random cells)  mean {null.mean():+.4f}  "
          f"sd {null.std():.4f}  "
          f"p10 {np.percentile(null, 10):+.4f}  p50 {np.percentile(null, 50):+.4f}  "
          f"p90 {np.percentile(null, 90):+.4f}")
    print(f"  memorised mean sits at the {pct:.1f}th percentile of the null; "
          f"z = {(mem_mean - null.mean()) / max(null.std(), 1e-9):+.2f}")


if __name__ == "__main__":
    main()
