"""Where, and how often, does `q` fail to point at the goal?

P1 of `docs/EXPERIMENTS_NAV_P2.md`. No policy is involved: this is a property
of the encoder, the scaffold and the Hopfield, so it bounds what any exploit
policy can do before one is trained.

Phase 1 measured this as a *mean* `dir_acc` over eight envs and got two
different answers on two different worlds -- 1.8% and 23.3% of cells below
cos 0.5 at ten distractors. So the mean was never the measurement. This module
reports every cell of every env of every distractor draw and saves the raw
arrays; the summary is computed from them, not instead of them.

**The decomposition that matters.** "`q` points the wrong way" is two failures
with different fixes, and separating them is the point of this module:

  * **lock failure** -- the Hopfield settled into a *distractor's* attractor
    rather than the goal's. Then `q` is the tangent-plane projection of a
    displacement to a pattern hundreds of scaffold cells away, so its direction
    in this env's frame is near-arbitrary. Fixing this needs an encoder whose
    cross-env repulsion holds up.
  * **readout error** -- the recall landed on the goal and `q` still points
    somewhat wrong, because projecting a D=1024 displacement onto a 2-D tangent
    plane is lossy. This is the irreducible floor, and it caps `mean_steps`
    even with a perfect memory.

Reporting `dir_err` alone conflates them, and the conflated number is the one
that swung 13x between worlds. Conditioning on the lock separates "how often
does the lock fail" from "how bad is it when it does not", and the first is
what varies between worlds.

Distractors are drawn from scaffold cells *outside* the env's footprint (see
`rollout/distractors.py`), so "points at a known distractor" cannot mean an
angle towards an in-env location -- there is none. It means the attractor the
recall settled into belongs to a distractor, which is what `lock` records.

Usage:
    python -m analysis.nav_p2.q_failure_map --ckpt <any nav ckpt> \
        --envs 32 --draws 8 --out results/nav_p2/qmap_seed0.npz

Any nav checkpoint will do -- it is read only for the config (encoder path,
lambdas, Npos, fwhm, hopfield beta).
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from hopfield import Hopfield
from hopfield_nav.encoder_io import load_encoder
from hopfield_nav.evaluation.checkpoint_io import (
    build_eval_world, cfg_from_checkpoint,
)
from hopfield_nav.rollout.distractors import goal_encoding, sample_distractors

# Lock categories, stored as small ints so the arrays stay compact.
LOCK_GOAL, LOCK_DISTRACTOR, LOCK_MIXTURE = 0, 1, 2
LOCK_NAMES = {LOCK_GOAL: "goal", LOCK_DISTRACTOR: "distractor",
              LOCK_MIXTURE: "mixture"}


def _all_cells(size: int) -> np.ndarray:
    """Every cell of the arena, (size*size, 2) int32 -- not a sample."""
    gx, gy = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.int32)


@torch.no_grad()
def _env_geometry(vh, cells, offset, device):
    """Encoded states and tangent bases for a set of cells.

    Independent of what is stored in the Hopfield, so it is computed once per
    env and reused across every distractor draw and count -- which is the
    difference between this module running in minutes and in hours.
    """
    emb_np = vh.get_encoded_state(cells, offset)
    emb = torch.from_numpy(emb_np).float().to(device)
    W = vh.gram_schmidt_projection(cells, offset)
    return emb_np, emb, W


@torch.no_grad()
def _recall_q(vh, hop, emb_np, emb, W):
    """(q, recalled) for a batch of cells against one memory."""
    if hop.num_memories == 0:
        n = emb_np.shape[0]
        return np.zeros((n, 2), dtype=np.float32), None
    recalled = hop.recall_batch(emb, steps=1, beta=hop.beta, alpha=1.0)
    q = vh.project_displacement(emb_np, recalled.cpu().numpy(), W)
    return np.asarray(q), recalled


def _cos_to(recalled: torch.Tensor, patterns: np.ndarray) -> np.ndarray:
    """cos(recall_i, pattern_j) -> (n_cells, n_patterns)."""
    P = torch.from_numpy(patterns).float().to(recalled.device)
    r = torch.nn.functional.normalize(recalled, dim=-1)
    p = torch.nn.functional.normalize(P, dim=-1)
    return (r @ p.T).cpu().numpy()


def _classify_lock(cos_goal, cos_dist_max, thresh):
    """Which attractor the recall settled into, per cell.

    A recall that is close to neither is a spurious mixture -- the Hopfield's
    classic failure -- and is recorded separately rather than being forced into
    whichever of the two it happens to be marginally nearer. That third
    category is also the P3 fixed-point cue seen from the other side.
    """
    best_is_goal = cos_goal >= cos_dist_max
    best = np.where(best_is_goal, cos_goal, cos_dist_max)
    lock = np.where(best_is_goal, LOCK_GOAL, LOCK_DISTRACTOR).astype(np.int8)
    lock[best < thresh] = LOCK_MIXTURE
    return lock


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True,
                   help="read only for config; no policy is evaluated")
    p.add_argument("--n_distractors", type=int, nargs="+",
                   default=list(range(11)))
    p.add_argument("--envs", type=int, default=32)
    p.add_argument("--draws", type=int, default=8,
                   help="independent distractor draws per (env, count). Phase 1 "
                        "reached two different wrong conclusions from two draws.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lock_thresh", type=float, default=0.9,
                   help="cos below which a recall is called a mixture rather "
                        "than a lock onto either the goal or a distractor")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True, help="destination .npz")
    p.add_argument("--recorded_world", action="store_true",
                   help="Replay the checkpoint's recorded eval world instead of "
                        "drawing a fresh one. That world has however many envs "
                        "training was configured for -- 6 for the phase-1 runs "
                        "-- so it silently caps --envs, which is the opposite "
                        "of what this module is for. Use it only to compare "
                        "against a specific trained model's own world.")
    p.add_argument("--npos", type=int, default=None,
                   help="Shrink the scaffold for tool validation only. It "
                        "changes the very geometry this module measures -- the "
                        "distractor exclusion region shrinks with it -- so it "
                        "is for exercising the code path, never for a result.")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = cfg_from_checkpoint(ck["config"])
    cfg.num_val_envs = args.envs
    if args.npos is not None:
        print(f"  WARNING: --npos {args.npos} shrinks the scaffold. "
              f"Tool-validation mode; numbers are NOT a result.")
        cfg.vectorhash.Npos = args.npos

    encoder, enc_cfg, gain = load_encoder(cfg.encoder_checkpoint, str(device),
        getattr(cfg, "encoder_gain", None))
    if cfg.hopfield.beta is None:
        cfg.hopfield.beta = float(gain)
    embed_dim = enc_cfg.out_dim
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Default to a FRESH world. The recorded one carries whatever env count
    # training used -- 6 for every phase-1 run -- and replaying it silently
    # caps --envs at that, which would reduce this module to the eight-env mean
    # whose failure is the reason it exists. A fresh draw re-samples env
    # placement, which is exactly the axis being measured.
    replay = args.recorded_world and args.npos is None
    envs, vh, offsets = build_eval_world(
        cfg, encoder, str(device), ckpt_path=(args.ckpt if replay else None))
    if len(envs) != args.envs:
        print(f"  NOTE: asked for {args.envs} envs, world has {len(envs)}"
              + ("  (recorded world governs)" if replay else ""))

    size = envs[0].size
    cells = _all_cells(size)
    n_env, n_draw, n_cell = len(envs), args.draws, cells.shape[0]
    n_lvl = len(args.n_distractors)

    print(f"encoder   : {cfg.encoder_checkpoint}")
    print(f"scaffold  : Npos={vh.Npos}  embed_dim={embed_dim}  "
          f"beta={cfg.hopfield.beta:.4f}")
    print(f"grid      : {n_env} envs x {n_draw} draws x {n_cell} cells "
          f"x {n_lvl} distractor levels = "
          f"{n_env * n_draw * n_cell * n_lvl:,} recalls (x2 for goal-absent)")
    print(f"predicted |q| ratio distractor/goal ~ sqrt(2/D) = "
          f"{np.sqrt(2.0 / embed_dim):.4f}\n")

    shape = (n_lvl, n_env, n_draw, n_cell)
    out = {
        "dir_err": np.full(shape, np.nan, dtype=np.float32),      # radians
        "dir_cos": np.full(shape, np.nan, dtype=np.float32),
        "qnorm_goal": np.zeros(shape, dtype=np.float32),
        "qnorm_absent": np.zeros(shape, dtype=np.float32),
        "cos_goal": np.zeros(shape, dtype=np.float32),
        "cos_dist_max": np.zeros(shape, dtype=np.float32),
        "lock": np.full(shape, LOCK_MIXTURE, dtype=np.int8),
    }
    goal_dist = np.zeros((n_env, n_cell), dtype=np.float32)
    wall_dist = np.zeros((n_env, n_cell), dtype=np.float32)
    goals = np.zeros((n_env, 2), dtype=np.int32)

    rng = np.random.RandomState(args.seed)
    t0 = time.time()

    for ei, env in enumerate(envs):
        off, goal = offsets[ei], env.goal_location
        goals[ei] = goal
        emb_np, emb, W = _env_geometry(vh, cells, off, device)

        true_dir = (np.asarray(goal, dtype=np.float64)[None, :]
                    - cells.astype(np.float64))
        goal_dist[ei] = np.linalg.norm(true_dir, axis=1)
        # Distance to the nearest of the four boundary planes, for the
        # wall-conditioned breakdown H-wall needs.
        wall_dist[ei] = np.minimum(
            cells.min(axis=1), (size - 1) - cells.max(axis=1)).astype(np.float32)

        g_pat = goal_encoding(vh, off, goal)
        for li, n_d in enumerate(args.n_distractors):
            for di in range(n_draw):
                d_pats = (sample_distractors(vh, off, size, n_d, rng)
                          if n_d > 0 else [])

                # Goal present: the exploit regime's memory.
                hop_g = Hopfield(embed_dim, beta=cfg.hopfield.beta,
                                 device=str(device))
                pats = [g_pat] + list(d_pats)
                rng.shuffle(pats)
                for pat in pats:
                    hop_g.input_memory(torch.from_numpy(pat).float())
                q_g, rec_g = _recall_q(vh, hop_g, emb_np, emb, W)

                # Goal absent: the explore regime's memory, for the |q|
                # separability picture. Same draw, so the two differ only by
                # the goal being stored.
                hop_d = Hopfield(embed_dim, beta=cfg.hopfield.beta,
                                 device=str(device))
                for pat in d_pats:
                    hop_d.input_memory(torch.from_numpy(pat).float())
                q_d, _ = _recall_q(vh, hop_d, emb_np, emb, W)

                qn = np.linalg.norm(q_g, axis=1)
                out["qnorm_goal"][li, ei, di] = qn
                out["qnorm_absent"][li, ei, di] = np.linalg.norm(q_d, axis=1)

                denom = qn * np.linalg.norm(true_dir, axis=1)
                ok = denom > 1e-8
                cosv = np.full(n_cell, np.nan, dtype=np.float64)
                cosv[ok] = ((q_g[ok] * true_dir[ok]).sum(axis=1) / denom[ok])
                out["dir_cos"][li, ei, di] = cosv
                out["dir_err"][li, ei, di] = np.arccos(np.clip(cosv, -1.0, 1.0))

                cg = _cos_to(rec_g, g_pat[None, :])[:, 0]
                cd = (_cos_to(rec_g, np.stack(d_pats)).max(axis=1)
                      if d_pats else np.full(n_cell, -1.0, dtype=np.float32))
                out["cos_goal"][li, ei, di] = cg
                out["cos_dist_max"][li, ei, di] = cd
                out["lock"][li, ei, di] = _classify_lock(cg, cd,
                                                         args.lock_thresh)

        done = ei + 1
        rate = (time.time() - t0) / done
        print(f"  env {done}/{n_env}  ({rate:.1f} s/env, "
              f"{rate * (n_env - done) / 60:.1f} min left)", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        n_distractors=np.asarray(args.n_distractors),
        cells=cells, goals=goals, goal_dist=goal_dist, wall_dist=wall_dist,
        size=np.asarray(size), seed=np.asarray(args.seed),
        lock_thresh=np.asarray(args.lock_thresh),
        embed_dim=np.asarray(embed_dim), npos=np.asarray(vh.Npos),
        **out)
    print(f"\nwrote {args.out}")

    # A summary printed from the saved arrays, so what is read here and what a
    # plot reads later cannot drift apart.
    print(f"\n{'n_d':>4} {'lock=goal':>10} {'lock=dist':>10} {'lock=mix':>9} "
          f"{'dir_cos|goal':>13} {'dir_cos|dist':>13} {'frac<0.5':>9} "
          f"{'|q| ratio':>10}")
    for li, n_d in enumerate(args.n_distractors):
        lk = out["lock"][li].ravel()
        dc = out["dir_cos"][li].ravel()
        f_goal = float((lk == LOCK_GOAL).mean())
        f_dist = float((lk == LOCK_DISTRACTOR).mean())
        f_mix = float((lk == LOCK_MIXTURE).mean())
        m_goal = float(np.nanmean(dc[lk == LOCK_GOAL])) if f_goal else float("nan")
        m_dist = float(np.nanmean(dc[lk == LOCK_DISTRACTOR])) if f_dist else float("nan")
        below = float(np.nanmean(dc < 0.5))
        # At zero distractors the goal-absent memory is empty, so its |q| is
        # identically zero and the ratio is not a number about anything.
        absent = float(np.mean(out["qnorm_absent"][li]))
        ratio = (f"{float(np.mean(out['qnorm_goal'][li])) / absent:>10.2f}"
                 if absent > 1e-9 else f"{'--':>10}")
        print(f"{n_d:>4} {f_goal:>10.4f} {f_dist:>10.4f} {f_mix:>9.4f} "
              f"{m_goal:>13.4f} {m_dist:>13.4f} {below:>9.4f} {ratio}")

    # Between-world spread, the finding-19 number, at the highest level run.
    li = len(args.n_distractors) - 1
    per_env = [float(np.nanmean(out["dir_cos"][li, e] < 0.5)) for e in range(n_env)]
    print(f"\nfrac(dir_cos < 0.5) at n_d={args.n_distractors[li]}, per env: "
          f"min {min(per_env):.4f}  median {np.median(per_env):.4f}  "
          f"max {max(per_env):.4f}")
    print("Phase 1 reported this as a single mean over 8 envs and got 1.8% on "
          "one world and 23.3% on another. The spread above is the reason.")


if __name__ == "__main__":
    main()
