"""How many steps does it take to tell "the goal is in my memory" from "it isn't"?

`signal_separability.py` asks whether a **single** `q` says which regime the
agent is in, and finds it does not (AUC 0.35-0.49 on |q|). That is not the end
of the question, because the policy is recurrent: it sees a *sequence* of `q`,
and the two cases have different temporal signatures even when their marginals
coincide.

  goal present -- every `q_t` points at ONE fixed cell. So the implied targets
                  `x_t + q_t` cluster, `|q|` shrinks as the agent approaches,
                  and the direction rotates smoothly and predictably with the
                  agent's own motion.
  goal absent  -- the attractor the recall lands in depends on where the agent
                  is standing, so the implied targets scatter and the direction
                  is not tied to any one point.

This module measures how much that is worth, as an **ideal-observer** bound: it
extracts a fixed feature vector from the first T values of `q` along a
trajectory and fits a logistic regression to separate the two conditions,
reporting held-out AUC as a function of T. No policy is trained and none is
loaded -- the answer is a property of the encoder, the scaffold and the
Hopfield, so it is an upper bound on what any architecture could extract, and a
low number is a **structural** obstacle rather than a tuning failure.

Read it as follows:

  AUC(T=1) ~ 0.5 and AUC(T=10) high   the cue is temporal. A recurrent policy
                                      can find it, but only after ~T steps of
                                      evidence -- which is a real cost in a
                                      200-step episode and an argument for
                                      interleaving rather than blocking.
  AUC stays near 0.5 at every T       the regimes are genuinely ambiguous from
                                      the observation. A single model then
                                      cannot be asked to do both without some
                                      other disambiguator, and that is a
                                      finding to report, not to tune around.

The trajectory the features are read along matters, so both are run:

  walk    -- unit steps in uniform random directions. The realistic case: the
             decision has to be made WHILE exploring, before the policy has
             committed to following anything.
  follow  -- steps along `q`. The optimistic case: acting on the signal is
             itself informative, because chasing a real goal converges and
             chasing a phantom does not.

Usage:
    python -m analysis.nav_tri.temporal_separability --ckpt <any nav ckpt> \
        --n_distractors 3 10 --steps 20 --json out.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from hopfield import Hopfield
from hopfield_nav.encoder_io import load_encoder
from hopfield_nav.evaluation.checkpoint_io import (
    build_eval_world, cfg_from_checkpoint,
)
from hopfield_nav.rollout.distractors import goal_encoding, sample_distractors


@torch.no_grad()
def _trajectory_q(vh, hop, *, size, offset, n_traj, steps, mode, rng, device):
    """(n_traj, steps, 2) of `q`, plus the (n_traj, steps, 2) cell positions.

    One shared Hopfield across the batch, which is exactly how training and
    evaluation use it (`training/explore.py:80`, `exploit.py:85`).
    """
    pos = rng.uniform(0, size - 1, size=(n_traj, 2))
    qs = np.zeros((n_traj, steps, 2), dtype=np.float32)
    xs = np.zeros((n_traj, steps, 2), dtype=np.float32)
    theta = rng.uniform(-np.pi, np.pi, size=n_traj)

    for t in range(steps):
        cell = np.clip(np.round(pos), 0, size - 1).astype(np.int32)
        emb_np = vh.get_encoded_state(cell, offset)
        emb = torch.from_numpy(emb_np).float().to(device)
        W = vh.gram_schmidt_projection(cell, offset)
        if hop.num_memories == 0:
            q = np.zeros((n_traj, 2), dtype=np.float32)
        else:
            rec = hop.recall_batch(emb, steps=1, beta=hop.beta, alpha=1.0)
            q = np.asarray(vh.project_displacement(
                emb_np, rec.cpu().numpy(), W), dtype=np.float32)
        qs[:, t] = q
        xs[:, t] = pos

        if mode == "follow":
            n = np.linalg.norm(q, axis=1, keepdims=True)
            step = np.divide(q, np.maximum(n, 1e-8))
            # A dead signal must not freeze the agent in place, or the two
            # conditions would differ by whether it moved at all.
            dead = (n[:, 0] < 1e-8)
            if dead.any():
                ph = rng.uniform(-np.pi, np.pi, size=int(dead.sum()))
                step[dead] = np.stack([np.cos(ph), np.sin(ph)], 1)
        else:
            theta = rng.uniform(-np.pi, np.pi, size=n_traj)
            step = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        pos = np.clip(pos + step, 0.0, float(size - 1))
    return qs, xs


def _features(q, x, T):
    """Fixed-length descriptor of the first T steps of one trajectory batch.

    Every feature is something a recurrent net could compute online; the point
    is to bound what is *available*, not to propose an architecture.
    """
    q, x = q[:, :T], x[:, :T]
    mag = np.linalg.norm(q, axis=2)                              # (N, T)
    feats = [mag.mean(1), mag.std(1), mag.min(1), mag.max(1)]

    if T >= 2:
        # Direction persistence: cos between consecutive q.
        a, b = q[:, 1:], q[:, :-1]
        na, nb = np.linalg.norm(a, 2, axis=2), np.linalg.norm(b, 2, axis=2)
        cos = (a * b).sum(2) / np.maximum(na * nb, 1e-8)
        feats += [cos.mean(1), cos.std(1)]

        # Implied-target scatter: if q points at ONE cell, x + q is constant.
        tgt = x + q
        feats += [np.linalg.norm(tgt.std(1), axis=1),
                  np.linalg.norm(tgt - tgt.mean(1, keepdims=True),
                                 axis=2).mean(1)]

        # Does |q| fall as the agent moves along it? (slope of mag vs t)
        t = np.arange(T) - (T - 1) / 2.0
        feats.append((mag * t).sum(1) / max((t * t).sum(), 1e-8))
    return np.stack(feats, axis=1)


def _auc(pos_f, neg_f, seed=0):
    X = np.concatenate([pos_f, neg_f])
    y = np.concatenate([np.ones(len(pos_f)), np.zeros(len(neg_f))])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]
    cut = len(y) // 2
    sc = StandardScaler().fit(X[:cut])
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(X[:cut]), y[:cut])
    return float(roc_auc_score(
        y[cut:], clf.predict_proba(sc.transform(X[cut:]))[:, 1]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True, help="read for config + eval world")
    p.add_argument("--n_distractors", type=int, nargs="+", default=[3, 10])
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--at", type=int, nargs="+", default=[1, 2, 5, 10, 20],
                   help="sequence lengths T to report AUC at")
    p.add_argument("--sets", type=int, default=8,
                   help="distinct memory draws per env per condition")
    p.add_argument("--traj", type=int, default=32, help="trajectories per set")
    p.add_argument("--envs", type=int, default=None)
    p.add_argument("--modes", nargs="+", default=["walk", "follow"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--npos", type=int, default=None,
                   help="shrink the scaffold; tool validation only")
    p.add_argument("--json", default=None)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = cfg_from_checkpoint(ck["config"])
    if args.envs is not None:
        cfg.num_val_envs = args.envs
    if args.npos is not None:
        print(f"  WARNING: --npos {args.npos}; tool-validation mode.")
        cfg.vectorhash.Npos = args.npos
    encoder, enc_cfg, gain = load_encoder(cfg.encoder_checkpoint, str(device))
    if cfg.hopfield.beta is None:
        cfg.hopfield.beta = float(gain)
    embed_dim = enc_cfg.out_dim
    torch.manual_seed(0)
    np.random.seed(0)
    envs, vh, offsets = build_eval_world(
        cfg, encoder, str(device),
        ckpt_path=(None if args.npos is not None else args.ckpt))

    print(f"encoder : {cfg.encoder_checkpoint}")
    print(f"Npos    : {vh.Npos}  envs: {len(envs)}  size: {envs[0].size}  "
          f"beta: {cfg.hopfield.beta:.3f}")
    print(f"per condition: {len(envs)} envs x {args.sets} memory draws x "
          f"{args.traj} trajectories = "
          f"{len(envs) * args.sets * args.traj} sequences\n")

    out: dict = {"npos": int(vh.Npos), "results": {}}
    for mode in args.modes:
        for n_d in args.n_distractors:
            rng = np.random.RandomState(args.seed)
            bank: dict[str, list] = {"pos": [], "neg": []}
            for i, env in enumerate(envs):
                off, goal, size = offsets[i], env.goal_location, env.size
                g_pat = goal_encoding(vh, off, goal)
                for _ in range(args.sets):
                    d_pats = sample_distractors(vh, off, size, n_d, rng)
                    for label, pats in (("pos", [g_pat] + list(d_pats)),
                                        ("neg", list(d_pats))):
                        hop = Hopfield(embed_dim, beta=cfg.hopfield.beta,
                                       device=str(device))
                        ps = list(pats)
                        rng.shuffle(ps)
                        for pat in ps:
                            hop.input_memory(torch.from_numpy(pat).float())
                        bank[label].append(_trajectory_q(
                            vh, hop, size=size, offset=off, n_traj=args.traj,
                            steps=args.steps, mode=mode, rng=rng, device=device))

            qp = np.concatenate([b[0] for b in bank["pos"]])
            xp = np.concatenate([b[1] for b in bank["pos"]])
            qn = np.concatenate([b[0] for b in bank["neg"]])
            xn = np.concatenate([b[1] for b in bank["neg"]])

            row = {}
            for T in args.at:
                if T > args.steps:
                    continue
                row[f"T{T}"] = _auc(_features(qp, xp, T), _features(qn, xn, T),
                                    seed=args.seed)
            out["results"][f"{mode}_d{n_d}"] = row
            print(f"--- mode={mode}  n_distractors={n_d} ---")
            print("  " + "  ".join(f"AUC@T={k[1:]}: {v:.3f}"
                                   for k, v in row.items()))
            print()

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
