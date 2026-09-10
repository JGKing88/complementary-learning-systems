"""Is "the recalled pattern belongs to the env I am in" decidable from the input?

Jack's framing of the whole problem: to explore the agent must **ignore** the
Hopfield (during explore the memory holds only distractors), and to exploit it
must **follow** it exactly. One policy can do both only if the observation says
which case it is in. This module measures whether it does — with no policy
involved, so the answer is a property of the encoder, the scaffold and the
Hopfield, not of any particular training run.

The geometric argument for why it should be decidable, which this checks:

`q = W_x (recall(x) − x)`, where `W_x` is the orthonormal 2-D basis of the
embedding manifold's tangent plane at the agent's cell (`world/scaffold.py:366`).
Both vectors are unit-norm.

  * an **in-env goal** is a few cells away, so `recall(x) − x` lies almost
    inside the tangent plane and the projection keeps nearly all of its norm;
  * a **distractor** is hundreds of scaffold cells away, so `recall(x) − x` is
    an essentially unrelated direction in D=1024 dimensions and its projection
    onto a 2-D plane keeps ~sqrt(2/D) of it — roughly 4% of the norm.

If that holds, **`|q|` is the discriminator**, which is exactly what
`--input_hopfield_raw 1` puts in the observation (the normalized signal throws
the magnitude away). `--input_hopfield_multistep 1 2 3` adds the recall
*dynamics*, which is the second candidate discriminator: iterating toward a
genuine stored neighbour converges, iterating toward a far pattern need not.

Reported per condition:

  |q|                 mean and percentiles
  q_dir_acc           cos(q, goal − cell): does the signal point at the goal?
  recall_is_goal      cos(recall, goal pattern) − max cos(recall, distractor);
                      >0 means the attractor landed on the goal, not a decoy
  AUC(|q|)            probability that a random goal-present cell has larger
                      |q| than a random goal-absent one. 1.0 = perfectly
                      separable by magnitude alone; 0.5 = no information.

Usage:
    python -m analysis.nav_tri.signal_separability --ckpt <any nav ckpt> \
        --n_distractors 0 1 3 10 --json out.json

Any checkpoint will do — it is read only for the config (encoder path, lambdas,
Npos, fwhm, hopfield beta) and the recorded eval world.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from hopfield import Hopfield
from hopfield_nav.encoder_io import load_encoder
from hopfield_nav.evaluation.checkpoint_io import (
    build_eval_world, cfg_from_checkpoint,
)
from hopfield_nav.rollout.distractors import goal_encoding, sample_distractors


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(a random `pos` exceeds a random `neg`), ties counted as half."""
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, allv.size + 1)
    # average ranks for ties
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size)
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    r_pos = ranks[:pos.size].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


@torch.no_grad()
def q_at(vh, hop, cells, offset, device, multistep):
    """(q, q_multistep, recalled) for a batch of cells against one memory."""
    pos = np.asarray(cells, dtype=np.int32)
    emb_np = vh.get_encoded_state(pos, offset)
    emb = torch.from_numpy(emb_np).float().to(device)
    W = vh.gram_schmidt_projection(pos, offset)
    if hop.num_memories == 0:
        z = np.zeros((len(pos), 2), dtype=np.float32)
        return z, {s: z for s in multistep}, None
    recalled = hop.recall_batch(emb, steps=1, beta=hop.beta, alpha=1.0)
    q = vh.project_displacement(emb_np, recalled.cpu().numpy(), W)
    ms = {}
    if multistep:
        traj = hop.recall_batch_trajectory(emb, list(multistep),
                                           beta=hop.beta, alpha=1.0)
        for s, X in traj.items():
            ms[s] = vh.project_displacement(emb_np, X.cpu().numpy(), W)
    return np.asarray(q), ms, recalled.cpu().numpy()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True,
                   help="read only for config + recorded eval world")
    p.add_argument("--n_distractors", type=int, nargs="+", default=[0, 1, 3, 10])
    p.add_argument("--cells", type=int, default=200,
                   help="cells sampled per env per condition")
    p.add_argument("--envs", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--json", default=None)
    p.add_argument("--chart_k", type=int, default=64,
                   help="Rank of the per-env chart basis used for the "
                        "`d1_chart` reference (§7.7). 0 skips it. This basis "
                        "is fitted from the env's own cells and is NOT a free "
                        "feature -- it is the ceiling the one-scalar "
                        "`chart_frac` is being compared against.")
    p.add_argument("--npos", type=int, default=None,
                   help="Shrink the scaffold for tool validation. See the same "
                        "flag on behavior_probe.py -- it changes the geometry "
                        "this module is measuring, so it is for exercising the "
                        "code path, never for a result.")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = cfg_from_checkpoint(ck["config"])
    if args.envs is not None:
        cfg.num_val_envs = args.envs
    if args.npos is not None:
        print(f"  WARNING: --npos {args.npos} overrides the checkpoint's "
              f"scaffold. Tool-validation mode; numbers are not comparable.")
        cfg.vectorhash.Npos = args.npos
    encoder, enc_cfg, gain = load_encoder(cfg.encoder_checkpoint, str(device),
        getattr(cfg, "encoder_gain", None))
    if cfg.hopfield.beta is None:
        cfg.hopfield.beta = float(gain)
    embed_dim = enc_cfg.out_dim
    torch.manual_seed(0)
    np.random.seed(0)
    # See behavior_probe: a shrunken scaffold is incompatible with the recorded
    # world's offsets, so validation mode replays the seed stream instead.
    envs, vh, offsets = build_eval_world(
        cfg, encoder, str(device),
        ckpt_path=(None if args.npos is not None else args.ckpt))
    multistep = list(cfg.agent.input_hopfield_multistep or [])

    print(f"encoder    : {cfg.encoder_checkpoint}")
    print(f"embed_dim  : {embed_dim}   beta: {cfg.hopfield.beta:.4f}   "
          f"Npos: {vh.Npos}   envs: {len(envs)}   size: {envs[0].size}")
    print(f"predicted |q| ratio distractor/goal ~ sqrt(2/D) = "
          f"{np.sqrt(2.0 / embed_dim):.4f}\n")

    out: dict = {"encoder": cfg.encoder_checkpoint, "embed_dim": embed_dim,
                 "conditions": {}}
    rng = np.random.RandomState(args.seed)

    for n_d in args.n_distractors:
        acc: dict[str, list] = {k: [] for k in (
            "q_goal", "dir_goal", "margin", "q_dist", "ms_goal", "ms_dist",
            "frac_goal", "frac_dist", "disp_goal", "disp_dist",
            "chart_goal", "chart_dist")}
        for i, env in enumerate(envs):
            off, goal, size = offsets[i], env.goal_location, env.size
            cells = np.stack([rng.randint(0, size, args.cells),
                              rng.randint(0, size, args.cells)], axis=1)
            true_dir = np.stack([goal[0] - cells[:, 0],
                                 goal[1] - cells[:, 1]], axis=1).astype(float)

            # Chart basis for this env: top-k right singular vectors of ALL
            # its cells' codes. Fitted from the env, which is exactly why it
            # is a ceiling and not a free feature (§7.7.1 point 1).
            basis_env = None
            if args.chart_k > 0:
                gx, gy = np.meshgrid(np.arange(size), np.arange(size),
                                     indexing="ij")
                allc = np.stack([gx.ravel(), gy.ravel()], axis=1)
                E = vh.get_encoded_state(allc.astype(np.int32), off)
                E = E.astype(np.float64)
                E = E - E.mean(0, keepdims=True)
                _u, _s, Vt = np.linalg.svd(E, full_matrices=False)
                basis_env = Vt[:args.chart_k].astype(np.float32)

            g_pat = goal_encoding(vh, off, goal)
            d_pats = (sample_distractors(vh, off, size, n_d, rng)
                      if n_d > 0 else [])

            # goal present (the exploit regime's memory)
            hop_g = Hopfield(embed_dim, beta=cfg.hopfield.beta, device=str(device))
            pats = [g_pat] + list(d_pats)
            rng.shuffle(pats)
            for pat in pats:
                hop_g.input_memory(torch.from_numpy(pat).float())
            q_g, ms_g, rec_g = q_at(vh, hop_g, cells, off, device, multistep)

            # goal absent (the explore regime's memory)
            hop_d = Hopfield(embed_dim, beta=cfg.hopfield.beta, device=str(device))
            for pat in d_pats:
                hop_d.input_memory(torch.from_numpy(pat).float())
            q_d, ms_d, rec_d = q_at(vh, hop_d, cells, off, device, multistep)

            acc["q_goal"].append(np.linalg.norm(q_g, axis=1))
            acc["q_dist"].append(np.linalg.norm(q_d, axis=1))

            # §7.7.1: the policy receives only q, the 2-D tangent projection of
            # (recall - x). `chart_frac` = ||q|| / ||recall - x|| is the
            # FRACTION of that displacement the local chart explains -- the
            # decoder-free, one-scalar shadow of group D's `d1_chart`, needing
            # no env-specific basis because W is already computed every step.
            # The docstring's geometry predicts it separates: an in-env goal
            # keeps nearly all of the norm, a distractor keeps ~sqrt(2/D).
            #
            # It is reported ALONGSIDE ||q||, not instead of it: ||q|| conflates
            # the chart-explained fraction with how far away the target is,
            # and the point is to see whether the fraction alone carries the
            # discrimination.
            emb_here = vh.get_encoded_state(np.asarray(cells, dtype=np.int32),
                                            off)
            for tag, rec in (("goal", rec_g), ("dist", rec_d)):
                if rec is None:
                    continue
                disp = rec - emb_here
                dn = np.linalg.norm(disp, axis=1)
                qq = np.linalg.norm(q_g if tag == "goal" else q_d, axis=1)
                acc[f"frac_{tag}"].append(qq / np.maximum(dn, 1e-12))
                acc[f"disp_{tag}"].append(dn)

            # The §7.7 reference the compression is being judged against:
            # d1_chart, the residual of the recall outside this env's chart
            # subspace (top-`chart_k` right singular vectors of its cells).
            # This one DOES need the env-specific basis, which is the cost
            # §7.11 records as unmeasured -- it is here as the ceiling, not as
            # a candidate channel.
            if args.chart_k > 0 and basis_env is not None:
                for tag, rec in (("goal", rec_g), ("dist", rec_d)):
                    if rec is None:
                        continue
                    proj = (rec @ basis_env.T) @ basis_env
                    resid = np.linalg.norm(rec - proj, axis=1)
                    acc[f"chart_{tag}"].append(
                        resid / np.maximum(np.linalg.norm(rec, axis=1), 1e-12))
            nrm = np.linalg.norm(q_g, axis=1) * np.linalg.norm(true_dir, axis=1)
            ok = nrm > 1e-8
            c = np.zeros(len(cells))
            c[ok] = (q_g[ok] * true_dir[ok]).sum(1) / nrm[ok]
            acc["dir_goal"].append(c)

            if rec_g is not None:
                cg = rec_g @ g_pat
                if d_pats:
                    cd = np.max(np.stack([rec_g @ d for d in d_pats], 1), axis=1)
                else:
                    cd = np.zeros_like(cg)
                acc["margin"].append(cg - cd)
            if multistep:
                s = multistep[-1]
                acc["ms_goal"].append(np.linalg.norm(ms_g[s], axis=1))
                acc["ms_dist"].append(np.linalg.norm(ms_d[s], axis=1))

        cat = {k: (np.concatenate(v) if v else np.array([]))
               for k, v in acc.items()}
        row = {
            "q_goal_mean": float(cat["q_goal"].mean()),
            "q_goal_p10": float(np.percentile(cat["q_goal"], 10)),
            "q_goal_p90": float(np.percentile(cat["q_goal"], 90)),
            "q_dist_mean": float(cat["q_dist"].mean()) if n_d else 0.0,
            "q_dist_p90": float(np.percentile(cat["q_dist"], 90)) if n_d else 0.0,
            "dir_acc_goal": float(cat["dir_goal"].mean()),
            # The MEAN cosine is ambiguous and was over-read once already:
            # 0.70 is equally consistent with "every cell is ~46 degrees
            # off" and with "70% of cells are near-perfect and 30% are
            # essentially random". Those say different things about whether
            # navigation is uniformly degraded or degraded in a subset of
            # the arena, so report the distribution, not just its mean.
            "dir_acc_p10": float(np.percentile(cat["dir_goal"], 10)),
            "dir_acc_p25": float(np.percentile(cat["dir_goal"], 25)),
            "dir_acc_median": float(np.median(cat["dir_goal"])),
            "dir_acc_p75": float(np.percentile(cat["dir_goal"], 75)),
            "dir_acc_p90": float(np.percentile(cat["dir_goal"], 90)),
            # cos < 0.5 is worse than 60 degrees off, i.e. the step makes
            # little progress toward the goal. cos < 0 is actively away.
            "dir_acc_frac_below_0.5": float((cat["dir_goal"] < 0.5).mean()),
            "dir_acc_frac_negative": float((cat["dir_goal"] < 0.0).mean()),
            "recall_margin": float(cat["margin"].mean()) if cat["margin"].size else float("nan"),
            "recall_is_goal_frac": float((cat["margin"] > 0).mean()) if cat["margin"].size else float("nan"),
            "auc_qmag": _auc(cat["q_goal"], cat["q_dist"]) if n_d else float("nan"),
        }
        # §7.7.1's question, in one number each.
        if n_d and cat["frac_goal"].size and cat["frac_dist"].size:
            row["frac_goal_mean"] = float(cat["frac_goal"].mean())
            row["frac_dist_mean"] = float(cat["frac_dist"].mean())
            row["auc_chart_frac"] = _auc(cat["frac_goal"], cat["frac_dist"])
        if n_d and cat["chart_goal"].size and cat["chart_dist"].size:
            row["chart_resid_goal"] = float(cat["chart_goal"].mean())
            row["chart_resid_dist"] = float(cat["chart_dist"].mean())
            # sign-corrected: goal-present has the SMALLER residual
            row["auc_d1_chart"] = _auc(-cat["chart_goal"], -cat["chart_dist"])
        if multistep:
            row[f"auc_qmag_step{multistep[-1]}"] = (
                _auc(cat["ms_goal"], cat["ms_dist"]) if n_d else float("nan"))
        out["conditions"][f"d{n_d}"] = row
        print(f"--- n_distractors = {n_d} ---")
        for k, v in row.items():
            print(f"  {k:<24s} {v:.4f}")
        print()

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
