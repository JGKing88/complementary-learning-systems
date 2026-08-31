"""Are the imperfect recalls non-converged, or converged onto mixtures?

The question Jack asked: a spurious fixed point is still a fixed point, so
`‖recall(x) − recall²(x)‖` cannot tell a blend from a clean retrieval. He is
right, and it matters because P1 found that direction error is governed almost
entirely by recall fidelity -- cells whose settled state sits at cos ≥ 0.99 to
the goal pattern have a 0.01% rate of bad direction, cells below it carry
99.7% of all the bad ones.

Two possibilities, with opposite consequences:

  * **Non-converged.** One step simply has not arrived yet. Then the residual
    *does* flag those cells, the multistep channel the policy already receives
    can see it, and running the recall for more steps would improve the readout
    outright -- a one-line change.
  * **Converged onto a mixture.** The state is a genuine attractor of the
    Hebbian dynamics that happens not to be any stored pattern. Then the
    residual is blind to exactly the cells that matter, the policy has no
    *local* cue for its own unreliability, and the only thing left is the
    motion-based evidence in EXPERIMENTS_NAV_P2 §7.2 groups A and B -- which is
    the case for P3 doing the work rather than a cheap static statistic.

This measures which. `q_failure_map` uses `steps=1`, matching the trainer, so
the states it scored are one step from the encoded cue and their convergence
was never checked.

    python -m analysis.nav_p2.recall_convergence --ckpt <any nav ckpt> \
        --envs 16 --draws 4 --n_distractors 10
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from hopfield import Hopfield
from hopfield_nav.encoder_io import load_encoder
from hopfield_nav.evaluation.checkpoint_io import (
    build_eval_world, cfg_from_checkpoint,
)
from hopfield_nav.rollout.distractors import goal_encoding, sample_distractors


def _all_cells(size: int) -> np.ndarray:
    gx, gy = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.int32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--envs", type=int, default=16)
    p.add_argument("--draws", type=int, default=4)
    p.add_argument("--n_distractors", type=int, default=10)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = cfg_from_checkpoint(ck["config"])
    cfg.num_val_envs = args.envs
    encoder, enc_cfg, gain = load_encoder(cfg.encoder_checkpoint, str(device),
        getattr(cfg, "encoder_gain", None))
    if cfg.hopfield.beta is None:
        cfg.hopfield.beta = float(gain)
    D = enc_cfg.out_dim
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    envs, vh, offsets = build_eval_world(cfg, encoder, str(device), ckpt_path=None)

    size = envs[0].size
    cells = _all_cells(size)
    K = args.steps
    snaps = list(range(1, K + 1))

    cos_k, res_k, dcos_k = [], [], []          # per step, pooled over everything
    rng = np.random.RandomState(args.seed)

    for ei, env in enumerate(envs):
        off, goal = offsets[ei], env.goal_location
        emb_np = vh.get_encoded_state(cells, off)
        emb = torch.from_numpy(emb_np).float().to(device)
        W = vh.gram_schmidt_projection(cells, off)
        true_dir = (np.asarray(goal, float)[None, :] - cells.astype(float))
        tn = np.linalg.norm(true_dir, axis=1)

        g_pat = goal_encoding(vh, off, goal)
        g_t = torch.nn.functional.normalize(
            torch.from_numpy(g_pat).float().to(device), dim=-1)

        for _ in range(args.draws):
            d_pats = sample_distractors(vh, off, size, args.n_distractors, rng)
            hop = Hopfield(D, beta=cfg.hopfield.beta, device=str(device))
            pats = [g_pat] + list(d_pats)
            rng.shuffle(pats)
            for pat in pats:
                hop.input_memory(torch.from_numpy(pat).float())

            traj = hop.recall_batch_trajectory(emb, snaps, beta=hop.beta, alpha=1.0)
            X = [traj[s] for s in snaps]
            cos_k.append(np.stack([
                (torch.nn.functional.normalize(x, dim=-1) @ g_t).cpu().numpy()
                for x in X]))                                    # (K, n_cell)
            # K-1 residuals for K snapshots; the last step has no successor,
            # so it is NaN rather than a padded zero -- a zero here reads as
            # "converged", which is the opposite of what it means.
            res_k.append(np.stack([
                (X[i + 1] - X[i]).norm(dim=-1).cpu().numpy()
                for i in range(K - 1)]
                + [np.full(len(cells), np.nan, np.float32)]))
            dq = []
            for x in X:
                q = vh.project_displacement(emb_np, x.cpu().numpy(), W)
                n = np.linalg.norm(q, axis=1) * tn
                c = np.full(len(cells), np.nan)
                ok = n > 1e-8
                c[ok] = (q[ok] * true_dir[ok]).sum(1) / n[ok]
                dq.append(c)
            dcos_k.append(np.stack(dq))
        print(f"  env {ei + 1}/{len(envs)}", flush=True)

    cos = np.concatenate(cos_k, axis=1)        # (K, N)
    res = np.concatenate(res_k, axis=1)
    dco = np.concatenate(dcos_k, axis=1)
    N = cos.shape[1]
    print(f"\n{N:,} cells x {K} recall steps, {args.n_distractors} distractors\n")

    # Grouped by fidelity AFTER ONE STEP -- the state q_failure_map scored.
    groups = (("cos1 >= 0.99", cos[0] >= 0.99),
              ("0.90 <= cos1 < 0.99", (cos[0] >= 0.90) & (cos[0] < 0.99)),
              ("cos1 < 0.90", cos[0] < 0.90))
    for name, m in groups:
        if m.sum() < 50:
            continue
        print(f"{name}   n={m.sum():,} ({m.mean() * 100:.1f}% of cells)")
        print(f"{'step':>6} {'cos to goal':>12} {'step residual':>14} "
              f"{'dir_cos':>9} {'%dir<0.5':>9}")
        for si, s in enumerate(snaps):
            if s not in (1, 2, 3, 5, 8, K):
                continue
            print(f"{s:>6} {np.median(cos[si, m]):>12.4f} "
                  f"{np.nanmedian(res[si, m]):>14.5f} "
                  f"{np.nanmedian(dco[si, m]):>9.4f} "
                  f"{np.nanmean(dco[si, m] < 0.5) * 100:>8.2f}%")
        print()

    # The decisive number: of the imperfect-after-one-step cells, how many are
    # still moving, and how many have settled somewhere that is not the goal?
    imperfect = cos[0] < 0.99
    settled = res[-2] < 1e-3
    print("Of the cells imperfect after one step "
          f"({imperfect.sum():,}, {imperfect.mean() * 100:.1f}% of all):")
    print(f"  still moving at step {K}      : "
          f"{(imperfect & ~settled).sum() / imperfect.sum() * 100:5.1f}%")
    print(f"  settled by step {K}           : "
          f"{(imperfect & settled).sum() / imperfect.sum() * 100:5.1f}%")
    st = imperfect & settled
    if st.sum():
        print(f"     ...of those, settled ON the goal (cos>=0.99): "
              f"{(cos[-1][st] >= 0.99).mean() * 100:5.1f}%")
        print(f"     ...settled on something ELSE  : "
              f"{(cos[-1][st] < 0.99).mean() * 100:5.1f}%   <-- spurious fixed points")
    print(f"\nDirection quality, all cells:  1 step {np.nanmean(dco[0] < 0.5) * 100:.2f}% "
          f"bad -> {K} steps {np.nanmean(dco[-1] < 0.5) * 100:.2f}% bad")


if __name__ == "__main__":
    main()
