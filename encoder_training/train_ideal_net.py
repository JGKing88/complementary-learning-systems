#!/usr/bin/env python3
"""Stage A: can gradient descent find the ideal network's integer weights?

``IdealNet`` (analysis/hopfield_probe/ideal_net.py) is the ideal encoder as
layers: a fixed phase readout, atan2, a 6 -> 512 harmonic layer, and cos/sin.
The ideal code needs that layer to hold integers. Here it is trainable and the
question is whether training finds integers, and which frequencies they
implement, from the same data the production encoders see.

Data: the att0.5 recipe's patches (118 random, non-overlapping 50 x 50 patches,
~10% of the scaffold, fwhm 0.25, batch 4096, pairs within one patch only), read
from that checkpoint's ``train_config``.

Objective: a well-posed kernel target instead of the campaign's attract/repel
step (EXPERIMENTS_UNIQUE_RADIUS Sec 9 found the step target itself is what makes
an equivariant code ring). For every within-patch pair in a batch,

    loss = mean ( z(p).z(q) - exp(-|p - q|^2 / 2 r^2) )^2 .

Layers 1-2 are fixed, so the six phases per training position are computed
once, and each step only needs ``z(p).z(q) = mean_f cos(theta_pf - theta_qf)``
on the sampled pairs.

Writes ``history.json`` (loss and integrality every ``--log_every`` epochs),
checkpoints (``save_ideal_net``) at a few epochs, and ``diagnostics.json`` from
``ideal_net_diagnostics`` at the end.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from analysis.hopfield_probe.ideal_net import (IdealNet, load_ideal_net,
                                               save_ideal_net)
from encoder_training.data import (build_patch_codes, mixed_batch_iterator,
                                   sample_nonoverlapping_patches)
from encoder_training.ideal_net_diagnostics import diagnose, integrality
from encoder_training.losses import coding_rate_loss

LAMBDAS = [11, 12, 13]
NPOS = 1716
RECIPE = dict(npos_list=[50] * 118, placement="random", fwhm_ratio=0.25,
              batch_size=4096)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--r", type=float, default=16.0)
    ap.add_argument("--init", choices=["integer", "noisy", "random"],
                    default="random")
    ap.add_argument("--seed", type=int, default=0,
                    help="patch placement, batch order and the init draw")
    ap.add_argument("--freq_seed", type=int, default=0,
                    help="which ideal frequency draw defines the integer init")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--rate_lambda", type=float, default=0.0,
                    help="weight on losses.coding_rate_loss of the batch codes "
                         "(the recipe's far-field term; Sec 12 chose 0.5)")
    ap.add_argument("--rate_eps", type=float, default=1.0)
    ap.add_argument("--inv_lambda", type=float, default=0.0,
                    help="translation-invariance penalty: variance of z(p).z(q) "
                         "across within-batch pairs with the same displacement "
                         "(Delta and -Delta pooled)")
    ap.add_argument("--inv_ramp_epochs", type=int, default=0,
                    help="ramp inv_lambda linearly from 0 over this many epochs "
                         "(0 = full strength from the start)")
    ap.add_argument("--inv_delay_epochs", type=int, default=0,
                    help="keep inv_lambda at 0 for this many epochs, then ramp")
    ap.add_argument("--noise_std", type=float, default=0.0,
                    help="Gaussian noise added to the harmonic weights after each "
                         "step, decaying linearly to 0 at the last epoch")
    ap.add_argument("--init_ckpt", default=None,
                    help="start from this saved IdealNet's harmonic weights")
    ap.add_argument("--kick", type=float, default=0.0,
                    help="add uniform(-kick, kick) to every harmonic weight at "
                         "the start (basin test)")
    ap.add_argument("--pair_sampling", choices=["batch", "uniform_delta"],
                    default="batch")
    ap.add_argument("--n_delta", type=int, default=4096,
                    help="uniform_delta: displacements drawn per step")
    ap.add_argument("--anchors_per_delta", type=int, default=16)
    ap.add_argument("--near_weight", action="store_true",
                    help="weight each pair's MSE by 1/(pairs in its 2-cell "
                         "distance bin), so near pairs count as much as far ones")
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--save_at", type=int, nargs="*", default=[0, 50, 100, 250])
    ap.add_argument("--patch_arena", type=int, default=0,
                    help="confine patches to the [0, N)^2 corner (0 = whole)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)

    arena = a.patch_arena or NPOS
    y0s, x0s, sizes = sample_nonoverlapping_patches(
        arena, arena, RECIPE["npos_list"], None, placement=RECIPE["placement"])
    Phi, coords, env_ids = build_patch_codes(LAMBDAS, y0s, x0s, sizes, dev,
                                             RECIPE["fwhm_ratio"])
    N = Phi.shape[0]
    print(f"{len(sizes)} patches, N={N} ({100 * N / NPOS ** 2:.1f}% of the "
          f"scaffold), device {dev}", flush=True)

    net = IdealNet(r=a.r, seed=a.freq_seed, trainable_harmonics=True,
                   init=a.init).to(dev)
    with torch.no_grad():                          # layers 1-2 are fixed
        phases = torch.cat([net.phase(net.readout(Phi[i:i + 65536].double()))
                            for i in range(0, N, 65536)])
    del Phi
    W = net.harmonics.weight                       # (F, 6), the only parameter
    if a.init_ckpt:
        src, _ = load_ideal_net(a.init_ckpt)
        with torch.no_grad():
            W.copy_(src.harmonics.weight.to(W))
    if a.kick > 0:
        with torch.no_grad():
            W.add_(torch.empty_like(W).uniform_(-a.kick, a.kick))
    opt = torch.optim.Adam([W], lr=a.lr)
    F = W.shape[0]
    two_r2 = 2.0 * a.r * a.r

    hist = []

    # --- pair sampling ------------------------------------------------------
    # "batch": all within-patch pairs among a random batch of points (the
    # recipe; short displacements are rare). "uniform_delta": displacement
    # vectors drawn uniformly from the (2s-1)^2 window, several anchors each.
    sizes_t = torch.tensor(sizes, device=dev)
    starts = torch.cumsum(sizes_t * sizes_t, 0) - sizes_t * sizes_t
    steps_per_epoch = N // RECIPE["batch_size"]

    def pair_batches():
        if a.pair_sampling == "batch":
            for idx in mixed_batch_iterator(N, RECIPE["batch_size"]):
                idx = idx.to(dev).long()
                e = env_ids[idx]
                ii, jj = torch.triu_indices(len(idx), len(idx), 1, device=dev)
                keep = e[ii] == e[jj]
                yield idx[ii[keep]], idx[jj[keep]], idx
            return
        s = int(sizes[0])
        assert all(int(x) == s for x in sizes), "uniform_delta needs equal patch sizes"
        D, A = a.n_delta, a.anchors_per_delta
        for _ in range(steps_per_epoch):
            dy = torch.randint(-(s - 1), s, (D,), device=dev)
            dx = torch.randint(-(s - 1), s, (D,), device=dev)
            zero = (dy == 0) & (dx == 0)
            dx = torch.where(zero, torch.ones_like(dx), dx)   # no self-pairs
            dy, dx = dy[:, None].expand(D, A), dx[:, None].expand(D, A)
            e = torch.randint(0, len(sizes), (D, A), device=dev)
            ay = (torch.rand(D, A, device=dev) * (s - dy.abs())).long() + (-dy).clamp_min(0)
            ax = (torch.rand(D, A, device=dev) * (s - dx.abs())).long() + (-dx).clamp_min(0)
            pi = (starts[e] + ay * s + ax).reshape(-1)
            pj = (starts[e] + (ay + dy) * s + (ax + dx)).reshape(-1)
            # every pair lies in one patch, at exactly the drawn displacement
            assert bool((env_ids[pi] == env_ids[pj]).all())
            got = (coords[pj] - coords[pi]).long()
            assert bool((got[:, 0] == dy.reshape(-1)).all()
                        and (got[:, 1] == dx.reshape(-1)).all())
            ridx = torch.randint(0, N, (RECIPE["batch_size"],), device=dev)
            yield pi, pj, ridx

    def log(ep, loss, parts=None):
        d = integrality(W.detach().cpu().numpy())
        d.update(epoch=ep, loss=loss, time=time.time() - t0, **(parts or {}))
        hist.append(d)
        pt = "".join(f"  {k} {v:.5f}" for k, v in (parts or {}).items())
        print(f"ep {ep:4d}  loss {loss:.5f}{pt}  row dev median {d['dev_median']:.3f}"
              f"  integral(<0.05) {d['frac_int_05']:.3f}"
              f"  (<0.01) {d['frac_int_01']:.3f}", flush=True)

    def snap(ep):
        save_ideal_net(net, os.path.join(a.out, f"ideal_net_ep{ep}.pt"),
                       epoch=ep, args=vars(a))

    t0 = time.time()
    log(0, float("nan"))
    if 0 in a.save_at:
        snap(0)
    for ep in range(1, a.epochs + 1):
        run, nb = 0.0, 0
        acc = {"mse": 0.0, "inv": 0.0, "rate": 0.0}
        e_on = ep - 1 - a.inv_delay_epochs           # epochs since the delay ended
        if e_on < 0:
            inv_lam = 0.0
        else:
            inv_lam = a.inv_lambda * (min(1.0, e_on / a.inv_ramp_epochs)
                                      if a.inv_ramp_epochs > 0 else 1.0)
        noise_t = a.noise_std * max(0.0, 1.0 - (ep - 1) / a.epochs)
        for pi, pj, ridx in pair_batches():
            def code(ix):
                th = phases[ix] @ W.T
                return torch.cat([torch.cos(th), torch.sin(th)], 1) / math.sqrt(F)
            k = (code(pi) * code(pj)).sum(1)
            d2 = (coords[pi] - coords[pj]).square().sum(1).double()
            err2 = (k - torch.exp(-d2 / two_r2)).square()
            if a.near_weight:
                dbin = (d2.sqrt() / 2).long()
                w = 1.0 / torch.bincount(dbin)[dbin].double()
                mse = (w * err2).sum() / w.sum()
            else:
                mse = err2.mean()
            loss = mse
            acc["mse"] += mse.item()
            if a.inv_lambda > 0:
                # pairs with the same displacement should have the same similarity
                dxy = (coords[pj] - coords[pi]).long()
                flip = (dxy[:, 0] < 0) | ((dxy[:, 0] == 0) & (dxy[:, 1] < 0))
                dxy = torch.where(flip[:, None], -dxy, dxy)
                key = (dxy[:, 0] + 128) * 512 + (dxy[:, 1] + 128)
                _, gid = torch.unique(key, return_inverse=True)
                cnt = torch.bincount(gid).double()
                gmean = torch.zeros_like(cnt).scatter_add(0, gid, k) / cnt
                valid = cnt[gid] > 1
                inv = (k - gmean[gid])[valid].square().mean()
                loss = loss + inv_lam * inv
                acc["inv"] += inv.item()
            if a.rate_lambda > 0:
                rate = coding_rate_loss(code(ridx), eps=a.rate_eps)
                loss = loss + a.rate_lambda * rate
                acc["rate"] += rate.item()
            opt.zero_grad()
            loss.backward()
            if a.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_([W], a.grad_clip)
            opt.step()
            if noise_t > 0:
                with torch.no_grad():
                    W.add_(torch.randn_like(W) * noise_t)
            run += loss.item()
            nb += 1
        if ep % a.log_every == 0 or ep == a.epochs:
            log(ep, run / max(nb, 1),
                {k_: v / max(nb, 1) for k_, v in acc.items()})
        if ep in a.save_at:
            snap(ep)

    snap(a.epochs)
    with open(os.path.join(a.out, "history.json"), "w") as f:
        json.dump(hist, f, indent=1)
    diag = diagnose(W.detach().cpu().numpy(), r=a.r, device=dev)
    diag["args"] = vars(a)
    with open(os.path.join(a.out, "diagnostics.json"), "w") as f:
        json.dump(diag, f, indent=1)
    print(json.dumps({k: v for k, v in diag.items()
                      if not isinstance(v, (list, dict))}, indent=1))


if __name__ == "__main__":
    main()
