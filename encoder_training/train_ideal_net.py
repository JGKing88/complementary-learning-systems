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

from analysis.hopfield_probe.ideal_net import IdealNet, save_ideal_net
from encoder_training.data import (build_patch_codes, mixed_batch_iterator,
                                   sample_nonoverlapping_patches)
from encoder_training.ideal_net_diagnostics import diagnose, integrality

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
    opt = torch.optim.Adam([W], lr=a.lr)
    F = W.shape[0]
    two_r2 = 2.0 * a.r * a.r

    hist = []

    def log(ep, loss):
        d = integrality(W.detach().cpu().numpy())
        d.update(epoch=ep, loss=loss, time=time.time() - t0)
        hist.append(d)
        print(f"ep {ep:4d}  loss {loss:.5f}  row dev median {d['dev_median']:.3f}"
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
        for idx in mixed_batch_iterator(N, RECIPE["batch_size"]):
            idx = idx.to(dev).long()
            e = env_ids[idx]
            ii, jj = torch.triu_indices(len(idx), len(idx), 1, device=dev)
            keep = e[ii] == e[jj]
            ii, jj = ii[keep], jj[keep]
            th = phases[idx] @ W.T                               # (B, F)
            z = torch.cat([torch.cos(th), torch.sin(th)], 1) / math.sqrt(F)
            k = (z[ii] * z[jj]).sum(1)
            d2 = (coords[idx[ii]] - coords[idx[jj]]).square().sum(1).double()
            loss = (k - torch.exp(-d2 / two_r2)).square().mean()
            opt.zero_grad()
            loss.backward()
            if a.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_([W], a.grad_clip)
            opt.step()
            run += loss.item()
            nb += 1
        if ep % a.log_every == 0 or ep == a.epochs:
            log(ep, run / max(nb, 1))
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
