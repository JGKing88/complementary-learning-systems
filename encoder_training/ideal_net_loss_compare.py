#!/usr/bin/env python3
"""Evaluate several IdealNet harmonic layers on the SAME training batches.

docs/EXPERIMENTS_IDEAL_ENCODER.md Sec 13.1. Paired comparison of the loss
components Stage A trains on: within-patch kernel MSE (r = 16), the coding-rate
term (``losses.coding_rate_loss``, eps = 1) and the translation-invariance
penalty, plus totals under MSE + 0.5 x rate (with and without + 300 x inv).
Patches are the att0.5 recipe's (seed 0); batches come from one fixed RNG seed,
so every network sees identical pairs.
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from analysis.hopfield_probe.ideal_net import IdealNet, load_ideal_net
from encoder_training.data import (build_patch_codes, mixed_batch_iterator,
                                   sample_nonoverlapping_patches)
from encoder_training.losses import coding_rate_loss

LAMBDAS, NPOS, R = [11, 12, 13], 1716, 16.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nets", nargs="+", required=True,
                    help="label=path.pt, or label=ideal for the exact integers")
    ap.add_argument("--n_batches", type=int, default=40)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    np.random.seed(0); torch.manual_seed(0)
    y0s, x0s, sizes = sample_nonoverlapping_patches(NPOS, NPOS, [50] * 118, None,
                                                    placement="random")
    Phi, coords, env = build_patch_codes(LAMBDAS, y0s, x0s, sizes, dev, 0.25)
    ref = IdealNet(r=R).to(dev)
    with torch.no_grad():
        ph = torch.cat([ref.phase(ref.readout(Phi[i:i + 65536].double()))
                        for i in range(0, Phi.shape[0], 65536)])
    del Phi

    nets = {}
    for spec in a.nets:
        lab, _, path = spec.partition("=")
        if path == "ideal":
            nets[lab] = ref.integer_harmonics.to(dev).double()
        else:
            net, _ = load_ideal_net(path)
            nets[lab] = net.harmonics.weight.detach().to(dev).double()

    torch.manual_seed(1)
    batches = []
    for idx in mixed_batch_iterator(ph.shape[0], 4096):
        batches.append(idx.to(dev).long())
        if len(batches) == a.n_batches:
            break

    acc = {k: {"mse": [], "rate": [], "inv": []} for k in nets}
    with torch.no_grad():
        for idx in batches:
            e = env[idx]
            ii, jj = torch.triu_indices(len(idx), len(idx), 1, device=dev)
            keep = e[ii] == e[jj]; ii, jj = ii[keep], jj[keep]
            d2 = (coords[idx[ii]] - coords[idx[jj]]).square().sum(1).double()
            t = torch.exp(-d2 / (2 * R * R))
            dxy = (coords[idx[jj]] - coords[idx[ii]]).long()
            flip = (dxy[:, 0] < 0) | ((dxy[:, 0] == 0) & (dxy[:, 1] < 0))
            dxy = torch.where(flip[:, None], -dxy, dxy)
            _, gid = torch.unique((dxy[:, 0] + 128) * 512 + (dxy[:, 1] + 128),
                                  return_inverse=True)
            cnt = torch.bincount(gid).double(); valid = cnt[gid] > 1
            for lab, W in nets.items():
                th = ph[idx] @ W.T
                z = torch.cat([torch.cos(th), torch.sin(th)], 1) / math.sqrt(W.shape[0])
                k = (z[ii] * z[jj]).sum(1)
                acc[lab]["mse"].append(float((k - t).square().mean()))
                acc[lab]["rate"].append(float(coding_rate_loss(z, eps=1.0)))
                g = torch.zeros_like(cnt).scatter_add(0, gid, k) / cnt
                acc[lab]["inv"].append(float((k - g[gid])[valid].square().mean()))

    base = None
    print(f"{len(batches)} paired batches, device {dev}")
    print(f"{'net':26s} {'integral':>8s} {'mse':>9s} {'rate':>9s} {'inv':>10s} "
          f"{'MSE+0.5rate':>12s} {'+300inv':>10s} {'Δ vs first (MSE+0.5rate)':>26s}")
    for lab, W in nets.items():
        dev_ = (W - torch.round(W)).abs().max(1).values
        m = {k: np.mean(v) for k, v in acc[lab].items()}
        tot = np.array(acc[lab]["mse"]) + 0.5 * np.array(acc[lab]["rate"])
        if base is None:
            base = tot
        diff = tot - base
        print(f"{lab:26s} {float((dev_ < 0.05).double().mean()):8.2f} {m['mse']:9.5f} "
              f"{m['rate']:9.5f} {m['inv']:10.2e} {tot.mean():12.5f} "
              f"{tot.mean() + 300 * m['inv']:10.5f} "
              f"{diff.mean():+.5f} ± {diff.std() / math.sqrt(len(diff)):.5f}")


if __name__ == "__main__":
    main()
