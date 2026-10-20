#!/usr/bin/env python3
"""Keep the K strongest waves of the flat-table optimum and refit their amplitudes.

docs/EXPERIMENTS_IDEAL_ENCODER.md Sec 14. The flat table (every half-plane
integer frequency with |n| <= 40, ~2500 waves) has a good optimum under kernel
MSE + 0.5 x coding rate (Sec 12), but it spreads weight over ~1700 waves, more
than the code's 1024 numbers hold. This keeps the K waves with the largest
weight and re-solves the same convex problem with only those allowed (D = 2K),
then saves ``freqs_K.npy`` and ``weights_K.npy`` for
``ideal:...,n_freq=K,freqs=...,weights=...``.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from encoder_training.ideal_net_rate_optimum import (describe, menus, optimise,
                                                     stats_for)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True,
                    help="weights_flat_lam0.5.npy from ideal_net_rate_optimum")
    ap.add_argument("--k", type=int, default=512)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--r", type=float, default=16.0)
    ap.add_argument("--flat_radius", type=int, default=40)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out, exist_ok=True)

    table = menus(a.r, a.flat_radius)["flat"]
    w = np.load(a.src)
    assert w.shape == (len(table),), (w.shape, len(table))
    keep = np.argsort(-w)[:a.k]
    n = table[keep]
    print(f"table {len(table)} waves; kept {a.k} carrying "
          f"{w[keep].sum():.3f} of the weight", flush=True)

    G, b, tt, M = stats_for(n, dev, a.r, 1_000_000, 60_000)
    out = {}
    for name, init in (("renormalised", None), ("refit", True)):
        if init is None:
            amp = torch.from_numpy(w[keep] / w[keep].sum()).to(dev)
            mse = float(amp @ G @ amp - 2 * amp @ b + tt)
            rate = float("nan")
        else:
            amp, mse, rate = optimise(G, b, tt, M, a.lam, 1.0, a.iters, dev)
        d = describe(amp, n, a.r, dev)
        d.update(mse=mse, rate=rate)
        out[name] = {k: v for k, v in d.items() if k != "kernel_axis_mean"}
        print(f"{name:13s} mse {mse:.6f}  waves(PR) {d['participation']:.1f}  "
              f"C(1) {d['c1']:.3f}  r_half {d['r_half']}  max>71 "
              f"{d['max_abs_beyond_71']:.3f}  far_sd {d['far_sd']:.4f}  "
              f"r_eff {d['r_eff']:.1f}", flush=True)
        if init is not None:
            np.save(os.path.join(a.out, f"freqs_{a.k}.npy"), n.astype(np.int64))
            np.save(os.path.join(a.out, f"weights_{a.k}.npy"), amp.cpu().numpy())
    with open(os.path.join(a.out, f"topk_{a.k}.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"saved freqs_{a.k}.npy, weights_{a.k}.npy -> {a.out}")


if __name__ == "__main__":
    main()
