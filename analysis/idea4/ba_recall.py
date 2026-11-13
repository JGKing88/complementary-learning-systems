"""Sanity check for idea 4: associative retrieval in the format of Ba et al.
2016 ("Using fast weights to attend to the recent past"), then one change at
a time toward idea 4's format, to find what stops a recurrent net learning.

    python -m analysis.idea4.ba_recall --cell lstm --H 100 --pairs 4

A sequence holds P key-value pairs and then a query key; the target is the
query's value. Ba et al. present each key and each value as its own step,
keys from 26 letters and values from 10 digits, and report that LSTMs of 50
or more units learn it. Options move toward idea 4:
  --together      key and value in the same step (idea 4's writes)
  --values phase  values are 12 random unit phase numbers, cosine loss
  --keys random   keys are random +-1 vectors (--key_dim), fresh per sequence
Readout is linear from the last hidden state (Ba et al.), or idea 4's MLP on
the hidden state and the query input with --mlp_read.
"""
from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cell", default="lstm", choices=["lstm", "gru"])
    p.add_argument("--H", type=int, default=100)
    p.add_argument("--pairs", type=int, default=4)
    p.add_argument("--n_keys", type=int, default=26)
    p.add_argument("--keys", default="onehot", choices=["onehot", "random"])
    p.add_argument("--key_dim", type=int, default=26)
    p.add_argument("--values", default="digit", choices=["digit", "phase"])
    p.add_argument("--together", action="store_true")
    p.add_argument("--mlp_read", action="store_true")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    P, B = a.pairs, a.batch
    kdim = a.n_keys if a.keys == "onehot" else a.key_dim
    vdim = 10 if a.values == "digit" else 12
    D = kdim + vdim + 1                                   # + query marker
    rnn = (nn.LSTM if a.cell == "lstm" else nn.GRU)(D, a.H).to(dev)
    if a.cell == "lstm":
        with torch.no_grad():
            rnn.bias_ih_l0[a.H:2 * a.H] = 1.0
    read = (nn.Sequential(nn.Linear(a.H + D, 128), nn.ReLU(), nn.Linear(128, vdim))
            if a.mlp_read else nn.Linear(a.H, vdim)).to(dev)
    opt = torch.optim.Adam(list(rnn.parameters()) + list(read.parameters()), lr=a.lr)
    tag = (f"{a.cell} H{a.H} pairs {P} keys {a.keys}{'' if a.keys == 'onehot' else a.key_dim} "
           f"values {a.values} {'together' if a.together else 'separate'}"
           f"{' mlp_read' if a.mlp_read else ''}")
    t0, run = time.time(), []
    for step in range(1, a.steps + 1):
        if a.keys == "onehot":
            idx = torch.rand(B, a.n_keys, device=dev).argsort(1)[:, :P]          # distinct keys
            keys = F.one_hot(idx, a.n_keys).float()                              # (B, P, kdim)
        else:
            keys = torch.randint(0, 2, (B, P, kdim), device=dev).float() * 2 - 1
        if a.values == "digit":
            vidx = torch.randint(0, 10, (B, P), device=dev)
            vals = F.one_hot(vidx, 10).float()
        else:
            ang = torch.rand(B, P, 6, device=dev) * 2 * math.pi
            vals = torch.stack([ang.cos(), ang.sin()], -1).reshape(B, P, 12)
        z_k, z_v = torch.zeros(B, P, kdim, device=dev), torch.zeros(B, P, vdim, device=dev)
        z_m = torch.zeros(B, P, 1, device=dev)
        if a.together:
            seq = torch.cat([keys, vals, z_m], -1)                               # (B, P, D)
        else:
            ks = torch.cat([keys, z_v, z_m], -1)
            vs = torch.cat([z_k, vals, z_m], -1)
            seq = torch.stack([ks, vs], 2).reshape(B, 2 * P, D)
        qi = torch.randint(0, P, (B,), device=dev)
        ar = torch.arange(B, device=dev)
        query = torch.cat([keys[ar, qi], torch.zeros(B, vdim, device=dev),
                           torch.ones(B, 1, device=dev)], -1)[:, None]
        x = torch.cat([seq, query], 1).transpose(0, 1)                           # (L, B, D)
        y, _ = rnn(x)
        hT = y[-1]
        out = read(torch.cat([hT, x[-1]], -1) if a.mlp_read else hT)
        if a.values == "digit":
            loss = F.cross_entropy(out, vidx[ar, qi])
            score = (out.argmax(-1) == vidx[ar, qi]).float().mean()
        else:
            o = out.reshape(B, 6, 2)
            o = o / (o.norm(dim=-1, keepdim=True) + 1e-6)
            cos = (o * vals[ar, qi].reshape(B, 6, 2)).sum(-1).mean(-1)
            loss, score = (1 - cos).mean(), cos.mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(rnn.parameters()) + list(read.parameters()), 1.0)
        opt.step()
        run.append(float(score))
        if step % 1000 == 0:
            print(f"{tag} | step {step}: {'accuracy' if a.values == 'digit' else 'cosine'} "
                  f"{sum(run) / len(run):.3f} ({time.time() - t0:.0f}s)", flush=True)
            run = []


if __name__ == "__main__":
    main()
