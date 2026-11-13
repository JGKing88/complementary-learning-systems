"""Which recurrent memories learn plain associative recall? (GRID_MLP_NAV_PLAN §7b)

    python -m analysis.idea4.recall_probe --cell gru --pairs 2 --steps 20000

A stripped-down version of idea 4's memory test, with no arenas or views.
Each lifetime draws fresh random keys (+-1 vectors, as distal panoramas are)
and random goals (unit phase pairs). The memory is written key + goal for
each pair in turn, each write repeated --rep steps, then queried with each key
for --dwell steps from a copy of the state; the loss is 1 - cos between the
read-out and that key's goal. Outputting the bisector of two goals scores
2/pi = 0.64 on both, so 0.64 at two pairs means no binding at all.

Cells: gru, lstm (nn modules), migru (GRU with multiplicative integration,
Wu et al. 2016), and fw, a positive control whose state is an outer-product
fast-weight matrix (M += e(key) g^T, read M^T e(query)).
"""
from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn as nn

N_PHASE = 12


class MIGRUCell(nn.Module):
    """GRU whose gates multiply input and recurrent projections:
    act(alpha * Wx * Uh + b1 * Wx + b2 * Uh + b)."""

    def __init__(self, D, H):
        super().__init__()
        self.W = nn.Linear(D, 3 * H)
        self.U = nn.Linear(H, 3 * H, bias=False)
        self.alpha = nn.Parameter(torch.ones(3 * H))
        self.b1 = nn.Parameter(torch.ones(3 * H))
        self.b2 = nn.Parameter(torch.ones(3 * H))
        self.H = H

    def forward(self, x, h):
        wx, uh = self.W(x), self.U(h)
        H = self.H
        a, b1, b2 = self.alpha, self.b1, self.b2

        def mix(i, w, u):
            s = slice(i * H, (i + 1) * H)
            return a[s] * w * u + b1[s] * w + b2[s] * u

        r = torch.sigmoid(mix(0, wx[..., :H], uh[..., :H]))
        z = torch.sigmoid(mix(1, wx[..., H:2 * H], uh[..., H:2 * H]))
        n = torch.tanh(mix(2, wx[..., 2 * H:], r * uh[..., 2 * H:]))
        return (1 - z) * n + z * h


class Memory(nn.Module):
    def __init__(self, cell, H, key_in, enc=128, read=128, layers=1, chrono=0, enc_kind="relu"):
        super().__init__()
        self.cell, self.H, self.layers = cell, H, layers
        # "relu": idea 4's encoder, ending in a ReLU (all codes in the positive
        # orthant); "linear": the same without the last ReLU (signed codes);
        # "none": the raw input.
        if enc_kind == "none":
            self.enc, enc = nn.Identity(), key_in
        elif enc_kind == "linear":
            self.enc = nn.Sequential(nn.Linear(key_in, enc), nn.ReLU(), nn.Linear(enc, enc))
        else:
            self.enc = nn.Sequential(nn.Linear(key_in, enc), nn.ReLU(), nn.Linear(enc, enc), nn.ReLU())
        D = enc + N_PHASE + 1
        if cell == "gru":
            self.rnn = nn.GRU(D, H, num_layers=layers)
        elif cell == "lstm":
            self.rnn = nn.LSTM(D, H, num_layers=layers)
            with torch.no_grad():
                for k in range(layers):
                    getattr(self.rnn, f"bias_ih_l{k}")[H:2 * H] = 1.0     # forget gate
        elif cell == "migru":
            self.rnn = MIGRUCell(D, H)
        elif cell == "fw":
            self.kproj = nn.Linear(enc, H)
            self.vproj = nn.Linear(N_PHASE, N_PHASE)
        if cell == "gru" and chrono:
            with torch.no_grad():
                for k in range(layers):
                    getattr(self.rnn, f"bias_ih_l{k}")[H:2 * H] = torch.log(
                        torch.empty(H).uniform_(1, chrono - 1))
                    getattr(self.rnn, f"bias_hh_l{k}")[H:2 * H] = 0.0
        out_in = (N_PHASE if cell == "fw" else H) + enc
        self.read = nn.Sequential(nn.Linear(out_in, read), nn.ReLU(), nn.Linear(read, N_PHASE))

    def init_state(self, B, device):
        z = torch.zeros(self.layers, B, self.H, device=device)
        if self.cell == "lstm":
            return (z, z.clone())
        if self.cell == "migru":
            return z[0]
        if self.cell == "fw":
            return torch.zeros(B, self.H, N_PHASE, device=device)
        return z

    def step(self, x, h, goal=None):
        """x (T, B, key_in) -> (read-out (T, B, 12) of the last step, h)."""
        e = self.enc(x)
        T, B = x.shape[:2]
        if goal is None:
            w = torch.zeros(T, B, N_PHASE + 1, device=x.device)
        else:
            w = torch.cat([goal, torch.ones(T, B, 1, device=x.device)], -1)
        if self.cell in ("gru", "lstm"):
            y, h = self.rnn(torch.cat([e, w], -1), h)
            y = y[-1]
        elif self.cell == "migru":
            inp = torch.cat([e, w], -1)
            for t in range(T):
                h = self.rnn(inp[t], h)
            y = h
        else:                                   # fw: outer-product fast weights
            k = self.kproj(e[-1])
            k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
            if goal is not None:
                h = h + k[:, :, None] * self.vproj(goal[-1])[:, None, :]
            y = (k[:, :, None] * h).sum(1)
        out = self.read(torch.cat([y, e[-1]], -1)).reshape(B, 6, 2)
        return (out / (out.norm(dim=-1, keepdim=True) + 1e-6)).reshape(B, N_PHASE), h


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cell", default="gru", choices=["gru", "lstm", "migru", "fw"])
    p.add_argument("--H", type=int, default=256)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--pairs", type=int, default=2)
    p.add_argument("--rep", type=int, default=1, help="steps per write")
    p.add_argument("--dwell", type=int, default=1, help="steps per query")
    p.add_argument("--key_dim", type=int, default=180,
                   help="random +-1 key entries; the rest of the 180 slots are zero")
    p.add_argument("--enc", default="relu", choices=["relu", "linear", "none"],
                   help="view encoder before the cell: idea 4's (ends in ReLU), without "
                        "its last ReLU, or none")
    p.add_argument("--fixed_keys", action="store_true",
                   help="control: the same keys in every lifetime, so recall is by write order")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--chrono", type=int, default=0)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    m = Memory(a.cell, a.H, 360, layers=a.layers, chrono=a.chrono, enc_kind=a.enc).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=a.lr)
    P, B = a.pairs, a.batch
    tag = (f"{a.cell} H{a.H} L{a.layers} enc {a.enc} pairs {P} rep {a.rep} dwell {a.dwell} "
           f"key_dim {a.key_dim} lr {a.lr:g} chrono {a.chrono}")
    t0, run = time.time(), []
    fixed = torch.randint(0, 2, (P, 180), device=dev).float() * 2 - 1
    for step in range(1, a.steps + 1):
        keys = torch.zeros(P, B, 180, device=dev)
        keys[..., :a.key_dim] = torch.randint(0, 2, (P, B, a.key_dim), device=dev).float() * 2 - 1
        if a.fixed_keys:                   # control: the same P keys in every lifetime
            keys = fixed[:, None].expand(P, B, 180)
        x = torch.cat([keys, (keys != 0).float()], -1)
        ang = torch.rand(P, B, 6, device=dev) * 2 * math.pi
        g = torch.stack([ang.cos(), ang.sin()], -1).reshape(P, B, N_PHASE)
        h = m.init_state(B, dev)
        for i in range(P):
            _, h = m.step(x[i:i + 1].expand(a.rep, B, 360), h, g[i:i + 1].expand(a.rep, B, N_PHASE))
        q = x.reshape(1, P * B, 360).expand(a.dwell, P * B, 360)
        ph, _ = m.step(q, _tile(h, P))
        cos = (ph.reshape(P, B, 6, 2) * g.reshape(P, B, 6, 2)).sum(-1).mean(-1)
        loss = (1 - cos).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        run.append(cos.detach().mean(1))
        if step % 1000 == 0:
            c = torch.stack(run).mean(0)
            print(f"{tag} | step {step}: cos per pair " + " ".join(f"{float(v):.2f}" for v in c)
                  + f" ({time.time() - t0:.0f}s)", flush=True)
            run = []


def _tile(h, P):
    """Repeat a batch-B state P times along batch (pair-major), matching the
    (P, B) -> P * B reshape of the queries."""
    if isinstance(h, tuple):
        return tuple(_tile(t, P) for t in h)
    if h.dim() == 3 and h.shape[-1] == N_PHASE:        # fw: (B, H, 12)
        return h.repeat(P, 1, 1)
    if h.dim() == 2:                                   # migru: (B, H)
        return h.repeat(P, 1)
    return h.repeat(1, P, 1)                           # (layers, B, H)


if __name__ == "__main__":
    main()
