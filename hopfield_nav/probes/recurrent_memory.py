"""How long can this trunk remember, as trained?

The explore task's remaining headroom needs *spatial memory*: a policy that
covers more than ~0.51 has to know where it has already been, over a rollout
of 200-400 steps. Whether the trunk can hold anything that long is a property
of its recurrent weights, and it is readable off a checkpoint.

For a vanilla Elman cell the answer is the spectral radius ρ of `W_hh`. In the
linear regime around a fixed point, a perturbation decays as ρ^t, so the memory
time constant is `τ = -1 / ln ρ` steps. ReLU only ever attenuates further (it
zeroes half the pre-activations), so this is an *upper* bound on what the cell
retains. PyTorch initializes `W_hh` uniform on ±1/√hidden, which puts ρ near
0.5 and τ near 1.5 steps — so a fresh trunk remembers essentially nothing, and
every step of memory it ends up with is something training had to build.

A GRU is not a single matrix and does not have one ρ. Its retention is the
update gate: `h_t = (1-z) h_{t-1} + z * candidate`, so `1-z` is the per-step
retention and `τ = -1 / ln(1-z)`. Evaluated at zero input, i.e. from the gate
bias alone, which is what the cell does in the absence of evidence.

    python -m hopfield_nav.probes.recurrent_memory --ckpt <a.pt> <b.pt> ...
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import torch

from ..evaluation.checkpoint_io import cfg_from_checkpoint


def _tau(retention: float) -> float:
    """Steps for a perturbation to decay by 1/e. inf when non-contracting."""
    if retention >= 1.0:
        return float("inf")
    if retention <= 0.0:
        return 0.0
    return -1.0 / math.log(retention)


def report(path: str) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = cfg_from_checkpoint(ck["config"])
    sd = ck["agent_state_dict"]
    hidden = cfg.agent.hidden_size
    cell = cfg.agent.rnn_cell
    w = sd["rnn.weight_hh_l0"].double().numpy()

    out = {"ckpt": path, "cell": cell,
           "nonlinearity": cfg.agent.rnn_nonlinearity, "hidden": hidden}

    if w.shape[0] == hidden:                     # vanilla Elman
        rho = float(np.abs(np.linalg.eigvals(w)).max())
        out.update(spectral_radius=rho, retention=rho, tau_steps=_tau(rho))
    else:                                        # GRU: r, z, n stacked
        # torch order is (reset, update, new); the update gate is the middle.
        b_ih = sd.get("rnn.bias_ih_l0")
        b_hh = sd.get("rnn.bias_hh_l0")
        bias = torch.zeros(w.shape[0]) if b_ih is None else b_ih.clone()
        if b_hh is not None:
            bias = bias + b_hh
        z = torch.sigmoid(bias[hidden:2 * hidden]).double().numpy()
        keep = float(np.mean(1.0 - z))
        out.update(mean_update_gate=float(z.mean()), retention=keep,
                   tau_steps=_tau(keep),
                   spectral_radius=float(
                       np.abs(np.linalg.eigvals(w[hidden:2 * hidden])).max()))
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", nargs="+", required=True)
    p.add_argument("--output_json", default=None)
    args = p.parse_args()

    rows = [report(c) for c in args.ckpt]
    print(f"{'cell':<12} {'hidden':>6} {'retention':>10} {'tau (steps)':>12}  ckpt")
    for r in rows:
        tau = r["tau_steps"]
        print(f"{r['cell'] + '/' + r['nonlinearity']:<12} {r['hidden']:>6} "
              f"{r['retention']:>10.4f} "
              f"{('inf' if tau == float('inf') else f'{tau:.1f}'):>12}  "
              f"{r['ckpt'].split('/')[-2]}/{r['ckpt'].split('/')[-1]}")
    print("\nA rollout is 200-400 steps. tau is how many steps a perturbation "
          "survives;\nfor the vanilla cell it is an upper bound, since ReLU "
          "only attenuates further.")
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
