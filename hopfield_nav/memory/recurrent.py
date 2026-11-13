"""Idea 4's memory: a recurrent network whose activity holds the goals
(GRID_MLP_NAV_PLAN §7b).

The arrangement is ideas 1 and 2's, with only the memory module changed. Its
weights are fixed after training, so a stored goal can live only in the
hidden state, which is never reset within a lifetime.

Each step the module reads the same query as ideas 1 and 2, the live view on
the direction-aligned canvas (``plastic.canvas``). During a write it also gets
the goal's grid phases (12 numbers) and a write flag of 1; at other steps both
are zero. A view encoder (idea 2's 360 -> 256 -> 256 ReLU trunk) feeds a GRU,
and a read MLP on the GRU's output and the encoded view returns the goal
phases (unit cos, sin pairs, as ``plastic.forward``) and the logit of ``c``,
the confidence that this env's goal is stored.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..world.env import PANORAMA_BINS
from .plastic import N_PHASE


class RecurrentMemory(nn.Module):
    def __init__(self, hidden: int = 512, layers: int = 1, enc: int = 256,
                 read: int = 256, in_dim: int = 2 * PANORAMA_BINS, chrono: int = 0,
                 enc_kind: str = "relu"):
        super().__init__()
        self.hidden, self.layers = hidden, layers
        # The view encoder. "relu" (the first runs) ends in a ReLU, so every
        # code lies in the positive orthant and different arenas' codes
        # overlap heavily; "linear" drops that last ReLU (signed codes);
        # "none" feeds the raw canvas.
        if enc_kind == "none":
            self.enc, enc = nn.Identity(), in_dim
        elif enc_kind == "linear":
            self.enc = nn.Sequential(nn.Linear(in_dim, enc), nn.ReLU(), nn.Linear(enc, enc))
        elif enc_kind == "relu":
            self.enc = nn.Sequential(nn.Linear(in_dim, enc), nn.ReLU(),
                                     nn.Linear(enc, enc), nn.ReLU())
        else:
            raise ValueError(f"enc_kind must be relu, linear or none, got {enc_kind!r}")
        self.gru = nn.GRU(enc + N_PHASE + 1, hidden, num_layers=layers)
        self.read = nn.Sequential(nn.Linear(hidden + enc, read), nn.ReLU(),
                                  nn.Linear(read, N_PHASE + 1))
        if chrono:
            self.chrono_init(chrono)

    @torch.no_grad()
    def chrono_init(self, T_max: int):
        """Chrono initialisation (Tallec & Ollivier 2018). PyTorch's default
        leaves the update gate z near 0.5, a memory half-life of about one
        step, so a goal written 40 steps before the next write leaves no trace
        and no gradient. Here z's bias is log(u), u ~ U(1, T_max - 1), so unit i
        keeps its state for about u_i steps at the start of training. PyTorch
        packs the gates as (reset, update, new) and sets
        h' = (1 - z) * n + z * h."""
        H = self.hidden
        for k in range(self.layers):
            b_ih = getattr(self.gru, f"bias_ih_l{k}")
            b_hh = getattr(self.gru, f"bias_hh_l{k}")
            u = torch.empty(H).uniform_(1.0, T_max - 1.0)
            b_ih[H:2 * H] = torch.log(u)
            b_hh[H:2 * H] = 0.0

    def init_state(self, B: int, device) -> torch.Tensor:
        return torch.zeros(self.layers, B, self.hidden, device=device)

    def forward(self, x, h, goal=None, extra=None):
        """Step through a sequence.

        x (T, B, 360) canvas views; h (layers, B, hidden). The 13 extra inputs
        are either ``extra`` (T, B, 13) as given, which the experience-based
        trainer uses for the agent's own grid phases and the at-goal flag, or
        ``goal`` (T, B, 12) goal phases plus a write flag of 1, or zeros.
        Returns phases (T, B, 12), the c logit (T, B) and the new state."""
        e = self.enc(x)
        flag = torch.zeros(*x.shape[:2], 1, device=x.device)
        if extra is not None:
            w = extra
        elif goal is None:
            w = torch.zeros(*x.shape[:2], N_PHASE + 1, device=x.device)
        else:
            w = torch.cat([goal, flag + 1.0], -1)
        y, h = self.gru(torch.cat([e, w], -1), h)
        out = self.read(torch.cat([y, e], -1))
        ph = out[..., :N_PHASE].reshape(*out.shape[:-1], 6, 2)
        ph = ph / (ph.norm(dim=-1, keepdim=True) + 1e-6)
        return ph.reshape(*out.shape[:-1], N_PHASE), out[..., N_PHASE], h
