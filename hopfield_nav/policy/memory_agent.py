"""Idea 4's memory inside the policy, trained by RL (GRID_MLP_NAV_PLAN §7b.15,
arms 3 and 4).

``MemoryNavAgent`` is the idea 1 agent (``NavAgent``) with idea 4's recurrent
memory built in. Each step the memory reads the ``memory_input`` channel --
the view on the direction-aligned canvas, the agent's own grid phases and the
oracle store flag, the inputs ``analysis/idea4`` trained it on -- and returns
goal phases and a confidence logit. Those become idea 1's two policy inputs:

- d: the goal phases turned into a grid code and passed, with the current
  cell's code, through the frozen grid MLP (``memory/grid_mlp.py``). The
  phases become a code by bilinear interpolation of each module's smoothed
  table (``memory/grid_code.py``) at the continuous phase, exact at whole
  cells, so gradients reach the memory (the frozen arms pick the best cell
  instead, which has none);
- c: the sigmoid of the logit times ``recurrent_c_scale``.

They replace whatever the collector put in the ``hopfield_signal`` and
``memory_conf`` slots, and the rest of the input goes to the policy core as
in ``NavAgent``. The memory's state rides behind the policy's in the one
recurrent state the rest of the code passes around, (1, B, policy width +
memory width), so the collector, ``freeze_done_state`` and PPO's replay of a
packed lifetime from a zero state need no changes. PPO trains the memory
with everything else; there is no memory loss.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..memory.recurrent import RecurrentMemory
from .agent import NavAgent
from .channels import MEMORY_CANVAS_WIDTH, MEMORY_INPUT_WIDTH


def channel_offsets(specs) -> dict[str, int]:
    out, off = {}, 0
    for s in specs:
        out[s.name] = off
        off += s.width
    return out


class MemoryNavAgent(NavAgent):
    has_trainable_memory = True

    def __init__(self, cfg, input_dim: int, action_bounds=None, *, specs,
                 lambdas, grid_tables, grid_mlp):
        super().__init__(cfg, input_dim - MEMORY_INPUT_WIDTH, action_bounds)
        # The observation is wider than the policy core's input (the memory
        # channel is consumed here); buffers that store observations size by this.
        self.obs_dim = int(input_dim)
        if int(getattr(cfg, "num_rnn_layers", 1)) != 1:
            raise ValueError("MemoryNavAgent keeps the memory state beside a "
                             "1-layer policy state")
        off = channel_offsets(specs)
        for name in ("hopfield_signal", "memory_conf", "memory_input"):
            if name not in off:
                raise ValueError(f"MemoryNavAgent needs the {name!r} channel")
        widths = {s.name: s.width for s in specs}
        if widths["hopfield_signal"] != 2 or widths["memory_conf"] != 1:
            raise ValueError("MemoryNavAgent fills a 2-wide d slot and a 1-wide c slot "
                             f"(continuous hopfield_mode); got {widths}")
        self._hs, self._mc, self._mi = off["hopfield_signal"], off["memory_conf"], off["memory_input"]
        if not (self._hs < self._mc < self._mi == input_dim - MEMORY_INPUT_WIDTH):
            raise ValueError(f"unexpected channel order {off}")
        self._hp = int(cfg.hidden_size)
        self.memory = RecurrentMemory(hidden=int(cfg.trainable_memory_hidden), enc_kind="none",
                                      cell="lstm")
        init = getattr(cfg, "trainable_memory_init", "") or ""
        if init:
            ck = torch.load(init, map_location="cpu", weights_only=False)
            a = ck["args"]
            if (a["hidden"], a.get("cell"), a.get("enc_kind")) != (
                    int(cfg.trainable_memory_hidden), "lstm", "none"):
                raise ValueError(f"{init}: hidden {a['hidden']} cell {a.get('cell')} "
                                 f"enc {a.get('enc_kind')}; this agent's memory is "
                                 f"{cfg.trainable_memory_hidden} lstm none")
            self.memory.load_state_dict(ck["model"])
        self.lambdas = [int(l) for l in lambdas]
        for m, t in enumerate(grid_tables):
            self.register_buffer(f"grid_table_{m}", torch.as_tensor(t, dtype=torch.float32))
        self.grid_mlp = grid_mlp
        for p in self.grid_mlp.parameters():
            p.requires_grad_(False)
        self.c_scale = float(getattr(cfg, "recurrent_c_scale", 0.5))

    def soft_code(self, ph: torch.Tensor) -> torch.Tensor:
        """(..., 12) phases -> (..., Ng) smoothed grid code, bilinear in the
        continuous phase of each module and axis."""
        parts = []
        two_pi = 2 * math.pi
        for m, lam in enumerate(self.lambdas):
            table = getattr(self, f"grid_table_{m}")              # (lam, lam, lam^2)
            p = ph[..., 4 * m:4 * m + 4]
            px = torch.remainder(torch.atan2(p[..., 1], p[..., 0]), two_pi) * (lam / two_pi)
            py = torch.remainder(torch.atan2(p[..., 3], p[..., 2]), two_pi) * (lam / two_pi)
            x0, y0 = torch.floor(px), torch.floor(py)
            fx, fy = (px - x0).unsqueeze(-1), (py - y0).unsqueeze(-1)
            i0, j0 = x0.long() % lam, y0.long() % lam
            i1, j1 = (i0 + 1) % lam, (j0 + 1) % lam
            parts.append((1 - fx) * (1 - fy) * table[i0, j0] + fx * (1 - fy) * table[i1, j0]
                         + (1 - fx) * fy * table[i0, j1] + fx * fy * table[i1, j1])
        return torch.cat(parts, dim=-1)

    def memory_read(self, mi: torch.Tensor, h_mem: torch.Tensor):
        """mi (B, T, MEMORY_INPUT_WIDTH) -> d (B, T, 2), c (B, T), new state."""
        canv = mi[..., :MEMORY_CANVAS_WIDTH].transpose(0, 1)
        extra = mi[..., MEMORY_CANVAS_WIDTH:].transpose(0, 1)
        ph, logit, h_mem = self.memory(canv, h_mem, extra=extra)
        ph, logit = ph.transpose(0, 1), logit.transpose(0, 1)
        own = mi[..., MEMORY_CANVAS_WIDTH:MEMORY_CANVAS_WIDTH + 12]
        out = self.grid_mlp.head(self.grid_mlp.core.net(
            torch.cat([self.soft_code(own), self.soft_code(ph)], dim=-1)))
        d = out / (out.norm(dim=-1, keepdim=True) + 1e-6)
        return d, torch.sigmoid(logit) * self.c_scale, h_mem

    def forward(self, x, h=None, return_features: bool = False):
        B = x.shape[0]
        if h is None:
            h_pol, h_mem = None, self.memory.init_state(B, x.device)
        else:
            h_pol = h[..., :self._hp].contiguous()
            h_mem = h[..., self._hp:].contiguous()
        d, c, h_mem = self.memory_read(x[..., self._mi:], h_mem)
        xp = torch.cat([x[..., :self._hs], d, x[..., self._hs + 2:self._mc],
                        c.unsqueeze(-1), x[..., self._mc + 1:self._mi]], dim=-1)
        res = super().forward(xp, h_pol, return_features=return_features)
        h_next = torch.cat([res[3], h_mem], dim=-1)
        return (*res[:3], h_next, *res[4:])


def build_agent(cfg, input_dim: int, action_bounds=None, *, specs=None):
    """``NavAgent``, or ``MemoryNavAgent`` when ``cfg.agent.trainable_memory``.

    ``cfg`` is the full TrainConfig (the memory agent needs the scaffold's
    lambdas and smoothing for its grid tables, and the grid MLP path)."""
    a = cfg.agent
    if not getattr(a, "trainable_memory", False):
        return NavAgent(a, input_dim, action_bounds=action_bounds)
    from ..memory.grid_code import SmoothedGridCode
    from ..memory.grid_mlp import load_grid_mlp
    lambdas = list(cfg.vectorhash.lambdas)
    npos = cfg.vectorhash.Npos or int(math.prod(lambdas))
    gc = SmoothedGridCode(lambdas, cfg.fwhm_ratio, npos)
    return MemoryNavAgent(a, input_dim, action_bounds, specs=specs, lambdas=gc.lambdas,
                          grid_tables=gc._tables, grid_mlp=load_grid_mlp(a.grid_mlp_checkpoint))


__all__ = ["MemoryNavAgent", "build_agent", "channel_offsets"]
