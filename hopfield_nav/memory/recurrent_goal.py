"""Idea 4's trained memory in the agent (GRID_MLP_NAV_PLAN §7b.14, stage 3).

A frozen ``RecurrentMemory`` (trained by ``analysis/idea4/train_protocol.py``
or ``train_experience.py``) whose state holds the goals. It stands in for
idea 1's sensory key-value table, so every caller that handles
``memory_backend='sensory_kv'`` (the controller's d and c channels, the
evaluators' kv branches) works unchanged; ``GridMemoryReadout.signal`` hands
reads of these objects to ``recurrent_signal``.

Each read steps the network once on what it was trained on: the live view on
the direction-aligned canvas (``plastic.canvas``), the agent's own grid
phases, and the store flag. The flag must be set before the read
(``set_store_flag``), because the write happens inside that step;
``input_memory``, called after the read as for the other backends, only
counts. The oracle store is the only store this supports.

The read returns the goal phases and a confidence logit. The phases are
decoded to the best-matching cell of the current arena, whose grid code
goes through the frozen grid MLP with the current cell's to give d, as idea
1's table read does. c is the sigmoid of the logit times ``c_scale`` (0.5
puts a held goal near idea 1's own-goal cosine of about 0.5); below
``c_min`` d is zero and the read counts as empty, as idea 1's empty table.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

from .plastic import canvas, decode_cells, goal_phases
from .recurrent import RecurrentMemory


@lru_cache(maxsize=4)
def _load(path: str, device: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["args"]
    model = RecurrentMemory(hidden=a["hidden"], layers=a.get("layers", 1), enc=a.get("enc", 256),
                            read=a.get("read", 256), enc_kind=a.get("enc_kind", "relu"),
                            cell=a.get("cell", "gru"))
    model.load_state_dict(ck["model"])
    model.eval().requires_grad_(False)
    return model.to(device)


class RecurrentGoalMemory:
    """One lifetime's recurrent memory: the network (shared, frozen) and its
    state (layers, 1, width), never reset."""

    is_recurrent_goal_memory = True                 # GridMemoryReadout.signal dispatches on it

    def __init__(self, path: str, device="cpu", arena_size: int = 20,
                 c_scale: float = 0.5, c_min: float = 0.5, skip_stays: bool = False):
        self.path, self.device = str(path), str(device)
        self.model = _load(self.path, self.device)
        self.h = self.model.init_state(1, self.device)
        self.flag = 0.0
        self.num_memories = 0                      # stores, for the logs
        self.arena_size = int(arena_size)
        self.c_scale, self.c_min = float(c_scale), float(c_min)
        # skip_stays: step the network only when the agent's cell changes, and
        # repeat the last read otherwise. The agent moves continuously and is
        # snapped to cells, so its cell often stays put between steps; the
        # memory was trained on walks that move one cell every step.
        self.skip_stays = bool(skip_stays)
        self.last_cell, self.last_out = None, None

    def set_store_flag(self, on: bool) -> None:
        """Mark the next read as the store step (the oracle store fires)."""
        self.flag = 1.0 if on else 0.0

    def tag(self, **info) -> None:
        """Diagnostics only: the protocol says which block and arena the next
        reads are in, that arena's true goal (local cell) and whether its goal
        is stored. ``recurrent_signal`` then logs how the reads compare."""
        self.info = info
        self.last_cell, self.last_out = None, None      # a new mini-episode always steps
        if not hasattr(self, "stats"):
            self.stats = {}

    def block_stats(self, block: int) -> dict:
        """Per arena of ``block``: reads with the goal stored (far-read hit
        rate, cosine of d with the true direction, mean c) and before it
        (mean c, fraction with d on)."""
        out = {}
        for (b, j, known), s in sorted(getattr(self, "stats", {}).items()):
            if b != block:
                continue
            r = out.setdefault(j, {})
            n = max(s["n"], 1)
            tail = "" if known else "_unknown"
            r["stay_frac" + tail] = s["stay"] / n
            r["diag_frac" + tail] = s["diag"] / n
            if known:
                r.update(far_hit=s["hit"] / max(s["far"], 1), cos_d=s["cos"] / max(s["ncos"], 1),
                         c=s["c"] / n, reads=s["n"])
            else:
                r.update(c_unknown=s["c"] / n, d_on_unknown=s["on"] / n, reads_unknown=s["n"])
        return out

    def input_memory(self, entry) -> None:
        """The write happened inside the flagged read; only count it."""
        self.num_memories += 1

    def copy(self) -> "RecurrentGoalMemory":
        new = object.__new__(RecurrentGoalMemory)
        new.__dict__.update(self.__dict__)
        new.h = self.h.clone()
        return new

    @torch.no_grad()
    def step(self, view_canvas: np.ndarray, phases: np.ndarray):
        """One network step. Returns the goal phases (12,) and the c logit."""
        x = torch.from_numpy(view_canvas).float().to(self.device)[None, None]
        ex = torch.cat([torch.from_numpy(phases).float(),
                        torch.tensor([self.flag])]).to(self.device)[None, None]
        ph, c, self.h = self.model(x, self.h, extra=ex)
        self.flag = 0.0
        return ph[0, 0].cpu().numpy(), float(c[0, 0])


def recurrent_signal(readout, memories, views, psi, positions, env_offset):
    """``GridMemoryReadout.signal`` for ``RecurrentGoalMemory`` rows: returns
    ``(channel (B,2), q (B,2), has_memory (B,), c (B,))``."""
    lam = readout.grid_code.lambdas
    positions = np.asarray(positions).reshape(-1, 2)
    ox, oy = int(env_offset[0]), int(env_offset[1])
    canv = canvas(np.asarray(views), np.asarray(psi).reshape(-1))
    own = goal_phases(positions[:, 0] + ox, positions[:, 1] + oy, lam)
    B = len(memories)
    ph = np.zeros((B, 12), dtype=np.float32)
    sig = np.zeros(B, dtype=np.float32)
    moves = []
    for b, m in enumerate(memories):
        cell = (int(positions[b, 0]), int(positions[b, 1]))
        prev = m.last_cell
        moves.append(None if prev is None else
                     (abs(cell[0] - prev[0]) > 0) + (abs(cell[1] - prev[1]) > 0))
        if m.skip_stays and prev == cell and m.flag == 0.0 and m.last_out is not None:
            ph[b], logit = m.last_out                    # the cell did not change: no step
        else:
            ph[b], logit = m.step(canv[b], own[b])
        m.last_cell, m.last_out = cell, (ph[b].copy(), logit)
        sig[b] = 1.0 / (1.0 + np.exp(-logit))
    size = memories[0].arena_size
    gx, gy = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    cells = np.stack([gx.ravel(), gy.ravel()], 1)                       # local (C, 2)
    goal_cell = cells[decode_cells(ph, cells + np.array([ox, oy]), lam)]
    has = sig >= np.array([m.c_min for m in memories])
    d = np.zeros((B, 2), dtype=np.float32)
    if has.any():
        g_now = readout.grid_code.at_local(positions[has], env_offset)
        g_goal = readout.grid_code.at_local(goal_cell[has], env_offset)
        d[has] = readout.mlp.direction(torch.from_numpy(g_now),
                                       torch.from_numpy(g_goal)).numpy()
    c = (sig * np.array([m.c_scale for m in memories])).astype(np.float32)
    for b, m in enumerate(memories):                   # diagnostics (tag), no effect on the read
        info = getattr(m, "info", None)
        if info is None:
            continue
        key = (info["block"], info["arena"], bool(info["known"]))
        s = m.stats.setdefault(key, dict(n=0, far=0, hit=0, cos=0.0, ncos=0, c=0.0, on=0,
                                         stay=0, diag=0))
        if moves[b] is not None:
            s["stay"] += int(moves[b] == 0)
            s["diag"] += int(moves[b] == 2)
        goal = np.asarray(info["goal"], dtype=np.float64)
        to_goal = goal - positions[b]
        s["n"] += 1
        s["c"] += float(sig[b])
        s["on"] += int(has[b])
        if np.linalg.norm(to_goal) > 1.5:
            s["far"] += 1
            s["hit"] += int(np.linalg.norm(goal_cell[b] - goal) <= 1.5)
        if has[b] and np.linalg.norm(to_goal) > 0:
            s["cos"] += float(d[b] @ to_goal / (np.linalg.norm(d[b]) * np.linalg.norm(to_goal) + 1e-9))
            s["ncos"] += 1
    channel = d * c[:, None] if readout.scale_q_by_c else d
    return channel.astype(np.float32), channel.astype(np.float32), has, c
