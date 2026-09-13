"""The scripted two-step frame estimator: gate B2-C4 (plan sec 4B.5).

A hand-written agent with the `act(x, h, deterministic)` interface that runs
through the same lifetime evaluator as the GRUs. It reads `gbook(p)` and
`gbook(g)` out of the input at the layout's offsets and keeps its own per-row
state; nothing else about it is learned. Its algorithm, per lifetime:

1. Measure the frame. Step 0 acts along +X, step 1 along +Y (the arena's
   axes; "N" and "E" in the plan's words). For each, the bump centroid of
   every module before and after, the wrapped difference averaged across
   modules, is `R_theta a / s` for the action `a` taken. Two actions give
   the two columns of `M = R_theta / s`; `theta` is the rotation nearest `M`
   (Procrustes) and `s = 1 / mean column norm`. A step whose measured shift
   is not a unit step -- clipped at the arena (~0) or a teleport at an
   episode boundary (large) -- is discarded and retried in the opposite
   direction. Frame measured in two steps.
2. Every step after that: the wrapped phase difference between `gbook(g)`
   and `gbook(p)` per module, then the Chinese-remainder step to the one
   rotated displacement `(dX', dY')` within the arena consistent with all
   three moduli. Direction = `normalize(R_theta^T (dX', dY'))`; `s` cancels.

While the frame is unknown the agent's directional output is undefined, so
it acts along the measuring axes and scores ~90 deg on average: that is
readout 2's expected shape for this agent, ~90 at steps 0-1 of episode 0 and
<= 5 deg from step 2 of the lifetime onward, flat across episodes. That
curve is the ceiling a GRU is being read against, and its passing is the
proof that the information is in the trajectory at all.
"""
from __future__ import annotations

import numpy as np
import torch

from gridcode.lattice import code_phases, crt_displacement, wrapped_phase_diff
from ..policy.agent_rnn import rnn_input_layout

_AXIS_ACTIONS = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)   # +X then +Y


class ScriptedFrameAgent:
    """`act(x, h)` -> unit direction from a measured frame and a CRT decode.

    `cfg` is a grid-mode `RNNAgentConfig` (continuous movement); `obs` and
    `Ng` fix where the two grid channels sit in the layout. `size` bounds the
    CRT search to one arena.
    """

    def __init__(self, cfg, obs: int, lambdas, size: int, *, unit_tol=(0.5, 2.0),
                 seed: int = 0) -> None:
        if cfg.movement_mode != "continuous":
            raise ValueError("ScriptedFrameAgent is a continuous-action agent")
        if not (cfg.input_grid_state and getattr(cfg, "input_goal_grid_state", False)):
            raise ValueError("ScriptedFrameAgent needs both grid channels (grid mode)")
        self.cfg = cfg
        self.lambdas = [int(l) for l in lambdas]
        self.lam_col = np.array(self.lambdas, dtype=np.float64)[:, None]
        Ng = sum(l * l for l in self.lambdas)
        off = 0
        self.slices = {}
        for name, width in rnn_input_layout(cfg, obs, Ng):
            self.slices[name] = slice(off, off + width)
            off += width
        self.input_dim = off
        self.max_abs = float(size) * np.sqrt(2.0)
        self.unit_tol = unit_tol
        self.rng = np.random.RandomState(seed)
        self.B = None

    # -- the evaluator calls this once per batch of lifetimes ----------------
    def begin_lifetimes(self, n: int) -> None:
        self.B = n
        self.stage = np.zeros(n, dtype=np.int64)          # 0: measure +X, 1: measure +Y, 2: navigate
        self.sign = np.ones(n, dtype=np.float64)          # flipped after a failed measuring step
        self.prev_phase = np.full((n, len(self.lambdas), 2), np.nan)
        self.prev_action = np.zeros((n, 2), dtype=np.float64)
        self.columns = np.zeros((n, 2, 2), dtype=np.float64)   # M[:, :, j] = shift for axis j
        self.theta = np.zeros(n, dtype=np.float64)
        self.scale = np.ones(n, dtype=np.float64)
        self.steps = 0

    def eval(self):
        return self

    def train(self, mode: bool = True):
        return self

    # -- one step -------------------------------------------------------------
    @torch.no_grad()
    def act(self, x: torch.Tensor, h=None, deterministic: bool = False) -> dict:
        xs = x.detach().cpu().numpy().reshape(x.shape[0], -1)
        B = xs.shape[0]
        if self.B != B:
            self.begin_lifetimes(B)
        gp = xs[:, self.slices["grid_state"]]
        gg = xs[:, self.slices["goal_grid_state"]]
        ph_p = code_phases(gp, self.lambdas)                    # (B, M, 2)
        ph_g = code_phases(gg, self.lambdas)

        # 1. Finish the previous measuring step, where there was one.
        measuring = self.stage < 2
        had_prev = measuring & ~np.isnan(self.prev_phase[:, 0, 0])
        if had_prev.any():
            idx = np.where(had_prev)[0]
            shift = wrapped_phase_diff(ph_p[idx], self.prev_phase[idx], self.lam_col).mean(axis=1)   # (n, 2)
            norm = np.linalg.norm(shift, axis=1)
            ok = (norm >= self.unit_tol[0]) & (norm <= self.unit_tol[1])
            for i, b in enumerate(idx):
                j = self.stage[b]
                if ok[i]:
                    # The step taken was sign * e_j; undo the sign so the
                    # column is M e_j.
                    self.columns[b, :, j] = shift[i] / self.sign[b]
                    self.stage[b] += 1
                    self.sign[b] = 1.0
                else:
                    self.sign[b] = -self.sign[b]
                if self.stage[b] == 2:
                    M = self.columns[b]
                    self.theta[b] = np.arctan2(M[1, 0] - M[0, 1], M[0, 0] + M[1, 1])
                    self.scale[b] = 1.0 / max(np.linalg.norm(M, axis=0).mean(), 1e-6)

        # 2. Choose the action.
        action = np.zeros((B, 2), dtype=np.float64)
        nav = self.stage >= 2
        if nav.any():
            idx = np.where(nav)[0]
            dph = wrapped_phase_diff(ph_g[idx], ph_p[idx], self.lam_col)       # (n, M, 2)
            d_rot = crt_displacement(dph, self.lambdas, self.max_abs)            # (n, 2) = R_theta d / s
            c, s = np.cos(self.theta[idx]), np.sin(self.theta[idx])
            # R_theta^T applied row-wise.
            d = np.stack([c * d_rot[:, 0] + s * d_rot[:, 1],
                          -s * d_rot[:, 0] + c * d_rot[:, 1]], axis=1)
            n = np.linalg.norm(d, axis=1, keepdims=True)
            # On the goal CELL but not inside the goal's L2 ball (the code is
            # cell-resolution; the sub-cell offset is not in the input) the
            # decode is exactly zero. A deterministic zero action would sit
            # there until the episode times out, so step in a random
            # direction instead -- the honest choice with no information,
            # and what a sampled policy does by itself.
            zero = n[:, 0] < 0.5
            if zero.any():
                r = self.rng.standard_normal((int(zero.sum()), 2))
                d[zero] = r / np.linalg.norm(r, axis=1, keepdims=True)
                n[zero] = 1.0
            action[idx] = d / np.maximum(n, 1e-8)
        meas = np.where(~nav)[0]
        if len(meas) > 0:
            action[meas] = self.sign[meas, None] * _AXIS_ACTIONS[self.stage[meas]]
            self.prev_phase[meas] = ph_p[meas]
        self.prev_action = action
        self.steps += 1
        return {"move_action": torch.from_numpy(action.astype(np.float32)).to(x.device),
                "move_log_prob": torch.zeros(B, device=x.device), "h_next": None}

    # -- diagnostics ------------------------------------------------------------
    def frame(self) -> dict:
        return {"theta": self.theta.copy(), "scale": self.scale.copy(),
                "measured": (self.stage >= 2).copy()}


__all__ = ["ScriptedFrameAgent"]
