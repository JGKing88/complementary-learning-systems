"""Per-step memory readout for the grid-MLP navigator: ``(d, c)``.

``d = MLP(g_now, ĝ)`` is the frozen grid MLP's unit direction from the agent's
grid code to the recalled goal's (zeros when memory is empty); ``c`` is the
recall's similarity. Both reach the policy: ``d`` (or ``c · d`` under
``scale_q_by_c``) on the ``hopfield_signal`` channel that carries Agent-HaSH's
``q``, and ``c`` on its own ``memory_conf`` channel. The same ``d`` is returned
as ``q`` so the teachers and follow-q diagnostics that read ``q`` keep working.
"""
from __future__ import annotations

import numpy as np
import torch

from .grid_code import SmoothedGridCode
from .grid_mlp import load_grid_mlp
from .sensory_kv import SensoryKVMemory, omni_key, read_batch


class GridMemoryReadout:
    def __init__(self, grid_code: SmoothedGridCode, mlp, *,
                 scale_q_by_c: bool = False) -> None:
        self.grid_code = grid_code
        self.mlp = mlp
        self.scale_q_by_c = bool(scale_q_by_c)

    @classmethod
    def from_config(cls, cfg) -> "GridMemoryReadout":
        a = cfg.agent
        if not a.grid_mlp_checkpoint:
            raise ValueError("memory_backend='sensory_kv' needs --grid_mlp_checkpoint")
        gc = SmoothedGridCode(cfg.vectorhash.lambdas, cfg.fwhm_ratio,
                              cfg.vectorhash.Npos or int(np.prod(cfg.vectorhash.lambdas)))
        mlp = load_grid_mlp(a.grid_mlp_checkpoint, "cpu")
        if mlp.input_dim != 2 * gc.Ng:
            raise ValueError(f"grid MLP takes {mlp.input_dim} inputs; this scaffold's "
                             f"grid code pair is {2 * gc.Ng} (lambdas {gc.lambdas})")
        return cls(gc, mlp, scale_q_by_c=a.scale_q_by_c)

    def goal_value(self, goal, env_offset) -> np.ndarray:
        return self.grid_code.at_local(np.asarray(goal)[None], env_offset)[0]

    def write_goal(self, memory: SensoryKVMemory, env, goal, env_offset) -> None:
        """The oracle store: (four-heading view at the goal, its grid code)."""
        memory.write(omni_key(env, goal), self.goal_value(goal, env_offset))

    def signal(self, memories, views, psi, positions, env_offset):
        """Returns ``(channel (B,2), q (B,2), has_memory (B,), c (B,))``."""
        g_goal, c, has = read_batch(memories, views, psi, self.grid_code.Ng)
        d = np.zeros((len(memories), 2), dtype=np.float32)
        if has.any():
            g_now = self.grid_code.at_local(positions, env_offset)
            dd = self.mlp.direction(torch.from_numpy(g_now[has]),
                                    torch.from_numpy(g_goal[has])).numpy()
            d[has] = dd
        channel = d * c[:, None] if self.scale_q_by_c else d
        return channel.astype(np.float32), channel.astype(np.float32), has, c.astype(np.float32)
