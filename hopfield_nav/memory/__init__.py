"""Alternative goal memories for the grid-MLP navigator (docs/GRID_MLP_NAV_PLAN.md).

Agent-HaSH's memory is a Hopfield store over encoder embeddings; the policy
reads its recalled displacement ``q``. The modules here provide the idea-1
alternative behind the same seam:

- ``grid_code``  -- the smoothed grid code at any scaffold position, per-module
  tables instead of the 5 GB smoothed codebook;
- ``grid_mlp``   -- the frozen Phase-1 grid MLP, ``(g_now, g_goal) -> direction``;
- ``sensory_kv`` -- a sensory-keyed, argmax key-value goal store;
- ``readout``    -- the per-step readout ``(d, c)`` the policy sees.

Selected by ``AgentConfig.memory_backend``; ``"hopfield"`` (default) leaves
every existing path untouched.
"""
