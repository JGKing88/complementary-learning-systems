## cls

A minimal, extensible codebase for generating navigation environments with discrete grid worlds and simple wall-based sensory input.

### Install

This repo is pure Python stdlib. Add the folder to your PYTHONPATH or install as an editable package if you prefer a project setup.

### Quickstart

```python
from cls import WMEnv

env = WMEnv(size=8, speed=1, seed=42)

cur, goal, ob = env.reset()
print("start:", cur, "goal:", goal, "obs:", ob)

# Vector actions: (dx, dy) in {-1,0,1}; scaled by speed per step.
# Movement proceeds in sub-steps; if a boundary is reached mid-step,
# the agent stops at the boundary (partial movement allowed).
for a in [(0, 1), (1, 0), (1, 1), (0, 0), (-1, 0)]:
    cur, goal, ob = env.step(a)
    print(a, cur, goal, ob)
```

### Design Notes

- Discrete coordinates with origin at bottom-left `(0, 0)`.
- Actions are relative to the agent's heading. Non-forward actions rotate the heading and then move.
- Sensory input is a slice of the facing wall's pattern; default FOV narrows near walls and widens with distance: `visible_width = max(1, distance_to_wall + 1)`.
- You can customize FOV with `fov_fn(distance, size)` and optional `max_fov`:

```python
from math import log

# Example: capped linear FOV, or a mild non-linear FOV
env = WMEnv(
    size=8,
    speed=1,
    fov_fn=lambda d, s: min(5, d + 1),  # capped linear
    # fov_fn=lambda d, s: max(1, int(2 * log(d + 1) + 1)),  # non-linear
    max_fov=6,
)
```
- Walls use simple binary patterns for now; extend as needed.

### Roadmap

- Configurable sensing models and wall pattern generators
- Termination conditions and rewards
- Continuous variants and richer observations
- Rendering utilities


