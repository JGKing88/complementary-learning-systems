"""How much coverage does a *constant turn bias* cost, at W6's own statistics?

3v: W6 covers 0.325 where a walk with its own stride (0.93) and its own turn
sigma (0.88) covers 0.472. The matched walk draws each turn independently and
zero-mean; a recurrent policy that has settled into a limit cycle instead turns
the same way every step. That is the one degree of freedom the matched-walk
control does NOT hold fixed, so it is the first candidate for the -0.147.

This sweeps a constant per-step drift added to the same zero-mean turn noise
and asks which drift, if any, reproduces the measured deficit.
"""
import numpy as np
from hopfield_nav.world.walks import simulate_coverage, random_starts, unit_vectors

SIZE, STEPS, B = 20, 400, 512
STRIDE, TURN_SIGMA = 0.930, 0.877


def biased_walk_coverage(drift, rng):
    state = {"theta": rng.uniform(0, 2 * np.pi, B)}

    def direction_fn(t, blocked, pos):
        state["theta"] = state["theta"] + drift + TURN_SIGMA * rng.randn(B)
        return STRIDE * unit_vectors(state["theta"])

    pos0 = random_starts(B, SIZE, rng)
    return float(np.mean(simulate_coverage(pos0, SIZE, STEPS, direction_fn, rng)))


print(f"{SIZE}x{SIZE}, {STEPS} steps, stride {STRIDE}, turn sigma {TURN_SIGMA} rad")
print("W6 measured coverage 0.325; unbiased matched walk 0.472\n")
print(f"{'drift (rad/step)':>17} {'turn radius (cells)':>21} {'coverage':>10}")
for drift in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8):
    cov = biased_walk_coverage(drift, np.random.RandomState(0))
    r = f"{STRIDE / drift:.1f}" if drift > 0 else "inf"
    print(f"{drift:>17.2f} {r:>21} {cov:>10.3f}")
