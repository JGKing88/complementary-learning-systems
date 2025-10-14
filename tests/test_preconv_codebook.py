from __future__ import annotations

import numpy as np

from cls.envs.environments import GridWMEnv, GridWMVecEnv
from cls.utils.GridUtils import VectorHash


def assert_allclose(a: np.ndarray, b: np.ndarray, atol: float = 1e-6, rtol: float = 1e-6) -> None:
    if not np.allclose(a, b, atol=atol, rtol=rtol):
        max_abs = np.max(np.abs(a - b))
        raise AssertionError(f"Arrays not close. max_abs={max_abs}")


def run_check(size: int = 8, batch_size: int = 16, lambdas: list[int] = [4, 5, 7], Np: int = 1600) -> None:
    # Base env and vectorhash
    base = GridWMEnv(size=size, speed=1, seed=0, observation_size=256, input_type="g_idx")
    vh = VectorHash(Np=Np, lambdas=lambdas, size=size)
    vh.initiate_vectorhash([base])

    # Two vectorized envs: with and without preconverted codebook
    vec_pre = GridWMVecEnv(base, batch_size=batch_size, seed=123, use_preconv_codebook=True)
    vec_dyn = GridWMVecEnv(base, batch_size=batch_size, seed=123, use_preconv_codebook=False)

    # Align internal batched state so comparisons are meaningful
    vec_dyn._pos = vec_pre._pos.copy()
    vec_dyn._heading = vec_pre._heading.copy()

    indices = list(range(batch_size))

    # Check observations equivalence for different addenda
    for add in (None, "goal", "diff"):
        obs_pre = vec_pre.obs_batch(indices, add)
        obs_dyn = vec_dyn.obs_batch(indices, add)
        assert_allclose(obs_pre, obs_dyn)

    # Check best action equivalence
    best_pre = vec_pre.best_action_to_goal_batch(indices, randomize=False)
    best_dyn = vec_dyn.best_action_to_goal_batch(indices, randomize=False)
    if best_pre != best_dyn:
        raise AssertionError("best_action_to_goal_batch mismatch between pre/dyn paths")

    # Step once with random actions and compare next observations
    rng = np.random.RandomState(0)
    action_indices = rng.randint(0, 4, size=batch_size).tolist()
    actions = [((0, 1), (1, 0), (0, -1), (-1, 0))[i] for i in action_indices]

    vec_pre.step_batch(indices, actions)
    vec_dyn.step_batch(indices, actions)

    for add in (None, "goal", "diff"):
        obs_pre = vec_pre.obs_batch(indices, add)
        obs_dyn = vec_dyn.obs_batch(indices, add)
        assert_allclose(obs_pre, obs_dyn)


if __name__ == "__main__":
    run_check()
    print("Preconverted codebook matches dynamic recall outputs. OK.")


