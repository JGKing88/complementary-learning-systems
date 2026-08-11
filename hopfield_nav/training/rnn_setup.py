"""Env and architecture setup for the RNN baseline.

Both of these were in `train_rnn.py`, which meant `analysis/continual/baseline.py`
-- the paper's continual-learning figure pipeline -- imported them from a
training CLI. That is an analysis module depending on an entry point, and with
`train_rnn`'s deferred import of `analysis.continual.plotting` going the other
way it formed a mutual dependency: legal under the layering rules, since 8 -> 7
is downward and the 7 -> 8 edge is declared, but a cycle nonetheless.

Neither function is CLI plumbing. `build_envs` makes one `GridEnv` per seed;
`restore_arch_from_ckpt` replays a checkpoint's architecture fields over a
freshly parsed config. This is the RNN baseline's counterpart to
`training/world_setup.py`, which took the same treatment out of `train_phased`.
"""
from __future__ import annotations

import numpy as np

from ..config import RNNTrainConfig
from ..world.env import GridEnv


def restore_arch_from_ckpt(cfg: RNNTrainConfig, ckpt: dict) -> None:
    """Auto-restore architecture-affecting fields from a saved ckpt's cfg dict.

    Mutates ``cfg`` in place. Prints a NOTE for each field where the CLI value
    is being overridden. Restores fields whose values affect either the agent's
    parameter shapes (movement_mode, hidden_size, num_rnn_layers, rnn_cell,
    input_prev_*, input_grid_state) or the gbook lookup encoding (lambdas,
    fwhm_ratio). Fields the ckpt doesn't have (legacy ckpts) are left as-is --
    which is why a checkpoint predating ``rnn_cell`` correctly finetunes as the
    GRU it was trained as.
    """
    saved = ckpt.get("cfg", {}) or {}
    saved_agent = saved.get("agent", {}) or {}

    def _restore(obj, attr: str, saved_value, label: str) -> None:
        if saved_value is None:
            return
        cur = getattr(obj, attr)
        if cur != saved_value:
            print(f"  NOTE: --{label} {cur!r} ignored; using ckpt's {saved_value!r}")
        setattr(obj, attr, saved_value)

    _restore(cfg.agent, "movement_mode",     saved_agent.get("movement_mode"),     "movement_mode")
    _restore(cfg.agent, "hidden_size",       saved_agent.get("hidden_size"),       "hidden_size")
    _restore(cfg.agent, "num_rnn_layers",    saved_agent.get("num_rnn_layers"),    "num_rnn_layers")
    _restore(cfg.agent, "dropout",           saved_agent.get("dropout"),           "dropout")
    # rnn_cell changes the trunk's parameter shapes (a GRU's weight_ih_l0 is
    # 3H x D against a vanilla cell's H x D), so a mismatch here surfaces as a
    # load_state_dict shape error rather than a wrong-but-runnable model.
    _restore(cfg.agent, "rnn_cell",          saved_agent.get("rnn_cell"),          "rnn_cell")
    _restore(cfg.agent, "rnn_nonlinearity",  saved_agent.get("rnn_nonlinearity"),  "rnn_nonlinearity")
    _restore(cfg.agent, "input_prev_action", saved_agent.get("input_prev_action"), "input_prev_action")
    _restore(cfg.agent, "input_prev_reward", saved_agent.get("input_prev_reward"), "input_prev_reward")
    _restore(cfg.agent, "input_grid_state",  saved_agent.get("input_grid_state"),  "input_grid_state")
    # Env-side movement_mode must mirror agent-side (VecEnv vs ContinuousVecEnv).
    cfg.env.movement_mode = cfg.agent.movement_mode
    _restore(cfg, "lambdas",    saved.get("lambdas"),    "lambdas")
    _restore(cfg, "fwhm_ratio", saved.get("fwhm_ratio"), "fwhm_ratio")


def build_envs(cfg: RNNTrainConfig, rng: np.random.RandomState) -> list[GridEnv]:
    """One GridEnv per seed; each gets its own codebook + goal."""
    envs: list[GridEnv] = []
    for _ in range(cfg.n_envs):
        seed = int(rng.randint(0, 10_000_000))
        # Continuous-mode env factory left as a future extension; sequential
        # currently uses the discrete VecEnv path.
        envs.append(GridEnv(
            size=cfg.env.size,
            speed=cfg.env.speed,
            observation_size=cfg.env.observation_size,
            seed=seed,
            time_penalty=cfg.env.time_penalty,
            goals_active=cfg.env.goals_active,
            # goal_reward and goal_radius were previously left at the GridEnv
            # defaults, so VecEnv (which reads them off the base env) ignored
            # the configured values.
            goal_reward=cfg.env.goal_reward,
            goal_radius=cfg.env.goal_radius,
            egocentric_heading=cfg.env.egocentric_heading,
        ))
    return envs

__all__ = ["build_envs", "restore_arch_from_ckpt"]
