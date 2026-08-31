"""Env and architecture setup for the RNN baseline.

Both of these were in `train_rnn.py`, which meant `analysis/continual/baseline.py`
-- the paper's continual-learning figure pipeline -- imported them from a
training CLI. That is an analysis module depending on an entry point, and with
`train_rnn`'s deferred import of `analysis.continual.plotting` going the other
way it formed a mutual dependency: legal under the layering rules, since 8 -> 7
is downward and the 7 -> 8 edge is declared, but a cycle nonetheless.

Neither function is CLI plumbing. `build_envs_from_config` makes one `GridEnv`
per seed;
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
    # Two extra input columns when set, so a mismatch here is a load-time shape
    # error rather than a wrong-but-runnable model -- the same reason rnn_cell
    # is restored.
    _restore(cfg.agent, "goal_channel",      saved_agent.get("goal_channel"),      "goal_channel")
    # Env-side movement_mode must mirror agent-side (VecEnv vs ContinuousVecEnv).
    cfg.env.movement_mode = cfg.agent.movement_mode
    _restore(cfg, "lambdas",    saved.get("lambdas"),    "lambdas")
    _restore(cfg, "fwhm_ratio", saved.get("fwhm_ratio"), "fwhm_ratio")


def build_envs_from_config(cfg: RNNTrainConfig,
                           rng: np.random.RandomState) -> list[GridEnv]:
    """One GridEnv per seed; each gets its own codebook + goal.

    Named apart from ``world.generate.build_envs``, which realizes resolved
    ``EnvSpec``s. The two took the same name and did unrelated things -- one
    draws envs from a config, the other rebuilds envs from a record -- which is
    exactly the collision to avoid while moving this stack onto the generator.
    """
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
            wall_resolution=cfg.env.wall_resolution,
        ))
    return envs

__all__ = ["build_envs_from_config", "restore_arch_from_ckpt",
           "rnn_world", "write_rnn_world_spec"]


# ---------------------------------------------------------------------------
# The declared-domain path, shared with train_navigate
# ---------------------------------------------------------------------------

def rnn_world(cfg: RNNTrainConfig, rng: np.random.RandomState):
    """Envs, offsets and the split describing them, for the RNN stack.

    Two paths, the same two `train_navigate` has. With ``cfg.env_generator`` the
    envs come from declared domains and their offsets are *recorded*; without
    it, the historical draw runs and is described after the fact. Either way a
    ``GeneratedSplit`` comes back, so both can write a ``world.json`` and a
    baseline run and an agent-hash run can be handed the same declared world
    instead of being talked into agreement by a draw-order convention
    (``analysis/continual/agenthash.py:325-333``).

    **The scaffold exists for the generator, not only for the agent.** Under
    ``input_grid_state=False`` the RNN never observes where its envs sit -- but
    the placement is still part of the world's identity, and the run still has
    to be able to say what it used. An agent-hash run pointed at the same
    ``world.json`` *does* observe those offsets, and the split's separation
    guarantees are stated in scaffold coordinates either way. So the generator
    builds a scaffold whenever it is asked to, and the offsets are recorded
    whether or not this particular agent can see them.

    Only the *legacy* path is conditional: it places envs only under grid state,
    because that is what it historically did and its draw must not move.

    Returns ``(envs, offsets | None, split, field | None, generator)``.
    """
    from ..world import domains as dom
    from ..world import generate as gen
    from ..world.scaffold import VectorHash, place_envs
    from ..world.spec import EnvSpec, GeneratedSplit, TraitDomains
    from ..config import VectorHashConfig

    size = int(cfg.env.size)
    declared = bool(getattr(cfg, "env_generator", False))
    grid_state = bool(cfg.agent.input_grid_state)
    field = None
    if declared or grid_state:
        # `build_scaffold` only; `precompute_encoded_phi` needs an encoder and
        # this stack has none. Placement reads Npos and lambdas, nothing else,
        # and the build is well under a second.
        field = VectorHash(VectorHashConfig(lambdas=list(cfg.lambdas),
                                            static_vectorhash=True))
        field.build_scaffold()

    if not declared:
        envs = build_envs_from_config(cfg, rng)
        offsets = (place_envs(len(envs), size, field.Npos, np.random,
                              placement="spread") if grid_state else None)
        specs = [EnvSpec(int(e.seed), size,
                         tuple(offsets[i]) if offsets else (0, 0),
                         tuple(e.goal_location)) for i, e in enumerate(envs)]
        all_cells = frozenset((x, y) for x in range(size) for y in range(size))
        goals = frozenset(s.goal for s in specs)
        split = GeneratedSplit(
            domains=TraitDomains(place=dom.Anywhere(),
                                 wall=dom.SeedRange(0, 10_000_000),
                                 goal=dom.AnyCells(), size=dom.Sizes((size,))),
            train=specs, base_val=[], goal_cells_train=goals,
            goal_cells_val=all_cells - goals, margin=0,
            period=int(np.prod(cfg.lambdas)) if grid_state else 0,
            Npos=int(field.Npos) if grid_state else 0)
        split.record_used(specs)
        return envs, offsets, split, field, "legacy"

    if cfg.place_margin is None:
        raise SystemExit(
            "--env_generator needs an explicit --place_margin here. The agent "
            "stack derives one from its scaffold's cosine-vs-distance curve, "
            "which needs an encoder; this stack has none, and a borrowed "
            f"constant would be wrong at lambdas={list(cfg.lambdas)} "
            f"(Npos={field.Npos}) anyway.")

    domains = TraitDomains(place=dom.parse_place(cfg.place_region),
                           wall=dom.parse_seed_range(cfg.wall_seeds),
                           goal=dom.parse_goal(cfg.goal_region),
                           size=dom.Sizes((size,)))
    split = gen.generate_split(
        field, cfg.env, domains, int(cfg.n_envs), int(cfg.n_val_envs),
        seed=int(cfg.seed), margin=int(cfg.place_margin),
        val_frac=float(cfg.goal_val_frac), diagnostics=False)
    envs = gen.build_envs(split.train, cfg.env, "discrete")
    return envs, [s.offset for s in split.train], split, field, "declared"


def write_rnn_world_spec(cfg: RNNTrainConfig, split, field, *, generator: str,
                         save_dir) -> str | None:
    """Record the RNN stack's world beside its checkpoints.

    Same file and same reader as `train_navigate`'s, which is the point: an
    agent-hash run and a baseline run become comparable by pointing at one
    record rather than by matching draw orders.
    """
    from ..world import generate as gen
    from ..world.spec import WorldSpec

    if save_dir is None:
        return None
    split.diagnostics = gen.split_diagnostics(field, cfg.env, split) if field \
        else {}
    spec = WorldSpec(
        scaffold={"lambdas": list(field.lambdas) if field else [],
                  "Npos": int(field.Npos) if field else 0,
                  "fwhm_ratio": float(cfg.fwhm_ratio),
                  "static_vectorhash": True, "encoder": None},
        generator=generator, split=split)
    path = spec.write(save_dir)
    print(f"  world.json: generator={generator} margin={split.margin} "
          f"n_envs={len(split.train)}", flush=True)
    return path
