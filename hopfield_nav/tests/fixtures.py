"""Shared test fixtures: a stub scaffold and deterministic world builders.

``StubVectorHash`` mimics the geometry the rollout collector needs while
skipping the heavy scaffold build (no encoder, no pbook, no Wsp/Wps), so tests
run in milliseconds. It was originally defined inside ``test_audit.py``; it
lives here because the golden fixtures depend on it byte-for-byte and two test
modules must not drift apart on what "the stub world" means.

Everything here is deterministic: same inputs, same arrays, every run. That is
what makes the golden files in ``tests/golden/`` meaningful.
"""
from __future__ import annotations

import numpy as np
import torch

from hopfield_nav.config import (
    AgentConfig, BCConfig, EnvConfig, HopfieldConfig, PPOConfig, TrainConfig,
    VectorHashConfig,
)


class StubVectorHash:
    """Minimal stand-in for VectorHash with deterministic encodings.

    ``encoded_Phi[gx, gy]`` is a smooth, position-tied (D,) embedding.
    Translation-similarity is enough for the Gram-Schmidt projection to be
    well-defined; the real projection and displacement code is reused by
    delegating to the VectorHash methods.
    """

    def __init__(self, Npos: int = 16, embed_dim: int = 8):
        self.Npos = Npos
        rng = np.random.default_rng(0)
        coords = np.stack(np.meshgrid(
            np.arange(Npos), np.arange(Npos), indexing="ij"), axis=-1)
        phase = coords[..., None, :] * np.array([[0.31, 0.71]])
        feat = np.concatenate(
            [np.sin(phase * np.pi).reshape(Npos, Npos, 2),
             np.cos(phase * np.pi).reshape(Npos, Npos, 2)],
            axis=-1)
        pad = embed_dim - feat.shape[-1]
        if pad > 0:
            feat = np.concatenate(
                [feat, rng.standard_normal((Npos, Npos, pad)) * 0.01], axis=-1)
        feat = feat / np.linalg.norm(feat, axis=-1, keepdims=True).clip(1e-9)
        self.encoded_Phi = feat.astype(np.float32)

    def get_encoded_state(self, positions: np.ndarray, env_offset):
        gx = np.clip(positions[:, 0] + env_offset[0], 0, self.Npos - 1)
        gy = np.clip(positions[:, 1] + env_offset[1], 0, self.Npos - 1)
        return self.encoded_Phi[gx, gy]

    def get_store_patterns(self, positions, env_offset, *, at_goal_mask=None,
                           goal=None, allow_offcell=True):
        from hopfield_nav.world.scaffold import VectorHash
        return VectorHash.get_store_patterns(
            self, positions, env_offset, at_goal_mask=at_goal_mask,
            goal=goal, allow_offcell=allow_offcell,
        )

    def gram_schmidt_projection(self, positions, env_offset, cached_W=None,
                                recompute_mask=None):
        from hopfield_nav.world.scaffold import VectorHash
        return VectorHash.gram_schmidt_projection(
            self, positions, env_offset, cached_W, recompute_mask,
        )

    def project_displacement(self, current, recalled, W):
        return np.einsum("bij,bj->bi", W, recalled - current)


def make_stub_cfg(
    *,
    movement_mode: str = "discrete",
    explore_steps=None,
    novelty_reward: float = 0.0,
    revisit_penalty: float = 0.0,
    wall_penalty: float = 0.0,
    epsilon_explore: float = 0.0,
    auto_nav_warmup: int = 0,
    input_goal_in_memory: bool = False,
    allow_store: bool = True,
    continuous_normalize: bool = False,
    # Observation-channel knobs -- the axis the golden fixtures sweep.
    input_encoded_state: bool = False,
    input_hopfield_signal: bool = True,
    input_hopfield_raw: bool = False,
    # AgentConfig's own default is [], not None. Both are falsy so the
    # channel is skipped either way, but matching the dataclass keeps the
    # stub honest for any code that calls len() on it.
    input_hopfield_multistep=(),
    input_prev_action: bool = False,
    input_prev_reward: bool = False,
    input_sensory: bool = False,
    hopfield_mode: str | None = None,
    batch_envs: int = 4,
    steps_per_rollout: int = 8,
) -> TrainConfig:
    """A TrainConfig wired for the stub world.

    Defaults reproduce the config ``test_audit.py`` has always used, so tests
    moved onto this helper keep their previous behavior.
    """
    return TrainConfig(
        env=EnvConfig(size=6, observation_size=12, time_penalty=0.01,
                      movement_mode=movement_mode,
                      continuous_normalize=continuous_normalize),
        vectorhash=VectorHashConfig(),
        hopfield=HopfieldConfig(
            beta=1.0, alpha=1.0, steps=1,
            allow_store=allow_store,
            novelty_reward=novelty_reward,
            revisit_penalty=revisit_penalty,
            wall_penalty=wall_penalty,
            epsilon_explore=epsilon_explore,
            auto_nav_warmup=auto_nav_warmup,
        ),
        agent=AgentConfig(
            hidden_size=16, num_rnn_layers=1,
            hopfield_mode=hopfield_mode or movement_mode,
            movement_mode=movement_mode,
            input_encoded_state=input_encoded_state,
            input_hopfield_signal=input_hopfield_signal,
            input_hopfield_raw=input_hopfield_raw,
            input_hopfield_multistep=list(input_hopfield_multistep or []),
            input_prev_action=input_prev_action,
            input_prev_reward=input_prev_reward,
            input_sensory=input_sensory,
            input_goal_in_memory=input_goal_in_memory,
        ),
        ppo=PPOConfig(),
        bc=BCConfig(),
        encoder_checkpoint="dummy",
        explore_steps=explore_steps,
        batch_envs=batch_envs,
        steps_per_rollout=steps_per_rollout,
        device="cpu",
    )


def make_collector(cfg: TrainConfig, embed_dim: int = 8, *, seed: int = 0):
    """(collector, agent, stub_vectorhash) for the stub world.

    The agent's weights are seeded, so its outputs -- and therefore the whole
    rollout -- are reproducible.
    """
    from hopfield_nav.policy.agent import NavAgent, compute_input_dim
    from hopfield_nav.rollout.collector import RolloutCollector

    vh = StubVectorHash(Npos=16, embed_dim=embed_dim)
    device = torch.device("cpu")
    collector = RolloutCollector(vh, cfg, embed_dim, device)
    torch.manual_seed(seed)
    input_dim = compute_input_dim(cfg.agent, embed_dim, cfg.env.observation_size)
    agent = NavAgent(cfg.agent, input_dim).to(device)
    agent.eval()
    return collector, agent, vh


class RecordingAgent(torch.nn.Module):
    """Wraps a NavAgent and records every assembled observation passed to it.

    The rollout collector builds its policy input in two places (the main loop
    and the truncation bootstrap) and ``eval.agent_step`` builds a third. None
    of them return the assembled tensor, and only the main loop's copy is kept
    (in ``RolloutBatch.obs``). Wrapping the agent is the one place all three
    are observable, which is what lets the goldens pin every site.
    """

    def __init__(self, agent):
        super().__init__()
        self.agent = agent
        self.inputs: list[torch.Tensor] = []

    def forward(self, x, h=None, return_features=False):
        self.inputs.append(x.detach().clone())
        return self.agent(x, h, return_features)

    def get_action_and_value(self, x, *args, **kwargs):
        self.inputs.append(x.detach().clone())
        return self.agent.get_action_and_value(x, *args, **kwargs)

    def __getattr__(self, name):
        # nn.Module.__getattr__ handles submodules/params; fall through to the
        # wrapped agent for everything else (cfg, log_std, eval(), ...).
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.agent, name)

    @property
    def recorded(self) -> np.ndarray:
        """(n_calls, B, D) stack of every observation the agent has seen."""
        return np.stack([t.squeeze(1).cpu().numpy() for t in self.inputs])


class ScriptedAgent(torch.nn.Module):
    """An agent whose policy is written by hand, so the answer is computable.

    The golden fixtures pin the evaluators against *their own previous output*
    on an untrained network. That catches drift, but it cannot say whether a
    metric is correct -- if `mean_coverage` were off by a factor of two, the
    golden would happily pin the wrong number forever. Nothing in the suite
    knew the right answer.

    This does. Give it a movement rule and a store rule and it obeys them
    exactly, so on a small grid every metric has a value you can work out with
    a pencil and assert against.

    ``move``  int | (dx,dy) | callable(step, B) -> per-row action
    ``store`` float in [0,1] | callable(step, B) -> per-row store probability.
              The evaluators threshold at > 0.5, so 1.0 means "always fire" and
              0.0 means "never".

    Only ``move_action``, ``store_action`` and ``h_next`` are read off the
    result by any evaluator, which is why that is all this returns.
    """

    def __init__(self, move, store=0.0, *, movement_mode: str = "discrete"):
        super().__init__()
        self._move = move
        self._store = store
        self.movement_mode = movement_mode
        self.step = 0
        # A parameter so .to(device) / .eval() behave like the real thing.
        self._anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def _resolve(self, rule, B: int, default):
        if callable(rule):
            return rule(self.step, B)
        if rule is None:
            return default
        return rule

    def get_action_and_value(self, x, h=None, deterministic=True,
                            move_action_override=None, move_override_mask=None,
                            action_temperature=1.0) -> dict:
        B = x.shape[0]
        mv = self._resolve(self._move, B, 0)
        if self.movement_mode == "discrete":
            arr = np.full(B, mv, dtype=np.int64) if np.isscalar(mv) else np.asarray(mv, dtype=np.int64)
            move_action = torch.from_numpy(arr)
        else:
            arr = np.asarray(mv, dtype=np.float32)
            if arr.ndim == 1:
                arr = np.tile(arr, (B, 1))
            move_action = torch.from_numpy(arr)
        st = self._resolve(self._store, B, 0.0)
        st_arr = np.full(B, st, dtype=np.float32) if np.isscalar(st) else np.asarray(st, dtype=np.float32)
        self.step += 1
        return {
            "move_action": move_action.to(x.device),
            "store_action": torch.from_numpy(st_arr).to(x.device),
            "h_next": h,
        }

    def reset(self) -> None:
        """Rewind the step counter, for a rule that keys on it."""
        self.step = 0
