import numpy as np
import torch
import types

import cls.envs.environments as envs_module
from cls.envs.environments import WMEnv, WMVecEnv
from train import generate_episode, generate_episodes_vectorized


class DummyModel:
    def __init__(self, num_actions: int = 4, hidden_size: int = 8):
        self.num_actions = num_actions
        self.hidden_size = hidden_size

    def to(self, device: str):
        return self

    def train(self):
        pass

    def eval(self):
        pass

    def __call__(self, x: torch.Tensor, h):
        # x shape: (B, 1, F)
        B = x.shape[0]
        logits = torch.zeros((B, 1, self.num_actions), dtype=torch.float32, device=x.device)
        values = torch.zeros((B, 1), dtype=torch.float32, device=x.device)
        h_next = torch.zeros((1, B, self.hidden_size), dtype=torch.float32, device=x.device)
        return logits, values, h_next


def _wrap_reset_and_capture_state(monkeypatch):
    """Monkeypatch WMEnv.reset to capture start state into envs_module._test_align_state."""
    assert hasattr(envs_module, 'WMEnv')
    orig_reset = envs_module.WMEnv.reset

    def wrapped(self):
        result = orig_reset(self)
        envs_module._test_align_state = (self.current_location, self.heading)
        return result

    monkeypatch.setattr(envs_module.WMEnv, 'reset', wrapped, raising=True)


def _wrap_vec_reset_all_to_align(monkeypatch):
    """Monkeypatch WMVecEnv.reset_all to align first batch element with captured WMEnv state."""
    orig_reset_all = envs_module.WMVecEnv.reset_all

    def aligned(self):
        # Call original to initialize shapes
        orig_reset_all(self)
        start = getattr(envs_module, '_test_align_state', None)
        if start is None:
            return
        (pos, heading) = start
        # Align only active batch element 0 (test uses batch_episodes=1)
        self._pos[0, 0] = pos[0]
        self._pos[0, 1] = pos[1]
        self._heading[0, 0] = heading[0]
        self._heading[0, 1] = heading[1]

    monkeypatch.setattr(envs_module.WMVecEnv, 'reset_all', aligned, raising=True)


def _run_pair(env: WMEnv, model: DummyModel, steps: int, input_addendum):
    # Serial episode
    obs_seq, labels = generate_episode(
        env=env,
        model=model,
        device='cpu',
        max_steps=steps,
        randomize_best=False,
        epsilon=0.0,
        input_addendum=input_addendum,
        ppo_input_reward=False,
    )

    # Vectorized episode with batch_episodes=1
    episodes = generate_episodes_vectorized(
        env=env,
        model=model,
        device='cpu',
        max_steps=steps,
        batch_episodes=1,
        randomize_best=False,
        epsilon=0.0,
        input_addendum=input_addendum,
        ppo_input_reward=False,
        action_selection='greedy',
        profile=False,
        prof=None,
        use_preconv_codebook=False,
    )
    assert episodes and len(episodes) == 1
    ep = episodes[0]
    return obs_seq, labels, ep['obs'], ep['labels']


@torch.no_grad()
def test_generate_episodes_vectorized_matches_serial(monkeypatch):
    # Arrange
    model = DummyModel(num_actions=4, hidden_size=8)
    env = WMEnv(size=6, speed=1, seed=123, observation_size=32)
    steps = 12

    # Align vectorized initial state with the serial reset state
    _wrap_reset_and_capture_state(monkeypatch)
    _wrap_vec_reset_all_to_align(monkeypatch)

    # None, goal, diff
    for input_addendum in (None, 'goal', 'diff'):
        obs_seq, labels, obs_vec, labels_vec = _run_pair(env, model, steps, input_addendum)

        # Assert
        assert obs_seq.shape == obs_vec.shape
        assert labels.shape == labels_vec.shape
        np.testing.assert_allclose(obs_seq, obs_vec, atol=0.0, rtol=0.0)
        np.testing.assert_array_equal(labels, labels_vec)


