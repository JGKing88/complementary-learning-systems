"""Tests for GridEncoder and Hopfield integration."""

import numpy as np
import torch
import pytest

from cls.encoder import GridEncoder
from cls.hopfield import Hopfield
from cls.envs.environments import GridWMEnv, GridWMVecEnv
from cls.utils.GridUtils import VectorHash


# ============== Fixtures ==============

@pytest.fixture
def lambdas():
    return [3, 4]  # Small for fast tests


@pytest.fixture
def grid_size():
    return 4


@pytest.fixture
def encoder(lambdas):
    g_hot_dim = sum(l**2 for l in lambdas)
    enc = GridEncoder(in_dim=g_hot_dim, hidden=32, out_dim=16)
    enc.eval()
    return enc


@pytest.fixture
def basic_env(grid_size):
    """Basic GridWMEnv without vectorhash."""
    return GridWMEnv(size=grid_size, speed=1, seed=42, input_type="g_hot")


@pytest.fixture
def env_with_vectorhash(grid_size, lambdas):
    """GridWMEnv with vectorhash initialized."""
    env = GridWMEnv(size=grid_size, speed=1, seed=42, input_type="g_hot")
    vh = VectorHash(Np=100, lambdas=lambdas, size=grid_size)
    vh.initiate_vectorhash([env])
    return env


@pytest.fixture
def env_with_encoder(grid_size, lambdas, encoder):
    """GridWMEnv with encoder and vectorhash."""
    env = GridWMEnv(size=grid_size, speed=1, seed=42, input_type="encoded_g", encoder=encoder)
    env._encoder_device = None  # CPU
    vh = VectorHash(Np=100, lambdas=lambdas, size=grid_size)
    vh.initiate_vectorhash([env])
    return env


@pytest.fixture
def env_with_hopfield(grid_size, lambdas):
    """GridWMEnv with hopfield-enabled vectorhash."""
    env = GridWMEnv(size=grid_size, speed=1, seed=42, input_type="g_hot")
    vh = VectorHash(
        Np=100, lambdas=lambdas, size=grid_size,
        use_hopfield=True, hopfield_beta=2.0, hopfield_steps=1
    )
    vh.initiate_vectorhash([env])
    return env


# ============== GridEncoder Tests ==============

class TestGridEncoder:
    def test_encoder_forward(self, encoder, lambdas):
        """Test encoder forward pass."""
        g_hot_dim = sum(l**2 for l in lambdas)
        x = torch.randn(1, g_hot_dim)
        out = encoder(x)
        assert out.shape == (1, 16)
        # Output should be L2 normalized
        norm = torch.norm(out, p=2, dim=-1)
        assert torch.allclose(norm, torch.ones(1), atol=1e-5)

    def test_encoder_batch(self, encoder, lambdas):
        """Test encoder with batch input."""
        g_hot_dim = sum(l**2 for l in lambdas)
        x = torch.randn(8, g_hot_dim)
        out = encoder(x)
        assert out.shape == (8, 16)

    def test_encoder_deterministic(self, encoder, lambdas):
        """Test encoder is deterministic."""
        g_hot_dim = sum(l**2 for l in lambdas)
        x = torch.randn(1, g_hot_dim)
        out1 = encoder(x)
        out2 = encoder(x)
        assert torch.allclose(out1, out2)
    
    def test_different_inputs_different_outputs(self, encoder, lambdas):
        """Test that different inputs produce different outputs."""
        g_hot_dim = sum(l**2 for l in lambdas)
        x1 = torch.zeros(1, g_hot_dim)
        x1[0, 0] = 1.0
        x2 = torch.zeros(1, g_hot_dim)
        x2[0, 1] = 1.0
        out1 = encoder(x1)
        out2 = encoder(x2)
        assert not torch.allclose(out1, out2), "Different inputs should produce different outputs"


# ============== GridWMEnv Encoder Integration Tests ==============

class TestGridWMEnvEncoder:
    def test_get_input_size_encoded_g(self, env_with_encoder):
        """Test get_input_size returns encoder output dim for encoded_g."""
        assert env_with_encoder.get_input_size() == 16

    def test_obs_returns_encoded(self, env_with_encoder, encoder, lambdas):
        """Test obs() returns encoded output."""
        obs = env_with_encoder.obs()
        assert obs.shape == (16,)
        # Should be normalized (encoder normalizes output)
        norm = np.linalg.norm(obs)
        assert np.isclose(norm, 1.0, atol=1e-5)

    def test_obs_at_goal_encoded(self, env_with_encoder):
        """Test obs_at_goal() returns encoded output."""
        goal_obs = env_with_encoder.obs_at_goal()
        assert goal_obs.shape == (16,)

    def test_clone_preserves_encoder(self, env_with_encoder):
        """Test clone() preserves encoder reference."""
        cloned = env_with_encoder.clone()
        assert cloned.encoder is env_with_encoder.encoder
        assert cloned.input_type == "encoded_g"
    
    def test_obs_deterministic_for_position(self, env_with_encoder):
        """Test that obs is deterministic for a given position."""
        env_with_encoder.reset()
        pos1 = env_with_encoder.current_location
        obs1 = env_with_encoder.obs().copy()
        
        # Move around
        env_with_encoder.step((1, 0))
        env_with_encoder.step((0, 1))
        
        # Manually set back to original position
        env_with_encoder._pos = pos1
        obs2 = env_with_encoder.obs()
        
        # Same position should give same observation
        assert np.allclose(obs1, obs2), "Same position should give same observation"
    
    def test_encoded_g_matches_manual_encoding(self, grid_size, lambdas, encoder):
        """Test that encoded_g output matches manually encoding g_hot."""
        # Create env with g_hot
        env_ghot = GridWMEnv(size=grid_size, speed=1, seed=42, input_type="g_hot")
        vh = VectorHash(Np=100, lambdas=lambdas, size=grid_size)
        vh.initiate_vectorhash([env_ghot])
        
        # Create env with encoded_g  
        env_enc = GridWMEnv(size=grid_size, speed=1, seed=42, input_type="encoded_g", encoder=encoder)
        env_enc._encoder_device = None
        vh2 = VectorHash(Np=100, lambdas=lambdas, size=grid_size)
        vh2.initiate_vectorhash([env_enc])
        
        # Reset both to same position
        env_ghot.reset()
        env_enc.reset()
        env_enc._pos = env_ghot._pos
        env_enc._heading = env_ghot._heading
        
        # Get observations
        g_hot = env_ghot.obs()
        encoded_obs = env_enc.obs()
        
        # Manually encode g_hot
        with torch.no_grad():
            g_t = torch.from_numpy(g_hot).float().unsqueeze(0)
            manual_encoded = encoder(g_t).squeeze(0).numpy()
        
        assert np.allclose(encoded_obs, manual_encoded, atol=1e-5), \
            "encoded_g obs should match manually encoded g_hot"


# ============== VectorHash Hopfield Tests ==============

class TestVectorHashHopfield:
    def test_hopfield_initialized(self, env_with_hopfield):
        """Test Hopfield network is initialized when use_hopfield=True."""
        vh = env_with_hopfield.vectorhash
        assert vh.hopfield is not None
        assert isinstance(vh.hopfield, Hopfield)

    def test_goal_patterns_stored(self, env_with_hopfield):
        """Test goal patterns are stored in Hopfield network."""
        vh = env_with_hopfield.vectorhash
        # One env = one goal pattern stored
        assert vh.hopfield.num_memories == 1

    def test_hopfield_recall_shape(self, env_with_hopfield):
        """Test hopfield_recall returns correct shape."""
        vh = env_with_hopfield.vectorhash
        obs = env_with_hopfield.obs()
        recalled = vh.hopfield_recall(obs)
        assert recalled.shape == obs.shape

    def test_hopfield_recall_batch_shape(self, env_with_hopfield):
        """Test hopfield_recall_batch returns correct shape."""
        vh = env_with_hopfield.vectorhash
        obs = env_with_hopfield.obs()
        obs_batch = np.stack([obs, obs, obs])
        recalled = vh.hopfield_recall_batch(obs_batch)
        assert recalled.shape == obs_batch.shape

    def test_hopfield_not_initialized_without_flag(self, env_with_vectorhash):
        """Test Hopfield is None when use_hopfield=False."""
        vh = env_with_vectorhash.vectorhash
        assert vh.hopfield is None
    
    def test_hopfield_recall_produces_output(self, env_with_hopfield):
        """Test that Hopfield recall produces valid output from arbitrary input."""
        vh = env_with_hopfield.vectorhash
        
        # Get goal observation (stored pattern)
        goal_obs = env_with_hopfield.obs_at_goal()
        
        # Test recall from random input
        random_input = np.random.randn(*goal_obs.shape).astype(np.float32)
        recalled = vh.hopfield_recall(random_input)
        
        # Recalled should have same shape
        assert recalled.shape == goal_obs.shape
        # Recalled should be finite
        assert np.all(np.isfinite(recalled))
        # Recalled should not be all zeros (hopfield should do something)
        assert not np.allclose(recalled, 0)
    
    def test_hopfield_stores_correct_goal(self, grid_size, lambdas):
        """Test that Hopfield stores the actual goal observation."""
        env = GridWMEnv(size=grid_size, speed=1, seed=42, input_type="g_hot")
        vh = VectorHash(
            Np=100, lambdas=lambdas, size=grid_size,
            use_hopfield=True, hopfield_beta=2.0, hopfield_steps=1
        )
        vh.initiate_vectorhash([env])
        
        # Get goal obs
        goal_obs = env.obs_at_goal()
        
        # Recall starting from goal should return something very close to goal
        recalled = vh.hopfield_recall(goal_obs)
        
        # Should be very similar (not exactly equal due to hopfield dynamics)
        cos_sim = np.dot(recalled, goal_obs) / (np.linalg.norm(recalled) * np.linalg.norm(goal_obs) + 1e-8)
        assert cos_sim > 0.9, f"Recalled goal should be similar to stored goal, got cos_sim={cos_sim}"


# ============== GridWMVecEnv Tests ==============

class TestGridWMVecEnvEncoder:
    def test_obs_batch_encoded_g(self, env_with_encoder):
        """Test obs_batch with encoded_g input type."""
        vec_env = GridWMVecEnv(env_with_encoder, batch_size=4)
        obs = vec_env.obs_batch([0, 1, 2, 3])
        assert obs.shape == (4, 16)

    def test_preconv_codebook_encoded_g(self, env_with_encoder):
        """Test precomputed codebook works for encoded_g."""
        vec_env = GridWMVecEnv(env_with_encoder, batch_size=4, use_preconv_codebook=True)
        assert vec_env._preconv_codebook is not None
        # Should have correct output dimension
        assert vec_env._preconv_codebook.shape[-1] == 16
    
    def test_preconv_matches_onthefly_encoded_g(self, env_with_encoder):
        """Test precomputed codebook gives same results as on-the-fly for encoded_g."""
        vec_env_preconv = GridWMVecEnv(env_with_encoder, batch_size=4, use_preconv_codebook=True)
        vec_env_onthefly = GridWMVecEnv(env_with_encoder, batch_size=4, use_preconv_codebook=False)
        
        # Sync positions
        vec_env_onthefly._pos = vec_env_preconv._pos.copy()
        vec_env_onthefly._heading = vec_env_preconv._heading.copy()
        
        obs_preconv = vec_env_preconv.obs_batch([0, 1, 2, 3])
        obs_onthefly = vec_env_onthefly.obs_batch([0, 1, 2, 3])
        
        assert np.allclose(obs_preconv, obs_onthefly, atol=1e-5), \
            "Precomputed and on-the-fly should give same results"


class TestGridWMVecEnvHopfield:
    def test_obs_batch_hopfield_addendum(self, env_with_hopfield):
        """Test obs_batch with hopfield addendum."""
        vec_env = GridWMVecEnv(env_with_hopfield, batch_size=4)
        base_obs = vec_env.obs_batch([0, 1, 2, 3])
        hopfield_obs = vec_env.obs_batch([0, 1, 2, 3], input_addendum="hopfield")
        
        # Hopfield addendum should double the observation size
        assert hopfield_obs.shape == (4, base_obs.shape[1] * 2)
        
        # First half should be the base observation
        assert np.allclose(hopfield_obs[:, :base_obs.shape[1]], base_obs)

    def test_obs_batch_hopfield_preconv(self, env_with_hopfield):
        """Test hopfield addendum works with precomputed codebook."""
        vec_env = GridWMVecEnv(env_with_hopfield, batch_size=4, use_preconv_codebook=True)
        assert vec_env._preconv_codebook is not None
        
        obs = vec_env.obs_batch([0, 1, 2], input_addendum="hopfield")
        base_dim = env_with_hopfield.get_input_size()
        assert obs.shape == (3, base_dim * 2)
    
    def test_hopfield_addendum_second_half_is_recall(self, env_with_hopfield):
        """Test that second half of hopfield addendum is actually the recall output."""
        vec_env = GridWMVecEnv(env_with_hopfield, batch_size=4)
        base_obs = vec_env.obs_batch([0, 1, 2, 3])
        hopfield_obs = vec_env.obs_batch([0, 1, 2, 3], input_addendum="hopfield")
        
        # Manually compute recall
        vh = env_with_hopfield.vectorhash
        expected_recall = vh.hopfield_recall_batch(base_obs)
        
        # Second half should match
        second_half = hopfield_obs[:, base_obs.shape[1]:]
        assert np.allclose(second_half, expected_recall, atol=1e-5), \
            "Second half of hopfield addendum should be the recall output"


# ============== Integration Tests ==============

class TestEncoderHopfieldIntegration:
    def test_encoded_g_with_hopfield(self, grid_size, lambdas, encoder):
        """Test encoded_g input type with hopfield addendum."""
        env = GridWMEnv(
            size=grid_size, speed=1, seed=42,
            input_type="encoded_g", encoder=encoder
        )
        env._encoder_device = None
        
        vh = VectorHash(
            Np=100, lambdas=lambdas, size=grid_size,
            use_hopfield=True, hopfield_beta=2.0, hopfield_steps=1
        )
        vh.initiate_vectorhash([env])
        
        # Hopfield should have encoded patterns
        assert vh.hopfield.num_units == 16  # encoder output dim
        
        # obs_batch with hopfield should work
        vec_env = GridWMVecEnv(env, batch_size=4)
        obs = vec_env.obs_batch([0, 1, 2, 3], input_addendum="hopfield")
        assert obs.shape == (4, 32)  # 16 * 2

    def test_multiple_envs_hopfield(self, grid_size, lambdas):
        """Test hopfield with multiple environments."""
        envs = [
            GridWMEnv(size=grid_size, speed=1, seed=i, input_type="g_hot")
            for i in range(3)
        ]
        
        vh = VectorHash(
            Np=100, lambdas=lambdas, size=grid_size,
            use_hopfield=True, hopfield_beta=2.0, hopfield_steps=1
        )
        vh.initiate_vectorhash(envs)
        
        # Should have 3 goal patterns stored
        assert vh.hopfield.num_memories == 3


# ============== Error Handling Tests ==============

class TestErrorHandling:
    def test_encoded_g_without_encoder_raises(self, grid_size, lambdas):
        """Test that encoded_g without encoder raises error."""
        env = GridWMEnv(size=grid_size, speed=1, seed=42, input_type="encoded_g")
        vh = VectorHash(Np=100, lambdas=lambdas, size=grid_size)
        vh.initiate_vectorhash([env])
        
        with pytest.raises(ValueError, match="Encoder required"):
            env.obs()

    def test_hopfield_recall_without_init_raises(self, env_with_vectorhash):
        """Test hopfield_recall raises when hopfield not initialized."""
        vh = env_with_vectorhash.vectorhash
        obs = env_with_vectorhash.obs()
        
        with pytest.raises(ValueError, match="Hopfield network not initialized"):
            vh.hopfield_recall(obs)


# ============== Training Integration Tests ==============

class TestTrainingIntegration:
    """Integration tests that verify the new functionality works with training code."""
    
    def test_training_with_encoded_g(self, grid_size, lambdas, encoder):
        """Test that training loop works with encoded_g input type."""
        from cls.models import Agent
        
        # Setup env
        env = GridWMEnv(size=grid_size, speed=1, seed=42, input_type="encoded_g", encoder=encoder)
        env._encoder_device = None
        vh = VectorHash(Np=100, lambdas=lambdas, size=grid_size)
        vh.initiate_vectorhash([env])
        
        # Create agent with correct input size
        input_size = env.get_input_size()
        assert input_size == 16
        
        agent = Agent(input_size=input_size, hidden_size=32, num_model_layers=1)
        agent.eval()
        
        # Simulate a forward pass like in training
        env.reset()
        obs = env.obs()
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).unsqueeze(0)  # (1, 1, F)
        
        logits, values, h = agent(obs_tensor, None)
        
        assert logits.shape == (1, 1, 4)  # 4 actions
        assert values.shape == (1, 1)
    
    def test_training_with_hopfield_addendum(self, grid_size, lambdas):
        """Test that training loop works with hopfield addendum."""
        from cls.models import Agent
        
        # Setup env with hopfield
        env = GridWMEnv(size=grid_size, speed=1, seed=42, input_type="g_hot")
        vh = VectorHash(
            Np=100, lambdas=lambdas, size=grid_size,
            use_hopfield=True, hopfield_beta=2.0, hopfield_steps=1
        )
        vh.initiate_vectorhash([env])
        
        # Input size doubles with hopfield addendum
        base_size = env.get_input_size()
        input_size = base_size * 2
        
        agent = Agent(input_size=input_size, hidden_size=32, num_model_layers=1)
        agent.eval()
        
        # Get observation with hopfield addendum
        env.reset()
        obs = env.obs()
        recalled = vh.hopfield_recall(obs)
        obs_with_hopfield = np.concatenate([obs, recalled])
        
        assert obs_with_hopfield.shape == (input_size,)
        
        obs_tensor = torch.from_numpy(obs_with_hopfield).float().unsqueeze(0).unsqueeze(0)
        logits, values, h = agent(obs_tensor, None)
        
        assert logits.shape == (1, 1, 4)
    
    def test_vectorized_training_encoded_g(self, grid_size, lambdas, encoder):
        """Test vectorized episode generation with encoded_g."""
        from cls.models import Agent
        
        # Setup env
        env = GridWMEnv(size=grid_size, speed=1, seed=42, input_type="encoded_g", encoder=encoder)
        env._encoder_device = None
        vh = VectorHash(Np=100, lambdas=lambdas, size=grid_size)
        vh.initiate_vectorhash([env])
        
        # Create vectorized env
        vec_env = GridWMVecEnv(env, batch_size=8)
        
        # Get batch of observations
        obs_batch = vec_env.obs_batch(list(range(8)))
        assert obs_batch.shape == (8, 16)
        
        # Create agent and forward
        agent = Agent(input_size=16, hidden_size=32, num_model_layers=1)
        agent.eval()
        
        obs_tensor = torch.from_numpy(obs_batch).float().unsqueeze(1)  # (B, 1, F)
        logits, values, h = agent(obs_tensor, None)
        
        assert logits.shape == (8, 1, 4)
    
    def test_vectorized_training_hopfield(self, grid_size, lambdas):
        """Test vectorized episode generation with hopfield addendum."""
        from cls.models import Agent
        
        # Setup env with hopfield
        env = GridWMEnv(size=grid_size, speed=1, seed=42, input_type="g_hot")
        vh = VectorHash(
            Np=100, lambdas=lambdas, size=grid_size,
            use_hopfield=True, hopfield_beta=2.0, hopfield_steps=1
        )
        vh.initiate_vectorhash([env])
        
        # Create vectorized env
        vec_env = GridWMVecEnv(env, batch_size=8)
        
        # Get batch with hopfield addendum
        obs_batch = vec_env.obs_batch(list(range(8)), input_addendum="hopfield")
        base_size = env.get_input_size()
        assert obs_batch.shape == (8, base_size * 2)
        
        # Create agent and forward
        agent = Agent(input_size=base_size * 2, hidden_size=32, num_model_layers=1)
        agent.eval()
        
        obs_tensor = torch.from_numpy(obs_batch).float().unsqueeze(1)
        logits, values, h = agent(obs_tensor, None)
        
        assert logits.shape == (8, 1, 4)
    
    def test_full_episode_rollout_encoded_g(self, grid_size, lambdas, encoder):
        """Test a full episode rollout with encoded_g."""
        from cls.models import Agent
        
        # Setup
        env = GridWMEnv(size=grid_size, speed=1, seed=42, input_type="encoded_g", encoder=encoder)
        env._encoder_device = None
        vh = VectorHash(Np=100, lambdas=lambdas, size=grid_size)
        vh.initiate_vectorhash([env])
        
        agent = Agent(input_size=16, hidden_size=32, num_model_layers=1)
        agent.eval()
        
        # Rollout episode
        env.reset()
        h = None
        observations = []
        actions = []
        
        for _ in range(10):  # Max 10 steps
            obs = env.obs()
            observations.append(obs)
            
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                logits, values, h = agent(obs_tensor, h)
            
            action_idx = logits[0, 0].argmax().item()
            actions.append(action_idx)
            
            action_vec = [(0, 1), (1, 0), (0, -1), (-1, 0)][action_idx]
            cur, goal, _, _ = env.step(action_vec)
            
            if cur == goal:
                break
        
        assert len(observations) > 0
        assert len(actions) == len(observations)
        assert all(obs.shape == (16,) for obs in observations)
    
    def test_full_episode_rollout_hopfield(self, grid_size, lambdas):
        """Test a full episode rollout with hopfield addendum."""
        from cls.models import Agent
        
        # Setup
        env = GridWMEnv(size=grid_size, speed=1, seed=42, input_type="g_hot")
        vh = VectorHash(
            Np=100, lambdas=lambdas, size=grid_size,
            use_hopfield=True, hopfield_beta=2.0, hopfield_steps=1
        )
        vh.initiate_vectorhash([env])
        
        base_size = env.get_input_size()
        agent = Agent(input_size=base_size * 2, hidden_size=32, num_model_layers=1)
        agent.eval()
        
        # Rollout episode
        env.reset()
        h = None
        observations = []
        
        for _ in range(10):
            obs = env.obs()
            recalled = vh.hopfield_recall(obs)
            obs_with_hopfield = np.concatenate([obs, recalled])
            observations.append(obs_with_hopfield)
            
            obs_tensor = torch.from_numpy(obs_with_hopfield).float().unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                logits, values, h = agent(obs_tensor, h)
            
            action_idx = logits[0, 0].argmax().item()
            action_vec = [(0, 1), (1, 0), (0, -1), (-1, 0)][action_idx]
            cur, goal, _, _ = env.step(action_vec)
            
            if cur == goal:
                break
        
        assert len(observations) > 0
        assert all(obs.shape == (base_size * 2,) for obs in observations)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
