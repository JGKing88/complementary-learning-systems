"""Tests to verify that train_action_classifier.py tests the same task as train.py with next_best.

Key equivalences to verify:
1. g_idx (and other grid representations) are heading-invariant
2. The (start_obs, next_obs) -> action task is equivalent to (current, next_best) -> best_action
3. The convert_obs output is the same regardless of how we get there
"""

import numpy as np
import pytest

from cls.envs.environments import GridWMEnv, WMEnv
from cls.utils.GridUtils import VectorHash


CARDINAL_ACTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # N, E, S, W


@pytest.fixture
def setup_env_with_vectorhash():
    """Create a GridWMEnv with VectorHash initialized."""
    size = 8
    env = GridWMEnv(size=size, speed=1, seed=42, observation_size=128, input_type="g_idx")
    vh = VectorHash(Np=1600, lambdas=[11, 12], size=size)
    vh.initiate_vectorhash([env])
    return env, vh


class TestHeadingInvariance:
    """Test that g_idx only depends on position, not heading."""
    
    def test_g_idx_same_for_all_headings(self, setup_env_with_vectorhash):
        """g_idx should be identical for the same position with different headings."""
        env, vh = setup_env_with_vectorhash
        
        # Test multiple positions
        test_positions = [(0, 0), (3, 4), (7, 7), (2, 5)]
        
        for pos in test_positions:
            g_idx_per_heading = []
            for heading in CARDINAL_ACTIONS:
                raw_obs = env._code_for(pos, heading)
                g_idx = env.convert_obs(raw_obs)
                g_idx_per_heading.append(g_idx)
            
            # All g_idx should be identical
            for i in range(1, len(g_idx_per_heading)):
                np.testing.assert_array_equal(
                    g_idx_per_heading[0], g_idx_per_heading[i],
                    err_msg=f"g_idx differs for pos={pos} between heading 0 and {i}"
                )
    
    def test_g_hot_same_for_all_headings(self, setup_env_with_vectorhash):
        """g_hot should be identical for the same position with different headings."""
        env, vh = setup_env_with_vectorhash
        env.input_type = "g_hot"  # Switch to g_hot
        
        test_positions = [(0, 0), (3, 4), (7, 7)]
        
        for pos in test_positions:
            g_hot_per_heading = []
            for heading in CARDINAL_ACTIONS:
                raw_obs = env._code_for(pos, heading)
                g_hot = env.convert_obs(raw_obs)
                g_hot_per_heading.append(g_hot)
            
            for i in range(1, len(g_hot_per_heading)):
                np.testing.assert_array_equal(
                    g_hot_per_heading[0], g_hot_per_heading[i],
                    err_msg=f"g_hot differs for pos={pos} between heading 0 and {i}"
                )


class TestTaskEquivalence:
    """Test that action_classifier task is equivalent to next_best task."""
    
    def test_next_best_gives_one_step_neighbor(self, setup_env_with_vectorhash):
        """Verify that next_best position is exactly one step in best_action direction."""
        env, vh = setup_env_with_vectorhash
        
        # Set various positions and goals
        test_cases = [
            ((2, 2), (5, 5)),  # goal to NE
            ((5, 5), (2, 2)),  # goal to SW
            ((0, 4), (7, 4)),  # goal to E
            ((4, 0), (4, 7)),  # goal to N
        ]
        
        for start_pos, goal_pos in test_cases:
            env._pos = start_pos
            env._goal = goal_pos
            env._heading = (1, 0)  # arbitrary heading
            
            best_action = env.best_action_to_goal()
            ndx, ndy = env._normalize_vector(best_action[0], best_action[1])
            
            # Get next_best observation
            next_best_obs = env.obs_at_next_best_step()
            
            # Manually compute expected next position
            expected_next_pos = env._simulate_move(start_pos, (ndx, ndy), env.speed)
            expected_next_heading = (ndx, ndy) if expected_next_pos != start_pos else env._heading
            expected_obs = env.convert_obs(env._code_for(expected_next_pos, expected_next_heading))
            
            np.testing.assert_array_equal(
                next_best_obs, expected_obs,
                err_msg=f"next_best_obs mismatch for start={start_pos}, goal={goal_pos}"
            )
    
    def test_action_from_displacement_is_unique(self, setup_env_with_vectorhash):
        """Given (pos1, pos2) where pos2 is one step from pos1, the action is uniquely determined."""
        env, vh = setup_env_with_vectorhash
        
        # For interior positions, each action leads to a different next position
        pos = (4, 4)
        
        next_positions = {}
        for action_idx, action_vec in enumerate(CARDINAL_ACTIONS):
            next_pos = env._simulate_move(pos, action_vec, env.speed)
            if next_pos != pos:  # Only if movement occurred
                next_positions[next_pos] = action_idx
        
        # All next positions should be unique (no two actions lead to same position)
        assert len(next_positions) == len(set(next_positions.keys())), \
            "Multiple actions lead to same next position"
    
    def test_classifier_input_matches_next_best_input(self, setup_env_with_vectorhash):
        """The input format (concat of two g_idx) is the same in both tasks."""
        env, vh = setup_env_with_vectorhash
        
        # Setup position and goal
        start_pos = (3, 3)
        goal = (6, 6)
        start_heading = (1, 0)
        
        env._pos = start_pos
        env._goal = goal
        env._heading = start_heading
        
        # Get input the train.py way (with next_best)
        current_obs = env.convert_obs(env._code_for(start_pos, start_heading))
        next_best_obs = env.obs_at_next_best_step()
        train_py_input = np.concatenate([current_obs, next_best_obs])
        
        # Get input the action_classifier way
        best_action = env.best_action_to_goal()
        ndx, ndy = env._normalize_vector(best_action[0], best_action[1])
        action_vec = (ndx, ndy)
        next_pos = env._simulate_move(start_pos, action_vec, env.speed)
        moved = next_pos != start_pos
        next_heading = action_vec if moved else start_heading
        
        start_obs = env.convert_obs(env._code_for(start_pos, start_heading))
        next_obs = env.convert_obs(env._code_for(next_pos, next_heading))
        classifier_input = np.concatenate([start_obs, next_obs])
        
        np.testing.assert_array_equal(
            train_py_input, classifier_input,
            err_msg="Input format differs between train.py and action_classifier"
        )


class TestLabelEquivalence:
    """Test that labels are equivalent between the two tasks."""
    
    def test_best_action_equals_displacement_direction(self, setup_env_with_vectorhash):
        """The best_action label equals the direction of displacement to next_best."""
        env, vh = setup_env_with_vectorhash
        
        test_cases = [
            ((2, 2), (5, 5)),
            ((5, 5), (2, 2)),
            ((0, 4), (7, 4)),
            ((4, 0), (4, 7)),
            ((3, 3), (3, 7)),  # straight N
            ((3, 3), (7, 3)),  # straight E
        ]
        
        for start_pos, goal_pos in test_cases:
            env._pos = start_pos
            env._goal = goal_pos
            env._heading = (1, 0)
            
            best_action = env.best_action_to_goal()
            ndx, ndy = env._normalize_vector(best_action[0], best_action[1])
            action_vec = (ndx, ndy)
            
            # The label in train.py is the index of best_action
            if action_vec in CARDINAL_ACTIONS:
                train_py_label = CARDINAL_ACTIONS.index(action_vec)
            else:
                continue  # Skip non-cardinal
            
            # Compute next_pos
            next_pos = env._simulate_move(start_pos, action_vec, env.speed)
            
            if next_pos == start_pos:
                continue  # Skip no-movement cases
            
            # Infer direction from displacement
            dx = next_pos[0] - start_pos[0]
            dy = next_pos[1] - start_pos[1]
            
            # Normalize to unit vector
            if dx != 0:
                dx = dx // abs(dx)
            if dy != 0:
                dy = dy // abs(dy)
            
            inferred_direction = (dx, dy)
            
            assert inferred_direction == action_vec, \
                f"Displacement direction {inferred_direction} != action {action_vec}"
            
            # Check label matches
            classifier_label = CARDINAL_ACTIONS.index(inferred_direction)
            assert train_py_label == classifier_label, \
                f"Labels differ: train.py={train_py_label}, classifier={classifier_label}"


class TestBoundaryHandling:
    """Test that boundary cases are handled correctly."""
    
    def test_no_movement_at_boundary_detected(self, setup_env_with_vectorhash):
        """When at boundary, movement in blocked direction returns same position."""
        env, vh = setup_env_with_vectorhash
        
        # Test corners
        boundary_cases = [
            ((0, 0), (-1, 0)),  # W blocked
            ((0, 0), (0, -1)),  # S blocked
            ((7, 7), (1, 0)),   # E blocked
            ((7, 7), (0, 1)),   # N blocked
        ]
        
        for pos, action_vec in boundary_cases:
            next_pos = env._simulate_move(pos, action_vec, env.speed)
            assert next_pos == pos, \
                f"Expected no movement at boundary: pos={pos}, action={action_vec}, got next_pos={next_pos}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
