#!/usr/bin/env python3
"""Unit tests for MAPPODogfightWrapper class."""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from uav_intent_rl.envs.MAPPODogfightWrapper import MAPPODogfightWrapper
from uav_intent_rl.envs.MultiDroneDogfightAviary import MultiDroneDogfightAviary
from uav_intent_rl.policies.team_scripted_red import TeamScriptedRedPolicy, TeamTactic


class TestMAPPODogfightWrapper(unittest.TestCase):
    """Test cases for MAPPODogfightWrapper class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock environment that mimics MultiDroneDogfightAviary
        class MockDogfightEnv(gym.Env):
            def __init__(self):
                self.NUM_DRONES = 6
                self.drones_per_team = 3
                self.PYB_FREQ = 240
                self.EPISODE_LEN_SEC = 30
                self.step_counter = 0
                
                # Mock action space
                self.action_space = spaces.Box(
                    low=np.array([-1, -1, -1, -1]),
                    high=np.array([1, 1, 1, 1]),
                    dtype=np.float32
                )
                
                # Mock observation space
                self.observation_space = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(6, 12),
                    dtype=np.float32
                )
                
                # Mock drone states (pos, vel, rpy, ang_vel)
                self.mock_states = {
                    0: np.array([0, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  # Blue 0
                    1: np.array([1, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  # Blue 1
                    2: np.array([2, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  # Blue 2
                    3: np.array([3, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  # Red 3
                    4: np.array([4, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  # Red 4
                    5: np.array([5, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  # Red 5
                }
                
                # Mock alive status and health
                self.alive_status = [True] * 6
                self.health = [100.0] * 6
                self.elimination_count = [0, 0]
                
                # Mock methods
                self.reset = Mock(return_value=(np.zeros((6, 12)), {}))
                self.step = Mock(return_value=(
                    np.zeros((6, 12)),  # obs
                    np.array([1.0, 2.0, 3.0, -1.0, -2.0, -3.0]),  # rewards
                    False,  # terminated
                    False,  # truncated
                    {}  # info
                ))
                
                def mock_get_state(drone_id):
                    return self.mock_states[drone_id]
                
                self._getDroneStateVector = Mock(side_effect=mock_get_state)
        
        self.mock_env = MockDogfightEnv()

    def test_initialization_defaults(self):
        """Test wrapper initialization with default parameters."""
        wrapper = MAPPODogfightWrapper(env=self.mock_env)
        
        # Test that shared_reward attribute is properly set (FIX 1)
        self.assertEqual(wrapper.shared_reward, False)
        self.assertEqual(wrapper.centralized_critic, True)
        self.assertEqual(wrapper.communication_radius, 10.0)
        self.assertEqual(wrapper.reward_sharing_ratio, 0.3)
        self.assertEqual(wrapper.add_agent_id, True)
        self.assertEqual(wrapper.normalize_rewards, True)
        self.assertEqual(wrapper.num_blue_drones, 3)

    def test_initialization_custom_params(self):
        """Test wrapper initialization with custom parameters."""
        wrapper = MAPPODogfightWrapper(
            env=self.mock_env,
            shared_reward=True,
            centralized_critic=False,
            communication_radius=15.0,
            reward_sharing_ratio=0.5,
            add_agent_id=False,
            normalize_rewards=False
        )
        
        # Test that shared_reward attribute is properly set (FIX 1)
        self.assertEqual(wrapper.shared_reward, True)
        self.assertEqual(wrapper.centralized_critic, False)
        self.assertEqual(wrapper.communication_radius, 15.0)
        self.assertEqual(wrapper.reward_sharing_ratio, 0.5)
        self.assertEqual(wrapper.add_agent_id, False)
        self.assertEqual(wrapper.normalize_rewards, False)

    def test_observation_space_shape(self):
        """Test that observation space has correct shape (FIX 2)."""
        wrapper = MAPPODogfightWrapper(env=self.mock_env)
        
        # Calculate expected observation dimension
        # Own state: 12 (pos3 + vel3 + rpy3 + ang_vel3)
        # Teammates: (3-1) * 7 = 14 (rel_pos3 + vel3 + health1)
        # Enemies: 3 * 8 = 24 (rel_pos3 + rel_vel3 + health1 + dist1)
        # Agent ID: 3 (one-hot)
        expected_dim = 12 + 14 + 24 + 3
        
        # Test that observation space is single-agent (FIX 2)
        self.assertEqual(wrapper.observation_space.shape, (expected_dim,))
        self.assertNotEqual(wrapper.observation_space.shape, (3, expected_dim))

    def test_observation_space_without_agent_id(self):
        """Test observation space without agent ID."""
        wrapper = MAPPODogfightWrapper(env=self.mock_env, add_agent_id=False)
        
        # Calculate expected observation dimension without agent ID
        expected_dim = 12 + 14 + 24  # No +3 for agent ID
        
        self.assertEqual(wrapper.observation_space.shape, (expected_dim,))

    def test_state_space_centralized_critic(self):
        """Test state space for centralized critic."""
        wrapper = MAPPODogfightWrapper(env=self.mock_env, centralized_critic=True)
        
        # Calculate expected state dimension
        # All drones: 6 * 10 = 60 (pos3 + vel3 + rpy3 + health1)
        # Team stats: 4 (blue_alive, red_alive, blue_elim, red_elim)
        # Time: 1
        expected_dim = 60 + 4 + 1
        
        self.assertEqual(wrapper.state_space.shape, (expected_dim,))

    def test_state_space_no_centralized_critic(self):
        """Test that state_space is None when centralized_critic=False."""
        wrapper = MAPPODogfightWrapper(env=self.mock_env, centralized_critic=False)
        
        self.assertFalse(hasattr(wrapper, 'state_space'))

    def test_reset_reward_normalization(self):
        """Test that reset properly resets reward normalization (FIX 3)."""
        wrapper = MAPPODogfightWrapper(env=self.mock_env, normalize_rewards=True)
        
        # Modify normalization stats
        wrapper.reward_mean = 5.0
        wrapper.reward_m2 = 10.0
        wrapper.reward_count = 100
        
        # Reset
        obs, info = wrapper.reset()
        
        # Check that stats are reset
        self.assertEqual(wrapper.reward_mean, 0.0)
        self.assertEqual(wrapper.reward_m2, 0.0)
        self.assertEqual(wrapper.reward_count, 0)

    def test_reward_sharing_works(self):
        """Test that reward sharing works correctly (FIX 1)."""
        wrapper = MAPPODogfightWrapper(
            env=self.mock_env,
            shared_reward=True,
            reward_sharing_ratio=0.5,
            normalize_rewards=False  # Disable normalization for this test
        )
        
        # Test individual rewards
        individual_rewards = np.array([1.0, 2.0, 3.0])
        
        # Process rewards
        processed_rewards = wrapper._process_mappo_rewards(individual_rewards)
        
        # Expected: 50% individual + 50% team average
        team_average = np.mean(individual_rewards)  # 2.0
        expected = 0.5 * individual_rewards + 0.5 * team_average
        expected = np.array([1.5, 2.0, 2.5])
        
        np.testing.assert_array_almost_equal(processed_rewards, expected)

    def test_reward_sharing_disabled(self):
        """Test that reward sharing is disabled when shared_reward=False."""
        wrapper = MAPPODogfightWrapper(
            env=self.mock_env,
            shared_reward=False,
            reward_sharing_ratio=0.5,
            normalize_rewards=False
        )
        
        individual_rewards = np.array([1.0, 2.0, 3.0])
        processed_rewards = wrapper._process_mappo_rewards(individual_rewards)
        
        # Should return original rewards unchanged
        np.testing.assert_array_equal(processed_rewards, individual_rewards)

    def test_reward_normalization_welford_algorithm(self):
        """Test reward normalization using Welford's algorithm (FIX 3)."""
        wrapper = MAPPODogfightWrapper(
            env=self.mock_env,
            normalize_rewards=True,
            shared_reward=False
        )
        
        # Test with a sequence of rewards
        rewards_sequence = [
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0]),
            np.array([5.0, 6.0])
        ]
        
        processed_rewards = []
        for rewards in rewards_sequence:
            processed = wrapper._process_mappo_rewards(rewards)
            processed_rewards.append(processed)
        
        # Check that normalization stats are updated correctly
        self.assertGreater(wrapper.reward_count, 0)
        self.assertNotEqual(wrapper.reward_mean, 0.0)
        self.assertGreater(wrapper.reward_m2, 0.0)
        
        # Check that later rewards are normalized
        # The first rewards should be less normalized than later ones
        self.assertTrue(np.any(np.abs(processed_rewards[0]) > 0.1))
        self.assertTrue(np.any(np.abs(processed_rewards[-1]) < 2.0))

    def test_extract_observations_structure(self):
        """Test that _extract_observations returns correct structure."""
        wrapper = MAPPODogfightWrapper(env=self.mock_env)
        
        observations = wrapper._extract_observations()
        
        # Should return array of shape (num_blue_drones, obs_dim)
        self.assertEqual(observations.shape, (3, wrapper.observation_space.shape[0]))
        self.assertEqual(observations.dtype, np.float32)

    def test_extract_observations_with_agent_id(self):
        """Test observations include agent ID when add_agent_id=True."""
        wrapper = MAPPODogfightWrapper(env=self.mock_env, add_agent_id=True)
        
        observations = wrapper._extract_observations()
        
        # Check that each agent has its one-hot ID
        for i in range(3):
            # Agent ID should be at the end of the observation
            agent_id_slice = observations[i, -3:]  # Last 3 elements
            expected_id = np.zeros(3)
            expected_id[i] = 1.0
            np.testing.assert_array_equal(agent_id_slice, expected_id)

    def test_extract_observations_without_agent_id(self):
        """Test observations don't include agent ID when add_agent_id=False."""
        wrapper = MAPPODogfightWrapper(env=self.mock_env, add_agent_id=False)
        
        observations = wrapper._extract_observations()
        
        # Should have smaller observation dimension
        expected_dim = 12 + 14 + 24  # No +3 for agent ID
        self.assertEqual(observations.shape, (3, expected_dim))

    def test_get_global_state(self):
        """Test global state extraction for centralized critic."""
        wrapper = MAPPODogfightWrapper(env=self.mock_env, centralized_critic=True)
        
        state = wrapper.get_global_state()
        
        # Check state dimension
        expected_dim = 60 + 4 + 1  # drones + stats + time
        self.assertEqual(state.shape, (expected_dim,))
        self.assertEqual(state.dtype, np.float32)
        
        # Check that state contains drone information
        self.assertTrue(np.any(state != 0))  # Should have some non-zero values

    def test_communication_graph(self):
        """Test communication graph generation."""
        wrapper = MAPPODogfightWrapper(env=self.mock_env, communication_radius=5.0)
        
        comm_graph = wrapper._get_communication_graph()
        
        # Should be adjacency matrix for blue team
        self.assertEqual(comm_graph.shape, (3, 3))
        self.assertEqual(comm_graph.dtype, np.float32)
        
        # Diagonal should be 0 (no self-communication)
        np.testing.assert_array_equal(np.diag(comm_graph), np.zeros(3))

    def test_step_with_centralized_critic(self):
        """Test step method with centralized critic enabled."""
        wrapper = MAPPODogfightWrapper(env=self.mock_env, centralized_critic=True)
        
        # Mock the parent step method
        with patch.object(wrapper, '_extract_observations') as mock_extract:
            mock_extract.return_value = np.random.rand(3, wrapper.observation_space.shape[0])
            
            obs, rewards, terminated, truncated, info = wrapper.step(np.random.rand(3, 4))
            
            # Check that global state is added to info
            self.assertIn('global_state', info)
            self.assertIn('communication_graph', info)
            
            # Check global state shape
            self.assertEqual(info['global_state'].shape, (wrapper.state_space.shape[0],))

    def test_step_without_centralized_critic(self):
        """Test step method without centralized critic."""
        wrapper = MAPPODogfightWrapper(env=self.mock_env, centralized_critic=False)
        
        with patch.object(wrapper, '_extract_observations') as mock_extract:
            mock_extract.return_value = np.random.rand(3, wrapper.observation_space.shape[0])
            
            obs, rewards, terminated, truncated, info = wrapper.step(np.random.rand(3, 4))
            
            # Should not have global_state in info
            self.assertNotIn('global_state', info)
            self.assertIn('communication_graph', info)

    def test_reward_normalization_edge_cases(self):
        """Test reward normalization edge cases."""
        wrapper = MAPPODogfightWrapper(env=self.mock_env, normalize_rewards=True)
        
        # Test with zero rewards
        zero_rewards = np.array([0.0, 0.0, 0.0])
        processed = wrapper._process_mappo_rewards(zero_rewards)
        
        # Should handle zero rewards gracefully
        self.assertEqual(processed.shape, zero_rewards.shape)
        
        # Test with single reward
        single_reward = np.array([1.0])
        processed = wrapper._process_mappo_rewards(single_reward)
        self.assertEqual(processed.shape, single_reward.shape)

    def test_observation_calculation_dimensions(self):
        """Test that observation dimensions are calculated correctly."""
        wrapper = MAPPODogfightWrapper(env=self.mock_env)
        
        obs_dim = wrapper._calculate_obs_dim()
        state_dim = wrapper._calculate_state_dim()
        
        # Verify dimensions match observation space
        self.assertEqual(obs_dim, wrapper.observation_space.shape[0])
        self.assertEqual(state_dim, wrapper.state_space.shape[0])

    def test_red_policy_integration(self):
        """Test integration with red policy."""
        mock_red_policy = Mock(spec=TeamScriptedRedPolicy)
        
        wrapper = MAPPODogfightWrapper(
            env=self.mock_env,
            red_policy=mock_red_policy,
            red_tactic=TeamTactic.AGGRESSIVE
        )
        
        # Check that red policy is properly set
        self.assertEqual(wrapper.red_policy, mock_red_policy)
        self.assertEqual(wrapper.red_tactic, TeamTactic.AGGRESSIVE)


if __name__ == '__main__':
    unittest.main() 