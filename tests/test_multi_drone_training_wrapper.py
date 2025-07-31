import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from uav_intent_rl.envs.MultiDroneTrainingWrapper import (
    MultiDroneTrainingWrapper, 
    MultiDroneObservationType
)
from uav_intent_rl.policies.team_scripted_red import TeamScriptedRedPolicy, TeamTactic


class TestMultiDroneObservationType(unittest.TestCase):
    """Test the observation type constants."""
    
    def test_observation_type_values(self):
        """Test that observation type values are strings."""
        self.assertIsInstance(MultiDroneObservationType.LOCAL, str)
        self.assertIsInstance(MultiDroneObservationType.TEAM, str)
        self.assertIsInstance(MultiDroneObservationType.GLOBAL, str)
        self.assertIsInstance(MultiDroneObservationType.COMMUNICATION, str)
    
    def test_observation_type_names(self):
        """Test that observation type names are descriptive."""
        self.assertEqual(MultiDroneObservationType.LOCAL, "local")
        self.assertEqual(MultiDroneObservationType.TEAM, "team")
        self.assertEqual(MultiDroneObservationType.GLOBAL, "global")
        self.assertEqual(MultiDroneObservationType.COMMUNICATION, "communication")


class TestMultiDroneTrainingWrapper(unittest.TestCase):
    """Test the MultiDroneTrainingWrapper class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock environment that inherits from gym.Env
        class MockEnv(gym.Env):
            def __init__(self):
                self.NUM_DRONES = 6
                self.drones_per_team = 3
                
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
                
                # Mock drone states
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
                
                # Add CTRL_FREQ for the red policy
                self.CTRL_FREQ = 240.0
                
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
        
        self.mock_env = MockEnv()

        # Create wrapper
        self.wrapper = MultiDroneTrainingWrapper(
            env=self.mock_env,
            obs_type=MultiDroneObservationType.TEAM
        )
    
    def test_initialization(self):
        """Test wrapper initialization."""
        self.assertEqual(self.wrapper.num_blue_drones, 3)
        self.assertEqual(self.wrapper.obs_type, MultiDroneObservationType.TEAM)
        self.assertTrue(self.wrapper.shared_reward)
        self.assertEqual(self.wrapper.reward_sharing_decay, 0.8)
        self.assertIsInstance(self.wrapper._red_policy, TeamScriptedRedPolicy)
    
    def test_action_space(self):
        """Test that action space is correctly shaped for blue team only."""
        expected_shape = (3, 4)  # 3 blue drones, 4 actions each
        self.assertEqual(self.wrapper.action_space.shape, expected_shape)
        self.assertEqual(self.wrapper.action_space.dtype, np.float32)
    
    def test_observation_space_team(self):
        """Test observation space for team observation type."""
        # Team obs: own state(12) + 2 teammates(12*2) + 3 enemies(13*3) = 12 + 24 + 39 = 75
        expected_shape = (3, 75)
        self.assertEqual(self.wrapper.observation_space.shape, expected_shape)
        self.assertEqual(self.wrapper.observation_space.dtype, np.float32)
    
    def test_observation_space_local(self):
        """Test observation space for local observation type."""
        wrapper = MultiDroneTrainingWrapper(
            env=self.mock_env,
            obs_type=MultiDroneObservationType.LOCAL
        )
        # Local obs: own state(12) + 3 enemies(4*3) = 12 + 12 = 24
        expected_shape = (3, 24)
        self.assertEqual(wrapper.observation_space.shape, expected_shape)
    
    def test_observation_space_global(self):
        """Test observation space for global observation type."""
        wrapper = MultiDroneTrainingWrapper(
            env=self.mock_env,
            obs_type=MultiDroneObservationType.GLOBAL
        )
        # Global obs: 6 drones * 13 values = 78
        expected_shape = (3, 78)
        self.assertEqual(wrapper.observation_space.shape, expected_shape)
    
    def test_observation_space_communication(self):
        """Test observation space for communication observation type."""
        wrapper = MultiDroneTrainingWrapper(
            env=self.mock_env,
            obs_type=MultiDroneObservationType.COMMUNICATION
        )
        # Comm obs: own state(12) + 4 messages(8*4) + 3 enemies(4*3) = 12 + 32 + 12 = 56
        expected_shape = (3, 56)
        self.assertEqual(wrapper.observation_space.shape, expected_shape)
    
    def test_invalid_observation_type(self):
        """Test that invalid observation type raises error."""
        with self.assertRaises(ValueError):
            MultiDroneTrainingWrapper(
                env=self.mock_env,
                obs_type="invalid_type"
            )
    
    def test_reset(self):
        """Test environment reset."""
        # Mock the red policy's reset method
        with patch.object(self.wrapper._red_policy, 'reset') as mock_reset:
            obs, info = self.wrapper.reset()
            
            # Check that environment was reset
            self.mock_env.reset.assert_called_once()
            
            # Check that red policy was reset
            mock_reset.assert_called_once()
            
            # Check observation shape
            self.assertEqual(obs.shape, (3, 75))  # 3 blue drones, team obs
    
    def test_step_valid_action(self):
        """Test step with valid action."""
        action = np.random.randn(3, 4)
        
        # Mock the entire red policy object
        mock_red_policy = Mock()
        mock_red_policy.return_value = np.random.randn(6, 4)
        self.wrapper._red_policy = mock_red_policy
        
        obs, rewards, terminated, truncated, info = self.wrapper.step(action)
        
        # Check that red policy was called
        mock_red_policy.assert_called_once_with(self.mock_env)
        
        # Check that environment was stepped
        self.mock_env.step.assert_called_once()
        
        # Check observation shape
        self.assertEqual(obs.shape, (3, 75))
        
        # Check reward shape
        self.assertEqual(rewards.shape, (3,))
        
        # Check info contains team reward
        self.assertIn('blue_team_reward', info)
        self.assertIn('individual_rewards', info)
    
    def test_step_invalid_action_shape(self):
        """Test step with invalid action shape."""
        action = np.random.randn(2, 4)  # Wrong number of drones
        
        with self.assertRaises(ValueError):
            self.wrapper.step(action)
    
    def test_local_observations(self):
        """Test local observation extraction."""
        wrapper = MultiDroneTrainingWrapper(
            env=self.mock_env,
            obs_type=MultiDroneObservationType.LOCAL
        )
        
        obs = wrapper._get_local_observations()
        
        # Check shape
        self.assertEqual(obs.shape, (3, 24))
        
        # Check that each drone has its own state + enemy relative positions
        for i in range(3):
            # Own state: pos(3) + rpy(3) + vel(3) + ang_vel(3) = 12
            # 3 enemies: rel_pos(3) + health(1) = 4 each, total 12
            # Total: 12 + 12 = 24
            self.assertEqual(len(obs[i]), 24)
    
    def test_team_observations(self):
        """Test team observation extraction."""
        obs = self.wrapper._get_team_observations()
        
        # Check shape
        self.assertEqual(obs.shape, (3, 75))
        
        # Check that each drone has own state + teammates + enemies
        for i in range(3):
            # Own state: 12
            # 2 teammates: 12 each = 24
            # 3 enemies: 13 each = 39
            # Total: 12 + 24 + 39 = 75
            self.assertEqual(len(obs[i]), 75)
    
    def test_global_observations(self):
        """Test global observation extraction."""
        wrapper = MultiDroneTrainingWrapper(
            env=self.mock_env,
            obs_type=MultiDroneObservationType.GLOBAL
        )
        
        obs = wrapper._get_global_observations()
        
        # Check shape
        self.assertEqual(obs.shape, (3, 78))
        
        # Check that all drones have the same global observation
        for i in range(3):
            # 6 drones * 13 values each = 78
            self.assertEqual(len(obs[i]), 78)
    
    def test_communication_observations(self):
        """Test communication observation extraction."""
        wrapper = MultiDroneTrainingWrapper(
            env=self.mock_env,
            obs_type=MultiDroneObservationType.COMMUNICATION
        )
        
        obs = wrapper._get_communication_observations()
        
        # Check shape
        self.assertEqual(obs.shape, (3, 56))
        
        # Check that each drone has own state + messages + enemy info
        for i in range(3):
            # Own state: 12
            # 4 messages: 8 each = 32
            # 3 enemies: 4 each = 12
            # Total: 12 + 32 + 12 = 56
            self.assertEqual(len(obs[i]), 56)
    
    def test_reward_processing_no_sharing(self):
        """Test reward processing without sharing."""
        wrapper = MultiDroneTrainingWrapper(
            env=self.mock_env,
            shared_reward=False
        )
        
        individual_rewards = np.array([1.0, 2.0, 3.0])
        processed = wrapper._process_rewards(individual_rewards)
        
        # Should be unchanged
        np.testing.assert_array_equal(processed, individual_rewards)
    
    def test_reward_processing_with_sharing(self):
        """Test reward processing with sharing."""
        individual_rewards = np.array([1.0, 2.0, 3.0])
        processed = self.wrapper._process_rewards(individual_rewards)
        
        # Should be modified due to sharing
        self.assertFalse(np.allclose(processed, individual_rewards))
        
        # Should have same shape
        self.assertEqual(processed.shape, individual_rewards.shape)
    
    def test_reward_processing_dead_drones(self):
        """Test reward processing when some drones are dead."""
        # Set some drones as dead
        self.mock_env.alive_status = [True, False, True, True, True, True]
        
        individual_rewards = np.array([1.0, 2.0, 3.0])
        processed = self.wrapper._process_rewards(individual_rewards)
        
        # Dead drone should not participate in sharing
        self.assertEqual(processed.shape, individual_rewards.shape)
    
    def test_custom_red_policy(self):
        """Test wrapper with custom red policy."""
        custom_policy = TeamScriptedRedPolicy(
            drones_per_team=3,
            tactic=TeamTactic.DEFENSIVE
        )
        
        wrapper = MultiDroneTrainingWrapper(
            env=self.mock_env,
            red_policy=custom_policy
        )
        
        self.assertIs(wrapper._red_policy, custom_policy)
    
    def test_communication_radius(self):
        """Test communication radius parameter."""
        wrapper = MultiDroneTrainingWrapper(
            env=self.mock_env,
            obs_type=MultiDroneObservationType.COMMUNICATION,
            communication_radius=5.0
        )
        
        self.assertEqual(wrapper.communication_radius, 5.0)
    
    def test_reward_sharing_decay(self):
        """Test reward sharing decay parameter."""
        wrapper = MultiDroneTrainingWrapper(
            env=self.mock_env,
            reward_sharing_decay=0.5
        )
        
        self.assertEqual(wrapper.reward_sharing_decay, 0.5)
    
    def test_red_tactic_parameter(self):
        """Test red tactic parameter."""
        wrapper = MultiDroneTrainingWrapper(
            env=self.mock_env,
            red_tactic=TeamTactic.FLANKING
        )
        
        self.assertEqual(wrapper._red_policy.tactic, TeamTactic.FLANKING)
    
    def test_observation_extraction_dispatch(self):
        """Test that observation extraction dispatches correctly."""
        # Test each observation type
        for obs_type in [
            MultiDroneObservationType.LOCAL,
            MultiDroneObservationType.TEAM,
            MultiDroneObservationType.GLOBAL,
            MultiDroneObservationType.COMMUNICATION
        ]:
            wrapper = MultiDroneTrainingWrapper(
                env=self.mock_env,
                obs_type=obs_type
            )
            
            obs = wrapper._extract_observations()
            self.assertEqual(obs.shape[0], 3)  # 3 blue drones
    
    def test_observation_with_dead_drones(self):
        """Test observation extraction when some drones are dead."""
        # Set some drones as dead
        self.mock_env.alive_status = [True, False, True, True, False, True]
        self.mock_env.health = [100.0, 0.0, 50.0, 75.0, 0.0, 25.0]
        
        obs = self.wrapper._get_team_observations()
        
        # Should still have correct shape
        self.assertEqual(obs.shape, (3, 75))
        
        # Dead drones should have zero values in observation
        # This is handled by the observation extraction logic


if __name__ == '__main__':
    unittest.main() 