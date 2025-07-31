"""Unit tests for MultiDroneDogfightAviary environment."""

import unittest
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from uav_intent_rl.envs.MultiDroneDogfightAviary import (
    MultiDroneDogfightAviary, 
    TeamColor
)
from gym_pybullet_drones.utils.enums import (
    ActionType,
    DroneModel,
    ObservationType,
    Physics,
)


class TestMultiDroneDogfightAviary(unittest.TestCase):
    """Test cases for MultiDroneDogfightAviary environment."""
    
    def setUp(self):
        """Set up test environment."""
        # Use minimal settings for faster tests
        self.env = MultiDroneDogfightAviary(
            drones_per_team=2,  # 2v2 for testing
            gui=False,
            physics=Physics.PYB,
            pyb_freq=240,
            ctrl_freq=30,
            obs=ObservationType.KIN,
            act=ActionType.VEL,
            enable_respawn=False,
        )
    
    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, 'env'):
            self.env.close()
    
    def test_initialization(self):
        """Test environment initialization."""
        # Check basic attributes
        self.assertEqual(self.env.drones_per_team, 2)
        self.assertEqual(self.env.NUM_DRONES, 4)  # 2 teams * 2 drones
        self.assertFalse(self.env.enable_respawn)
        
        # Check team assignments
        expected_teams = [TeamColor.BLUE, TeamColor.BLUE, TeamColor.RED, TeamColor.RED]
        np.testing.assert_array_equal(self.env.team_assignments, expected_teams)
        
        # Check health tracking arrays
        self.assertEqual(len(self.env.health), 4)
        self.assertEqual(len(self.env.alive_status), 4)
        self.assertEqual(len(self.env.respawn_timers), 4)
        
        # Check combat tracking arrays
        self.assertEqual(len(self.env.last_attacker), 4)
        self.assertEqual(len(self.env.hit_timestamps), 4)
        self.assertEqual(len(self.env.elimination_count), 2)
    
    def test_reset_health_and_status(self):
        """Test that reset properly initializes health and status."""
        # Damage some drones
        self.env.health[0] = 0.5
        self.env.health[2] = 0.0
        self.env.alive_status[2] = False
        
        # Reset environment
        obs, info = self.env.reset()
        
        # Check health is reset
        np.testing.assert_array_equal(self.env.health, np.ones(4))
        np.testing.assert_array_equal(self.env.alive_status, np.ones(4, dtype=bool))
        np.testing.assert_array_equal(self.env.respawn_timers, np.zeros(4))
        np.testing.assert_array_equal(self.env.last_attacker, np.full(4, -1))
        np.testing.assert_array_equal(self.env.hit_timestamps, np.zeros(4))
        np.testing.assert_array_equal(self.env.elimination_count, np.zeros(2))
    
    def test_formation_position_generation(self):
        """Test formation position generation for different team sizes."""
        rng = np.random.default_rng(42)
        
        # Test single drone
        center = np.array([0, 0, 1.5])
        positions = self.env._generate_formation_positions(center, 1, 2.0, rng)
        self.assertEqual(positions.shape, (1, 3))
        
        # Test two drones (line formation)
        positions = self.env._generate_formation_positions(center, 2, 2.0, rng)
        self.assertEqual(positions.shape, (2, 3))
        # Check they're separated horizontally
        self.assertGreater(np.linalg.norm(positions[0] - positions[1]), 1.0)
        
        # Test three drones (triangle formation)
        positions = self.env._generate_formation_positions(center, 3, 2.0, rng)
        self.assertEqual(positions.shape, (3, 3))
        # Check they form a triangle (not all in a line)
        pairwise_dists = []
        for i in range(3):
            for j in range(i+1, 3):
                pairwise_dists.append(np.linalg.norm(positions[i] - positions[j]))
        self.assertGreater(np.std(pairwise_dists), 0.1)  # Not all equal
    
    def test_hit_detection(self):
        """Test hit detection logic."""
        # Mock drone states for testing
        with patch.object(self.env, '_getDroneStateVector') as mock_get_state:
            # Set up two drones facing each other within damage radius
            mock_get_state.side_effect = [
                # Drone 0 (Blue team) - at origin, facing right
                np.array([0, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                # Drone 1 (Blue team) - far away
                np.array([10, 10, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                # Drone 2 (Red team) - 1.0m to the right, facing left
                np.array([1.0, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, np.pi, 0, 0, 0, 0, 0]),
                # Drone 3 (Red team) - far away
                np.array([-10, -10, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            ]
            
            # Both drones should be alive
            self.env.alive_status[0] = True
            self.env.alive_status[2] = True
            
            hits = self.env._detect_hits()
            
            # Should detect a hit from drone 0 to drone 2
            self.assertIn((0, 2), hits)
    
    def test_hit_detection_same_team(self):
        """Test that hit detection ignores same-team drones."""
        with patch.object(self.env, '_getDroneStateVector') as mock_get_state:
            # Set up two blue team drones close to each other
            mock_get_state.side_effect = [
                # Drone 0 (Blue team)
                np.array([0, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                # Drone 1 (Blue team) - close to drone 0
                np.array([1.0, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                # Drone 2 (Red team) - far away
                np.array([10, 10, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                # Drone 3 (Red team) - far away
                np.array([-10, -10, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            ]
            
            self.env.alive_status[0] = True
            self.env.alive_status[1] = True
            
            hits = self.env._detect_hits()
            
            # Should not detect hits between same team
            self.assertNotIn((0, 1), hits)
            self.assertNotIn((1, 0), hits)
    
    def test_hit_detection_fov(self):
        """Test that hit detection respects field of view."""
        with patch.object(self.env, '_getDroneStateVector') as mock_get_state:
            # Set up drone facing right, target behind it
            mock_get_state.side_effect = [
                # Drone 0 (Blue team) - facing right
                np.array([0, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                # Drone 1 (Blue team) - far away
                np.array([10, 10, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                # Drone 2 (Red team) - behind drone 0
                np.array([-1.0, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, np.pi, 0, 0, 0, 0, 0]),
                # Drone 3 (Red team) - far away
                np.array([-10, -10, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            ]
            
            self.env.alive_status[0] = True
            self.env.alive_status[2] = True
            
            hits = self.env._detect_hits()
            
            # Should not detect hit because target is behind drone
            self.assertNotIn((0, 2), hits)
    
    def test_damage_and_elimination(self):
        """Test damage application and elimination logic."""
        # Set up initial health
        self.env.health = np.array([1.0, 1.0, 0.5, 1.0])  # Drone 2 at 50% health
        self.env.alive_status = np.array([True, True, True, True])
        
        # Simulate a hit that should eliminate drone 2
        attacker_id, target_id = 0, 2
        rewards = np.zeros(4)
        
        self.env._handle_elimination(attacker_id, target_id, rewards)
        
        # Check elimination
        self.assertFalse(self.env.alive_status[target_id])
        self.assertEqual(self.env.health[target_id], 0.5)  # Health unchanged
        
        # Check rewards
        self.assertEqual(rewards[attacker_id], self.env.ELIMINATION_REWARD)
        self.assertEqual(rewards[target_id], self.env.DEATH_PENALTY)
        
        # Check team elimination count
        self.assertEqual(self.env.elimination_count[0], 1)  # Blue team got elimination
        self.assertEqual(self.env.elimination_count[1], 0)  # Red team got eliminated
    
    def test_formation_evaluation(self):
        """Test formation evaluation logic."""
        # Mock drone positions for blue team
        with patch.object(self.env, 'pos') as mock_pos:
            # Good formation - drones close together
            mock_pos.__getitem__.side_effect = lambda i: {
                0: np.array([0, 0, 1.5]),
                1: np.array([1, 0, 1.5]),
                2: np.array([0, 1, 1.5]),
                3: np.array([1, 1, 1.5])
            }[i]
            
            # All blue team drones alive
            self.env.alive_status = np.array([True, True, False, False])
            
            blue_indices = [0, 1]
            score = self.env._evaluate_formation(blue_indices)
            
            # Should get good formation score
            self.assertGreater(score, 0.5)
    
    def test_formation_evaluation_single_drone(self):
        """Test formation evaluation with single drone."""
        self.env.alive_status = np.array([True, False, False, False])
        
        blue_indices = [0]
        score = self.env._evaluate_formation(blue_indices)
        
        # Single drone should get zero formation score
        self.assertEqual(score, 0.0)
    
    def test_targeting_rewards(self):
        """Test targeting reward computation."""
        with patch.object(self.env, '_getDroneStateVector') as mock_get_state:
            # Set up drone facing target
            mock_get_state.side_effect = [
                # Drone 0 (Blue team) - facing right
                np.array([0, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                # Drone 1 (Blue team) - far away
                np.array([10, 10, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                # Drone 2 (Red team) - in front of drone 0
                np.array([2, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, np.pi, 0, 0, 0, 0, 0]),
                # Drone 3 (Red team) - far away
                np.array([-10, -10, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            ]
            
            self.env.alive_status = np.array([True, True, True, True])
            self.env.health = np.array([1.0, 1.0, 0.5, 1.0])  # Drone 2 damaged
            
            rewards = self.env._compute_targeting_rewards()
            
            # Drone 0 should get targeting reward for aiming at damaged drone 2
            self.assertGreater(rewards[0], 0)
    
    def test_episode_termination(self):
        """Test episode termination conditions."""
        # Test normal state (no termination)
        self.env.alive_status = np.array([True, True, True, True])
        self.assertFalse(self.env._computeTerminated())
        
        # Test blue team eliminated
        self.env.alive_status = np.array([False, False, True, True])
        self.assertTrue(self.env._computeTerminated())
        
        # Test red team eliminated
        self.env.alive_status = np.array([True, True, False, False])
        self.assertTrue(self.env._computeTerminated())
        
        # Test both teams eliminated (draw)
        self.env.alive_status = np.array([False, False, False, False])
        self.assertTrue(self.env._computeTerminated())
    
    def test_episode_truncation(self):
        """Test episode truncation on time limit."""
        # Test before time limit
        self.env.step_counter = 0
        self.assertFalse(self.env._computeTruncated())
        
        # Test after time limit
        self.env.step_counter = self.env.EPISODE_LEN_SEC * self.env.PYB_FREQ + 1
        self.assertTrue(self.env._computeTruncated())
    
    def test_info_computation(self):
        """Test info dictionary computation."""
        # Set up some test state
        self.env.alive_status = np.array([True, False, True, False])
        self.env.health = np.array([0.8, 0.0, 0.6, 0.0])
        self.env.elimination_count = np.array([1, 1])
        self.env.step_counter = 100
        
        info = self.env._computeInfo()
        
        # Check info contents
        self.assertEqual(info['blue_alive'], 1)
        self.assertEqual(info['red_alive'], 1)
        self.assertEqual(info['blue_health'], [0.8, 0.0])
        self.assertEqual(info['red_health'], [0.6, 0.0])
        self.assertEqual(info['elimination_count'], [1, 1])
        self.assertAlmostEqual(info['episode_time'], 100 / self.env.PYB_FREQ)
    
    def test_respawn_functionality(self):
        """Test respawn functionality when enabled."""
        # Create environment with respawn enabled
        respawn_env = MultiDroneDogfightAviary(
            drones_per_team=2,
            enable_respawn=True,
            gui=False
        )
        
        # Eliminate a drone
        respawn_env.alive_status[0] = False
        respawn_env.respawn_timers[0] = respawn_env.RESPAWN_TIME_SEC
        
        # Step environment to trigger respawn countdown
        action = np.zeros((4, 4))  # Zero action for all drones
        obs, rewards, terminated, truncated, info = respawn_env.step(action)
        
        # Timer should be decremented
        self.assertLess(respawn_env.respawn_timers[0], respawn_env.RESPAWN_TIME_SEC)
        
        respawn_env.close()
    
    def test_reward_computation(self):
        """Test overall reward computation."""
        # Mock hit detection to return a hit
        with patch.object(self.env, '_detect_hits') as mock_hits:
            mock_hits.return_value = [(0, 2)]  # Drone 0 hits drone 2
            
            # Set up drone states
            with patch.object(self.env, '_getDroneStateVector') as mock_get_state:
                # Provide mock states for all 4 drones
                mock_get_state.side_effect = [
                    np.array([0, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  # Drone 0
                    np.array([10, 10, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  # Drone 1
                    np.array([1, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, np.pi, 0, 0, 0, 0, 0]),  # Drone 2
                    np.array([-10, -10, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # Drone 3
                ]
                rewards = self.env._computeReward()
                
                # Should have rewards for all drones
                self.assertEqual(len(rewards), 4)
                
                # Attacker should get positive reward
                self.assertGreater(rewards[0], 0)
                
                # Target should get negative reward
                self.assertLess(rewards[2], 0)
    
    def test_team_victory_bonus(self):
        """Test team victory bonus distribution."""
        # Set up red team eliminated
        self.env.alive_status = np.array([True, True, False, False])
        
        rewards = self.env._computeReward()
        
        # Blue team should get victory bonus
        self.assertGreater(rewards[0], 0)
        self.assertGreater(rewards[1], 0)
        
        # Red team should get penalty
        self.assertLess(rewards[2], 0)
        self.assertLess(rewards[3], 0)
    
    def test_invalid_initialization(self):
        """Test that invalid parameters raise appropriate errors."""
        # Test with zero drones per team
        with self.assertRaises(ValueError):
            MultiDroneDogfightAviary(drones_per_team=0)
        
        # Test with negative drones per team
        with self.assertRaises(ValueError):
            MultiDroneDogfightAviary(drones_per_team=-1)


class TestTeamColor(unittest.TestCase):
    """Test cases for TeamColor enum."""
    
    def test_team_color_values(self):
        """Test TeamColor enum values."""
        self.assertEqual(TeamColor.BLUE.value, 0)
        self.assertEqual(TeamColor.RED.value, 1)
    
    def test_team_color_names(self):
        """Test TeamColor enum names."""
        self.assertEqual(TeamColor.BLUE.name, 'BLUE')
        self.assertEqual(TeamColor.RED.name, 'RED')


if __name__ == '__main__':
    unittest.main() 