import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from uav_intent_rl.policies.team_scripted_red import (
    TeamScriptedRedPolicy,
    TeamRole,
    TeamTactic
)


class TestTeamRole(unittest.TestCase):
    """Test TeamRole enum."""
    
    def test_team_role_values(self):
        """Test that team roles have correct values."""
        self.assertEqual(TeamRole.LEADER.value, "leader")
        self.assertEqual(TeamRole.WINGMAN.value, "wingman")
        self.assertEqual(TeamRole.SUPPORT.value, "support")
    
    def test_team_role_names(self):
        """Test that team roles have correct names."""
        self.assertEqual(TeamRole.LEADER.name, "LEADER")
        self.assertEqual(TeamRole.WINGMAN.name, "WINGMAN")
        self.assertEqual(TeamRole.SUPPORT.name, "SUPPORT")


class TestTeamTactic(unittest.TestCase):
    """Test TeamTactic enum."""
    
    def test_team_tactic_values(self):
        """Test that team tactics have correct values."""
        self.assertEqual(TeamTactic.AGGRESSIVE.value, "aggressive")
        self.assertEqual(TeamTactic.DEFENSIVE.value, "defensive")
        self.assertEqual(TeamTactic.FLANKING.value, "flanking")
        self.assertEqual(TeamTactic.FORMATION.value, "formation")
    
    def test_team_tactic_names(self):
        """Test that team tactics have correct names."""
        self.assertEqual(TeamTactic.AGGRESSIVE.name, "AGGRESSIVE")
        self.assertEqual(TeamTactic.DEFENSIVE.name, "DEFENSIVE")
        self.assertEqual(TeamTactic.FLANKING.name, "FLANKING")
        self.assertEqual(TeamTactic.FORMATION.name, "FORMATION")


class TestTeamScriptedRedPolicy(unittest.TestCase):
    """Test TeamScriptedRedPolicy class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.policy = TeamScriptedRedPolicy(
            drones_per_team=3,
            target_alt=1.5,
            formation_spacing=2.0,
            tactic=TeamTactic.AGGRESSIVE
        )
        
        # Mock environment
        self.mock_env = Mock()
        self.mock_env.NUM_DRONES = 6  # 3v3
        self.mock_env.drones_per_team = 3
        self.mock_env.CTRL_FREQ = 30
        self.mock_env.alive_status = np.array([True, True, True, True, True, True])
    
    def test_initialization(self):
        """Test policy initialization."""
        self.assertEqual(self.policy.drones_per_team, 3)
        self.assertEqual(self.policy.target_alt, 1.5)
        self.assertEqual(self.policy.formation_spacing, 2.0)
        self.assertEqual(self.policy.tactic, TeamTactic.AGGRESSIVE)
        self.assertEqual(self.policy.engagement_range, 8.0)
        self.assertEqual(self.policy.close_range, 3.0)
        self.assertEqual(self.policy.evasion_range, 1.5)
        self.assertEqual(self.policy.tactic_timer, 0)
        self.assertEqual(len(self.policy.target_assignments), 0)
        self.assertEqual(len(self.policy.formation_positions), 0)
    
    def test_role_assignment(self):
        """Test role assignment for different team sizes."""
        # Test 1 drone
        policy_1 = TeamScriptedRedPolicy(drones_per_team=1)
        self.assertEqual(policy_1.roles[0], TeamRole.LEADER)
        self.assertEqual(len(policy_1.roles), 1)
        
        # Test 2 drones
        policy_2 = TeamScriptedRedPolicy(drones_per_team=2)
        self.assertEqual(policy_2.roles[0], TeamRole.LEADER)
        self.assertEqual(policy_2.roles[1], TeamRole.WINGMAN)
        self.assertEqual(len(policy_2.roles), 2)
        
        # Test 3 drones
        policy_3 = TeamScriptedRedPolicy(drones_per_team=3)
        self.assertEqual(policy_3.roles[0], TeamRole.LEADER)
        self.assertEqual(policy_3.roles[1], TeamRole.WINGMAN)
        self.assertEqual(policy_3.roles[2], TeamRole.SUPPORT)
        self.assertEqual(len(policy_3.roles), 3)
        
        # Test 5 drones
        policy_5 = TeamScriptedRedPolicy(drones_per_team=5)
        self.assertEqual(policy_5.roles[0], TeamRole.LEADER)
        self.assertEqual(policy_5.roles[1], TeamRole.WINGMAN)
        for i in range(2, 5):
            self.assertEqual(policy_5.roles[i], TeamRole.SUPPORT)
        self.assertEqual(len(policy_5.roles), 5)
    
    def test_reset(self):
        """Test policy reset functionality."""
        # Set some state
        self.policy.target_assignments = {0: 1, 1: 2}
        self.policy.formation_positions = {0: np.array([1, 2, 3])}
        self.policy.last_positions = np.array([1, 2, 3])
        self.policy.tactic_timer = 5.0
        
        # Reset
        self.policy.reset()
        
        # Check reset
        self.assertEqual(len(self.policy.target_assignments), 0)
        self.assertEqual(len(self.policy.formation_positions), 0)
        self.assertIsNone(self.policy.last_positions)
        self.assertEqual(self.policy.tactic_timer, 0)
    
    def test_call_method(self):
        """Test that __call__ returns actions."""
        with patch.object(self.policy, '_compute_team_actions') as mock_compute:
            mock_compute.return_value = np.zeros((6, 4))
            actions = self.policy(self.mock_env)
            mock_compute.assert_called_once_with(self.mock_env)
            self.assertEqual(actions.shape, (6, 4))
    
    def test_compute_team_actions_no_red_alive(self):
        """Test action computation when no red team is alive."""
        self.mock_env.alive_status = np.array([True, True, True, False, False, False])
        
        actions = self.policy._compute_team_actions(self.mock_env)
        
        self.assertEqual(actions.shape, (6, 4))
        self.assertTrue(np.allclose(actions, np.zeros((6, 4))))
    
    def test_update_target_assignments(self):
        """Test target assignment logic."""
        alive_red = [(3, np.array([0, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
                     (4, np.array([2, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))]
        alive_blue = [(0, np.array([1, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
                      (1, np.array([5, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))]
        
        self.policy._update_target_assignments(alive_red, alive_blue)
        
        # Red drone 3 should target blue drone 0 (closer)
        self.assertEqual(self.policy.target_assignments[3], 0)
        # Red drone 4 should target blue drone 0 (closer)
        self.assertEqual(self.policy.target_assignments[4], 0)
    
    def test_update_target_assignments_no_blue(self):
        """Test target assignment when no blue team is alive."""
        alive_red = [(3, np.array([0, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))]
        alive_blue = []
        
        self.policy._update_target_assignments(alive_red, alive_blue)
        
        self.assertEqual(len(self.policy.target_assignments), 0)
    
    def test_select_new_tactic(self):
        """Test tactic selection based on team numbers."""
        alive_red = [(3, np.array([0, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
                     (4, np.array([2, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))]
        alive_blue = [(0, np.array([1, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))]
        
        # Red has advantage (2 vs 1)
        self.policy._select_new_tactic(alive_red, alive_blue, self.mock_env)
        self.assertEqual(self.policy.tactic, TeamTactic.AGGRESSIVE)
        
        # Red is outnumbered (1 vs 2)
        alive_red_single = [(3, np.array([0, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))]
        alive_blue_double = [(0, np.array([1, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
                             (1, np.array([2, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))]
        
        self.policy._select_new_tactic(alive_red_single, alive_blue_double, self.mock_env)
        self.assertEqual(self.policy.tactic, TeamTactic.DEFENSIVE)
        
        # Equal numbers (2 vs 2)
        alive_red_equal = [(3, np.array([0, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
                           (4, np.array([2, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))]
        alive_blue_equal = [(0, np.array([1, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
                            (1, np.array([3, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))]
        
        self.policy._select_new_tactic(alive_red_equal, alive_blue_equal, self.mock_env)
        self.assertEqual(self.policy.tactic, TeamTactic.FLANKING)
    
    def test_aggressive_tactic(self):
        """Test aggressive tactic behavior."""
        actions = np.zeros((6, 4))
        alive_red = [(3, np.array([0, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))]
        alive_blue = [(0, np.array([2, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))]
        
        # Set target assignment
        self.policy.target_assignments[3] = 0
        
        self.policy._aggressive_tactic(actions, alive_red, alive_blue, self.mock_env)
        
        # Check that red drone 3 has non-zero action towards blue drone 0
        self.assertFalse(np.allclose(actions[3], 0))
        self.assertTrue(np.linalg.norm(actions[3, 0:3]) > 0)
        self.assertTrue(0 <= actions[3, 3] <= 1.0)
    
    def test_defensive_tactic(self):
        """Test defensive tactic behavior."""
        actions = np.zeros((6, 4))
        alive_red = [(3, np.array([0, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
                     (4, np.array([1, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))]
        alive_blue = [(0, np.array([5, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))]
        
        self.policy._defensive_tactic(actions, alive_red, alive_blue, self.mock_env)
        
        # Check that red drones have actions towards defensive position
        for red_id in [3, 4]:
            self.assertFalse(np.allclose(actions[red_id], 0))
            self.assertTrue(np.linalg.norm(actions[red_id, 0:3]) > 0)
    
    def test_flanking_tactic(self):
        """Test flanking tactic behavior."""
        actions = np.zeros((6, 4))
        alive_red = [(3, np.array([0, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
                     (4, np.array([1, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))]
        alive_blue = [(0, np.array([3, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))]
        
        self.policy._flanking_tactic(actions, alive_red, alive_blue, self.mock_env)
        
        # Check that red drones have flanking actions
        for red_id in [3, 4]:
            self.assertFalse(np.allclose(actions[red_id], 0))
            self.assertTrue(np.linalg.norm(actions[red_id, 0:3]) > 0)
    
    def test_formation_tactic(self):
        """Test formation tactic behavior."""
        actions = np.zeros((6, 4))
        alive_red = [(3, np.array([0, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
                     (4, np.array([1, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))]
        alive_blue = [(0, np.array([2, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))]
        
        # Set target assignment for leader
        self.policy.target_assignments[3] = 0
        
        self.policy._formation_tactic(actions, alive_red, alive_blue, self.mock_env)
        
        # Check that leader has pursuit action
        self.assertFalse(np.allclose(actions[3], 0))
        self.assertTrue(np.linalg.norm(actions[3, 0:3]) > 0)
        
        # Check that follower has formation action
        self.assertFalse(np.allclose(actions[4], 0))
    
    def test_formation_tactic_no_enemies(self):
        """Test formation tactic when no enemies are present."""
        actions = np.zeros((6, 4))
        alive_red = [(3, np.array([0, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
                     (4, np.array([1, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))]
        alive_blue = []
        
        self.policy._formation_tactic(actions, alive_red, alive_blue, self.mock_env)
        
        # Should still maintain formation even without enemies
        self.assertFalse(np.allclose(actions[3], 0))
        self.assertFalse(np.allclose(actions[4], 0))
    
    def test_action_clipping_and_noise(self):
        """Test that actions are properly clipped and have noise."""
        # Mock the tactic methods to return known actions
        with patch.object(self.policy, '_aggressive_tactic') as mock_aggressive:
            def set_actions(actions, *args):
                actions[3] = [2.0, 2.0, 2.0, 2.0]  # Will be clipped
                actions[4] = [-2.0, -2.0, -2.0, -2.0]  # Will be clipped
            mock_aggressive.side_effect = set_actions
            
            # Mock environment state
            self.mock_env._getDroneStateVector.side_effect = [
                np.array([0, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  # Blue 0
                np.array([1, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  # Blue 1
                np.array([2, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  # Blue 2
                np.array([3, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  # Red 3
                np.array([4, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  # Red 4
                np.array([5, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  # Red 5
            ]
            
            actions = self.policy._compute_team_actions(self.mock_env)
            
            # Check that actions are clipped to [-1, 1]
            self.assertTrue(np.all(actions >= -1.0))
            self.assertTrue(np.all(actions <= 1.0))
            
            # Check that noise was added (actions should not be exactly 0)
            self.assertFalse(np.allclose(actions, np.zeros((6, 4))))
    
    def test_tactical_state_update(self):
        """Test tactical state update functionality."""
        alive_red = [(3, np.array([0, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))]
        alive_blue = [(0, np.array([1, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))]
        
        # Set tactic timer to trigger tactic change
        self.policy.tactic_timer = 11.0
        
        with patch.object(self.policy, '_select_new_tactic') as mock_select:
            self.policy._update_tactical_state(alive_red, alive_blue, self.mock_env)
            
            # Check that tactic selection was called
            mock_select.assert_called_once_with(alive_red, alive_blue, self.mock_env)
            
            # Check that timer was reset
            self.assertEqual(self.policy.tactic_timer, 0)
    
    def test_different_tactics(self):
        """Test that different tactics produce different behaviors."""
        actions1 = np.zeros((6, 4))
        actions2 = np.zeros((6, 4))
        alive_red = [(3, np.array([0, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))]
        alive_blue = [(0, np.array([2, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))]
        
        # Test aggressive tactic
        self.policy.tactic = TeamTactic.AGGRESSIVE
        self.policy.target_assignments[3] = 0
        self.policy._aggressive_tactic(actions1, alive_red, alive_blue, self.mock_env)
        
        # Test defensive tactic
        self.policy.tactic = TeamTactic.DEFENSIVE
        self.policy._defensive_tactic(actions2, alive_red, alive_blue, self.mock_env)
        
        # Actions should be different for different tactics
        self.assertFalse(np.allclose(actions1, actions2))


if __name__ == '__main__':
    unittest.main() 