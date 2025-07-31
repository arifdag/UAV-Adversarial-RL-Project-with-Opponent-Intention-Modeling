#!/usr/bin/env python3
"""Unit tests for evaluate_mappo.py module."""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import tempfile
import json
from pathlib import Path
import matplotlib.pyplot as plt

from uav_intent_rl.examples.evaluate_mappo import (
    run_mappo_episode,
    analyze_team_behavior,
    evaluate_mappo
)
from uav_intent_rl.policies.team_scripted_red import TeamTactic


class TestEvaluateMappo(unittest.TestCase):
    """Test cases for evaluate_mappo.py module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock MAPPO model
        class MockMAPPO:
            def __init__(self):
                self.policy = None
            
            def predict(self, obs, state=None, deterministic=True):
                # Return mock action and state
                action = np.random.randn(3, 4)  # 3 drones, 4 action dims
                return action, None
            
            @classmethod
            def load(cls, path):
                return cls()
        
        self.mock_model = MockMAPPO()
        
        # Mock environment
        class MockEnv:
            def __init__(self, num_blue_drones=3):
                self.num_blue_drones = num_blue_drones
                self.env = MockBaseEnv()
            
            def reset(self):
                # Return tuple as expected by the fix
                obs = np.random.randn(self.num_blue_drones, 20)  # Mock obs
                info = {'winner': 'blue', 'blue_alive': 3, 'red_alive': 0, 'elimination_count': [3, 0]}
                return obs, info
            
            def step(self, action):
                # Return proper step tuple
                obs = np.random.randn(self.num_blue_drones, 20)
                rewards = np.random.randn(self.num_blue_drones)
                terminated = False
                truncated = False
                info = {'winner': 'blue', 'blue_alive': 3, 'red_alive': 0, 'elimination_count': [3, 0]}
                return obs, rewards, terminated, truncated, info
            
            def render(self):
                pass
        
        class MockBaseEnv:
            def __init__(self):
                self.alive_status = [True, True, True]
                self.health = [100.0, 100.0, 100.0]
                self.drones_per_team = 3
            
            def _getDroneStateVector(self, i):
                # Return mock drone state
                return np.random.randn(16)  # 16-dim state vector
        
        self.mock_env = MockEnv()
        
        # Mock results for analysis
        self.mock_results = [
            {
                'positions_history': [[np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])] for _ in range(10)],
                'health_history': [[100, 100, 100] for _ in range(10)],
                'winner': 'blue',
                'tactic': 'test_tactic',
                'episode_reward': 10.5
            },
            {
                'positions_history': [[np.array([2, 0, 0]), np.array([0, 2, 0]), np.array([0, 0, 2])] for _ in range(10)],
                'health_history': [[80, 90, 95] for _ in range(10)],
                'winner': 'red',
                'tactic': 'test_tactic',
                'episode_reward': -5.2
            }
        ]

    def test_run_mappo_episode_basic(self):
        """Test basic functionality of run_mappo_episode."""
        results = run_mappo_episode(
            model=self.mock_model,
            env=self.mock_env,
            render=False,
            verbose=False,
            tactic="test_tactic"
        )
        
        # Check that results have expected structure
        self.assertIn('episode_reward', results)
        self.assertIn('episode_length', results)
        self.assertIn('individual_rewards', results)
        self.assertIn('winner', results)
        self.assertIn('tactic', results)
        self.assertEqual(results['tactic'], "test_tactic")
        
        # Check data types
        self.assertIsInstance(results['episode_reward'], float)
        self.assertIsInstance(results['episode_length'], int)
        self.assertIsInstance(results['individual_rewards'], list)
        self.assertIsInstance(results['winner'], str)

    def test_run_mappo_episode_with_render(self):
        """Test run_mappo_episode with rendering enabled."""
        results = run_mappo_episode(
            model=self.mock_model,
            env=self.mock_env,
            render=True,
            verbose=True,
            tactic="test_tactic"
        )
        
        # Should still work with render=True
        self.assertIn('episode_reward', results)
        self.assertIn('tactic', results)

    def test_run_mappo_episode_action_copying(self):
        """Test that action copying works with both numpy arrays and torch tensors."""
        # Test with numpy array
        mock_action_np = np.random.randn(3, 4)
        
        with patch.object(self.mock_model, 'predict', return_value=(mock_action_np, None)):
            results = run_mappo_episode(
                model=self.mock_model,
                env=self.mock_env,
                render=False,
                verbose=False,
                tactic="test_tactic"
            )
            
            self.assertIn('actions_history', results)
            self.assertIsInstance(results['actions_history'], list)
            self.assertGreater(len(results['actions_history']), 0)

    def test_analyze_team_behavior_basic(self):
        """Test basic functionality of analyze_team_behavior."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            try:
                metrics = analyze_team_behavior(self.mock_results, save_path=Path(tmp_file.name))
                
                # Check that metrics have expected structure
                self.assertIn('avg_formation_distance', metrics)
                self.assertIn('avg_survival_rate', metrics)
                self.assertIn('formation_consistency', metrics)
                
                # Check that values are floats (not numpy types)
                self.assertIsInstance(metrics['avg_formation_distance'], float)
                self.assertIsInstance(metrics['avg_survival_rate'], float)
                self.assertIsInstance(metrics['formation_consistency'], float)
                
                # Check that file was created
                self.assertTrue(Path(tmp_file.name).exists())
                
            finally:
                # Clean up temp file
                Path(tmp_file.name).unlink(missing_ok=True)

    def test_analyze_team_behavior_no_save(self):
        """Test analyze_team_behavior without saving plot."""
        metrics = analyze_team_behavior(self.mock_results, save_path=None)
        
        # Should still return metrics
        self.assertIn('avg_formation_distance', metrics)
        self.assertIn('avg_survival_rate', metrics)
        self.assertIn('formation_consistency', metrics)

    def test_analyze_team_behavior_empty_results(self):
        """Test analyze_team_behavior with empty results."""
        empty_results = []
        metrics = analyze_team_behavior(empty_results, save_path=None)
        
        # Should handle empty results gracefully
        self.assertIn('avg_formation_distance', metrics)
        self.assertIn('avg_survival_rate', metrics)
        self.assertIn('formation_consistency', metrics)
        
        # Should have default values
        self.assertEqual(metrics['avg_formation_distance'], 0.0)
        self.assertEqual(metrics['avg_survival_rate'], 0.0)
        self.assertEqual(metrics['formation_consistency'], 0.0)

    def test_analyze_team_behavior_mixed_results(self):
        """Test analyze_team_behavior with mixed win/loss results."""
        mixed_results = [
            {
                'positions_history': [[np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])] for _ in range(5)],
                'health_history': [[100, 100, 100] for _ in range(5)],
                'winner': 'blue',
                'tactic': 'aggressive',
                'episode_reward': 8.3
            },
            {
                'positions_history': [[np.array([2, 0, 0]), np.array([0, 2, 0]), np.array([0, 0, 2])] for _ in range(5)],
                'health_history': [[0, 0, 0] for _ in range(5)],  # All dead
                'winner': 'red',
                'tactic': 'defensive',
                'episode_reward': -12.7
            }
        ]
        
        metrics = analyze_team_behavior(mixed_results, save_path=None)
        
        # Should handle mixed results
        self.assertIn('avg_formation_distance', metrics)
        self.assertIn('avg_survival_rate', metrics)
        self.assertIn('formation_consistency', metrics)

    @patch('uav_intent_rl.examples.evaluate_mappo.MAPPO')
    @patch('uav_intent_rl.examples.evaluate_mappo.MultiDroneDogfightAviary')
    @patch('uav_intent_rl.examples.evaluate_mappo.MAPPODogfightWrapper')
    def test_evaluate_mappo_basic(self, mock_wrapper_class, mock_aviary_class, mock_mappo_class):
        """Test basic functionality of evaluate_mappo."""
        # Setup mocks
        mock_model = Mock()
        mock_mappo_class.load.return_value = mock_model
        
        mock_env = Mock()
        mock_env.num_blue_drones = 3
        mock_env.reset.return_value = (np.random.randn(3, 20), {})
        mock_env.step.return_value = (
            np.random.randn(3, 20),
            np.random.randn(3),
            False,
            False,
            {'winner': 'blue', 'blue_alive': 3, 'red_alive': 0, 'elimination_count': [3, 0]}
        )
        mock_env.close = Mock()
        mock_env.render = Mock()
        mock_wrapper_class.return_value = mock_env
        
        mock_aviary_class.return_value = Mock()
        
        # Create temporary model path
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_file:
            model_path = Path(tmp_file.name)
            
            try:
                # Test evaluation
                results = evaluate_mappo(
                    model_path=model_path,
                    n_episodes=2,  # Small number for testing
                    drones_per_team=3,
                    red_tactics=[TeamTactic.AGGRESSIVE],
                    gui=False,
                    verbose=False,
                    save_results=False,
                    analyze=False
                )
                
                # Check basic structure
                self.assertIn('model_path', results)
                self.assertIn('n_episodes', results)
                self.assertIn('drones_per_team', results)
                self.assertIn('tactics_results', results)
                self.assertIn('overall_win_rate', results)
                
                # Check that environment was closed
                mock_env.close.assert_called()
                
            finally:
                # Clean up temp file
                model_path.unlink(missing_ok=True)

    @patch('uav_intent_rl.examples.evaluate_mappo.MAPPO')
    @patch('uav_intent_rl.examples.evaluate_mappo.MultiDroneDogfightAviary')
    @patch('uav_intent_rl.examples.evaluate_mappo.MAPPODogfightWrapper')
    def test_evaluate_mappo_with_save(self, mock_wrapper_class, mock_aviary_class, mock_mappo_class):
        """Test evaluate_mappo with save_results=True."""
        # Setup mocks
        mock_model = Mock()
        mock_mappo_class.load.return_value = mock_model
        
        mock_env = Mock()
        mock_env.num_blue_drones = 3
        mock_env.reset.return_value = (np.random.randn(3, 20), {})
        mock_env.step.return_value = (
            np.random.randn(3, 20),
            np.random.randn(3),
            False,
            False,
            {'winner': 'blue', 'blue_alive': 3, 'red_alive': 0, 'elimination_count': [3, 0]}
        )
        mock_env.close = Mock()
        mock_env.render = Mock()
        mock_wrapper_class.return_value = mock_env
        
        mock_aviary_class.return_value = Mock()
        
        # Create temporary model path
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_file:
            model_path = Path(tmp_file.name)
            
            try:
                # Test evaluation with save
                results = evaluate_mappo(
                    model_path=model_path,
                    n_episodes=2,
                    drones_per_team=3,
                    red_tactics=[TeamTactic.AGGRESSIVE],
                    gui=False,
                    verbose=False,
                    save_results=True,
                    analyze=False
                )
                
                # Check that results file was created
                results_file = model_path.parent / f"eval_results_{model_path.stem}.json"
                self.assertTrue(results_file.exists())
                
                # Check JSON content
                with open(results_file, 'r') as f:
                    saved_data = json.load(f)
                
                self.assertIn('model_path', saved_data)
                self.assertIn('n_episodes', saved_data)
                self.assertIn('drones_per_team', saved_data)
                self.assertIn('overall_win_rate', saved_data)
                self.assertIn('tactics_summary', saved_data)
                
                # Clean up results file
                results_file.unlink(missing_ok=True)
                
            finally:
                # Clean up temp file
                model_path.unlink(missing_ok=True)

    def test_json_serialization_numpy_types(self):
        """Test that numpy types are properly converted to native Python types."""
        # Create data with numpy types
        data = {
            'avg_reward': np.float32(1.5),
            'avg_length': np.int32(100),
            'formation_distance': np.float64(2.5)
        }
        
        # This should work without TypeError
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        
        # Check that types are native Python types
        self.assertIsInstance(parsed['avg_reward'], float)
        self.assertIsInstance(parsed['avg_length'], int)
        self.assertIsInstance(parsed['formation_distance'], float)

    def test_done_logic_with_arrays(self):
        """Test that done logic handles both scalar and array terminated/truncated."""
        # Test with scalar values
        terminated_scalar = False
        truncated_scalar = False
        done_scalar = bool(np.asarray(terminated_scalar).any() or np.asarray(truncated_scalar).any())
        self.assertFalse(done_scalar)
        
        # Test with array values
        terminated_array = np.array([False, False, False])
        truncated_array = np.array([False, False, False])
        done_array = bool(np.asarray(terminated_array).any() or np.asarray(truncated_array).any())
        self.assertFalse(done_array)
        
        # Test with mixed values
        terminated_mixed = np.array([False, True, False])
        truncated_mixed = np.array([False, False, False])
        done_mixed = bool(np.asarray(terminated_mixed).any() or np.asarray(truncated_mixed).any())
        self.assertTrue(done_mixed)

    def test_action_copying_safety(self):
        """Test that action copying works safely with different input types."""
        # Test with numpy array
        action_np = np.random.randn(3, 4)
        copied_np = np.asarray(action_np).copy()
        self.assertIsInstance(copied_np, np.ndarray)
        self.assertEqual(copied_np.shape, action_np.shape)
        
        # Test with list
        action_list = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        copied_list = np.asarray(action_list).copy()
        self.assertIsInstance(copied_list, np.ndarray)
        
        # Test with torch tensor (if available)
        try:
            import torch
            action_torch = torch.randn(3, 4)
            copied_torch = np.asarray(action_torch).copy()
            self.assertIsInstance(copied_torch, np.ndarray)
        except ImportError:
            # Skip torch test if not available
            pass

    def test_float_conversion_safety(self):
        """Test that float conversion works safely with numpy types."""
        # Test various numpy types
        test_values = [
            np.float32(1.5),
            np.float64(2.5),
            np.int32(100),
            np.int64(200),
            np.mean([1, 2, 3]),  # numpy scalar
            np.std([1, 2, 3]),   # numpy scalar
        ]
        
        for value in test_values:
            converted = float(value)
            self.assertIsInstance(converted, float)
            self.assertNotIsInstance(converted, np.floating)
            self.assertNotIsInstance(converted, np.integer)

    def test_tactic_assignment(self):
        """Test that tactic is properly assigned in results."""
        results = run_mappo_episode(
            model=self.mock_model,
            env=self.mock_env,
            render=False,
            verbose=False,
            tactic="test_tactic"
        )
        
        self.assertIn('tactic', results)
        self.assertEqual(results['tactic'], "test_tactic")

    def test_removed_coordination_scores(self):
        """Test that coordination_scores variable was removed from analyze_team_behavior."""
        # This test ensures the unused variable was properly removed
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            try:
                metrics = analyze_team_behavior(self.mock_results, save_path=Path(tmp_file.name))
                
                # Should not have coordination_scores in the returned metrics
                self.assertNotIn('coordination_scores', metrics)
                
                # Should have the expected metrics
                self.assertIn('avg_formation_distance', metrics)
                self.assertIn('avg_survival_rate', metrics)
                self.assertIn('formation_consistency', metrics)
                
            finally:
                Path(tmp_file.name).unlink(missing_ok=True)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2) 