import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import tempfile
import shutil
from pathlib import Path
import json

from uav_intent_rl.examples.evaluate_multidrone import (
    run_episode,
    evaluate_multidrone,
    plot_results
)
from uav_intent_rl.policies.team_scripted_red import TeamTactic
from uav_intent_rl.envs.MultiDroneTrainingWrapper import MultiDroneObservationType


class TestRunEpisode(unittest.TestCase):
    """Test the run_episode function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_model = Mock()
        self.mock_env = Mock()
        
        # Mock environment properties
        self.mock_env.num_blue_drones = 3
        
        # Mock environment step and reset
        self.mock_env.reset.return_value = np.random.randn(3, 75)  # 3 drones, 75 obs dim
        
        # Create a side effect that always terminates episodes properly
        # Each episode needs at least 2 steps: one "not done" step, then one "done" step
        # Since run_episode might be called multiple times, we need to provide many responses
        
        # Define the step responses
        step_not_done = (
            np.random.randn(3, 75),  # obs
            np.array([1.0, 2.0, 3.0]),  # rewards
            False,  # terminated
            False,  # truncated
            {
                'winner': 'blue',
                'blue_alive': 3,
                'red_alive': 0,
                'elimination_count': [3, 0]
            }  # info
        )
        
        step_done = (
            np.random.randn(3, 75),  # obs
            np.array([4.0, 5.0, 6.0]),  # rewards
            True,  # terminated
            False,  # truncated
            {
                'winner': 'blue',
                'blue_alive': 3,
                'red_alive': 0,
                'elimination_count': [3, 0]
            }  # info
        )
        
        # Create a sequence that can handle multiple episodes
        # Each episode gets exactly 2 steps and then terminates
        self.mock_env.step.side_effect = [step_not_done, step_done] * 10  # Enough for 10 episodes
        
        # Mock model prediction
        self.mock_model.predict.return_value = (np.random.randn(3, 4), None)
    
    def test_run_episode_basic(self):
        """Test basic episode execution."""
        results = run_episode(
            self.mock_model,
            self.mock_env,
            render=False,
            verbose=False
        )
        
        # Check that environment was reset
        self.mock_env.reset.assert_called_once()
        
        # Check that model was called
        self.mock_model.predict.assert_called()
        
        # Check that environment was stepped
        self.mock_env.step.assert_called()
        
        # Check results structure
        self.assertIn('episode_reward', results)
        self.assertIn('episode_length', results)
        self.assertIn('drone_rewards', results)
        self.assertIn('winner', results)
        self.assertIn('blue_alive', results)
        self.assertIn('red_alive', results)
        self.assertIn('blue_eliminations', results)
        self.assertIn('red_eliminations', results)
        
        # Check data types
        self.assertIsInstance(results['episode_reward'], float)
        self.assertIsInstance(results['episode_length'], int)
        self.assertIsInstance(results['drone_rewards'], list)
        self.assertIsInstance(results['winner'], str)
        self.assertIsInstance(results['blue_alive'], int)
        self.assertIsInstance(results['red_alive'], int)
        self.assertIsInstance(results['blue_eliminations'], int)
        self.assertIsInstance(results['red_eliminations'], int)
    
    def test_run_episode_with_termination(self):
        """Test episode with early termination."""
        # Mock environment to terminate after first step
        # Override the side_effect with a simple return_value for this test
        self.mock_env.step.side_effect = None  # Clear the side_effect
        self.mock_env.step.return_value = (
            np.random.randn(3, 75),
            np.array([1.0, 2.0, 3.0]),
            True,  # terminated
            False,
            {'winner': 'red', 'blue_alive': 1, 'red_alive': 2, 'elimination_count': [2, 1]}
        )
        
        results = run_episode(
            self.mock_model,
            self.mock_env,
            render=False,
            verbose=False
        )
        
        # Check that episode terminated
        self.assertEqual(results['winner'], 'red')
        self.assertEqual(results['blue_alive'], 1)
        self.assertEqual(results['red_alive'], 2)
        self.assertEqual(results['blue_eliminations'], 2)
        self.assertEqual(results['red_eliminations'], 1)
    
    def test_run_episode_with_truncation(self):
        """Test episode with truncation."""
        # Mock environment to truncate after first step
        # Override the side_effect with a simple return_value for this test
        self.mock_env.step.side_effect = None  # Clear the side_effect
        self.mock_env.step.return_value = (
            np.random.randn(3, 75),
            np.array([1.0, 2.0, 3.0]),
            False,
            True,  # truncated
            {'winner': 'unknown', 'blue_alive': 2, 'red_alive': 1, 'elimination_count': [1, 2]}
        )
        
        results = run_episode(
            self.mock_model,
            self.mock_env,
            render=False,
            verbose=False
        )
        
        # Check that episode was truncated
        self.assertEqual(results['winner'], 'unknown')
        self.assertEqual(results['blue_alive'], 2)
        self.assertEqual(results['red_alive'], 1)
        self.assertEqual(results['blue_eliminations'], 1)
        self.assertEqual(results['red_eliminations'], 2)
    
    def test_run_episode_reward_accumulation(self):
        """Test that rewards are properly accumulated."""
        # Mock multiple steps with different rewards
        step_results = [
            (np.random.randn(3, 75), np.array([1.0, 2.0, 3.0]), False, False, {}),
            (np.random.randn(3, 75), np.array([4.0, 5.0, 6.0]), False, False, {}),
            (np.random.randn(3, 75), np.array([7.0, 8.0, 9.0]), True, False, {'winner': 'blue'})
        ]
        
        # Override the side_effect for this specific test
        self.mock_env.step.side_effect = step_results
        
        results = run_episode(
            self.mock_model,
            self.mock_env,
            render=False,
            verbose=False
        )
        
        # Check total reward (1+2+3 + 4+5+6 + 7+8+9 = 45)
        expected_total = sum(sum(step[1]) for step in step_results)
        self.assertEqual(results['episode_reward'], expected_total)
        
        # Check episode length
        self.assertEqual(results['episode_length'], 3)
        
        # Check individual drone rewards
        expected_drone_rewards = np.array([1+4+7, 2+5+8, 3+6+9])
        np.testing.assert_array_equal(results['drone_rewards'], expected_drone_rewards.tolist())


class TestEvaluateMultidrone(unittest.TestCase):
    """Test the evaluate_multidrone function."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = Path(self.temp_dir) / "test_model.zip"
        
        # Create a dummy model file
        with open(self.model_path, 'w') as f:
            f.write("dummy model content")
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    @patch('uav_intent_rl.examples.evaluate_multidrone.PPO.load')
    @patch('uav_intent_rl.examples.evaluate_multidrone.MultiDroneDogfightAviary')
    @patch('uav_intent_rl.examples.evaluate_multidrone.MultiDroneTrainingWrapper')
    @patch('uav_intent_rl.examples.evaluate_multidrone.run_episode')
    def test_evaluate_multidrone_basic(self, mock_run_episode, mock_wrapper, mock_aviary, mock_load):
        """Test basic evaluation setup."""
        # Mock model loading
        mock_model = Mock()
        mock_load.return_value = mock_model
        
        # Mock environment creation
        mock_base_env = Mock()
        mock_aviary.return_value = mock_base_env
        
        mock_env = Mock()
        mock_env.num_blue_drones = 3
        mock_env.close = Mock()
        mock_wrapper.return_value = mock_env
        
        # Mock episode results
        mock_run_episode.return_value = {
            'episode_reward': 10.0,
            'episode_length': 100,
            'drone_rewards': [3.0, 4.0, 3.0],
            'winner': 'blue',
            'blue_alive': 3,
            'red_alive': 0,
            'blue_eliminations': 3,
            'red_eliminations': 0,
        }
        
        # Run evaluation
        results = evaluate_multidrone(
            model_path=self.model_path,
            n_episodes=5,
            drones_per_team=3,
            obs_type=MultiDroneObservationType.TEAM,
            red_tactics=[TeamTactic.AGGRESSIVE],
            gui=False,
            verbose=False,
            save_results=False,
        )
        
        # Check results structure
        self.assertIn('model_path', results)
        self.assertIn('n_episodes', results)
        self.assertIn('drones_per_team', results)
        self.assertIn('obs_type', results)
        self.assertIn('tactics_results', results)
        self.assertIn('overall_win_rate', results)
        
        # Check that model was loaded
        mock_load.assert_called_once_with(self.model_path)
        
        # Check that environments were created
        mock_aviary.assert_called_once()
        mock_wrapper.assert_called_once()
        
        # Check that episodes were run
        self.assertEqual(mock_run_episode.call_count, 5)
        
        # Check that environment was closed
        mock_env.close.assert_called_once()
    
    @patch('uav_intent_rl.examples.evaluate_multidrone.PPO.load')
    @patch('uav_intent_rl.examples.evaluate_multidrone.MultiDroneDogfightAviary')
    @patch('uav_intent_rl.examples.evaluate_multidrone.MultiDroneTrainingWrapper')
    @patch('uav_intent_rl.examples.evaluate_multidrone.run_episode')
    def test_evaluate_multidrone_multiple_tactics(self, mock_run_episode, mock_wrapper, mock_aviary, mock_load):
        """Test evaluation against multiple tactics."""
        # Mock model loading
        mock_model = Mock()
        mock_load.return_value = mock_model
        
        # Mock environment creation
        mock_base_env = Mock()
        mock_aviary.return_value = mock_base_env
        
        mock_env = Mock()
        mock_env.num_blue_drones = 3
        mock_env.close = Mock()
        mock_wrapper.return_value = mock_env
        
        # Mock episode results - alternating wins and losses
        episode_results = [
            {'episode_reward': 10.0, 'episode_length': 100, 'drone_rewards': [3.0, 4.0, 3.0],
             'winner': 'blue', 'blue_alive': 3, 'red_alive': 0, 'blue_eliminations': 3, 'red_eliminations': 0},
            {'episode_reward': -5.0, 'episode_length': 80, 'drone_rewards': [-2.0, -1.0, -2.0],
             'winner': 'red', 'blue_alive': 0, 'red_alive': 3, 'blue_eliminations': 0, 'red_eliminations': 3},
        ]
        
        mock_run_episode.side_effect = episode_results * 3  # 3 episodes per tactic
        
        # Run evaluation with multiple tactics
        results = evaluate_multidrone(
            model_path=self.model_path,
            n_episodes=3,
            drones_per_team=3,
            obs_type=MultiDroneObservationType.TEAM,
            red_tactics=[TeamTactic.AGGRESSIVE, TeamTactic.DEFENSIVE],
            gui=False,
            verbose=False,
            save_results=False,
        )
        
        # Check that both tactics were evaluated
        self.assertIn('aggressive', results['tactics_results'])
        self.assertIn('defensive', results['tactics_results'])
        
        # Check that environments were created for each tactic
        self.assertEqual(mock_aviary.call_count, 2)
        self.assertEqual(mock_wrapper.call_count, 2)
        
        # Check that episodes were run for each tactic
        self.assertEqual(mock_run_episode.call_count, 6)  # 3 episodes * 2 tactics
    
    @patch('uav_intent_rl.examples.evaluate_multidrone.PPO.load')
    @patch('uav_intent_rl.examples.evaluate_multidrone.MultiDroneDogfightAviary')
    @patch('uav_intent_rl.examples.evaluate_multidrone.MultiDroneTrainingWrapper')
    @patch('uav_intent_rl.examples.evaluate_multidrone.run_episode')
    def test_evaluate_multidrone_save_results(self, mock_run_episode, mock_wrapper, mock_aviary, mock_load):
        """Test that results are saved to file."""
        # Mock model loading
        mock_model = Mock()
        mock_load.return_value = mock_model
        
        # Mock environment creation
        mock_base_env = Mock()
        mock_aviary.return_value = mock_base_env
        
        mock_env = Mock()
        mock_env.num_blue_drones = 3
        mock_env.close = Mock()
        mock_wrapper.return_value = mock_env
        
        # Mock episode results
        mock_run_episode.return_value = {
            'episode_reward': 10.0,
            'episode_length': 100,
            'drone_rewards': [3.0, 4.0, 3.0],
            'winner': 'blue',
            'blue_alive': 3,
            'red_alive': 0,
            'blue_eliminations': 3,
            'red_eliminations': 0,
        }
        
        # Run evaluation with save_results=True
        results = evaluate_multidrone(
            model_path=self.model_path,
            n_episodes=2,
            drones_per_team=3,
            obs_type=MultiDroneObservationType.TEAM,
            red_tactics=[TeamTactic.AGGRESSIVE],
            gui=False,
            verbose=False,
            save_results=True,
        )
        
        # Check that results file was created
        results_file = self.model_path.parent / f"eval_results_{self.model_path.stem}.json"
        self.assertTrue(results_file.exists())
        
        # Check that results can be loaded
        with open(results_file, 'r') as f:
            saved_results = json.load(f)
        
        # Check that saved results match returned results
        self.assertEqual(saved_results['model_path'], results['model_path'])
        self.assertEqual(saved_results['n_episodes'], results['n_episodes'])
        self.assertEqual(saved_results['drones_per_team'], results['drones_per_team'])
        self.assertEqual(saved_results['obs_type'], results['obs_type'])
    
    @patch('uav_intent_rl.examples.evaluate_multidrone.PPO.load')
    def test_evaluate_multidrone_default_tactics(self, mock_load):
        """Test that default tactics are used when none specified."""
        # Mock model loading
        mock_model = Mock()
        mock_load.return_value = mock_model
        
        # Mock environment creation
        with patch('uav_intent_rl.examples.evaluate_multidrone.MultiDroneDogfightAviary') as mock_aviary:
            with patch('uav_intent_rl.examples.evaluate_multidrone.MultiDroneTrainingWrapper') as mock_wrapper:
                with patch('uav_intent_rl.examples.evaluate_multidrone.run_episode') as mock_run_episode:
                    mock_base_env = Mock()
                    mock_aviary.return_value = mock_base_env
                    
                    mock_env = Mock()
                    mock_env.num_blue_drones = 3
                    mock_env.close = Mock()
                    mock_wrapper.return_value = mock_env
                    
                    mock_run_episode.return_value = {
                        'episode_reward': 10.0,
                        'episode_length': 100,
                        'drone_rewards': [3.0, 4.0, 3.0],
                        'winner': 'blue',
                        'blue_alive': 3,
                        'red_alive': 0,
                        'blue_eliminations': 3,
                        'red_eliminations': 0,
                    }
                    
                    # Run evaluation without specifying tactics
                    results = evaluate_multidrone(
                        model_path=self.model_path,
                        n_episodes=1,
                        drones_per_team=3,
                        obs_type=MultiDroneObservationType.TEAM,
                        red_tactics=None,  # Use defaults
                        gui=False,
                        verbose=False,
                        save_results=False,
                    )
                    
                    # Check that all default tactics were evaluated
                    expected_tactics = ['aggressive', 'defensive', 'flanking', 'formation']
                    for tactic in expected_tactics:
                        self.assertIn(tactic, results['tactics_results'])


class TestPlotResults(unittest.TestCase):
    """Test the plot_results function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_results = {
            'drones_per_team': 3,
            'tactics_results': {
                'aggressive': {
                    'win_rate': 0.7,
                    'avg_reward': 15.5,
                    'avg_blue_alive': 2.1,
                },
                'defensive': {
                    'win_rate': 0.8,
                    'avg_reward': 18.2,
                    'avg_blue_alive': 2.5,
                },
                'flanking': {
                    'win_rate': 0.6,
                    'avg_reward': 12.8,
                    'avg_blue_alive': 1.8,
                },
            }
        }
    
    @patch('uav_intent_rl.examples.evaluate_multidrone.plt')
    def test_plot_results_basic(self, mock_plt):
        """Test basic plotting functionality."""
        # Mock matplotlib components
        mock_fig = Mock()
        mock_axes = [Mock(), Mock(), Mock()]
        mock_plt.subplots.return_value = (mock_fig, mock_axes)
        
        # Test plotting
        plot_results(self.sample_results)
        
        # Check that subplots were created
        mock_plt.subplots.assert_called_once_with(1, 3, figsize=(15, 5))
        
        # Check that plots were created
        self.assertEqual(mock_axes[0].bar.call_count, 1)
        self.assertEqual(mock_axes[1].bar.call_count, 1)
        self.assertEqual(mock_axes[2].bar.call_count, 1)
        
        # Check that tight_layout was called
        mock_plt.tight_layout.assert_called_once()
        
        # Check that show was called
        mock_plt.show.assert_called_once()
    
    @patch('uav_intent_rl.examples.evaluate_multidrone.plt')
    def test_plot_results_save_to_file(self, mock_plt):
        """Test plotting with save to file."""
        # Mock matplotlib components
        mock_fig = Mock()
        mock_axes = [Mock(), Mock(), Mock()]
        mock_plt.subplots.return_value = (mock_fig, mock_axes)
        
        # Create temporary file path
        save_path = Path("test_plot.png")
        
        # Test plotting with save
        plot_results(self.sample_results, save_path)
        
        # Check that savefig was called
        mock_fig.savefig.assert_called_once_with(save_path, dpi=300, bbox_inches='tight')
        
        # Check that show was not called when saving
        mock_plt.show.assert_not_called()
        
        # Clean up
        if save_path.exists():
            save_path.unlink()


if __name__ == '__main__':
    unittest.main() 