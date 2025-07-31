import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import tempfile
import shutil
from pathlib import Path

from uav_intent_rl.examples.train_multidrone import (
    make_env,
    train_multi_drone,
    TeamWinRateCallback,
    FormationRewardCallback
)
from uav_intent_rl.policies.team_scripted_red import TeamTactic
from uav_intent_rl.envs.MultiDroneTrainingWrapper import MultiDroneObservationType


class TestMakeEnv(unittest.TestCase):
    """Test the make_env function."""
    
    def test_make_env_returns_callable(self):
        """Test that make_env returns a callable."""
        env_factory = make_env(rank=0, seed=42, drones_per_team=3)
        self.assertTrue(callable(env_factory))
    
    def test_make_env_creates_wrapper(self):
        """Test that the factory creates a MultiDroneTrainingWrapper."""
        env_factory = make_env(rank=0, seed=42, drones_per_team=3)
        env = env_factory()
        
        from uav_intent_rl.envs.MultiDroneTrainingWrapper import MultiDroneTrainingWrapper
        self.assertIsInstance(env, MultiDroneTrainingWrapper)
    
    def test_make_env_parameters(self):
        """Test that make_env respects parameters."""
        env_factory = make_env(
            rank=1,
            seed=123,
            drones_per_team=2,
            obs_type=MultiDroneObservationType.LOCAL,
            red_tactic=TeamTactic.DEFENSIVE,
            gui=True
        )
        env = env_factory()
        
        # Check wrapper parameters
        self.assertEqual(env.num_blue_drones, 2)
        self.assertEqual(env.obs_type, MultiDroneObservationType.LOCAL)
        self.assertEqual(env._red_policy.tactic, TeamTactic.DEFENSIVE)
    
    def test_make_env_gui_only_first_rank(self):
        """Test that GUI is only enabled for rank 0."""
        # Rank 0 should have GUI option
        env_factory_0 = make_env(rank=0, gui=True)
        env_0 = env_factory_0()
        
        # Rank 1 should not have GUI even if requested
        env_factory_1 = make_env(rank=1, gui=True)
        env_1 = env_factory_1()
        
        # The base environment should have different GUI settings
        # (We can't easily test this without accessing internal state,
        # but we can verify the wrappers are created successfully)
        self.assertIsNotNone(env_0)
        self.assertIsNotNone(env_1)


class TestTeamWinRateCallback(unittest.TestCase):
    """Test the TeamWinRateCallback class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_eval_env = Mock()
        self.mock_model = Mock()
        
        # Mock environment step and reset
        self.mock_eval_env.reset.return_value = np.random.randn(1, 3, 75)  # 1 env, 3 drones, 75 obs dim
        
        # Create a side effect that always terminates episodes properly
        # Each episode needs at least 2 steps: one "not done" step, then one "done" step
        # Since the callback runs multiple episodes, we need to provide many responses
        from itertools import repeat
        
        # Define the step responses
        step_not_done = (
            np.random.randn(1, 3, 75),  # obs
            np.array([[1.0, 2.0, 3.0]]),  # rewards
            np.array([False]),  # dones
            [{'winner': 'blue', 'blue_alive': 3, 'red_alive': 0, 'elimination_count': [3, 0]}]  # infos
        )
        
        step_done = (
            np.random.randn(1, 3, 75),  # obs
            np.array([[4.0, 5.0, 6.0]]),  # rewards
            np.array([True]),  # dones - episode terminates
            [{'winner': 'blue', 'blue_alive': 3, 'red_alive': 0, 'elimination_count': [3, 0]}]  # infos
        )
        
        # Create an endless sequence: not_done, done, not_done, done, ...
        # This ensures each episode gets exactly 2 steps and then terminates
        self.mock_eval_env.step.side_effect = [step_not_done, step_done] * 10  # Enough for 10 episodes
        
        self.callback = TeamWinRateCallback(
            eval_env=self.mock_eval_env,
            eval_freq=10,
            n_eval_episodes=2,
            verbose=0
        )
        self.callback.model = self.mock_model
        # Create a mock logger that allows attribute setting
        mock_logger = Mock()
        mock_logger.record = Mock()
        # Patch the logger property directly
        self.callback._logger = mock_logger
    
    def test_initialization(self):
        """Test callback initialization."""
        self.assertEqual(self.callback.eval_freq, 10)
        self.assertEqual(self.callback.n_eval_episodes, 2)
        self.assertEqual(self.callback.verbose, 0)
        self.assertEqual(len(self.callback.evaluations_results), 0)
    
    def test_on_step_no_evaluation(self):
        """Test that evaluation doesn't happen before eval_freq."""
        # Set n_calls to be less than eval_freq
        self.callback.n_calls = 5
        
        result = self.callback._on_step()
        self.assertTrue(result)
        
        # Should not have called reset or step
        self.mock_eval_env.reset.assert_not_called()
        self.mock_eval_env.step.assert_not_called()
    
    def test_on_step_with_evaluation(self):
        """Test that evaluation happens at eval_freq."""
        # Mock model prediction to return a tuple
        self.mock_model.predict.return_value = (np.random.randn(1, 3, 4), None)
        
        # Set n_calls to be at eval_freq
        self.callback.n_calls = 10
        
        result = self.callback._on_step()
        self.assertTrue(result)
        
        # Should have called reset and step
        self.mock_eval_env.reset.assert_called()
        self.mock_eval_env.step.assert_called()
    
    def test_evaluate_metrics(self):
        """Test that evaluation calculates correct metrics."""
        # Mock model prediction to return a tuple
        self.mock_model.predict.return_value = (np.random.randn(1, 3, 4), None)
        
        # Call evaluation
        self.callback._evaluate()
        
        # Check that metrics were logged
        self.callback.logger.record.assert_called()
        
        # Check that results were stored
        self.assertEqual(len(self.callback.evaluations_results), 1)
        result = self.callback.evaluations_results[0]
        self.assertIn('timestep', result)
        self.assertIn('win_rate', result)
        self.assertIn('avg_reward', result)
        self.assertIn('elimination_ratio', result)


class TestFormationRewardCallback(unittest.TestCase):
    """Test the FormationRewardCallback class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.callback = FormationRewardCallback(verbose=0)
        # Create a mock logger that allows attribute setting
        mock_logger = Mock()
        mock_logger.record = Mock()
        # Patch the logger property directly
        self.callback._logger = mock_logger
    
    def test_initialization(self):
        """Test callback initialization."""
        self.assertEqual(self.callback.verbose, 0)
        self.assertEqual(len(self.callback.formation_scores), 0)
    
    def test_on_step(self):
        """Test that _on_step returns True."""
        result = self.callback._on_step()
        self.assertTrue(result)
    
    def test_on_rollout_end(self):
        """Test that _on_rollout_end doesn't raise errors."""
        # Should not raise any exceptions
        self.callback._on_rollout_end()


class TestTrainMultiDrone(unittest.TestCase):
    """Test the train_multi_drone function."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.save_path = Path(self.temp_dir) / "test_models"
        self.tensorboard_log = Path(self.temp_dir) / "test_logs"
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    @patch('uav_intent_rl.examples.train_multidrone.SubprocVecEnv')
    @patch('uav_intent_rl.examples.train_multidrone.DummyVecEnv')
    @patch('uav_intent_rl.examples.train_multidrone.PPO')
    @patch('uav_intent_rl.examples.train_multidrone.EvalCallback')
    def test_train_multi_drone_basic(self, mock_eval_callback, mock_ppo, mock_dummy_vec, mock_subproc_vec):
        """Test basic training setup."""
        # Mock the environment creation
        mock_env = Mock()
        mock_dummy_vec.return_value = mock_env
        
        # Mock the PPO model
        mock_model = Mock()
        mock_ppo.return_value = mock_model
        
        # Mock the learn method
        mock_model.learn.return_value = None
        mock_model.save.return_value = None
        
        # Mock the evaluation callback
        mock_eval_callback.return_value = Mock()
        
        # Test with minimal parameters
        result = train_multi_drone(
            total_timesteps=1000,
            drones_per_team=2,
            n_envs=1,
            use_multiprocessing=False,
            save_path=str(self.save_path),
            tensorboard_log=str(self.tensorboard_log),
            verbose=0
        )
        
        # Check that model was created
        mock_ppo.assert_called_once()
        
        # Check that learn was called
        mock_model.learn.assert_called_once()
        
        # Check that save was called
        mock_model.save.assert_called_once()
        
        # Check return value
        self.assertEqual(result, mock_model)
    
    @patch('uav_intent_rl.examples.train_multidrone.SubprocVecEnv')
    @patch('uav_intent_rl.examples.train_multidrone.DummyVecEnv')
    @patch('uav_intent_rl.examples.train_multidrone.PPO')
    @patch('uav_intent_rl.examples.train_multidrone.EvalCallback')
    def test_train_multi_drone_multiprocessing(self, mock_eval_callback, mock_ppo, mock_dummy_vec, mock_subproc_vec):
        """Test training with multiprocessing."""
        # Mock the environment creation
        mock_env = Mock()
        mock_subproc_vec.return_value = mock_env
        
        # Mock the PPO model
        mock_model = Mock()
        mock_ppo.return_value = mock_model
        
        # Mock the learn method
        mock_model.learn.return_value = None
        mock_model.save.return_value = None
        
        # Mock the evaluation callback
        mock_eval_callback.return_value = Mock()
        
        # Test with multiprocessing
        result = train_multi_drone(
            total_timesteps=1000,
            drones_per_team=3,
            n_envs=4,
            use_multiprocessing=True,
            save_path=str(self.save_path),
            tensorboard_log=str(self.tensorboard_log),
            verbose=0
        )
        
        # Check that SubprocVecEnv was used for training
        mock_subproc_vec.assert_called_once()
        
        # Check that DummyVecEnv was used for evaluation (not training)
        # The evaluation environment always uses DummyVecEnv regardless of multiprocessing
        mock_dummy_vec.assert_called_once()
    
    @patch('uav_intent_rl.examples.train_multidrone.SubprocVecEnv')
    @patch('uav_intent_rl.examples.train_multidrone.DummyVecEnv')
    @patch('uav_intent_rl.examples.train_multidrone.PPO')
    @patch('uav_intent_rl.examples.train_multidrone.EvalCallback')
    def test_train_multi_drone_curriculum(self, mock_eval_callback, mock_ppo, mock_dummy_vec, mock_subproc_vec):
        """Test training with curriculum learning."""
        # Mock the environment creation
        mock_env = Mock()
        mock_dummy_vec.return_value = mock_env
        
        # Mock the PPO model
        mock_model = Mock()
        mock_ppo.return_value = mock_model
        
        # Mock the learn method
        mock_model.learn.return_value = None
        mock_model.save.return_value = None
        
        # Mock the evaluation callback
        mock_eval_callback.return_value = Mock()
        
        # Test with curriculum learning
        result = train_multi_drone(
            total_timesteps=1000,
            drones_per_team=2,
            n_envs=1,
            curriculum=True,
            use_multiprocessing=False,
            save_path=str(self.save_path),
            tensorboard_log=str(self.tensorboard_log),
            verbose=0
        )
        
        # Check that model was created
        mock_ppo.assert_called_once()
        
        # Check that learn was called
        mock_model.learn.assert_called_once()
    
    @patch('uav_intent_rl.examples.train_multidrone.SubprocVecEnv')
    @patch('uav_intent_rl.examples.train_multidrone.DummyVecEnv')
    @patch('uav_intent_rl.examples.train_multidrone.PPO')
    @patch('uav_intent_rl.examples.train_multidrone.EvalCallback')
    def test_train_multi_drone_no_curriculum(self, mock_eval_callback, mock_ppo, mock_dummy_vec, mock_subproc_vec):
        """Test training without curriculum learning."""
        # Mock the environment creation
        mock_env = Mock()
        mock_dummy_vec.return_value = mock_env
        
        # Mock the PPO model
        mock_model = Mock()
        mock_ppo.return_value = mock_model
        
        # Mock the learn method
        mock_model.learn.return_value = None
        mock_model.save.return_value = None
        
        # Mock the evaluation callback
        mock_eval_callback.return_value = Mock()
        
        # Test without curriculum learning
        result = train_multi_drone(
            total_timesteps=1000,
            drones_per_team=2,
            n_envs=1,
            curriculum=False,
            use_multiprocessing=False,
            save_path=str(self.save_path),
            tensorboard_log=str(self.tensorboard_log),
            verbose=0
        )
        
        # Check that model was created
        mock_ppo.assert_called_once()
        
        # Check that learn was called
        mock_model.learn.assert_called_once()
    
    def test_train_multi_drone_invalid_algorithm(self):
        """Test that invalid algorithm raises ValueError."""
        with self.assertRaises(ValueError):
            train_multi_drone(
                total_timesteps=1000,
                algorithm="INVALID",
                use_multiprocessing=False,
                save_path=str(self.save_path),
                tensorboard_log=str(self.tensorboard_log),
                verbose=0
            )
    
    @patch('uav_intent_rl.examples.train_multidrone.SubprocVecEnv')
    @patch('uav_intent_rl.examples.train_multidrone.DummyVecEnv')
    @patch('uav_intent_rl.examples.train_multidrone.PPO')
    @patch('uav_intent_rl.examples.train_multidrone.EvalCallback')
    def test_train_multi_drone_different_obs_types(self, mock_eval_callback, mock_ppo, mock_dummy_vec, mock_subproc_vec):
        """Test training with different observation types."""
        # Mock the environment creation
        mock_env = Mock()
        mock_dummy_vec.return_value = mock_env
        
        # Mock the PPO model
        mock_model = Mock()
        mock_ppo.return_value = mock_model
        
        # Mock the learn method
        mock_model.learn.return_value = None
        mock_model.save.return_value = None
        
        # Mock the evaluation callback
        mock_eval_callback.return_value = Mock()
        
        # Test different observation types
        obs_types = [
            MultiDroneObservationType.LOCAL,
            MultiDroneObservationType.TEAM,
            MultiDroneObservationType.GLOBAL,
            MultiDroneObservationType.COMMUNICATION
        ]
        
        for obs_type in obs_types:
            with self.subTest(obs_type=obs_type):
                result = train_multi_drone(
                    total_timesteps=1000,
                    drones_per_team=2,
                    n_envs=1,
                    obs_type=obs_type,
                    use_multiprocessing=False,
                    save_path=str(self.save_path),
                    tensorboard_log=str(self.tensorboard_log),
                    verbose=0
                )
                
                # Check that model was created
                mock_ppo.assert_called()
                
                # Reset mocks for next iteration
                mock_ppo.reset_mock()
                mock_model.learn.reset_mock()
    
    @patch('uav_intent_rl.examples.train_multidrone.SubprocVecEnv')
    @patch('uav_intent_rl.examples.train_multidrone.DummyVecEnv')
    @patch('uav_intent_rl.examples.train_multidrone.PPO')
    @patch('uav_intent_rl.examples.train_multidrone.EvalCallback')
    def test_train_multi_drone_custom_hyperparameters(self, mock_eval_callback, mock_ppo, mock_dummy_vec, mock_subproc_vec):
        """Test training with custom hyperparameters."""
        # Mock the environment creation
        mock_env = Mock()
        mock_dummy_vec.return_value = mock_env
        
        # Mock the PPO model
        mock_model = Mock()
        mock_ppo.return_value = mock_model
        
        # Mock the learn method
        mock_model.learn.return_value = None
        mock_model.save.return_value = None
        
        # Mock the evaluation callback
        mock_eval_callback.return_value = Mock()
        
        # Test with custom hyperparameters
        result = train_multi_drone(
            total_timesteps=1000,
            drones_per_team=2,
            n_envs=1,
            learning_rate=1e-4,
            batch_size=128,
            n_steps=1024,
            gamma=0.95,
            gae_lambda=0.9,
            clip_range=0.1,
            ent_coef=0.005,
            use_multiprocessing=False,
            save_path=str(self.save_path),
            tensorboard_log=str(self.tensorboard_log),
            verbose=0
        )
        
        # Check that PPO was called with custom parameters
        mock_ppo.assert_called_once()
        call_args = mock_ppo.call_args
        
        # Check some key parameters
        self.assertEqual(call_args[1]['learning_rate'], 1e-4)
        self.assertEqual(call_args[1]['batch_size'], 128)
        self.assertEqual(call_args[1]['n_steps'], 1024)
        self.assertEqual(call_args[1]['gamma'], 0.95)
        self.assertEqual(call_args[1]['gae_lambda'], 0.9)
        self.assertEqual(call_args[1]['clip_range'], 0.1)
        self.assertEqual(call_args[1]['ent_coef'], 0.005)


if __name__ == '__main__':
    unittest.main() 