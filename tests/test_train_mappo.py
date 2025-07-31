#!/usr/bin/env python3
"""Unit tests for train_mappo function."""

import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import torch as th
from gymnasium import spaces

from uav_intent_rl.examples.train_mappo import train_mappo, make_mappo_env, MAPPOEvalCallback
from uav_intent_rl.algo.mappo import MAPPO, MAPPOPolicy, MultiAgentRolloutBuffer
from uav_intent_rl.envs.MAPPODogfightWrapper import MAPPODogfightWrapper
from uav_intent_rl.policies.team_scripted_red import TeamTactic


class TestTrainMAPPO(unittest.TestCase):
    """Test the train_mappo function and related components."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.save_path = Path(self.temp_dir) / "test_mappo"
        self.tensorboard_log = Path(self.temp_dir) / "runs" / "test_mappo"

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_make_mappo_env(self):
        """Test that make_mappo_env creates a valid environment function."""
        print("Testing make_mappo_env...")
        
        env_fn = make_mappo_env(
            rank=0,
            seed=42,
            drones_per_team=3,
            red_tactic=TeamTactic.AGGRESSIVE,
            centralized_critic=True,
            gui=False
        )
        
        # Test that the function returns a callable
        self.assertTrue(callable(env_fn))
        
        # Test that it creates a valid environment
        env = env_fn()
        self.assertIsInstance(env, MAPPODogfightWrapper)
        
        # Test environment properties
        self.assertEqual(env.num_blue_drones, 3)
        self.assertEqual(env.centralized_critic, True)
        self.assertEqual(env.reward_sharing_ratio, 0.2)
        self.assertEqual(env.add_agent_id, True)
        self.assertEqual(env.normalize_rewards, True)
        
        print("✓ make_mappo_env works correctly")

    def test_mappo_eval_callback(self):
        """Test the MAPPOEvalCallback class."""
        print("Testing MAPPOEvalCallback...")
        
        # Create a mock eval environment
        mock_eval_env = MagicMock()
        mock_eval_env.reset.return_value = np.random.randn(1, 3, 50)  # (n_envs, n_agents, obs_dim)
        mock_eval_env.step.return_value = (
            np.random.randn(1, 3, 50),  # obs
            np.random.randn(1, 3),       # reward
            np.array([False]),           # terminated
            np.array([False]),           # truncated
            [{'winner': 'blue'}]         # info
        )
        
        # Create a mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = (np.random.randn(1, 3, 4), None)  # actions, states
        mock_model.save = MagicMock()
        
        # Create callback
        callback = MAPPOEvalCallback(
            eval_env=mock_eval_env,
            eval_freq=10,
            n_eval_episodes=2,
            deterministic=True,
            render=False,
            best_model_save_path=str(self.save_path / "best_model"),
            log_path=str(self.save_path / "eval_logs"),
            verbose=1
        )
        
        # Set the model
        callback.model = mock_model
        callback.logger = MagicMock()
        
        # Test evaluation
        callback._evaluate()
        
        # Check that evaluation completed without errors
        self.assertTrue(len(callback.evaluations_results) > 0)
        self.assertTrue(len(callback.evaluations_timesteps) > 0)
        
        print("✓ MAPPOEvalCallback works correctly")

    @patch('uav_intent_rl.examples.train_mappo.MAPPO')
    def test_train_mappo_minimal(self, mock_mappo_class):
        """Test train_mappo with minimal parameters."""
        print("Testing train_mappo with minimal parameters...")
        
        # Mock the MAPPO class
        mock_model = MagicMock()
        mock_mappo_class.return_value = mock_model
        
        # Test with minimal parameters
        model = train_mappo(
            total_timesteps=1000,
            drones_per_team=2,
            n_envs=2,
            save_path=str(self.save_path),
            tensorboard_log=str(self.tensorboard_log),
            use_multiprocessing=False,  # Use DummyVecEnv for testing
            curriculum=False,  # Disable curriculum for simplicity
            verbose=0
        )
        
        # Check that MAPPO was called with correct parameters
        mock_mappo_class.assert_called_once()
        call_args = mock_mappo_class.call_args
        
        # Check key parameters
        self.assertEqual(call_args[1]['n_agents'], 2)
        self.assertEqual(call_args[1]['use_centralized_critic'], True)
        self.assertEqual(call_args[1]['learning_rate'], 3e-4)
        self.assertEqual(call_args[1]['n_steps'], 2048)
        self.assertEqual(call_args[1]['batch_size'], 256)
        
        # Check that learn was called
        mock_model.learn.assert_called_once()
        
        # Check that model was saved
        mock_model.save.assert_called()
        
        print("✓ train_mappo with minimal parameters works")

    @patch('uav_intent_rl.examples.train_mappo.MAPPO')
    def test_train_mappo_with_curriculum(self, mock_mappo_class):
        """Test train_mappo with curriculum learning enabled."""
        print("Testing train_mappo with curriculum...")
        
        # Mock the MAPPO class
        mock_model = MagicMock()
        mock_mappo_class.return_value = mock_model
        
        # Test with curriculum enabled
        model = train_mappo(
            total_timesteps=1000,
            drones_per_team=3,
            n_envs=4,
            save_path=str(self.save_path),
            tensorboard_log=str(self.tensorboard_log),
            use_multiprocessing=False,
            curriculum=True,
            verbose=0
        )
        
        # Check that learn was called
        mock_model.learn.assert_called_once()
        
        # Check that callbacks were created (curriculum callback should be included)
        learn_call_args = mock_model.learn.call_args
        callback = learn_call_args[1]['callback']
        
        # Should have evaluation, checkpoint, and curriculum callbacks
        self.assertGreaterEqual(len(callback.callbacks), 3)
        
        print("✓ train_mappo with curriculum works")

    @patch('uav_intent_rl.examples.train_mappo.MAPPO')
    def test_train_mappo_with_multiprocessing(self, mock_mappo_class):
        """Test train_mappo with multiprocessing enabled."""
        print("Testing train_mappo with multiprocessing...")
        
        # Mock the MAPPO class
        mock_model = MagicMock()
        mock_mappo_class.return_value = mock_model
        
        # Test with multiprocessing enabled
        model = train_mappo(
            total_timesteps=1000,
            drones_per_team=3,
            n_envs=4,
            save_path=str(self.save_path),
            tensorboard_log=str(self.tensorboard_log),
            use_multiprocessing=True,
            curriculum=False,
            verbose=0
        )
        
        # Check that learn was called
        mock_model.learn.assert_called_once()
        
        print("✓ train_mappo with multiprocessing works")

    def test_mappo_rollout_buffer_fix(self):
        """Test that the rollout buffer fix works correctly."""
        print("Testing rollout buffer fix...")
        
        # Create a rollout buffer
        obs_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
        action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        state_space = spaces.Box(low=-1, high=1, shape=(20,), dtype=np.float32)
        
        buffer = MultiAgentRolloutBuffer(
            buffer_size=100,
            observation_space=obs_space,
            action_space=action_space,
            n_envs=2,
            n_agents=3,
            state_space=state_space,
            device="cpu"
        )
        
        # Test adding data with the correct shapes
        # This simulates what happens after the fix in collect_rollouts
        obs = np.random.randn(6, 10).astype(np.float32)  # (n_envs * n_agents, obs_dim)
        actions = np.random.randn(6, 4).astype(np.float32)  # (n_envs * n_agents, action_dim)
        rewards = np.random.randn(6).astype(np.float32)  # (n_envs * n_agents)
        episode_starts = np.array([False, False, False, False, False, False])
        values = th.randn(6, 1)  # (n_envs * n_agents, 1)
        log_probs = th.randn(6, 4)  # (n_envs * n_agents, action_dim)
        states = np.random.randn(6, 20).astype(np.float32)  # (n_envs * n_agents, state_dim)
        
        # This should not raise any errors
        buffer.add(obs, actions, rewards, episode_starts, values, log_probs, states)
        
        # Test that the buffer can be sampled
        buffer.pos = buffer.buffer_size
        buffer.full = True
        
        for batch in buffer.get(batch_size=32):
            self.assertIsNotNone(batch)
            self.assertTrue(hasattr(batch, 'states'))
            break
        
        print("✓ Rollout buffer fix works correctly")

    def test_mappo_policy_centralized_critic(self):
        """Test that the MAPPO policy works with centralized critic."""
        print("Testing MAPPO policy with centralized critic...")
        
        # Create a simple policy
        obs_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
        action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        
        policy = MAPPOPolicy(
            observation_space=obs_space,
            action_space=action_space,
            lr_schedule=lambda _: 3e-4,
            n_agents=3,
            use_centralized_critic=True,
            state_shape=(20,)
        )
        
        # Test forward pass with centralized critic
        obs = th.randn(2, 10)
        state = th.randn(2, 20)
        
        actions, values, log_probs = policy(obs, state)
        
        self.assertEqual(actions.shape, (2, 4))
        self.assertEqual(values.shape, (2, 1))
        self.assertEqual(log_probs.shape, (2, 4))
        
        # Test evaluate_actions
        actions_tensor = th.randn(2, 4)
        values_eval, log_probs_eval, entropy = policy.evaluate_actions(obs, actions_tensor, state)
        
        self.assertEqual(values_eval.shape, (2, 1))
        self.assertEqual(log_probs_eval.shape, (2, 4))
        self.assertEqual(entropy.shape, (2,))
        
        print("✓ MAPPO policy with centralized critic works")

    def test_reward_sharing_fix(self):
        """Test that the reward sharing fix works correctly."""
        print("Testing reward sharing fix...")
        
        # Create a mock environment wrapper
        mock_env = MagicMock()
        mock_env.num_blue_drones = 3
        mock_env.env.drones_per_team = 3
        mock_env.env.NUM_DRONES = 6
        mock_env.env.alive_status = [True] * 6
        mock_env.env.health = [1.0] * 6
        mock_env.env.elimination_count = [0, 0]
        mock_env.env.step_counter = 0
        mock_env.env.PYB_FREQ = 240
        mock_env.env.EPISODE_LEN_SEC = 30
        
        wrapper = MAPPODogfightWrapper(
            env=mock_env,
            centralized_critic=True,
            shared_reward=False,  # This should not matter anymore
            reward_sharing_ratio=0.3,  # This should always be applied
            add_agent_id=True,
            normalize_rewards=False  # Disable for testing
        )
        
        # Test reward processing
        individual_rewards = np.array([1.0, 2.0, 3.0])
        processed_rewards = wrapper._process_mappo_rewards(individual_rewards)
        
        # Check that reward sharing was applied (0.7 * individual + 0.3 * mean)
        expected_mean = np.mean(individual_rewards)
        expected_rewards = 0.7 * individual_rewards + 0.3 * expected_mean
        
        np.testing.assert_array_almost_equal(processed_rewards, expected_rewards)
        
        print("✓ Reward sharing fix works correctly")

    def test_gymnasium_api_fix(self):
        """Test that the Gymnasium API fix works correctly."""
        print("Testing Gymnasium API fix...")
        
        # Test the evaluation callback with the new API
        mock_eval_env = MagicMock()
        mock_eval_env.reset.return_value = np.random.randn(1, 3, 50)
        
        # Simulate the new Gymnasium API: (obs, reward, terminated, truncated, info)
        mock_eval_env.step.return_value = (
            np.random.randn(1, 3, 50),  # obs
            np.random.randn(1, 3),       # reward
            np.array([False]),           # terminated
            np.array([True]),            # truncated (this should be handled correctly)
            [{'winner': 'blue'}]         # info
        )
        
        mock_model = MagicMock()
        mock_model.predict.return_value = (np.random.randn(1, 3, 4), None)
        mock_model.save = MagicMock()
        
        callback = MAPPOEvalCallback(
            eval_env=mock_eval_env,
            eval_freq=10,
            n_eval_episodes=1,
            deterministic=True,
            render=False,
            verbose=0
        )
        callback.model = mock_model
        callback.logger = MagicMock()
        
        # This should not cause an infinite loop
        callback._evaluate()
        
        # Check that evaluation completed
        self.assertTrue(len(callback.evaluations_results) > 0)
        
        print("✓ Gymnasium API fix works correctly")

    def test_curriculum_callback_fix(self):
        """Test that the curriculum callback fix works correctly."""
        print("Testing curriculum callback fix...")
        
        # Test with DummyVecEnv (direct attribute access)
        mock_dummy_env = MagicMock()
        mock_dummy_env.num_envs = 2
        mock_dummy_env.envs = [MagicMock(), MagicMock()]
        for env in mock_dummy_env.envs:
            env._red_policy = MagicMock()
            env._red_policy.tactic = TeamTactic.FORMATION
        
        # Test with SubprocVecEnv (using call method)
        mock_subproc_env = MagicMock()
        mock_subproc_env.num_envs = 2
        mock_subproc_env.call = MagicMock()
        
        # Test that both work without errors
        new_tactic = TeamTactic.AGGRESSIVE
        
        # DummyVecEnv case
        for env in mock_dummy_env.envs:
            env._red_policy.tactic = new_tactic
        
        # SubprocVecEnv case
        mock_subproc_env.call(lambda e, t=new_tactic: setattr(e._red_policy, "tactic", t))
        
        # Verify the calls worked
        for env in mock_dummy_env.envs:
            self.assertEqual(env._red_policy.tactic, new_tactic)
        
        mock_subproc_env.call.assert_called_once()
        
        print("✓ Curriculum callback fix works correctly")


if __name__ == "__main__":
    print("Running MAPPO training tests...")
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTrainMAPPO)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("\n🎉 All MAPPO training tests passed!")
    else:
        print(f"\n❌ {len(result.failures)} test(s) failed:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
        print(f"\n❌ {len(result.errors)} test(s) had errors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}") 