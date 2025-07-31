#!/usr/bin/env python3
"""Unit tests for MAPPO implementation."""

import unittest
import numpy as np
import torch as th
from gymnasium import spaces

from uav_intent_rl.algo.mappo import (
    MultiAgentRolloutBuffer, 
    MultiAgentRolloutBufferSamples,
    MAPPOPolicy, 
    MAPPO
)


class TestMultiAgentRolloutBuffer(unittest.TestCase):
    """Test the MultiAgentRolloutBuffer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.obs_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.state_space = spaces.Box(low=-1, high=1, shape=(20,), dtype=np.float32)
        
        self.buffer = MultiAgentRolloutBuffer(
            buffer_size=100,
            observation_space=self.obs_space,
            action_space=self.action_space,
            n_envs=2,
            n_agents=3,
            state_space=self.state_space,
            device="cpu"
        )
    
    def test_initialization(self):
        """Test that the buffer initializes correctly."""
        self.assertEqual(self.buffer.n_agents, 3)
        self.assertEqual(self.buffer.n_envs, 2)
        self.assertTrue(hasattr(self.buffer, 'states'))
        self.assertEqual(self.buffer.states.shape, (100, 2, 20))
    
    def test_add_data(self):
        """Test adding data to the buffer."""
        obs = np.random.randn(2, 10).astype(np.float32)
        actions = np.random.randn(2, 4).astype(np.float32)
        rewards = np.random.randn(2).astype(np.float32)
        episode_starts = np.array([False, False])
        values = th.randn(2, 1)
        log_probs = th.randn(2, 1)
        states = np.random.randn(2, 20).astype(np.float32)
        
        # Should not raise any errors
        self.buffer.add(obs, actions, rewards, episode_starts, values, log_probs, states)
        
        # Check that states were stored
        self.assertTrue(np.array_equal(self.buffer.states[0], states))
    
    def test_get_with_none_batch_size(self):
        """Test that get() works with None batch_size."""
        # Fill the buffer
        for i in range(self.buffer.buffer_size):
            obs = np.random.randn(2, 10).astype(np.float32)
            actions = np.random.randn(2, 4).astype(np.float32)
            rewards = np.random.randn(2).astype(np.float32)
            episode_starts = np.array([False, False])
            values = th.randn(2, 1)
            log_probs = th.randn(2, 1)
            states = np.random.randn(2, 20).astype(np.float32)
            
            self.buffer.add(obs, actions, rewards, episode_starts, values, log_probs, states)
        
        # Test get with None batch_size
        batches = list(self.buffer.get(batch_size=None))
        self.assertEqual(len(batches), 1)  # Should return all data in one batch
        
        batch = batches[0]
        self.assertIsInstance(batch, MultiAgentRolloutBufferSamples)
        self.assertTrue(hasattr(batch, 'states'))
        self.assertEqual(batch.states.shape[1], 20)  # state dimension
    
    def test_get_with_regular_batch_size(self):
        """Test that get() works with regular batch_size."""
        # Fill the buffer
        for i in range(self.buffer.buffer_size):
            obs = np.random.randn(2, 10).astype(np.float32)
            actions = np.random.randn(2, 4).astype(np.float32)
            rewards = np.random.randn(2).astype(np.float32)
            episode_starts = np.array([False, False])
            values = th.randn(2, 1)
            log_probs = th.randn(2, 1)
            states = np.random.randn(2, 20).astype(np.float32)
            
            self.buffer.add(obs, actions, rewards, episode_starts, values, log_probs, states)
        
        # Test get with regular batch_size
        batches = list(self.buffer.get(batch_size=32))
        self.assertGreater(len(batches), 0)
        
        for batch in batches:
            self.assertIsInstance(batch, MultiAgentRolloutBufferSamples)
            self.assertTrue(hasattr(batch, 'states'))
            self.assertEqual(batch.states.shape[1], 20)  # state dimension


class TestMAPPOPolicy(unittest.TestCase):
    """Test the MAPPOPolicy class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.obs_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        
        self.policy = MAPPOPolicy(
            observation_space=self.obs_space,
            action_space=self.action_space,
            lr_schedule=lambda _: 0.001,
            n_agents=3,
            use_centralized_critic=True,
            state_shape=(5, 4),  # Multi-dimensional state
        )
    
    def test_initialization(self):
        """Test that the policy initializes correctly."""
        self.assertEqual(self.policy.n_agents, 3)
        self.assertTrue(self.policy.use_centralized_critic)
        self.assertEqual(self.policy.state_shape, (5, 4))
        self.assertTrue(hasattr(self.policy, 'centralized_value_net'))
        self.assertTrue(hasattr(self.policy, 'centralized_value_head'))
    
    def test_forward_with_centralized_critic(self):
        """Test forward pass with centralized critic."""
        obs = th.randn(2, 10)
        state = th.randn(2, 5, 4)  # Multi-dimensional state
        
        actions, values, log_probs = self.policy(obs, state)
        
        self.assertEqual(actions.shape, (2, 4))
        self.assertEqual(values.shape, (2, 1))
        self.assertEqual(log_probs.shape, (2, 4))
    
    def test_forward_without_centralized_critic(self):
        """Test forward pass without centralized critic."""
        obs = th.randn(2, 10)
        
        actions, values, log_probs = self.policy(obs, None)
        
        self.assertEqual(actions.shape, (2, 4))
        self.assertEqual(values.shape, (2, 1))
        self.assertEqual(log_probs.shape, (2, 4))
    
    def test_evaluate_actions_with_centralized_critic(self):
        """Test evaluate_actions with centralized critic."""
        obs = th.randn(2, 10)
        actions = th.randn(2, 4)
        state = th.randn(2, 5, 4)  # Multi-dimensional state
        
        values, log_prob, entropy = self.policy.evaluate_actions(obs, actions, state)
        
        self.assertEqual(values.shape, (2, 1))
        self.assertEqual(log_prob.shape, (2, 4))
        self.assertIsNotNone(entropy)
    
    def test_evaluate_actions_without_centralized_critic(self):
        """Test evaluate_actions without centralized critic."""
        obs = th.randn(2, 10)
        actions = th.randn(2, 4)
        
        values, log_prob, entropy = self.policy.evaluate_actions(obs, actions, None)
        
        self.assertEqual(values.shape, (2, 1))
        self.assertEqual(log_prob.shape, (2, 4))
        self.assertIsNotNone(entropy)
    
    def test_predict_values_with_centralized_critic(self):
        """Test predict_values with centralized critic."""
        obs = th.randn(2, 10)
        state = th.randn(2, 5, 4)  # Multi-dimensional state
        
        values = self.policy.predict_values(obs, state)
        
        self.assertEqual(values.shape, (2, 1))
    
    def test_predict_values_without_centralized_critic(self):
        """Test predict_values without centralized critic."""
        obs = th.randn(2, 10)
        
        values = self.policy.predict_values(obs, None)
        
        self.assertEqual(values.shape, (2, 1))
    
    def test_multi_dimensional_state_handling(self):
        """Test that multi-dimensional states are handled correctly."""
        obs = th.randn(2, 10)
        state_3d = th.randn(2, 5, 4, 3)  # 3D state
        state_4d = th.randn(2, 5, 4, 3, 2)  # 4D state
        
        # Should not raise shape mismatch errors
        actions, values, log_probs = self.policy(obs, state_3d)
        self.assertEqual(values.shape, (2, 1))
        
        actions, values, log_probs = self.policy(obs, state_4d)
        self.assertEqual(values.shape, (2, 1))


class TestMAPPO(unittest.TestCase):
    """Test the MAPPO class."""
    
    def test_initialization_parameters(self):
        """Test that MAPPO has the expected parameters."""
        # Test class attributes
        self.assertTrue(hasattr(MAPPO, 'n_agents'))
        self.assertTrue(hasattr(MAPPO, 'use_centralized_critic'))
        self.assertTrue(hasattr(MAPPO, 'state_shape'))
    
    def test_rollout_buffer_class(self):
        """Test that MAPPO uses the correct rollout buffer class."""
        # This would normally be tested with a proper environment
        # but we can test the class structure
        self.assertEqual(MAPPO.rollout_buffer_class, MultiAgentRolloutBuffer)


class TestMultiAgentRolloutBufferSamples(unittest.TestCase):
    """Test the MultiAgentRolloutBufferSamples named tuple."""
    
    def test_named_tuple_structure(self):
        """Test that the named tuple has the correct structure."""
        # Check that it has all the fields from RolloutBufferSamples plus states
        expected_fields = ('observations', 'actions', 'old_values', 'old_log_prob', 
                         'advantages', 'returns', 'episode_starts', 'states')
        
        for field in expected_fields:
            self.assertTrue(hasattr(MultiAgentRolloutBufferSamples, field))
    
    def test_named_tuple_creation(self):
        """Test creating a MultiAgentRolloutBufferSamples instance."""
        observations = th.randn(10, 5)
        actions = th.randn(10, 4)
        old_values = th.randn(10, 1)
        old_log_prob = th.randn(10, 4)
        advantages = th.randn(10, 1)
        returns = th.randn(10, 1)
        episode_starts = th.randn(10, 1)
        states = th.randn(10, 20)
        
        samples = MultiAgentRolloutBufferSamples(
            observations, actions, old_values, old_log_prob,
            advantages, returns, episode_starts, states
        )
        
        self.assertEqual(samples.observations.shape, (10, 5))
        self.assertEqual(samples.actions.shape, (10, 4))
        self.assertEqual(samples.states.shape, (10, 20))


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2) 