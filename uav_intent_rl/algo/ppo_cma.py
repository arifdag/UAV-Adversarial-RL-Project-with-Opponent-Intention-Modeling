from __future__ import annotations

"""PPO-CMA – PPO with Covariance Matrix Adaptation for adaptive exploration.

This algorithm extends PPO with an adaptive variance strategy inspired by CMA-ES
(Covariance Matrix Adaptation Evolution Strategy). The key innovation is that
the action noise (exploration variance) is automatically adjusted based on
performance feedback, preventing premature convergence to local optima.

Key features:
1. Adaptive variance scaling based on reward improvement
2. Covariance matrix adaptation for action correlations
3. Automatic exploration expansion when stuck in local optima
4. Variance contraction when making good progress
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.ppo.ppo import PPO

__all__ = ["PPOCMA"]


class PPOCMA(PPO):
    """PPO with Covariance Matrix Adaptation for adaptive exploration."""

    def __init__(
        self,
        policy: str | type,
        env: GymEnv | str | None = None,
        # CMA-specific parameters
        cma_learning_rate: float = 0.1,
        cma_memory_size: int = 100,
        cma_min_variance: float = 0.01,
        cma_max_variance: float = 2.0,
        cma_variance_decay: float = 0.95,
        cma_variance_expansion: float = 1.05,
        cma_performance_threshold: float = 0.1,
        **kwargs: Any,
    ) -> None:
        # Initialize CMA parameters
        self.cma_lr = float(cma_learning_rate)
        self.cma_memory_size = int(cma_memory_size)
        self.cma_min_variance = float(cma_min_variance)
        self.cma_max_variance = float(cma_max_variance)
        self.cma_variance_decay = float(cma_variance_decay)
        self.cma_variance_expansion = float(cma_variance_expansion)
        self.cma_performance_threshold = float(cma_performance_threshold)
        
        # Initialize CMA state
        self.cma_reward_history = []
        self.cma_variance_scale = 1.0
        self.cma_covariance_matrix = None
        self.cma_action_dim = None
        
        super().__init__(policy, env, **kwargs)

    def _initialize_cma(self, action_space: spaces.Space) -> None:
        """Initialize CMA parameters based on action space."""
        if isinstance(action_space, spaces.Box):
            self.cma_action_dim = action_space.shape[0]
            # Initialize covariance matrix as identity
            self.cma_covariance_matrix = np.eye(self.cma_action_dim)
        else:
            # For discrete actions, use scalar variance
            self.cma_action_dim = 1
            self.cma_covariance_matrix = np.eye(1)

    def _update_cma_variance(self, episode_rewards: list[float]) -> None:
        """Update CMA variance based on recent performance."""
        if len(episode_rewards) < 2:
            return
            
        # Calculate performance improvement
        recent_rewards = episode_rewards[-self.cma_memory_size:]
        if len(recent_rewards) < 2:
            return
            
        # Compute moving average improvement
        old_avg = np.mean(recent_rewards[:-1])
        new_avg = np.mean(recent_rewards[1:])
        improvement = new_avg - old_avg
        
        # Update variance scale based on performance
        if improvement >= self.cma_performance_threshold:
            # Good progress: contract variance
            self.cma_variance_scale *= self.cma_variance_decay
        elif improvement <= -self.cma_performance_threshold:
            # Poor progress: expand variance
            self.cma_variance_scale *= self.cma_variance_expansion
        
        # Clamp variance scale to bounds
        self.cma_variance_scale = np.clip(
            self.cma_variance_scale,
            self.cma_min_variance,
            self.cma_max_variance
        )
        
        # Update covariance matrix
        self.cma_covariance_matrix = self.cma_variance_scale * np.eye(self.cma_action_dim)

    def _apply_cma_noise(self, actions: np.ndarray) -> np.ndarray:
        """Apply CMA-adapted noise to actions."""
        if self.cma_covariance_matrix is None:
            return actions
            
        # Generate correlated noise using Cholesky decomposition
        try:
            L = np.linalg.cholesky(self.cma_covariance_matrix)
            noise = np.random.normal(0, 1, actions.shape)
            correlated_noise = noise @ L.T
        except np.linalg.LinAlgError:
            # Fallback to diagonal noise if covariance is not positive definite
            correlated_noise = np.random.normal(0, np.sqrt(self.cma_variance_scale), actions.shape)
        
        # Apply noise to actions
        noisy_actions = actions + correlated_noise
        
        # Clip to action space bounds
        if isinstance(self.action_space, spaces.Box):
            noisy_actions = np.clip(
                noisy_actions,
                self.action_space.low,
                self.action_space.high
            )
        
        return noisy_actions

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):  # type: ignore[override]
        """Collect rollouts with CMA-adapted exploration."""
        # Initialize CMA if not done yet
        if self.cma_covariance_matrix is None:
            self._initialize_cma(self.action_space)
        
        # Collect episode rewards for CMA update
        episode_rewards = []
        
        # Call parent method but intercept actions for CMA noise
        original_step = env.step
        
        def cma_step(actions):
            # Apply CMA noise to actions
            noisy_actions = self._apply_cma_noise(actions)
            return original_step(noisy_actions)
        
        # Temporarily replace env.step
        env.step = cma_step
        
        try:
            # Call parent collect_rollouts
            result = super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)
            
            # Update CMA based on recent performance
            if hasattr(self, '_last_episode_rewards'):
                episode_rewards.extend(self._last_episode_rewards)
                self._update_cma_variance(episode_rewards)
            
            return result
        finally:
            # Restore original step function
            env.step = original_step

    def train(self) -> None:  # type: ignore[override]
        """Train with CMA logging."""
        # Call parent training
        super().train()
        
        # Log CMA-specific metrics
        if self.cma_covariance_matrix is not None:
            self.logger.record("train/cma_variance_scale", self.cma_variance_scale)
            self.logger.record("train/cma_entropy", np.log(np.linalg.det(self.cma_covariance_matrix)))
            
            # Log variance for each action dimension
            variances = np.diag(self.cma_covariance_matrix)
            for i, var in enumerate(variances):
                self.logger.record(f"train/cma_variance_dim_{i}", var)

    def _setup_model(self) -> None:
        """Setup model with CMA initialization."""
        super()._setup_model()
        
        # Initialize CMA after model is created
        if hasattr(self, 'action_space'):
            self._initialize_cma(self.action_space)

    def save(self, path: str, include: Optional[list[str]] = None, exclude: Optional[list[str]] = None) -> None:
        """Save model with CMA state."""
        # Save CMA state
        cma_state = {
            'cma_variance_scale': self.cma_variance_scale,
            'cma_covariance_matrix': self.cma_covariance_matrix,
            'cma_reward_history': self.cma_reward_history,
        }
        
        # Temporarily attach CMA state to model
        self.cma_state = cma_state
        
        try:
            super().save(path, include, exclude)
        finally:
            # Clean up
            delattr(self, 'cma_state')

    @classmethod
    def load(cls, path: str, env: Optional[GymEnv] = None, device: str = "auto", custom_objects: Optional[Dict[str, Any]] = None, **kwargs: Any) -> "PPOCMA":
        """Load model with CMA state."""
        # Load base model
        model = super(PPOCMA, cls).load(path, env, device, custom_objects, **kwargs)
        # Restore CMA state if available
        if hasattr(model, 'cma_state'):
            model.cma_variance_scale = model.cma_state['cma_variance_scale']
            model.cma_covariance_matrix = model.cma_state['cma_covariance_matrix']
            model.cma_reward_history = model.cma_state['cma_reward_history']
            delattr(model, 'cma_state')
        return model 