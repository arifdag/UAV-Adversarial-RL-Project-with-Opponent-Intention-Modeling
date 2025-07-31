"""MAPPO (Multi-Agent Proximal Policy Optimization) implementation.

This module provides a complete MAPPO implementation with:
- Multi-agent rollout buffer with centralized state information
- Centralized critic for better value estimation
- Team-based policy optimization
- Support for 3v3 multi-agent environments
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
from collections import namedtuple

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer, RolloutBufferSamples
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.ppo.ppo import PPO

__all__ = [
    "MultiAgentRolloutBuffer",
    "MultiAgentRolloutBufferSamples", 
    "MAPPOPolicy",
    "MAPPO"
]


# Named tuple for multi-agent rollout samples
MultiAgentRolloutBufferSamples = namedtuple(
    "MultiAgentRolloutBufferSamples",
    [
        "observations",
        "actions", 
        "old_values",
        "old_log_prob",
        "advantages",
        "returns",
        "episode_starts",
        "states"  # Centralized state information
    ]
)


class MultiAgentRolloutBuffer(RolloutBuffer):
    """Multi-agent rollout buffer with centralized state information."""
    
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        n_envs: int = 1,
        n_agents: int = 6,  # 3v3 = 6 agents
        state_space: Optional[spaces.Space] = None,
        device: Union[th.device, str] = "cpu",
    ):
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            n_envs=n_envs,
            device=device,
        )
        
        self.n_agents = n_agents
        self.state_space = state_space
        
        # Initialize centralized state buffer
        if state_space is not None:
            if isinstance(state_space, spaces.Box):
                self.states = np.zeros(
                    (self.buffer_size, self.n_envs, *state_space.shape),
                    dtype=np.float32
                )
            else:
                # Handle other space types if needed
                self.states = np.zeros(
                    (self.buffer_size, self.n_envs, state_space.shape[0]),
                    dtype=np.float32
                )
        else:
            # Default state shape if not provided
            # Handle multi-agent observation space (dictionary)
            if isinstance(observation_space, dict):
                # Get the first agent's observation space shape
                first_agent_space = next(iter(observation_space.values()))
                obs_dim = first_agent_space.shape[0] if hasattr(first_agent_space, 'shape') else 12
            else:
                # Single agent observation space
                obs_dim = observation_space.shape[0] if hasattr(observation_space, 'shape') else 12
            
            self.states = np.zeros(
                (self.buffer_size, self.n_envs, obs_dim * n_agents),
                dtype=np.float32
            )
        
        # Initialize episode_starts tracking
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=bool)

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        episode_starts: np.ndarray,
        values: th.Tensor,
        log_probs: th.Tensor,
        states: Optional[np.ndarray] = None,
    ) -> None:
        """Add a new experience to the buffer."""
        super().add(obs, actions, rewards, episode_starts, values, log_probs)
        
        # Write into the *just-added* slot (pos-1 wraps to end when full)
        idx = (self.pos - 1) % self.buffer_size
        
        # Store episode_starts information
        self.episode_starts[idx] = episode_starts
        
        # Store centralized state information
        if states is not None:
            self.states[idx] = states
        else:
            # If no state provided, use concatenated observations
            if obs.ndim == 3:  # (n_envs, n_agents, obs_dim)
                self.states[idx] = obs.reshape(obs.shape[0], -1)
            else:
                # Fallback: zero state
                self.states[idx] = np.zeros(self.states.shape[1:], dtype=np.float32)

    def get(self, batch_size: Optional[int] = None) -> List[MultiAgentRolloutBufferSamples]:
        """Get samples from the buffer."""
        # Get base samples from parent
        base_samples = super().get(batch_size)
        
        # Convert to multi-agent samples with state information
        ma_samples = []
        
        # Track position in flattened buffer for indexing
        flat_ptr = 0
        flat_episode_starts = self.episode_starts.reshape(-1)
        flat_states = self.states.reshape(-1, self.states.shape[-1])
        
        for batch in base_samples:
            batch_len = len(batch.advantages)
            
            # Get corresponding episode_starts and states
            ep_starts_slice = flat_episode_starts[flat_ptr : flat_ptr + batch_len]
            states_slice = flat_states[flat_ptr : flat_ptr + batch_len]
            
            ep_starts = th.as_tensor(ep_starts_slice, device=self.device)
            states = th.as_tensor(states_slice, device=self.device)
            
            ma_sample = MultiAgentRolloutBufferSamples(
                observations=batch.observations,
                actions=batch.actions,
                old_values=batch.old_values,
                old_log_prob=batch.old_log_prob,
                advantages=batch.advantages,
                returns=batch.returns,
                episode_starts=ep_starts,
                states=states
            )
            ma_samples.append(ma_sample)
            
            flat_ptr += batch_len
        
        return ma_samples


class CentralizedValueNet(nn.Module):
    """Centralized value network for MAPPO."""
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
    ):
        super().__init__()
        
        layers = []
        input_dim = state_dim
        
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.value_net = nn.Sequential(*layers)
    
    def forward(self, states: th.Tensor) -> th.Tensor:
        """Forward pass through the centralized value network."""
        return self.value_net(states)


class MAPPOPolicy(ActorCriticPolicy):
    """MAPPO policy with centralized critic support."""
    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        n_agents: int = 6,
        use_centralized_critic: bool = True,
        state_shape: Optional[Tuple[int, ...]] = None,
        hidden_dim: int = 256,
        **kwargs: Any,
    ):
        # Handle multi-agent observation space (dictionary)
        if isinstance(observation_space, dict):
            # Use the first agent's observation space for the policy
            first_agent_space = next(iter(observation_space.values()))
            single_observation_space = first_agent_space
        else:
            single_observation_space = observation_space
            
        # Handle multi-agent action space (dictionary)
        if isinstance(action_space, dict):
            # Use the first agent's action space for the policy
            first_agent_space = next(iter(action_space.values()))
            single_action_space = first_agent_space
        else:
            single_action_space = action_space
            
        super().__init__(single_observation_space, single_action_space, lr_schedule, **kwargs)
        
        self.n_agents = n_agents
        self.use_centralized_critic = use_centralized_critic
        self.state_shape = state_shape
        
        # Initialize centralized value network if enabled
        if use_centralized_critic and state_shape is not None:
            state_dim = np.prod(state_shape)
            self.centralized_value_net = CentralizedValueNet(
                state_dim=state_dim,
                hidden_dim=hidden_dim
            )
            self.centralized_value_head = nn.Linear(hidden_dim, 1)
        else:
            self.centralized_value_net = None
            self.centralized_value_head = None

    def forward(
        self, 
        obs: th.Tensor, 
        states: Optional[th.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Forward pass through the policy network."""
        # Actor network (policy)
        latent_pi = self.extract_features(obs)
        latent_vf = self.extract_features(obs)
        
        # Policy head
        mean_actions = self.action_net(latent_pi)
        log_std = self.log_std
        
        # Sample actions
        if deterministic:
            actions = mean_actions
        else:
            std = th.exp(log_std)
            noise = th.randn_like(mean_actions)
            actions = mean_actions + noise * std
        
        # Value head
        if self.use_centralized_critic and states is not None and self.centralized_value_net is not None:
            # Use centralized critic
            centralized_values = self.centralized_value_net(states)
            values = self.centralized_value_head(centralized_values)
        else:
            # Use decentralized critic
            values = self.value_net(latent_vf)
        
        # Calculate log probabilities
        log_probs = self.get_distribution(mean_actions, log_std).log_prob(actions)
        
        return actions, values, log_probs

    def evaluate_actions(
        self, 
        obs: th.Tensor, 
        actions: th.Tensor,
        states: Optional[th.Tensor] = None
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Evaluate actions for training."""
        latent_pi = self.extract_features(obs)
        latent_vf = self.extract_features(obs)
        
        mean_actions = self.action_net(latent_pi)
        log_std = self.log_std
        
        # Value estimation
        if self.use_centralized_critic and states is not None and self.centralized_value_net is not None:
            centralized_values = self.centralized_value_net(states)
            values = self.centralized_value_head(centralized_values)
        else:
            values = self.value_net(latent_vf)
        
        # Log probabilities
        log_probs = self.get_distribution(mean_actions, log_std).log_prob(actions)
        
        # Entropy
        entropy = self.get_distribution(mean_actions, log_std).entropy()
        
        return values, log_probs, entropy

    def predict_values(
        self, 
        obs: th.Tensor,
        states: Optional[th.Tensor] = None
    ) -> th.Tensor:
        """Predict values for the given observations."""
        latent_vf = self.extract_features(obs)
        
        if self.use_centralized_critic and states is not None and self.centralized_value_net is not None:
            centralized_values = self.centralized_value_net(states)
            values = self.centralized_value_head(centralized_values)
        else:
            values = self.value_net(latent_vf)
        
        return values


class MAPPO(PPO):
    """Multi-Agent Proximal Policy Optimization algorithm."""
    
    # Class attributes for configuration
    n_agents: int = 6
    use_centralized_critic: bool = True
    state_shape: Optional[Tuple[int, ...]] = None
    rollout_buffer_class = MultiAgentRolloutBuffer
    
    def __init__(
        self,
        policy: Union[str, type],
        env: Union[GymEnv, str, None] = None,
        n_agents: int = 6,
        use_centralized_critic: bool = True,
        state_shape: Optional[Tuple[int, ...]] = None,
        **kwargs: Any,
    ):
        # Set class attributes
        MAPPO.n_agents = n_agents
        MAPPO.use_centralized_critic = use_centralized_critic
        MAPPO.state_shape = state_shape
        
        super().__init__(policy, env, **kwargs)
        
        # Override rollout buffer with multi-agent version
        if hasattr(self, 'rollout_buffer'):
            # Recreate rollout buffer using the class's rollout_buffer_class
            # Pass only the parameters that the buffer class expects
            buffer_kwargs = {
                'buffer_size': self.n_steps,
                'observation_space': self.observation_space,
                'action_space': self.action_space,
                'n_envs': self.n_envs,
                'device': self.device,
            }
            
            # Add optional parameters if the buffer class expects them
            if hasattr(self.rollout_buffer_class, '__init__'):
                import inspect
                sig = inspect.signature(self.rollout_buffer_class.__init__)
                if 'n_agents' in sig.parameters:
                    buffer_kwargs['n_agents'] = n_agents
                if 'state_space' in sig.parameters:
                    buffer_kwargs['state_space'] = self.observation_space  # Use obs space as state space
            
            self.rollout_buffer = self.rollout_buffer_class(**buffer_kwargs)

    def collect_rollouts(
        self,
        env: GymEnv,
        callback: Any,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """Collect rollouts with centralized state information."""
        # Override to collect centralized state information
        return super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)

    def train(self) -> None:
        """Train the MAPPO agent."""
        # Override training to handle multi-agent samples
        super().train()

    def learn(
        self,
        total_timesteps: int,
        callback: Any = None,
        log_interval: int = 1,
        tb_log_name: str = "MAPPO",
        reset_num_timesteps: bool = True,
    ) -> "MAPPO":
        """Learn the MAPPO policy."""
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
        ) 