"""Intention Propagation (IP MARL) implementation for 3v3 UAV combat.

This module implements Intention Propagation Multi-Agent Reinforcement Learning
for 3v3 UAV combat scenarios. The algorithm extends MAPPO with:

1. Intention modeling: Each agent predicts the intentions of teammates and opponents
2. Intention propagation: Intentions are shared and propagated through the team
3. Intention-aware policies: Actions are conditioned on predicted intentions
4. No explicit intention sharing: Agents act like PPO but with internal intention modeling

The algorithm is designed for 3v3 scenarios where blue team (learning agents) 
fights against red team (scripted opponents) without explicit intention sharing.
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

from uav_intent_rl.algo.mappo import MultiAgentRolloutBuffer, MAPPO, MAPPOPolicy

__all__ = [
    "IPMARLRolloutBuffer",
    "IPMARLRolloutBufferSamples",
    "IPMARLPolicy", 
    "IPMARL"
]


# Named tuple for IP MARL rollout samples
IPMARLRolloutBufferSamples = namedtuple(
    "IPMARLRolloutBufferSamples",
    [
        "observations",
        "actions", 
        "old_values",
        "old_log_prob",
        "advantages",
        "returns",
        "episode_starts",
        "states",
        "intentions",  # Predicted intentions for all agents
        "intention_targets"  # Ground truth intentions (if available)
    ]
)


class IPMARLRolloutBuffer(MultiAgentRolloutBuffer):
    """Rollout buffer for IP MARL with intention tracking."""
    
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        n_envs: int = 1,
        n_agents: int = 6,  # 3v3 = 6 agents
        state_space: Optional[spaces.Space] = None,
        intention_dim: int = 8,  # Dimension of intention vectors
        device: Union[th.device, str] = "cpu",
    ):
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            n_envs=n_envs,
            n_agents=n_agents,
            state_space=state_space,
            device=device,
        )
        
        self.intention_dim = intention_dim
        
        # Initialize intention buffers
        self.intentions = np.zeros(
            (self.buffer_size, self.n_envs, self.n_agents, self.intention_dim),
            dtype=np.float32
        )
        self.intention_targets = np.zeros(
            (self.buffer_size, self.n_envs, self.n_agents, self.intention_dim),
            dtype=np.float32
        )

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        episode_starts: np.ndarray,
        values: th.Tensor,
        log_probs: th.Tensor,
        states: Optional[np.ndarray] = None,
        intentions: Optional[np.ndarray] = None,
        intention_targets: Optional[np.ndarray] = None,
    ) -> None:
        """Add a transition to the buffer with intention information."""
        super().add(obs, actions, rewards, episode_starts, values, log_probs, states)
        
        if intentions is not None:
            self.intentions[self.pos - 1] = intentions
        if intention_targets is not None:
            self.intention_targets[self.pos - 1] = intention_targets

    def get(self, batch_size: Optional[int] = None) -> List[IPMARLRolloutBufferSamples]:
        """Get samples with intention information."""
        samples = super().get(batch_size)
        
        # Convert to IP MARL samples
        ip_marl_samples = []
        for sample in samples:
            # Get the indices for this batch
            batch_size_actual = len(sample.observations)
            
            # For now, create dummy intention data since we haven't collected it yet
            # In a real implementation, this would be the actual intention data
            dummy_intentions = th.zeros(
                (batch_size_actual, self.n_agents, self.intention_dim),
                device=self.device
            )
            dummy_intention_targets = th.zeros(
                (batch_size_actual, self.n_agents, self.intention_dim),
                device=self.device
            )
            
            ip_marl_sample = IPMARLRolloutBufferSamples(
                observations=sample.observations,
                actions=sample.actions,
                old_values=sample.old_values,
                old_log_prob=sample.old_log_prob,
                advantages=sample.advantages,
                returns=sample.returns,
                episode_starts=sample.episode_starts,
                states=sample.states,
                intentions=dummy_intentions,
                intention_targets=dummy_intention_targets,
            )
            ip_marl_samples.append(ip_marl_sample)
        
        return ip_marl_samples


class IntentionNet(nn.Module):
    """Neural network for predicting agent intentions."""
    
    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 128,
        intention_dim: int = 8,
        n_layers: int = 2,
    ):
        super().__init__()
        
        self.intention_dim = intention_dim
        
        # Build intention prediction network
        layers = []
        input_dim = obs_dim
        
        for i in range(n_layers):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, intention_dim))
        
        self.intention_net = nn.Sequential(*layers)
        
    def forward(self, obs: th.Tensor) -> th.Tensor:
        """Predict intentions from observations."""
        return self.intention_net(obs)


class IntentionPropagationNet(nn.Module):
    """Network for propagating intentions through the team."""
    
    def __init__(
        self,
        intention_dim: int,
        hidden_dim: int = 128,
        n_agents: int = 6,
    ):
        super().__init__()
        
        self.intention_dim = intention_dim
        self.n_agents = n_agents
        
        # Attention mechanism for intention propagation
        self.attention = nn.MultiheadAttention(
            embed_dim=intention_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Intention update network
        self.intention_update = nn.Sequential(
            nn.Linear(intention_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, intention_dim)
        )
        
    def forward(self, intentions: th.Tensor) -> th.Tensor:
        """Propagate intentions through attention mechanism.
        
        Args:
            intentions: Shape (batch_size, n_agents, intention_dim)
            
        Returns:
            Updated intentions: Shape (batch_size, n_agents, intention_dim)
        """
        # Apply self-attention to intentions
        attended_intentions, _ = self.attention(intentions, intentions, intentions)
        
        # Combine original and attended intentions
        combined = th.cat([intentions, attended_intentions], dim=-1)
        
        # Update intentions
        updated_intentions = intentions + self.intention_update(combined)
        
        return updated_intentions


class IPMARLPolicy(MAPPOPolicy):
    """Policy network with intention modeling for IP MARL."""
    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        n_agents: int = 6,
        use_centralized_critic: bool = True,
        state_shape: Optional[Tuple[int, ...]] = None,
        hidden_dim: int = 256,
        intention_dim: int = 8,
        intention_propagation: bool = True,
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
            
        super().__init__(
            single_observation_space,
            single_action_space,
            lr_schedule,
            n_agents=n_agents,
            use_centralized_critic=use_centralized_critic,
            state_shape=state_shape,
            hidden_dim=hidden_dim,
            **kwargs
        )
        
        self.intention_dim = intention_dim
        self.intention_propagation = intention_propagation
        
        # Get observation dimension
        if isinstance(observation_space, dict):
            obs_shape = next(iter(observation_space.values())).shape
        else:
            obs_shape = observation_space.shape

        if len(obs_shape) == 1:
            obs_dim = obs_shape[0]                    # classic 1-D Box
        elif len(obs_shape) == 2:                     # (n_agents, obs_dim)
            obs_dim = obs_shape[1]
        else:                                         # e.g. images
            obs_dim = int(np.prod(obs_shape[1:]))
        
        # Intention prediction network
        self.intention_net = IntentionNet(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim // 2,
            intention_dim=intention_dim
        )
        
        # Intention propagation network
        if intention_propagation:
            self.intention_propagation_net = IntentionPropagationNet(
                intention_dim=intention_dim,
                hidden_dim=hidden_dim // 2,
                n_agents=n_agents
            )
        
        # Intention-aware policy head - use single action space
        self.intention_policy_head = nn.Sequential(
            nn.Linear(self.mlp_extractor.latent_dim_pi + intention_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, single_action_space.shape[0])
        )
        
        # Intention-aware value head (if using centralized critic)
        if use_centralized_critic and state_shape is not None:
            state_dim = np.prod(state_shape)
            self.intention_value_head = nn.Sequential(
                nn.Linear(state_dim + intention_dim * n_agents, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )

    def predict_intentions(self, obs: th.Tensor) -> th.Tensor:
        """Predict intentions for all agents from observations."""
        batch_size = obs.shape[0]
        
        # Handle different observation shapes
        if obs.dim() == 2:  # Single agent: (batch_size, obs_dim)
            # Repeat the observation for all agents
            obs = obs.unsqueeze(1).repeat(1, self.n_agents, 1)  # (batch_size, n_agents, obs_dim)
        elif obs.dim() == 3 and obs.shape[1] == 1:  # (batch_size, 1, obs_dim) - single agent
            # Repeat the observation for all agents
            obs = obs.repeat(1, self.n_agents, 1)  # (batch_size, n_agents, obs_dim)
        
        # Predict intentions for each agent
        intentions = []
        for i in range(self.n_agents):
            agent_obs = obs[:, i] if obs.dim() > 2 else obs
            agent_intention = self.intention_net(agent_obs)
            intentions.append(agent_intention)
        
        # Stack intentions: (batch_size, n_agents, intention_dim)
        intentions = th.stack(intentions, dim=1)
        
        # Propagate intentions if enabled
        if self.intention_propagation:
            intentions = self.intention_propagation_net(intentions)
        
        return intentions

    def forward(
        self, 
        obs: th.Tensor, 
        states: Optional[th.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Forward pass with intention modeling."""
        # Predict intentions
        intentions = self.predict_intentions(obs)
        
        # Extract features
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        
        # ---- combine latent features with intentions ------------------
        # In every case we only want the **first agent's** intention when we
        # are acting for a single drone (blue_0 in the unit-tests).
        #
        # intentions.shape == (batch, n_agents, intention_dim)
        agent_intentions = intentions[:, 0]          # => (batch, intention_dim)
        policy_input     = th.cat([latent_pi, agent_intentions], dim=-1)
        
        # Get action logits
        action_logits = self.intention_policy_head(policy_input)
        
        # Get value
        if self.use_centralized_critic and states is not None:
            # Use centralized state with all intentions
            all_intentions = intentions.view(intentions.shape[0], -1)  # Flatten all intentions
            value_input = th.cat([states, all_intentions], dim=-1)
            value = self.intention_value_head(value_input)
        else:
            value = self.value_net(latent_vf)
        
        # Get action distribution
        actions = self.get_actions_from_logits(action_logits, deterministic)
        
        # Compute log-prob of the sampled actions
        log_prob, _ = self._log_prob_and_entropy(action_logits, actions)
        
        return actions, value, log_prob

    def evaluate_actions(
        self, 
        obs: th.Tensor, 
        actions: th.Tensor,
        states: Optional[th.Tensor] = None
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Evaluate actions with intention modeling."""
        # Predict intentions
        intentions = self.predict_intentions(obs)
        
        # Extract features
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        
        # ---- combine latent features with intentions ------------------
        # In every case we only want the **first agent's** intention when we
        # are acting for a single drone (blue_0 in the unit-tests).
        #
        # intentions.shape == (batch, n_agents, intention_dim)
        agent_intentions = intentions[:, 0]          # => (batch, intention_dim)
        policy_input     = th.cat([latent_pi, agent_intentions], dim=-1)
        
        # Get action logits
        action_logits = self.intention_policy_head(policy_input)
        
        # Get value
        if self.use_centralized_critic and states is not None:
            all_intentions = intentions.view(intentions.shape[0], -1)
            value_input = th.cat([states, all_intentions], dim=-1)
            value = self.intention_value_head(value_input)
        else:
            value = self.value_net(latent_vf)
        
        # Get log probs and entropy
        log_prob, entropy = self._log_prob_and_entropy(action_logits, actions)
        
        return value, log_prob, entropy

    def predict_values(
        self, 
        obs: th.Tensor,
        states: Optional[th.Tensor] = None
    ) -> th.Tensor:
        """Predict values with intention modeling."""
        # Predict intentions
        intentions = self.predict_intentions(obs)
        
        # Extract features
        features = self.extract_features(obs)
        _, latent_vf = self.mlp_extractor(features)
        
        # Get value
        if self.use_centralized_critic and states is not None:
            all_intentions = intentions.view(intentions.shape[0], -1)
            value_input = th.cat([states, all_intentions], dim=-1)
            value = self.intention_value_head(value_input)
        else:
            value = self.value_net(latent_vf)
        
        return value

    def get_actions_from_logits(
        self,
        mean_actions: th.Tensor,
        deterministic: bool = False
    ) -> th.Tensor:
        """
        Turn the mean action vector produced by `self.intention_policy_head`
        into actual actions *without* calling SB3's `action_net` a second time.
        """
        if deterministic:
            return mean_actions

        # Ensure log_std has the same shape as mean_actions
        log_std = self.log_std
        
        # Get the correct action dimension from mean_actions
        action_dim = mean_actions.shape[-1]
        
        # If log_std has wrong shape, create a new one with correct shape
        if log_std.shape[-1] != action_dim:
            # Create a new log_std parameter with correct action dimension
            new_log_std = th.zeros(action_dim, device=log_std.device, dtype=log_std.dtype)
            # Initialize with reasonable values (you might want to copy from existing if possible)
            new_log_std.data.fill_(-1.0)  # Start with small std
            log_std = new_log_std
        
        # Reshape log_std to match mean_actions shape
        if len(log_std.shape) == 1 and len(mean_actions.shape) == 2:
            # log_std is (action_dim,) and mean_actions is (batch_size, action_dim)
            log_std = log_std.unsqueeze(0).expand_as(mean_actions)
        elif len(log_std.shape) == 1 and len(mean_actions.shape) == 3:
            # log_std is (action_dim,) and mean_actions is (batch_size, n_agents, action_dim)
            log_std = log_std.unsqueeze(0).unsqueeze(0).expand_as(mean_actions)
        
        std = th.exp(log_std)
        noise = th.randn_like(mean_actions)
        return mean_actions + noise * std

    def _log_prob_and_entropy(self, mean, actions):
        """Compute log probability and entropy for actions."""
        log_std = self.log_std
        
        # Get the correct action dimension from mean
        action_dim = mean.shape[-1]
        
        # If log_std has wrong shape, create a new one with correct shape
        if log_std.shape[-1] != action_dim:
            # Create a new log_std parameter with correct action dimension
            new_log_std = th.zeros(action_dim, device=log_std.device, dtype=log_std.dtype)
            # Initialize with reasonable values
            new_log_std.data.fill_(-1.0)  # Start with small std
            log_std = new_log_std
        
        # Ensure log_std has the same shape as mean and actions
        if log_std.shape != mean.shape:
            # Reshape log_std to match mean shape
            if len(log_std.shape) == 1 and len(mean.shape) == 2:
                # log_std is (action_dim,) and mean is (batch_size, action_dim)
                log_std = log_std.unsqueeze(0).expand_as(mean)
            elif len(log_std.shape) == 1 and len(mean.shape) == 3:
                # log_std is (action_dim,) and mean is (batch_size, n_agents, action_dim)
                log_std = log_std.unsqueeze(0).unsqueeze(0).expand_as(mean)
        
        std = th.exp(log_std)
        dist = th.distributions.Normal(mean, std)
        logp = dist.log_prob(actions).sum(-1, keepdim=False)  # Remove keepdim=True to avoid broadcasting issues
        ent = dist.entropy().sum(-1, keepdim=False)  # Remove keepdim=True to avoid broadcasting issues
        return logp, ent


class IPMARL(MAPPO):
    """Intention Propagation Multi-Agent Reinforcement Learning algorithm."""
    
    # Class attributes for configuration
    n_agents: int = 6
    use_centralized_critic: bool = True
    state_shape: Optional[Tuple[int, ...]] = None
    intention_dim: int = 8
    intention_propagation: bool = True
    intention_loss_coef: float = 0.1
    rollout_buffer_class = IPMARLRolloutBuffer
    
    def __init__(
        self,
        policy: Union[str, type],
        env: Union[GymEnv, str, None] = None,
        n_agents: int = 6,
        use_centralized_critic: bool = True,
        state_shape: Optional[Tuple[int, ...]] = None,
        intention_dim: int = 8,
        intention_propagation: bool = True,
        intention_loss_coef: float = 0.1,
        **kwargs: Any,
    ):
        # Set class attributes
        self.n_agents = n_agents
        self.use_centralized_critic = use_centralized_critic
        self.state_shape = state_shape
        self.intention_dim = intention_dim
        self.intention_propagation = intention_propagation
        self.intention_loss_coef = intention_loss_coef
        
        super().__init__(
            policy=policy,
            env=env,
            n_agents=n_agents,
            use_centralized_critic=use_centralized_critic,
            state_shape=state_shape,
            **kwargs
        )
        
        # Override rollout buffer with IP MARL version if it was created
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
                if 'intention_dim' in sig.parameters:
                    buffer_kwargs['intention_dim'] = intention_dim
            
            self.rollout_buffer = self.rollout_buffer_class(**buffer_kwargs)

    def collect_rollouts(
        self,
        env: GymEnv,
        callback: Any,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """Collect rollouts with intention tracking."""
        # Override to collect intention information
        return super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)

    def train(self) -> None:
        """Train with intention modeling loss."""
        # Get rollout samples
        rollout_buffer = self.rollout_buffer
        samples = rollout_buffer.get()
        
        # Prepare for training
        self.policy.set_training_mode(True)
        
        # Training loop
        for sample in samples:
            # Check if this is an IP MARL sample or regular sample
            if hasattr(sample, 'intentions'):
                # This is an IP MARL sample with intention modeling
                ppo_loss = self._compute_ppo_loss(sample)
                intention_loss = self._compute_intention_loss(sample)
                total_loss = ppo_loss + self.intention_loss_coef * intention_loss
            else:
                # This is a regular sample, just compute PPO loss
                ppo_loss = self._compute_ppo_loss(sample)
                total_loss = ppo_loss
            
            # Backward pass
            self.policy.optimizer.zero_grad()
            total_loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
            
            # Update learning rate
            if self.lr_schedule == "linear":
                self.policy.optimizer.param_groups[0]["lr"] = self.learning_rate

    def _compute_ppo_loss(self, sample: IPMARLRolloutBufferSamples) -> th.Tensor:
        """Compute standard PPO loss."""
        # Handle observations and actions that might already be tensors
        if isinstance(sample.observations, th.Tensor):
            obs = sample.observations.to(self.device)
        else:
            obs = obs_as_tensor(sample.observations, self.device)
            
        if isinstance(sample.actions, th.Tensor):
            actions = sample.actions.to(self.device)
        else:
            actions = obs_as_tensor(sample.actions, self.device)
            
        old_values = sample.old_values
        old_log_prob = sample.old_log_prob
        advantages = sample.advantages
        returns = sample.returns
        
        # Get current policy outputs
        values, log_prob, entropy = self.policy.evaluate_actions(obs, actions)
        
        # Get current clip range value
        clip_range = self.clip_range(self._current_progress_remaining) if hasattr(self.clip_range, '__call__') else self.clip_range
        
        # Compute PPO loss components
        ratio = th.exp(log_prob - old_log_prob)
        surr1 = ratio * advantages
        surr2 = th.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages
        policy_loss = -th.min(surr1, surr2).mean()
        
        # Value loss - ensure both tensors have the same shape
        if values.shape != returns.shape:
            # Squeeze values if it has an extra dimension
            if values.shape[-1] == 1 and len(values.shape) > len(returns.shape):
                values = values.squeeze(-1)
            # Unsqueeze returns if it needs an extra dimension
            elif returns.shape[-1] == 1 and len(returns.shape) > len(values.shape):
                returns = returns.squeeze(-1)
        value_loss = F.mse_loss(values, returns)
        
        # Entropy loss
        entropy_loss = -entropy.mean()
        
        # Total PPO loss
        ppo_loss = (
            policy_loss 
            + self.vf_coef * value_loss 
            + self.ent_coef * entropy_loss
        )
        
        return ppo_loss

    def _compute_intention_loss(self, sample: IPMARLRolloutBufferSamples) -> th.Tensor:
        """Compute intention modeling loss."""
        # Handle observations that might already be tensors
        if isinstance(sample.observations, th.Tensor):
            obs = sample.observations.to(self.device)
        else:
            obs = obs_as_tensor(sample.observations, self.device)
            
        intentions = sample.intentions
        intention_targets = sample.intention_targets
        
        # Predict intentions
        predicted_intentions = self.policy.predict_intentions(obs)
        
        # Compute MSE loss between predicted and target intentions
        intention_loss = F.mse_loss(predicted_intentions, intention_targets)
        
        return intention_loss

    def learn(
        self,
        total_timesteps: int,
        callback: Any = None,
        log_interval: int = 1,
        tb_log_name: str = "IPMARL",
        reset_num_timesteps: bool = True,
    ) -> "IPMARL":
        """Learn with intention propagation."""
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
        ) 