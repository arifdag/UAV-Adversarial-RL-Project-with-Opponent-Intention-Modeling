"""Multi-agent vector environment wrapper for 3v3 dogfight."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.util import obs_space_info

from uav_intent_rl.envs.Dogfight3v3MultiAgentEnv import Dogfight3v3MultiAgentEnv


class Dogfight3v3MultiAgentVecEnv(VecEnv):
    """Vector environment wrapper for 3v3 multi-agent dogfight."""
    
    def __init__(self, env_fns: List[callable]):
        """Initialize the vector environment.
        
        Args:
            env_fns: List of environment creation functions
        """
        self.envs = [env_fn() for env_fn in env_fns]
        self.num_envs = len(self.envs)
        
        # Get observation and action spaces from first environment
        first_env = self.envs[0]
        self.observation_space = first_env.observation_space
        self.action_space = first_env.action_space
        
        # Get agent IDs
        self.agent_ids = list(first_env.observation_space.keys())
        self.num_agents = len(self.agent_ids)
        
        # Initialize observation and action buffers
        self._obs_buffer = None
        self._action_buffer = None
        
    def reset(self) -> Dict[str, np.ndarray]:
        """Reset all environments.
        
        Returns:
            Dictionary of observations for each agent
        """
        obs_list = []
        for env in self.envs:
            obs = env.reset()
            obs_list.append(obs)
        
        # Stack observations across environments
        stacked_obs = {}
        for agent_id in self.agent_ids:
            agent_obs = np.stack([obs[agent_id] for obs in obs_list])
            stacked_obs[agent_id] = agent_obs
        
        return stacked_obs
    
    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], 
                                                            Dict[str, np.ndarray], 
                                                            Dict[str, np.ndarray], 
                                                            List[Dict[str, Any]]]:
        """Step all environments.
        
        Args:
            actions: Dictionary of actions for each agent
            
        Returns:
            Tuple of (observations, rewards, dones, infos)
        """
        # Prepare actions for each environment
        env_actions = []
        for env_idx in range(self.num_envs):
            env_action = {}
            for agent_id in self.agent_ids:
                env_action[agent_id] = actions[agent_id][env_idx]
            env_actions.append(env_action)
        
        # Step each environment
        obs_list = []
        reward_list = []
        done_list = []
        info_list = []
        
        for env_idx, env in enumerate(self.envs):
            obs, rewards, dones, infos = env.step(env_actions[env_idx])
            obs_list.append(obs)
            reward_list.append(rewards)
            done_list.append(dones)
            info_list.append(infos)
        
        # Stack results across environments
        stacked_obs = {}
        stacked_rewards = {}
        stacked_dones = {}
        
        for agent_id in self.agent_ids:
            stacked_obs[agent_id] = np.stack([obs[agent_id] for obs in obs_list])
            stacked_rewards[agent_id] = np.stack([rewards[agent_id] for rewards in reward_list])
            stacked_dones[agent_id] = np.stack([dones[agent_id] for dones in done_list])
        
        return stacked_obs, stacked_rewards, stacked_dones, info_list
    
    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()
    
    def get_attr(self, attr_name: str, indices: Optional[List[int]] = None) -> List[Any]:
        """Get attribute from environments."""
        if indices is None:
            indices = list(range(self.num_envs))
        return [getattr(self.envs[i], attr_name) for i in indices]
    
    def set_attr(self, attr_name: str, value: Any, indices: Optional[List[int]] = None):
        """Set attribute in environments."""
        if indices is None:
            indices = list(range(self.num_envs))
        for i in indices:
            setattr(self.envs[i], attr_name, value)
    
    def env_method(self, method_name: str, *method_args, indices: Optional[List[int]] = None, **method_kwargs) -> List[Any]:
        """Call method on environments."""
        if indices is None:
            indices = list(range(self.num_envs))
        return [getattr(self.envs[i], method_name)(*method_args, **method_kwargs) for i in indices]
    
    def env_is_wrapped(self, wrapper_class) -> List[bool]:
        """Check if environments are wrapped with a specific wrapper."""
        return [hasattr(env, 'env') and isinstance(env.env, wrapper_class) for env in self.envs]
    
    def step_async(self, actions: Dict[str, np.ndarray]):
        """Step environments asynchronously (not implemented for this wrapper)."""
        raise NotImplementedError("Async stepping not implemented for this wrapper")
    
    def step_wait(self) -> Tuple[Dict[str, np.ndarray], 
                                Dict[str, np.ndarray], 
                                Dict[str, np.ndarray], 
                                List[Dict[str, Any]]]:
        """Wait for async step to complete (not implemented for this wrapper)."""
        raise NotImplementedError("Async stepping not implemented for this wrapper")


def make_mappo_vec_env(
    env_config: Optional[Dict[str, Any]] = None,
    n_envs: int = 1,
    seed: Optional[int] = None,
) -> Dogfight3v3MultiAgentVecEnv:
    """Create a multi-agent vector environment for MAPPO training.
    
    Args:
        env_config: Environment configuration
        n_envs: Number of parallel environments
        seed: Random seed
        
    Returns:
        Multi-agent vector environment
    """
    if env_config is None:
        env_config = {}
    
    def make_env():
        return Dogfight3v3MultiAgentEnv(env_config)
    
    env_fns = [make_env for _ in range(n_envs)]
    return Dogfight3v3MultiAgentVecEnv(env_fns) 