#!/usr/bin/env python3
"""Training script for Intention Propagation (IP MARL) for 3v3 UAV combat.

This script trains an IP MARL model for 3v3 UAV combat scenarios where
the blue team (learning agents) fights against the red team (scripted opponents).
The agents don't share intentions explicitly but model them internally.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch as th
import yaml
import gymnasium as gym
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
    BaseCallback,
)
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from uav_intent_rl.envs.Dogfight3v3MultiAgentEnv import Dogfight3v3MultiAgentEnv
from uav_intent_rl.policies.team_scripted_red import TeamScriptedRedPolicy
from uav_intent_rl.algo.ip_marl import IPMARL, IPMARLPolicy


class IPMARLMetricsCallback(BaseCallback):
    """Custom callback to log IP MARL specific metrics."""
    
    def __init__(
        self,
        eval_env: Optional[VecEnv] = None,
        eval_freq: int = 20000,
        n_eval_episodes: int = 10,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_reward = -np.inf
        
    def _on_step(self) -> bool:
        """Called after each step."""
        if self.n_calls % self.eval_freq == 0:
            self._evaluate_and_log_metrics()
        return True
    
    def _evaluate_and_log_metrics(self):
        """Evaluate the model and log detailed metrics."""
        if self.eval_env is None:
            return
            
        # Run evaluation episodes
        episode_rewards = []
        episode_lengths = []
        blue_wins = 0
        red_wins = 0
        draws = 0
        blue_hits = 0
        red_hits = 0
        team_coordination_scores = []
        formation_quality_scores = []
        intention_accuracy_scores = []
        
        for episode in range(self.n_eval_episodes):
            try:
                obs = self.eval_env.reset()
                episode_reward = 0
                episode_length = 0
                episode_blue_hits = 0
                episode_red_hits = 0
                
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, info = self.eval_env.step(action)
                    
                    # Handle reward properly
                    if isinstance(reward, (list, np.ndarray)):
                        episode_reward += reward[0] if len(reward) > 0 else 0
                    else:
                        episode_reward += reward
                    
                    episode_length += 1
                    
                    # Extract metrics from info if available
                    if isinstance(info, dict) and len(info) > 0:
                        first_info = list(info.values())[0] if isinstance(info, dict) else info
                        if isinstance(first_info, dict):
                            episode_blue_hits += first_info.get('blue_hits', 0)
                            episode_red_hits += first_info.get('red_hits', 0)
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                blue_hits += episode_blue_hits
                red_hits += episode_red_hits
                
                # Determine winner based on actual team elimination status
                # Get final info to check team status
                final_info = None
                if isinstance(info, dict) and len(info) > 0:
                    final_info = list(info.values())[0] if isinstance(info, dict) else info
                elif isinstance(info, list) and len(info) > 0:
                    final_info = info[0]
                
                # Check if episode was truncated (time ran out) vs terminated (team eliminated)
                episode_truncated = False
                if final_info and isinstance(final_info, dict):
                    # Check if episode was truncated due to time limit
                    episode_truncated = final_info.get('episode_truncated', False)
                    blue_team_down = final_info.get('blue_team_down', False)
                    red_team_down = final_info.get('red_team_down', False)
                    
                    # Determine winner based on team elimination
                    if red_team_down and not blue_team_down:
                        blue_wins += 1
                    elif blue_team_down and not red_team_down:
                        red_wins += 1
                    elif episode_truncated and not blue_team_down and not red_team_down:
                        # Draw: time ran out and neither team was eliminated
                        draws += 1
                    else:
                        # Fallback: use reward-based determination
                        if episode_reward > 0:
                            blue_wins += 1
                        elif episode_reward < 0:
                            red_wins += 1
                        else:
                            draws += 1
                else:
                    # Fallback: use reward-based determination
                    if episode_reward > 0:
                        blue_wins += 1
                    elif episode_reward < 0:
                        red_wins += 1
                    else:
                        draws += 1
                
                # Calculate team coordination and formation quality (simplified)
                team_coordination = max(0, episode_reward / 100)  # Normalized coordination score
                formation_quality = max(0, (episode_length - 100) / 500)  # Formation quality based on episode length
                intention_accuracy = max(0, episode_reward / 50)  # Intention accuracy based on performance
                
                team_coordination_scores.append(team_coordination)
                formation_quality_scores.append(formation_quality)
                intention_accuracy_scores.append(intention_accuracy)
                
            except Exception as e:
                print(f"Error in evaluation episode {episode}: {e}")
                # Add default values for this episode
                episode_rewards.append(0.0)
                episode_lengths.append(1)
                team_coordination_scores.append(0.0)
                formation_quality_scores.append(0.0)
                intention_accuracy_scores.append(0.0)
        
        # Calculate metrics
        mean_reward = np.mean(episode_rewards)
        mean_length = np.mean(episode_lengths)
        win_rate = blue_wins / self.n_eval_episodes
        blue_win_rate = blue_wins / self.n_eval_episodes
        red_win_rate = red_wins / self.n_eval_episodes
        draw_rate = draws / self.n_eval_episodes
        mean_blue_hits = blue_hits / self.n_eval_episodes
        mean_red_hits = red_hits / self.n_eval_episodes
        mean_team_coordination = np.mean(team_coordination_scores)
        mean_formation_quality = np.mean(formation_quality_scores)
        mean_intention_accuracy = np.mean(intention_accuracy_scores)
        
        # Log metrics
        if self.logger is not None:
            self.logger.record("eval/mean_ep_reward", mean_reward)
            self.logger.record("eval/mean_ep_length", mean_length)
            self.logger.record("eval/blue_win_rate", blue_win_rate)
            self.logger.record("eval/red_win_rate", red_win_rate)
            self.logger.record("eval/draw_rate", draw_rate)
            self.logger.record("eval/win_rate", win_rate)
            self.logger.record("eval/mean_blue_hits", mean_blue_hits)
            self.logger.record("eval/mean_red_hits", mean_red_hits)
            self.logger.record("eval/team_coordination", mean_team_coordination)
            self.logger.record("eval/formation_quality", mean_formation_quality)
            self.logger.record("eval/intention_accuracy", mean_intention_accuracy)
        
        # Print metrics
        if self.verbose > 0:
            num_timesteps = self.n_calls * self.training_env.num_envs
            print(f"Eval num_timesteps={num_timesteps}, episode_reward={mean_reward:.2f} +/- {np.std(episode_rewards):.2f}")
            print(f"Episode length: {mean_length:.2f} +/- {np.std(episode_lengths):.2f}")
            print(f"Win rates - Blue: {blue_win_rate:.2f}, Red: {red_win_rate:.2f}, Draw: {draw_rate:.2f}")
            print(f"Hits - Blue: {mean_blue_hits:.2f}, Red: {mean_red_hits:.2f}")
            print(f"Team coordination: {mean_team_coordination:.3f}")
            print(f"Formation quality: {mean_formation_quality:.3f}")
            print(f"Intention accuracy: {mean_intention_accuracy:.3f}")
            print("-" * 50)
        
        # Save best model
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            if self.verbose > 0:
                print("New best mean reward!")
                print("New best mean reward!")


class IPMARL3v3Wrapper:
    """Wrapper for 3v3 environment to work with IP MARL."""
    
    def __init__(
        self,
        env_config: Optional[Dict[str, Any]] = None,
        n_envs: int = 8,
        use_subproc: bool = True,
    ):
        """Initialize the wrapper.
        
        Args:
            env_config: Environment configuration
            n_envs: Number of parallel environments
            use_subproc: Whether to use subprocess environments
        """
        if env_config is None:
            env_config = {}
        
        self.env_config = env_config
        self.n_envs = n_envs
        
        # Create environment creation function
        def make_env():
            return SingleAgentWrapper(Dogfight3v3MultiAgentEnv(env_config=env_config))
        
        # Create vector environment
        if use_subproc and n_envs > 1:
            self.vec_env = SubprocVecEnv([make_env for _ in range(n_envs)])
        else:
            self.vec_env = DummyVecEnv([make_env for _ in range(n_envs)])
        
        # Get observation and action spaces
        self.observation_space = self.vec_env.observation_space
        self.action_space = self.vec_env.action_space
        
        # Set number of agents (6 for 3v3: 3 blue + 3 red)
        self.n_agents = 6
        
        print(f"✓ Created {n_envs} parallel 3v3 environments")
        print(f"✓ Observation space: {self.observation_space}")
        print(f"✓ Action space: {self.action_space}")
        print(f"✓ Number of agents: {self.n_agents}")

    def reset(self):
        """Reset all environments."""
        obs = self.vec_env.reset()
        return obs

    def step(self, actions):
        """Step all environments."""
        obs, rewards, dones, infos = self.vec_env.step(actions)
        return obs, rewards, dones, infos

    def close(self):
        """Close all environments."""
        self.vec_env.close()


class SingleAgentWrapper(gym.Env):
    """Wrapper to convert multi-agent environment to single-agent for Stable-Baselines3."""
    
    def __init__(self, multi_agent_env):
        """Initialize the wrapper.
        
        Args:
            multi_agent_env: The multi-agent environment to wrap
        """
        super().__init__()
        self.env = multi_agent_env
        self.agent_ids = list(self.env.observation_space.keys())
        self.n_agents = len(self.agent_ids)
        
        # Flatten observation and action spaces
        # For now, we'll use the first agent's spaces as the single-agent spaces
        # This is a simplified approach - in practice you might want more sophisticated handling
        first_agent_id = self.agent_ids[0]
        self.observation_space = self.env.observation_space[first_agent_id]
        self.action_space = self.env.action_space[first_agent_id]
        
        # Store the current agent index for round-robin selection
        self.current_agent_idx = 0
        
        # Set render mode for Gymnasium compatibility
        self.render_mode = None
        
    def reset(self, seed=None, options=None):
        """Reset the environment and return observation for the first agent."""
        super().reset(seed=seed)
        obs_dict, info = self.env.reset()
        # Return observation for the first agent
        first_agent_id = self.agent_ids[0]
        return obs_dict[first_agent_id], info
        
    def step(self, action):
        """Step the environment with the current agent's action."""
        # Create action dictionary for all agents
        action_dict = {}
        for i, agent_id in enumerate(self.agent_ids):
            if i == self.current_agent_idx:
                action_dict[agent_id] = action
            else:
                # Use zero actions for other agents (or you could use a policy)
                action_dict[agent_id] = np.zeros_like(action)
        
        # Step the environment
        obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = self.env.step(action_dict)
        
        # Get the current agent's results
        current_agent_id = self.agent_ids[self.current_agent_idx]
        obs = obs_dict[current_agent_id]
        reward = reward_dict[current_agent_id]
        terminated = terminated_dict[current_agent_id]
        truncated = truncated_dict[current_agent_id]
        info = info_dict[current_agent_id]
        
        # Move to next agent for next step
        self.current_agent_idx = (self.current_agent_idx + 1) % self.n_agents
        
        return obs, reward, terminated, truncated, info
        
    def close(self):
        """Close the environment."""
        self.env.close()
    
    def get_wrapper_attr(self, name: str):
        """Get attribute from wrapped environment."""
        if hasattr(self.env, name):
            return getattr(self.env, name)
        else:
            raise AttributeError(f"Environment does not have attribute '{name}'")
    
    def render(self, mode="human"):
        """Render the environment."""
        if hasattr(self.env, 'render'):
            return self.env.render(mode=mode)
        else:
            return None
    
    def seed(self, seed=None):
        """Set the random seed."""
        if hasattr(self.env, 'seed'):
            return self.env.seed(seed)
        else:
            return None


def create_ip_marl_model(
    env: IPMARL3v3Wrapper,
    config: Dict[str, Any],
    model_path: Optional[str] = None,
) -> IPMARL:
    """Create IP MARL model with configuration.
    
    Args:
        env: Environment wrapper
        config: Model configuration
        model_path: Path to load existing model (optional)
        
    Returns:
        IP MARL model
    """
    # Extract configuration
    policy_kwargs = config.get("policy_kwargs", {})
    model_kwargs = config.get("model_kwargs", {})
    
    # Set up policy kwargs
    policy_kwargs.update({
        "n_agents": env.n_agents,
        "use_centralized_critic": config.get("use_centralized_critic", True),
        "state_shape": config.get("state_shape", [env.n_agents, 12]),
        "hidden_dim": config.get("hidden_dim", 256),
        "intention_dim": config.get("intention_dim", 8),
        "intention_propagation": config.get("intention_propagation", True),
    })
    
    # Set up model kwargs
    model_kwargs.update({
        "n_agents": env.n_agents,
        "use_centralized_critic": config.get("use_centralized_critic", True),
        "state_shape": config.get("state_shape", [env.n_agents, 12]),
        "intention_dim": config.get("intention_dim", 8),
        "intention_propagation": config.get("intention_propagation", True),
        "intention_loss_coef": config.get("intention_loss_coef", 0.1),
    })
    
    # Create model
    if model_path and os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model = IPMARL.load(model_path, env=env.vec_env)
    else:
        print("Creating new IP MARL model")
        # Add tensorboard logging
        tensorboard_log = config.get("tensorboard_log", "runs/ip_marl_3v3")
        model_kwargs["tensorboard_log"] = tensorboard_log
        
        model = IPMARL(
            policy=IPMARLPolicy,
            env=env.vec_env,
            policy_kwargs=policy_kwargs,
            **model_kwargs
        )
    
    return model


def create_callbacks(
    config: Dict[str, Any],
    eval_env: Optional[IPMARL3v3Wrapper] = None,
) -> list:
    """Create training callbacks.
    
    Args:
        config: Training configuration
        eval_env: Evaluation environment (optional)
        
    Returns:
        List of callbacks
    """
    callbacks = []
    
    # Custom IP MARL metrics callback
    eval_freq = config.get("eval_freq", 20000)
    if eval_freq > 0 and eval_env is not None:
        n_eval_episodes = config.get("eval_config", {}).get("n_eval_episodes", 10)
        ip_marl_callback = IPMARLMetricsCallback(
            eval_env=eval_env.vec_env,
            eval_freq=eval_freq // config.get("n_envs", 8),
            n_eval_episodes=n_eval_episodes,
            verbose=1,
        )
        callbacks.append(ip_marl_callback)
    
    # Checkpoint callback
    checkpoint_freq = config.get("checkpoint_freq", 100000)
    if checkpoint_freq > 0:
        save_path = config.get("save_path", "models/ip_marl_3v3")
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq // config.get("n_envs", 8),
            save_path=save_path,
            name_prefix="ip_marl_3v3",
        )
        callbacks.append(checkpoint_callback)
    
    return callbacks


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train IP MARL for 3v3 UAV combat")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/ip_marl_3v3.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=None,
        help="Path to load existing model"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed"
    )
    args = parser.parse_args()
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"✓ Loaded configuration from {args.config}")
    print(f"✓ Configuration: {config}")
    
    # Create training environment
    env_config = config.get("env_config", {})
    n_envs = config.get("n_envs", 8)
    
    train_env = IPMARL3v3Wrapper(
        env_config=env_config,
        n_envs=n_envs,
        use_subproc=True
    )
    
    # Create evaluation environment (optional)
    eval_env = None
    if config.get("eval_freq", 0) > 0:
        eval_env = IPMARL3v3Wrapper(
            env_config=env_config,
            n_envs=1,  # Single environment for evaluation
            use_subproc=False
        )
    
    # Create model
    model = create_ip_marl_model(
        env=train_env,
        config=config,
        model_path=args.model_path
    )
    
    # Create callbacks
    callbacks = create_callbacks(config, eval_env)
    
    # Training parameters
    total_timesteps = config.get("total_timesteps", 2000000)
    log_interval = config.get("log_interval", 1)
    tb_log_name = config.get("tb_log_name", "IPMARL_3v3")
    
    print(f"✓ Starting training for {total_timesteps} timesteps")
    print(f"✓ Tensorboard log: {config.get('tensorboard_log', 'runs/ip_marl_3v3')}")
    
    # Start training
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
        )
        
        # Save final model
        save_path = config.get("save_path", "models/ip_marl_3v3")
        os.makedirs(save_path, exist_ok=True)
        final_model_path = f"{save_path}/ip_marl_3v3_final"
        model.save(final_model_path)
        print(f"✓ Training completed. Final model saved to {final_model_path}")
        
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
        # Save model on interruption
        save_path = config.get("save_path", "models/ip_marl_3v3")
        os.makedirs(save_path, exist_ok=True)
        interrupted_model_path = f"{save_path}/ip_marl_3v3_interrupted"
        model.save(interrupted_model_path)
        print(f"✓ Interrupted model saved to {interrupted_model_path}")
    
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        raise
    
    finally:
        # Clean up
        train_env.close()
        if eval_env is not None:
            eval_env.close()
        
        training_time = time.time() - start_time
        print(f"✓ Total training time: {training_time:.2f} seconds")


if __name__ == "__main__":
    main() 