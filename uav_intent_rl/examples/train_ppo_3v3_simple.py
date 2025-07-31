"""Simple PPO training script for 3v3 dogfight environment.

This script uses standard PPO with a custom environment wrapper to handle
the multi-agent 3v3 environment without the complexity of MAPPO.
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch as th
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import PPO

from uav_intent_rl.envs.Dogfight3v3Aviary import Dogfight3v3Aviary
from uav_intent_rl.policies.team_scripted_red import TeamScriptedRedPolicy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Simple3v3Wrapper(gym.Env):
    """Simple wrapper to convert 3v3 environment to single-agent format for PPO."""

    def __init__(self, env_config: Optional[Dict[str, Any]] = None):
        """Initialize the wrapper."""
        super().__init__()
        if env_config is None:
            env_config = {}
        self.env = Dogfight3v3Aviary(**env_config)
        self.scripted_red = TeamScriptedRedPolicy()

        # Use the observation space of the first drone
        self.observation_space = self.env.observation_space
        # Action space should be for 3 blue drones only (12 actions: 3 drones * 4 actions each)
        self.action_space = gym.spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(12,),  # 3 blue drones * 4 actions each
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        obs, info = self.env.reset()
        return obs, info

    def step(self, action):
        """Step the environment with blue team actions."""
        # Reshape action to (3, 4) for 3 blue drones
        blue_actions = action.reshape(3, 4)
        
        # Get red team actions from scripted policy
        red_actions = self.scripted_red(self.env)

        # Extract only red team actions (indices 3-5) and reshape to (3, 4)
        red_team_actions = red_actions[3:6]  # Shape: (3, 4)

        # Combine blue and red actions into (6, 4) array
        all_actions = np.vstack([blue_actions, red_team_actions])

        # Step the environment
        obs, reward, terminated, truncated, info = self.env.step(all_actions)

        return obs, reward, terminated, truncated, info

    def close(self):
        """Close the environment."""
        self.env.close()


class PPOEvalCallback(BaseCallback):
    """Callback for evaluating PPO agents during training."""
    
    def __init__(
        self,
        eval_env,
        eval_freq: int = 10000,
        n_eval_episodes: int = 20,
        deterministic: bool = True,
        verbose: int = 1,
        name: str = "ppo_eval",
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.name = name
        
        # Evaluation metrics
        self.eval_results = {
            "episode_rewards": [],
            "episode_lengths": [],
            "blue_wins": [],
            "red_wins": [],
            "draws": [],
        }
    
    def _on_step(self) -> bool:
        """Called after each step during training."""
        if self.n_calls % self.eval_freq == 0:
            self._evaluate_policy()
        return True
    
    def _evaluate_policy(self):
        """Evaluate the current policy."""
        logger.info(f"Evaluating policy at step {self.n_calls}")
        
        episode_rewards = []
        episode_lengths = []
        blue_wins = 0
        red_wins = 0
        draws = 0
        
        for episode in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            final_info = None
            blue_reward = 0
            red_reward = 0
            
            while not done:
                # Get actions from trained model
                actions, _ = self.model.predict(obs, deterministic=self.deterministic)
                
                # Step environment (DummyVecEnv returns 4 values, not 5)
                step_result = self.eval_env.step(actions)
                if len(step_result) == 4:
                    obs, reward, done, info = step_result
                    terminated = done
                    truncated = False
                else:
                    obs, reward, terminated, truncated, info = step_result
                
                episode_reward += reward
                episode_length += 1
                final_info = info  # Keep track of final info
                
                # Track blue and red rewards separately for debugging
                if isinstance(info, list):
                    step_info = info[0]
                else:
                    step_info = info
                
                blue_reward += step_info.get('blue_reward', 0)
                red_reward += step_info.get('red_reward', 0)
                
                done = terminated or truncated
            
            # Record episode results
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Determine winner using actual team status from environment info
            # DummyVecEnv returns a list of info dicts, we need the first one
            if isinstance(final_info, list):
                final_info = final_info[0]
            
            # Check actual team status
            episode_truncated = final_info.get('episode_truncated', False)
            blue_team_down = final_info.get('blue_team_down', False)
            red_team_down = final_info.get('red_team_down', False)
            
            if red_team_down and not blue_team_down:
                blue_wins += 1
            elif blue_team_down and not red_team_down:
                red_wins += 1
            elif episode_truncated and not blue_team_down and not red_team_down:
                # Draw: time ran out and neither team was eliminated
                draws += 1
            else:
                # Fallback: treat as draw if neither team is down
                draws += 1
            
            # Sanity check: episode reward should be close to zero
            if abs(float(episode_reward)) > 1e-6:
                logger.warning(f"Episode reward {float(episode_reward)} is not close to zero - check reward calculation")
        
        # Calculate metrics
        mean_reward = float(np.mean(episode_rewards))
        mean_length = float(np.mean(episode_lengths))
        blue_winrate = float(blue_wins) / self.n_eval_episodes
        red_winrate = float(red_wins) / self.n_eval_episodes
        draw_rate = float(draws) / self.n_eval_episodes
        
        # Record evaluation results
        self.eval_results["episode_rewards"].append(mean_reward)
        self.eval_results["episode_lengths"].append(mean_length)
        self.eval_results["blue_wins"].append(blue_wins)
        self.eval_results["red_wins"].append(red_wins)
        self.eval_results["draws"].append(draws)
        
        # Log to tensorboard
        if self.logger is not None:
            self.logger.record(f"{self.name}/eval_mean_reward", mean_reward)
            self.logger.record(f"{self.name}/eval_mean_length", mean_length)
            self.logger.record(f"{self.name}/eval_blue_winrate", blue_winrate)
            self.logger.record(f"{self.name}/eval_red_winrate", red_winrate)
            self.logger.record(f"{self.name}/eval_draw_rate", draw_rate)
            self.logger.record(f"{self.name}/eval_blue_wins", blue_wins)
            self.logger.record(f"{self.name}/eval_red_wins", red_wins)
            self.logger.record(f"{self.name}/eval_draws", draws)
        
        logger.info(f"Evaluation results:")
        logger.info(f"  Mean reward: {mean_reward:.3f}")
        logger.info(f"  Mean length: {mean_length:.1f}")
        logger.info(f"  Blue wins: {blue_wins}/{self.n_eval_episodes} (winrate: {blue_winrate:.2%})")
        logger.info(f"  Red wins: {red_wins}/{self.n_eval_episodes} (winrate: {red_winrate:.2%})")
        logger.info(f"  Draws: {draws}/{self.n_eval_episodes} (rate: {draw_rate:.2%})")
        logger.info(f"  Debug - Last episode: Blue reward: {float(blue_reward):.3f}, Red reward: {float(red_reward):.3f}, Total: {float(episode_reward):.3f}")


def make_simple_env(env_config: Optional[Dict[str, Any]] = None):
    """Create a simple environment wrapper."""
    return Simple3v3Wrapper(env_config)


def train_ppo_3v3_simple(
    total_timesteps: int = 1000000,
    n_envs: int = 8,
    eval_freq: int = 20000,
    checkpoint_freq: int = 50000,
    save_path: str = "models/ppo_3v3_simple",
    tensorboard_log: str = "runs/ppo_3v3_simple",
    seed: Optional[int] = None,
    n_eval_episodes: int = 20,
    env_config: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> PPO:
    """Train a PPO agent in the 3v3 environment using simple wrapper."""
    
    # Set random seeds
    if seed is not None:
        np.random.seed(seed)
        th.manual_seed(seed)
    
    # Create directories
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Default model kwargs
    if model_kwargs is None:
        model_kwargs = {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 256,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "verbose": 1,
        }
    
    # Create training environment
    env_fn = lambda: make_simple_env(env_config)
    env = DummyVecEnv([env_fn for _ in range(n_envs)])
    
    # Create evaluation environment
    eval_env_fn = lambda: make_simple_env(env_config)
    eval_env = DummyVecEnv([eval_env_fn])
    
    # Create model
    model = PPO(
        "MlpPolicy",
        env=env,
        tensorboard_log=tensorboard_log,
        **model_kwargs
    )
    
    # Setup callbacks
    callbacks = []
    
    # Evaluation callback
    eval_callback = PPOEvalCallback(
        eval_env=eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        verbose=1,
        name="ppo_3v3_simple"
    )
    callbacks.append(eval_callback)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq // n_envs,
        save_path=str(save_path),
        name_prefix="ppo_3v3_simple",
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Train the model
    logger.info(f"Starting PPO training for {total_timesteps} timesteps")
    start_time = time.time()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        log_interval=1,
        tb_log_name="ppo_3v3_simple",
        reset_num_timesteps=True,
    )
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Save final model
    final_save_path = save_path / "ppo_3v3_simple_final"
    model.save(str(final_save_path))
    logger.info(f"Final model saved to {final_save_path}")
    
    # Save evaluation results
    eval_results_path = save_path / "eval_results.json"
    # Convert numpy types to Python types for JSON serialization
    eval_results_serializable = {}
    for key, value in eval_callback.eval_results.items():
        if isinstance(value, list):
            eval_results_serializable[key] = [float(v) if isinstance(v, (np.integer, np.floating)) else int(v) if isinstance(v, np.integer) else v for v in value]
        else:
            eval_results_serializable[key] = float(value) if isinstance(value, np.floating) else int(value) if isinstance(value, np.integer) else value
    
    with open(eval_results_path, 'w') as f:
        json.dump(eval_results_serializable, f, indent=2)
    logger.info(f"Evaluation results saved to {eval_results_path}")
    
    return model


def main():
    """Main training function."""
    
    # Training configuration
    config = {
        "total_timesteps": 200000,
        "n_envs": 4,
        "eval_freq": 10000,
        "checkpoint_freq": 50000,
        "save_path": "models/ppo_3v3_simple",
        "tensorboard_log": "runs/ppo_3v3_simple",
        "seed": 42,
        
        # Environment configuration
        "env_config": {
            "gui": False,
            "record": False,
            "pyb_freq": 240,
            "ctrl_freq": 30,
        },
        
        # Model configuration
        "model_kwargs": {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 256,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "verbose": 1,
        },
    }
    
    # Start training
    model = train_ppo_3v3_simple(**config)
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main() 