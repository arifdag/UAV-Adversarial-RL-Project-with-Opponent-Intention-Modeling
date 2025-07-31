"""Training script for MAPPO in 3v3 dogfight environment.

This script provides a complete training pipeline for MAPPO agents in the
3v3 multi-agent dogfight environment with team coordination, evaluation,
and curriculum learning capabilities.
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable

import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecEnv

from uav_intent_rl.algo.mappo import MAPPO, MAPPOPolicy
from uav_intent_rl.envs.Dogfight3v3MultiAgentVecEnv import make_mappo_vec_env
from uav_intent_rl.policies.team_scripted_red import TeamScriptedRedPolicy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MAPPOEvalCallback(BaseCallback):
    """Callback for evaluating MAPPO agents during training."""
    
    def __init__(
        self,
        eval_env: VecEnv,
        eval_freq: int = 10000,
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        verbose: int = 1,
        name: str = "mappo_eval",
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
            "blue_hits": [],
            "red_hits": [],
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
        blue_hits = []
        red_hits = []
        
        for episode in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                # Get actions from trained model
                actions, _ = self.model.predict(obs, deterministic=self.deterministic)
                
                # Step environment
                obs, rewards, dones, infos = self.eval_env.step(actions)
                
                episode_reward += sum(rewards.values())
                episode_length += 1
                done = any(dones.values())
            
            # Record episode results
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Determine winner based on actual team elimination status
            final_info = None
            if infos:
                final_info = list(infos.values())[0]  # Get first agent's info
            
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
            
            # Record combat statistics
            if infos:
                info = list(infos.values())[0]  # Get first agent's info
                blue_hits.append(info.get("blue_hits_dealt", 0))
                red_hits.append(info.get("red_hits_dealt", 0))
        
        # Calculate metrics
        mean_reward = np.mean(episode_rewards)
        mean_length = np.mean(episode_lengths)
        blue_win_rate = blue_wins / self.n_eval_episodes
        red_win_rate = red_wins / self.n_eval_episodes
        draw_rate = draws / self.n_eval_episodes
        mean_blue_hits = np.mean(blue_hits) if blue_hits else 0
        mean_red_hits = np.mean(red_hits) if red_hits else 0
        
        # Log results
        logger.info(f"Evaluation Results:")
        logger.info(f"  Mean Reward: {mean_reward:.2f}")
        logger.info(f"  Mean Length: {mean_length:.2f}")
        logger.info(f"  Blue Win Rate: {blue_win_rate:.2f}")
        logger.info(f"  Red Win Rate: {red_win_rate:.2f}")
        logger.info(f"  Draw Rate: {draw_rate:.2f}")
        logger.info(f"  Mean Blue Hits: {mean_blue_hits:.2f}")
        logger.info(f"  Mean Red Hits: {mean_red_hits:.2f}")
        
        # Store results
        self.eval_results["episode_rewards"].append(mean_reward)
        self.eval_results["episode_lengths"].append(mean_length)
        self.eval_results["blue_wins"].append(blue_win_rate)
        self.eval_results["red_wins"].append(red_win_rate)
        self.eval_results["draws"].append(draw_rate)
        self.eval_results["blue_hits"].append(mean_blue_hits)
        self.eval_results["red_hits"].append(mean_red_hits)
        
        # Log to tensorboard
        self.logger.record(f"{self.name}/eval_mean_reward", mean_reward)
        self.logger.record(f"{self.name}/eval_mean_length", mean_length)
        self.logger.record(f"{self.name}/eval_blue_win_rate", blue_win_rate)
        self.logger.record(f"{self.name}/eval_red_win_rate", red_win_rate)
        self.logger.record(f"{self.name}/eval_draw_rate", draw_rate)
        self.logger.record(f"{self.name}/eval_mean_blue_hits", mean_blue_hits)
        self.logger.record(f"{self.name}/eval_mean_red_hits", mean_red_hits)


def make_mappo_env(
    env_config: Optional[Dict[str, Any]] = None,
    n_envs: int = 1,
    seed: Optional[int] = None,
):
    """Create a multi-agent vector environment for MAPPO training."""
    return make_mappo_vec_env(env_config, n_envs, seed)


def create_mappo_model(
    env: VecEnv,
    policy_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> MAPPO:
    """Create a MAPPO model with the given configuration."""
    
    # Default policy kwargs
    if policy_kwargs is None:
        policy_kwargs = {
            "n_agents": 6,
            "use_centralized_critic": True,
            "state_shape": (6, 12),  # 6 agents, 12 obs dims each
            "hidden_dim": 256,
        }
    
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
            "clip_range_vf": None,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "use_sde": False,
            "sde_sample_freq": -1,
            "target_kl": None,
            "tensorboard_log": None,
            "policy_kwargs": policy_kwargs,
            "verbose": 1,
            "seed": None,
            "device": "auto",
            "_init_setup_model": True,
        }
    
    # Create model
    model = MAPPO(
        policy=MAPPOPolicy,
        env=env,
        **model_kwargs
    )
    
    return model


def train_mappo_3v3(
    total_timesteps: int = 1000000,
    n_envs: int = 8,
    eval_freq: int = 20000,
    checkpoint_freq: int = 50000,
    save_path: str = "models/mappo_3v3",
    tensorboard_log: str = "runs/mappo_3v3",
    seed: Optional[int] = None,
    env_config: Optional[Dict[str, Any]] = None,
    policy_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    curriculum_config: Optional[Dict[str, Any]] = None,
) -> MAPPO:
    """Train a MAPPO agent in the 3v3 environment.
    
    Args:
        total_timesteps: Total number of timesteps to train
        n_envs: Number of parallel environments
        eval_freq: Frequency of evaluation (in timesteps)
        checkpoint_freq: Frequency of model checkpointing
        save_path: Path to save the trained model
        tensorboard_log: Path for tensorboard logs
        seed: Random seed
        env_config: Environment configuration
        policy_kwargs: Policy network configuration
        model_kwargs: Model training configuration
        curriculum_config: Curriculum learning configuration
    
    Returns:
        Trained MAPPO model
    """
    
    # Create directories
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    tensorboard_log = Path(tensorboard_log)
    tensorboard_log.mkdir(parents=True, exist_ok=True)
    
    # Set random seed
    if seed is not None:
        np.random.seed(seed)
        th.manual_seed(seed)
    
    # Create training environment
    env = make_mappo_env(env_config, n_envs, seed)
    
    # Create evaluation environment
    eval_env = make_mappo_env(env_config, 1, seed)
    
    # Create model
    model = create_mappo_model(env, policy_kwargs, model_kwargs)
    
    # Setup callbacks
    callbacks = []
    
    # Evaluation callback
    eval_callback = MAPPOEvalCallback(
        eval_env=eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
        name="mappo_3v3"
    )
    callbacks.append(eval_callback)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq // n_envs,
        save_path=str(save_path),
        name_prefix="mappo_3v3",
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Train the model
    logger.info(f"Starting MAPPO training for {total_timesteps} timesteps")
    start_time = time.time()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        log_interval=1,
        tb_log_name="mappo_3v3",
        reset_num_timesteps=True,
    )
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Save final model
    final_save_path = save_path / "mappo_3v3_final"
    model.save(str(final_save_path))
    logger.info(f"Final model saved to {final_save_path}")
    
    # Save evaluation results
    eval_results_path = save_path / "eval_results.json"
    with open(eval_results_path, 'w') as f:
        json.dump(eval_callback.eval_results, f, indent=2)
    logger.info(f"Evaluation results saved to {eval_results_path}")
    
    return model


def main():
    """Main training function."""
    
    # Training configuration
    config = {
        "total_timesteps": 2000000,
        "n_envs": 8,
        "eval_freq": 20000,
        "checkpoint_freq": 100000,
        "save_path": "models/mappo_3v3",
        "tensorboard_log": "runs/mappo_3v3",
        "seed": 42,
        
        # Environment configuration
        "env_config": {
            "gui": False,
            "record": False,
            "pyb_freq": 240,
            "ctrl_freq": 30,
        },
        
        # Policy configuration
        "policy_kwargs": {
            "n_agents": 6,
            "use_centralized_critic": True,
            "state_shape": (6, 12),
            "hidden_dim": 256,
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
    model = train_mappo_3v3(**config)
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main() 