#!/usr/bin/env python3
"""Enhanced CLI script for MAPPO 3v3 training with comprehensive options."""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

from uav_intent_rl.examples.train_mappo_3v3 import train_mappo_3v3
from uav_intent_rl.examples.train_mappo_3v3_curriculum import train_mappo_3v3_curriculum
from uav_intent_rl.examples.train_ppo_3v3_simple import train_ppo_3v3_simple


def validate_args(args):
    """Validate command line arguments."""
    if args.total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")
    
    if args.n_envs <= 0:
        raise ValueError("n_envs must be positive")
    
    if args.eval_freq <= 0:
        raise ValueError("eval_freq must be positive")
    
    if args.checkpoint_freq <= 0:
        raise ValueError("checkpoint_freq must be positive")
    
    if args.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    
    if args.n_eval_episodes <= 0:
        raise ValueError("n_eval_episodes must be positive")
    
    # Ensure checkpoint_freq is a multiple of eval_freq for better logging
    if args.checkpoint_freq % args.eval_freq != 0:
        print(f"Warning: checkpoint_freq ({args.checkpoint_freq}) is not a multiple of eval_freq ({args.eval_freq})")
    
    return True


def create_directories(save_path: str, tensorboard_log: str):
    """Create necessary directories."""
    Path(save_path).mkdir(parents=True, exist_ok=True)
    Path(tensorboard_log).mkdir(parents=True, exist_ok=True)


def main():
    """Main CLI function with enhanced features."""
    parser = argparse.ArgumentParser(
        description="Train MAPPO for 3v3 drone combat with comprehensive options",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training mode
    parser.add_argument(
        "--mode", 
        choices=["normal", "curriculum", "simple"], 
        default="normal",
        help="Training mode: normal MAPPO, curriculum learning, or simple PPO"
    )
    
    # Training parameters
    parser.add_argument(
        "--total_timesteps", 
        type=int, 
        default=2000000,
        help="Total training timesteps"
    )
    
    parser.add_argument(
        "--n_envs", 
        type=int, 
        default=8,
        help="Number of parallel environments"
    )
    
    parser.add_argument(
        "--eval_freq", 
        type=int, 
        default=20000,
        help="Evaluation frequency in timesteps"
    )
    
    parser.add_argument(
        "--checkpoint_freq", 
        type=int, 
        default=100000,
        help="Checkpoint frequency in timesteps"
    )
    
    parser.add_argument(
        "--n_eval_episodes",
        type=int,
        default=20,
        help="Number of episodes to evaluate during training"
    )
    
    # Paths
    parser.add_argument(
        "--save_path", 
        type=str, 
        default="models/mappo_3v3",
        help="Path to save models"
    )
    
    parser.add_argument(
        "--tensorboard_log", 
        type=str, 
        default="runs/mappo_3v3",
        help="Path for tensorboard logs"
    )
    
    # Model parameters
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=3e-4,
        help="Learning rate for the optimizer"
    )
    
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=256,
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--n_steps",
        type=int,
        default=2048,
        help="Number of steps to run for each environment per update"
    )
    
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=10,
        help="Number of epochs when optimizing the surrogate loss"
    )
    
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor"
    )
    
    parser.add_argument(
        "--gae_lambda",
        type=float,
        default=0.95,
        help="Factor for trade-off of bias vs variance for Generalized Advantage Estimator"
    )
    
    parser.add_argument(
        "--clip_range",
        type=float,
        default=0.2,
        help="Clipping parameter for PPO"
    )
    
    parser.add_argument(
        "--ent_coef",
        type=float,
        default=0.01,
        help="Entropy coefficient for exploration"
    )
    
    parser.add_argument(
        "--vf_coef",
        type=float,
        default=0.5,
        help="Value function coefficient"
    )
    
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=0.5,
        help="Maximum gradient norm for gradient clipping"
    )
    
    # Environment parameters
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Enable GUI for environment visualization"
    )
    
    parser.add_argument(
        "--record",
        action="store_true",
        help="Enable recording of environment"
    )
    
    parser.add_argument(
        "--pyb_freq",
        type=int,
        default=240,
        help="PyBullet physics frequency"
    )
    
    parser.add_argument(
        "--ctrl_freq",
        type=int,
        default=30,
        help="Control frequency"
    )
    
    # Policy parameters
    parser.add_argument(
        "--n_agents",
        type=int,
        default=6,
        help="Number of agents in the environment"
    )
    
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Hidden dimension for neural networks"
    )
    
    parser.add_argument(
        "--use_centralized_critic",
        action="store_true",
        default=True,
        help="Use centralized critic for MAPPO"
    )
    
    # Verbosity
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbosity level (0=minimal, 1=normal, 2=detailed)"
    )
    
    # Dry run
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print configuration without starting training"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    try:
        validate_args(args)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Create directories
    create_directories(args.save_path, args.tensorboard_log)
    
    # Base configuration
    config = {
        "total_timesteps": args.total_timesteps,
        "n_envs": args.n_envs,
        "eval_freq": args.eval_freq,
        "checkpoint_freq": args.checkpoint_freq,
        "save_path": args.save_path,
        "tensorboard_log": args.tensorboard_log,
        "seed": args.seed,
        "n_eval_episodes": args.n_eval_episodes,
        
        # Environment configuration
        "env_config": {
            "gui": args.gui,
            "record": args.record,
            "pyb_freq": args.pyb_freq,
            "ctrl_freq": args.ctrl_freq,
        },
        
        # Policy configuration
        "policy_kwargs": {
            "n_agents": args.n_agents,
            "use_centralized_critic": args.use_centralized_critic,
            "state_shape": (6, 12),
            "hidden_dim": args.hidden_dim,
        },
        
        # Model configuration
        "model_kwargs": {
            "learning_rate": args.learning_rate,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "clip_range": args.clip_range,
            "ent_coef": args.ent_coef,
            "vf_coef": args.vf_coef,
            "max_grad_norm": args.max_grad_norm,
            "verbose": args.verbose,
        },
    }
    
    # Print configuration
    print("=" * 60)
    print("MAPPO 3v3 Training Configuration")
    print("=" * 60)
    print(f"Mode:                    {args.mode}")
    print(f"Total timesteps:         {args.total_timesteps:,}")
    print(f"Number of environments:  {args.n_envs}")
    print(f"Evaluation frequency:    {args.eval_freq:,}")
    print(f"Checkpoint frequency:    {args.checkpoint_freq:,}")
    print(f"Evaluation episodes:     {args.n_eval_episodes}")
    print(f"Save path:              {args.save_path}")
    print(f"Tensorboard log:        {args.tensorboard_log}")
    print(f"Seed:                   {args.seed}")
    print(f"Learning rate:          {args.learning_rate}")
    print(f"Batch size:             {args.batch_size}")
    print(f"GUI enabled:            {args.gui}")
    print(f"Recording enabled:      {args.record}")
    print("=" * 60)
    
    if args.dry_run:
        print("Dry run mode - configuration printed above")
        return
    
    # Start training based on mode
    try:
        if args.mode == "normal":
            model = train_mappo_3v3(**config)
        elif args.mode == "curriculum":
            # Add curriculum configuration
            config["curriculum_config"] = {
                "phases": [
                    {
                        "name": "basic_formation",
                        "timesteps": args.total_timesteps // 4,
                        "difficulty": 0.3,
                        "description": "Basic team formation and coordination"
                    },
                    {
                        "name": "team_coordination",
                        "timesteps": args.total_timesteps // 2,
                        "difficulty": 0.6,
                        "description": "Enhanced team coordination and tactics"
                    },
                    {
                        "name": "advanced_tactics",
                        "timesteps": args.total_timesteps,
                        "difficulty": 1.0,
                        "description": "Advanced team tactics and combat"
                    }
                ]
            }
            model = train_mappo_3v3_curriculum(**config)
        else:  # simple
            # For simple mode, only pass the parameters that train_ppo_3v3_simple accepts
            simple_config = {
                "total_timesteps": config["total_timesteps"],
                "n_envs": config["n_envs"],
                "eval_freq": config["eval_freq"],
                "checkpoint_freq": config["checkpoint_freq"],
                "save_path": config["save_path"],
                "tensorboard_log": config["tensorboard_log"],
                "seed": config["seed"],
                "n_eval_episodes": config["n_eval_episodes"],
                "env_config": config["env_config"],
                "model_kwargs": config["model_kwargs"],
            }
            model = train_ppo_3v3_simple(**simple_config)
        
        print(f"\n✅ Training completed successfully!")
        print(f"📁 Model saved to: {args.save_path}")
        print(f"📊 Tensorboard logs: {args.tensorboard_log}")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 