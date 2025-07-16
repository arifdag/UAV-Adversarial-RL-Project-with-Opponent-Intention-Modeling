"""
Training script with League Manager integration

Trains UAV agents and automatically adds checkpoints to league every 0.5M steps
"""

import argparse
import os
from pathlib import Path

import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig

from uav_intent_rl.envs.DogfightMultiAgentEnv import DogfightMultiAgentEnv
from uav_intent_rl.utils.league_manager import LeagueManager


# TODO: Re-implement callback for periodic checkpoint saving
# class LeagueCheckpointCallback:
#     """Callback to save checkpoints to league every 0.5M steps"""
#     
#     def __init__(self, league_manager: LeagueManager):
#         self.league_manager = league_manager
#         self.last_checkpoint_steps = 0
#     
#     def __call__(self, trial):
#         """Called after each training iteration"""
#         current_steps = trial.last_result.get("num_env_steps_sampled_lifetime", 0)
#         
#         # Check if we've crossed a 0.5M step boundary
#         if current_steps - self.last_checkpoint_steps >= self.league_manager.checkpoint_interval:
#             # Get the latest checkpoint path
#             checkpoint_path = trial.get_best_checkpoint().path
#             
#             # Add to league
#             agent_id = self.league_manager.add_checkpoint(
#                 checkpoint_path=checkpoint_path,
#                 total_steps=current_steps,
#                 metadata={
#                     'trial_id': trial.trial_id,
#                     'episode_reward_mean': trial.last_result.get("env_runners/episode_return_mean", 0),
#                     'episode_len_mean': trial.last_result.get("env_runners/episode_len_mean", 0),
#                     'training_iteration': trial.last_result.get("training_iteration", 0)
#                 }
#             )
#             
#             print(f"Added checkpoint to league: {agent_id} at {current_steps} steps")
#             self.last_checkpoint_steps = current_steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop-timesteps", type=int, default=2000000, 
                       help="Total timesteps to train")
    parser.add_argument("--league-dir", type=str, default="league",
                       help="Directory for league data")
    parser.add_argument("--checkpoint-interval", type=int, default=500000,
                       help="Steps between league checkpoints")
    parser.add_argument("--num-workers", type=int, default=0,
                       help="Number of worker processes")
    parser.add_argument("--local-mode", action="store_true",
                       help="Run in local mode for debugging")
    
    args = parser.parse_args()
    
    # Initialize Ray
    ray.init(local_mode=args.local_mode, ignore_reinit_error=True)
    
    # Initialize league manager
    league_manager = LeagueManager(
        league_dir=args.league_dir,
        checkpoint_interval=args.checkpoint_interval
    )
    
    # Configure PPO for multi-agent training
    base_config = (
        PPOConfig()
        .environment(env=DogfightMultiAgentEnv)
        .framework("torch")
        .env_runners(num_env_runners=args.num_workers)
        .training(
            lr=5e-5,
            train_batch_size=4000,
            minibatch_size=128,
            num_epochs=30,
            gamma=0.99,
            lambda_=1.0,
            clip_param=0.3,
            vf_clip_param=10.0,
            entropy_coeff=0.0,
            kl_coeff=0.2,
            use_gae=True,
            use_critic=True,
            use_kl_loss=True
        )
        .debugging(
            log_level="WARN"
        )
        .resources(
            num_gpus=0
        )
    )
    
    # Convert to dict and add multi-agent config
    config_dict = base_config.to_dict()
    config_dict["multiagent"] = {
        "policies": {
            "shared_ppo": (None, None, None, {})
        },
        "policy_mapping_fn": lambda agent_id, *_: "shared_ppo",
    }
    
    # Run training
    results = tune.Tuner(
        "PPO",
        run_config=air.RunConfig(
            stop={"num_env_steps_sampled_lifetime": args.stop_timesteps},
            name="UAV_League_Training"
        ),
        param_space=config_dict,
    ).fit()
    
    # Get the best result and add final checkpoint to league
    best_result = results.get_best_result()
    if best_result and best_result.checkpoint:
        final_checkpoint = best_result.checkpoint
        final_steps = best_result.metrics.get("num_env_steps_sampled_lifetime", 0)
        
        final_agent_id = league_manager.add_checkpoint(
            checkpoint_path=final_checkpoint.path,
            total_steps=final_steps,
            metadata={
                'episode_reward_mean': best_result.metrics.get("env_runners/episode_return_mean", 0),
                'episode_len_mean': best_result.metrics.get("env_runners/episode_len_mean", 0),
                'training_iteration': best_result.metrics.get("training_iteration", 0),
                'is_final_checkpoint': True
            }
        )
        
        print(f"Added final checkpoint to league: {final_agent_id}")
    
    # Print league status
    league_manager.print_status()
    
    ray.shutdown()


if __name__ == "__main__":
    main() 