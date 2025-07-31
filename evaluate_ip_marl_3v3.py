#!/usr/bin/env python3
"""Evaluation script for IP MARL 3v3 model."""

import os
import sys
import argparse
import numpy as np
import time
from pathlib import Path
from typing import Dict, Any, List

import torch as th

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from uav_intent_rl.envs.Dogfight3v3MultiAgentEnv import Dogfight3v3MultiAgentEnv
from uav_intent_rl.policies.team_scripted_red import TeamScriptedRedPolicy
from uav_intent_rl.algo.ip_marl import IPMARL


class IPMARLEvaluator:
    """Evaluator for IP MARL 3v3 model."""
    
    def __init__(
        self,
        model_path: str,
        env_config: Dict[str, Any] = None,
        n_episodes: int = 10,
        render: bool = False,
    ):
        """Initialize the evaluator.
        
        Args:
            model_path: Path to the trained model
            env_config: Environment configuration
            n_episodes: Number of episodes to evaluate
            render: Whether to render the environment
        """
        self.model_path = model_path
        self.env_config = env_config or {}
        self.n_episodes = n_episodes
        self.render = render
        
        # Create environment
        self.env = Dogfight3v3MultiAgentEnv(
            env_config={**self.env_config, "gui": render}
        )
        
        # Load model
        self.model = IPMARL.load(model_path, env=self.env)
        
        print(f"✓ Loaded model from {model_path}")
        print(f"✓ Environment: {self.env}")
        print(f"✓ Model: {self.model}")
    
    def evaluate_episode(self, episode_idx: int) -> Dict[str, Any]:
        """Evaluate a single episode.
        
        Args:
            episode_idx: Episode index
            
        Returns:
            Episode statistics
        """
        obs = self.env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        episode_actions = []
        episode_positions = []
        
        while not done:
            # Get actions from model
            actions, _ = self.model.predict(obs, deterministic=True)
            
            # Step environment
            obs, rewards, dones, infos = self.env.step(actions)
            
            # Record episode data
            episode_reward += sum(rewards.values())
            episode_length += 1
            episode_actions.append(actions)
            
            # Record positions if available
            if hasattr(self.env, '_getDroneStateVector'):
                positions = []
                for i in range(6):  # 6 drones
                    try:
                        state = self.env._getDroneStateVector(i)
                        positions.append(state[0:3])  # x, y, z
                    except:
                        positions.append([0, 0, 0])
                episode_positions.append(positions)
            
            done = any(dones.values())
        
        # Get final episode info
        final_info = list(infos.values())[0] if infos else {}
        
        return {
            "episode_idx": episode_idx,
            "reward": episode_reward,
            "length": episode_length,
            "blue_hits": final_info.get("blue_hits_dealt", 0),
            "red_hits": final_info.get("red_hits_dealt", 0),
            "blue_survivors": final_info.get("blue_drones_alive", 3),
            "red_survivors": final_info.get("red_drones_alive", 3),
            "blue_team_down": final_info.get("blue_team_down", False),
            "red_team_down": final_info.get("red_team_down", False),
            "actions": episode_actions,
            "positions": episode_positions,
        }
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the model over multiple episodes.
        
        Returns:
            Evaluation statistics
        """
        print(f"🎯 Evaluating IP MARL model over {self.n_episodes} episodes...")
        
        episode_stats = []
        start_time = time.time()
        
        for episode_idx in range(self.n_episodes):
            print(f"Episode {episode_idx + 1}/{self.n_episodes}...")
            
            episode_stat = self.evaluate_episode(episode_idx)
            episode_stats.append(episode_stat)
            
            print(f"  Reward: {episode_stat['reward']:.2f}")
            print(f"  Length: {episode_stat['length']}")
            print(f"  Blue hits: {episode_stat['blue_hits']}")
            print(f"  Red hits: {episode_stat['red_hits']}")
            print(f"  Blue survivors: {episode_stat['blue_survivors']}")
            print(f"  Red survivors: {episode_stat['red_survivors']}")
        
        evaluation_time = time.time() - start_time
        
        # Compute statistics
        rewards = [ep["reward"] for ep in episode_stats]
        lengths = [ep["length"] for ep in episode_stats]
        blue_hits = [ep["blue_hits"] for ep in episode_stats]
        red_hits = [ep["red_hits"] for ep in episode_stats]
        blue_survivors = [ep["blue_survivors"] for ep in episode_stats]
        red_survivors = [ep["red_survivors"] for ep in episode_stats]
        
        # Win/loss statistics
        blue_wins = sum(1 for ep in episode_stats if ep["red_team_down"])
        red_wins = sum(1 for ep in episode_stats if ep["blue_team_down"])
        draws = self.n_episodes - blue_wins - red_wins
        
        results = {
            "n_episodes": self.n_episodes,
            "evaluation_time": evaluation_time,
            "rewards": {
                "mean": np.mean(rewards),
                "std": np.std(rewards),
                "min": np.min(rewards),
                "max": np.max(rewards),
            },
            "lengths": {
                "mean": np.mean(lengths),
                "std": np.std(lengths),
                "min": np.min(lengths),
                "max": np.max(lengths),
            },
            "hits": {
                "blue_mean": np.mean(blue_hits),
                "blue_std": np.std(blue_hits),
                "red_mean": np.mean(red_hits),
                "red_std": np.std(red_hits),
            },
            "survivors": {
                "blue_mean": np.mean(blue_survivors),
                "blue_std": np.std(blue_survivors),
                "red_mean": np.mean(red_survivors),
                "red_std": np.std(red_survivors),
            },
            "wins": {
                "blue_wins": blue_wins,
                "red_wins": red_wins,
                "draws": draws,
                "blue_win_rate": blue_wins / self.n_episodes,
                "red_win_rate": red_wins / self.n_episodes,
                "draw_rate": draws / self.n_episodes,
            },
            "episode_stats": episode_stats,
        }
        
        return results
    
    def print_results(self, results: Dict[str, Any]):
        """Print evaluation results.
        
        Args:
            results: Evaluation results
        """
        print("\n" + "=" * 60)
        print("🎯 IP MARL 3v3 Evaluation Results")
        print("=" * 60)
        
        print(f"📊 Episodes evaluated: {results['n_episodes']}")
        print(f"⏱️  Evaluation time: {results['evaluation_time']:.2f} seconds")
        
        print(f"\n💰 Rewards:")
        print(f"  Mean: {results['rewards']['mean']:.2f} ± {results['rewards']['std']:.2f}")
        print(f"  Range: [{results['rewards']['min']:.2f}, {results['rewards']['max']:.2f}]")
        
        print(f"\n⏱️  Episode lengths:")
        print(f"  Mean: {results['lengths']['mean']:.1f} ± {results['lengths']['std']:.1f}")
        print(f"  Range: [{results['lengths']['min']}, {results['lengths']['max']}]")
        
        print(f"\n🎯 Hits:")
        print(f"  Blue: {results['hits']['blue_mean']:.1f} ± {results['hits']['blue_std']:.1f}")
        print(f"  Red:  {results['hits']['red_mean']:.1f} ± {results['hits']['red_std']:.1f}")
        
        print(f"\n🛡️  Survivors:")
        print(f"  Blue: {results['survivors']['blue_mean']:.1f} ± {results['survivors']['blue_std']:.1f}")
        print(f"  Red:  {results['survivors']['red_mean']:.1f} ± {results['survivors']['red_std']:.1f}")
        
        print(f"\n🏆 Win rates:")
        print(f"  Blue wins: {results['wins']['blue_wins']} ({results['wins']['blue_win_rate']:.1%})")
        print(f"  Red wins:  {results['wins']['red_wins']} ({results['wins']['red_win_rate']:.1%})")
        print(f"  Draws:     {results['wins']['draws']} ({results['wins']['draw_rate']:.1%})")
        
        print("\n" + "=" * 60)
    
    def close(self):
        """Close the environment."""
        self.env.close()


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate IP MARL 3v3 model")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to the trained model"
    )
    parser.add_argument(
        "--n_episodes", 
        type=int, 
        default=10,
        help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--render", 
        action="store_true",
        help="Render the environment"
    )
    parser.add_argument(
        "--env_config", 
        type=str, 
        default="{}",
        help="Environment configuration as JSON string"
    )
    args = parser.parse_args()
    
    # Parse environment config
    import json
    env_config = json.loads(args.env_config)
    
    # Create evaluator
    evaluator = IPMARLEvaluator(
        model_path=args.model_path,
        env_config=env_config,
        n_episodes=args.n_episodes,
        render=args.render,
    )
    
    try:
        # Run evaluation
        results = evaluator.evaluate()
        
        # Print results
        evaluator.print_results(results)
        
        # Save results
        results_path = f"results/ip_marl_3v3_eval_{int(time.time())}.json"
        os.makedirs("results", exist_ok=True)
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Recursively convert numpy types
        def convert_dict(d):
            if isinstance(d, dict):
                return {k: convert_dict(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [convert_dict(v) for v in d]
            else:
                return convert_numpy(d)
        
        results_converted = convert_dict(results)
        
        with open(results_path, 'w') as f:
            json.dump(results_converted, f, indent=2)
        
        print(f"💾 Results saved to {results_path}")
        
    except Exception as e:
        print(f"❌ Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        evaluator.close()
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 