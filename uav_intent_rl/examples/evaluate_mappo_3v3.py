"""Evaluation script for MAPPO 3v3 models.

This script provides comprehensive evaluation capabilities for trained MAPPO
models in the 3v3 environment, including team behavior analysis, combat
statistics, and performance metrics.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3.common.vec_env import DummyVecEnv

from uav_intent_rl.algo.mappo import MAPPO
from uav_intent_rl.envs.Dogfight3v3MultiAgentEnv import Dogfight3v3MultiAgentEnv
from uav_intent_rl.policies.team_scripted_red import TeamScriptedRedPolicy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MAPPO3v3Evaluator:
    """Evaluator for MAPPO 3v3 models."""
    
    def __init__(
        self,
        model_path: str,
        n_eval_episodes: int = 100,
        render: bool = False,
        save_results: bool = True,
        results_path: str = "results/mappo_3v3_eval",
    ):
        self.model_path = model_path
        self.n_eval_episodes = n_eval_episodes
        self.render = render
        self.save_results = save_results
        self.results_path = Path(results_path)
        
        # Create results directory
        if self.save_results:
            self.results_path.mkdir(parents=True, exist_ok=True)
        
        # Load model
        self.model = MAPPO.load(model_path)
        logger.info(f"Loaded MAPPO model from {model_path}")
        
        # Create evaluation environment
        self.env = DummyVecEnv([lambda: Dogfight3v3MultiAgentEnv()])
        
        # Evaluation results
        self.results = {
            "episode_rewards": [],
            "episode_lengths": [],
            "blue_wins": 0,
            "red_wins": 0,
            "draws": 0,
            "blue_hits": [],
            "red_hits": [],
            "blue_survivors": [],
            "red_survivors": [],
            "team_coordination": [],
            "formation_quality": [],
        }
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive evaluation of the MAPPO model."""
        logger.info(f"Starting evaluation with {self.n_eval_episodes} episodes")
        
        start_time = time.time()
        
        for episode in range(self.n_eval_episodes):
            if episode % 10 == 0:
                logger.info(f"Evaluating episode {episode}/{self.n_eval_episodes}")
            
            episode_result = self._run_single_episode(episode)
            self._record_episode_result(episode_result)
        
        evaluation_time = time.time() - start_time
        logger.info(f"Evaluation completed in {evaluation_time:.2f} seconds")
        
        # Calculate final statistics
        final_stats = self._calculate_final_statistics()
        
        # Save results
        if self.save_results:
            self._save_results(final_stats)
        
        return final_stats
    
    def _run_single_episode(self, episode_idx: int) -> Dict[str, Any]:
        """Run a single evaluation episode."""
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
            if hasattr(self.env.envs[0], '_getDroneStateVector'):
                positions = []
                for i in range(6):  # 6 drones
                    try:
                        state = self.env.envs[0]._getDroneStateVector(i)
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
    
    def _record_episode_result(self, result: Dict[str, Any]):
        """Record results from a single episode."""
        self.results["episode_rewards"].append(result["reward"])
        self.results["episode_lengths"].append(result["length"])
        self.results["blue_hits"].append(result["blue_hits"])
        self.results["red_hits"].append(result["red_hits"])
        self.results["blue_survivors"].append(result["blue_survivors"])
        self.results["red_survivors"].append(result["red_survivors"])
        
        # Determine winner
        if result["blue_team_down"] and not result["red_team_down"]:
            self.results["red_wins"] += 1
        elif result["red_team_down"] and not result["blue_team_down"]:
            self.results["blue_wins"] += 1
        else:
            self.results["draws"] += 1
        
        # Calculate team coordination and formation quality
        if result["positions"]:
            coordination = self._calculate_team_coordination(result["positions"])
            formation = self._calculate_formation_quality(result["positions"])
            self.results["team_coordination"].append(coordination)
            self.results["formation_quality"].append(formation)
    
    def _calculate_team_coordination(self, positions: List[List[List[float]]]) -> float:
        """Calculate team coordination score based on position history."""
        if not positions:
            return 0.0
        
        # Use final positions for coordination calculation
        final_positions = positions[-1]
        blue_positions = final_positions[:3]
        red_positions = final_positions[3:]
        
        # Calculate blue team coordination
        blue_center = np.mean(blue_positions, axis=0)
        blue_distances = [np.linalg.norm(np.array(pos) - blue_center) for pos in blue_positions]
        
        # Coordination score: lower variance in distances = better coordination
        coordination_score = 1.0 / (1.0 + np.var(blue_distances))
        
        return coordination_score
    
    def _calculate_formation_quality(self, positions: List[List[List[float]]]) -> float:
        """Calculate formation quality score."""
        if not positions:
            return 0.0
        
        final_positions = positions[-1]
        blue_positions = final_positions[:3]
        
        # Calculate formation quality based on triangular formation
        if len(blue_positions) >= 3:
            # Calculate distances between all pairs
            distances = []
            for i in range(3):
                for j in range(i + 1, 3):
                    dist = np.linalg.norm(np.array(blue_positions[i]) - np.array(blue_positions[j]))
                    distances.append(dist)
            
            # Formation quality: more uniform distances = better formation
            formation_score = 1.0 / (1.0 + np.var(distances))
            return formation_score
        
        return 0.0
    
    def _calculate_final_statistics(self) -> Dict[str, Any]:
        """Calculate final evaluation statistics."""
        stats = {
            "n_episodes": self.n_eval_episodes,
            "blue_win_rate": self.results["blue_wins"] / self.n_eval_episodes,
            "red_win_rate": self.results["red_wins"] / self.n_eval_episodes,
            "draw_rate": self.results["draws"] / self.n_eval_episodes,
            "mean_episode_reward": np.mean(self.results["episode_rewards"]),
            "std_episode_reward": np.std(self.results["episode_rewards"]),
            "mean_episode_length": np.mean(self.results["episode_lengths"]),
            "mean_blue_hits": np.mean(self.results["blue_hits"]),
            "mean_red_hits": np.mean(self.results["red_hits"]),
            "mean_blue_survivors": np.mean(self.results["blue_survivors"]),
            "mean_red_survivors": np.mean(self.results["red_survivors"]),
        }
        
        if self.results["team_coordination"]:
            stats["mean_team_coordination"] = np.mean(self.results["team_coordination"])
            stats["std_team_coordination"] = np.std(self.results["team_coordination"])
        
        if self.results["formation_quality"]:
            stats["mean_formation_quality"] = np.mean(self.results["formation_quality"])
            stats["std_formation_quality"] = np.std(self.results["formation_quality"])
        
        return stats
    
    def _save_results(self, stats: Dict[str, Any]):
        """Save evaluation results to files."""
        # Save statistics
        stats_path = self.results_path / "evaluation_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save detailed results
        results_path = self.results_path / "detailed_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Create plots
        self._create_evaluation_plots()
        
        logger.info(f"Results saved to {self.results_path}")
    
    def _create_evaluation_plots(self):
        """Create evaluation plots."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].hist(self.results["episode_rewards"], bins=20, alpha=0.7)
        axes[0, 0].set_title("Episode Rewards Distribution")
        axes[0, 0].set_xlabel("Reward")
        axes[0, 0].set_ylabel("Frequency")
        
        # Episode lengths
        axes[0, 1].hist(self.results["episode_lengths"], bins=20, alpha=0.7)
        axes[0, 1].set_title("Episode Lengths Distribution")
        axes[0, 1].set_xlabel("Length")
        axes[0, 1].set_ylabel("Frequency")
        
        # Win rates
        win_rates = [
            self.results["blue_wins"],
            self.results["red_wins"],
            self.results["draws"]
        ]
        labels = ["Blue Wins", "Red Wins", "Draws"]
        axes[0, 2].pie(win_rates, labels=labels, autopct='%1.1f%%')
        axes[0, 2].set_title("Win Rate Distribution")
        
        # Combat statistics
        axes[1, 0].bar(["Blue Hits", "Red Hits"], 
                       [np.mean(self.results["blue_hits"]), np.mean(self.results["red_hits"])])
        axes[1, 0].set_title("Average Hits per Episode")
        axes[1, 0].set_ylabel("Hits")
        
        # Survivors
        axes[1, 1].bar(["Blue Survivors", "Red Survivors"],
                       [np.mean(self.results["blue_survivors"]), np.mean(self.results["red_survivors"])])
        axes[1, 1].set_title("Average Survivors per Episode")
        axes[1, 1].set_ylabel("Survivors")
        
        # Team coordination
        if self.results["team_coordination"]:
            axes[1, 2].hist(self.results["team_coordination"], bins=20, alpha=0.7)
            axes[1, 2].set_title("Team Coordination Distribution")
            axes[1, 2].set_xlabel("Coordination Score")
            axes[1, 2].set_ylabel("Frequency")
        
        plt.tight_layout()
        plot_path = self.results_path / "evaluation_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Plots saved to {plot_path}")


def evaluate_mappo_model(
    model_path: str,
    n_eval_episodes: int = 100,
    render: bool = False,
    save_results: bool = True,
    results_path: str = "results/mappo_3v3_eval",
) -> Dict[str, Any]:
    """Evaluate a MAPPO 3v3 model.
    
    Args:
        model_path: Path to the trained MAPPO model
        n_eval_episodes: Number of episodes to evaluate
        render: Whether to render episodes
        save_results: Whether to save evaluation results
        results_path: Path to save evaluation results
    
    Returns:
        Dictionary containing evaluation statistics
    """
    
    evaluator = MAPPO3v3Evaluator(
        model_path=model_path,
        n_eval_episodes=n_eval_episodes,
        render=render,
        save_results=save_results,
        results_path=results_path,
    )
    
    return evaluator.run_evaluation()


def main():
    """Main evaluation function."""
    
    # Example usage
    model_path = "models/mappo_3v3/mappo_3v3_final"
    
    if not os.path.exists(model_path + ".zip"):
        logger.error(f"Model not found at {model_path}")
        return
    
    logger.info("Starting MAPPO 3v3 evaluation")
    
    stats = evaluate_mappo_model(
        model_path=model_path,
        n_eval_episodes=100,
        render=False,
        save_results=True,
        results_path="results/mappo_3v3_eval",
    )
    
    # Print results
    logger.info("Evaluation Results:")
    logger.info(f"  Blue Win Rate: {stats['blue_win_rate']:.3f}")
    logger.info(f"  Red Win Rate: {stats['red_win_rate']:.3f}")
    logger.info(f"  Draw Rate: {stats['draw_rate']:.3f}")
    logger.info(f"  Mean Episode Reward: {stats['mean_episode_reward']:.3f}")
    logger.info(f"  Mean Episode Length: {stats['mean_episode_length']:.1f}")
    logger.info(f"  Mean Blue Hits: {stats['mean_blue_hits']:.2f}")
    logger.info(f"  Mean Red Hits: {stats['mean_red_hits']:.2f}")
    
    if 'mean_team_coordination' in stats:
        logger.info(f"  Mean Team Coordination: {stats['mean_team_coordination']:.3f}")
    
    if 'mean_formation_quality' in stats:
        logger.info(f"  Mean Formation Quality: {stats['mean_formation_quality']:.3f}")


if __name__ == "__main__":
    main() 