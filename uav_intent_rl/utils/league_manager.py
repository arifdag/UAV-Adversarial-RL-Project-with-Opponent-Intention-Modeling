"""
League Manager for UAV Adversarial RL

Handles:
- Checkpoint storage every 0.5M steps
- Round-robin evaluation tournaments
- Elo rating calculations
- CSV output generation
"""

import os
import csv
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import shutil
from pathlib import Path

import ray
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.algorithm import Algorithm


class EloRating:
    """Elo rating system for UAV agents"""
    
    def __init__(self, k_factor: float = 32, initial_rating: float = 1500):
        self.k_factor = k_factor
        self.initial_rating = initial_rating
    
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for player A against player B"""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_ratings(self, rating_a: float, rating_b: float, score_a: float) -> Tuple[float, float]:
        """
        Update Elo ratings after a match
        
        Args:
            rating_a: Current rating of player A
            rating_b: Current rating of player B
            score_a: Score of player A (1.0 = win, 0.5 = draw, 0.0 = loss)
        
        Returns:
            Tuple of new ratings (rating_a_new, rating_b_new)
        """
        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = self.expected_score(rating_b, rating_a)
        
        new_rating_a = rating_a + self.k_factor * (score_a - expected_a)
        new_rating_b = rating_b + self.k_factor * ((1 - score_a) - expected_b)
        
        return new_rating_a, new_rating_b


class LeagueManager:
    """Manages UAV agent league with checkpoints, evaluations, and Elo ratings"""
    
    def __init__(self, 
                 league_dir: str = "league",
                 checkpoint_interval: int = 500000,  # 0.5M steps
                 eval_episodes: int = 50):
        
        self.league_dir = Path(league_dir)
        self.checkpoint_interval = checkpoint_interval
        self.eval_episodes = eval_episodes
        
        # Create directories
        self.league_dir.mkdir(exist_ok=True)
        (self.league_dir / "checkpoints").mkdir(exist_ok=True)
        (self.league_dir / "artifacts").mkdir(exist_ok=True)
        (self.league_dir / "evaluations").mkdir(exist_ok=True)
        
        # Initialize Elo system
        self.elo_system = EloRating()
        
        # Load or initialize league state
        self.agents = {}  # agent_id -> metadata
        self.elo_ratings = {}  # agent_id -> current rating
        self.match_history = []  # List of match results
        
        self._load_league_state()
    
    def _load_league_state(self):
        """Load existing league state from disk"""
        state_file = self.league_dir / "league_state.json"
        if state_file.exists():
            with open(state_file, 'r') as f:
                state = json.load(f)
                self.agents = state.get('agents', {})
                self.elo_ratings = state.get('elo_ratings', {})
                self.match_history = state.get('match_history', [])
    
    def _save_league_state(self):
        """Save current league state to disk"""
        state = {
            'agents': self.agents,
            'elo_ratings': self.elo_ratings,
            'match_history': self.match_history,
            'last_updated': datetime.now().isoformat()
        }
        
        state_file = self.league_dir / "league_state.json"
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def add_checkpoint(self, checkpoint_path: str, total_steps: int, metadata: Optional[Dict] = None) -> str:
        """
        Add a new agent checkpoint to the league
        
        Args:
            checkpoint_path: Path to the RLlib checkpoint
            total_steps: Total training steps for this checkpoint
            metadata: Additional metadata (config, performance, etc.)
        
        Returns:
            agent_id: Unique identifier for this agent
        """
        # Generate agent ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        agent_id = f"agent_{total_steps}k_{timestamp}"
        
        # Copy checkpoint to league directory
        dest_path = self.league_dir / "checkpoints" / agent_id
        if os.path.exists(checkpoint_path):
            shutil.copytree(checkpoint_path, dest_path, dirs_exist_ok=True)
        
        # Store agent metadata
        self.agents[agent_id] = {
            'checkpoint_path': str(dest_path),
            'total_steps': total_steps,
            'created_at': timestamp,
            'metadata': metadata or {}
        }
        
        # Initialize Elo rating
        self.elo_ratings[agent_id] = self.elo_system.initial_rating
        
        self._save_league_state()
        print(f"Added agent {agent_id} with {total_steps} training steps")
        
        return agent_id
    
    def evaluate_matchup(self, agent_a_id: str, agent_b_id: str) -> Dict:
        """
        Evaluate a single matchup between two agents
        
        Returns:
            Dictionary with match results
        """
        from uav_intent_rl.envs.DogfightMultiAgentEnv import DogfightMultiAgentEnv
        
        print(f"Evaluating: {agent_a_id} vs {agent_b_id}")
        
        # Load agents
        checkpoint_a = self.agents[agent_a_id]['checkpoint_path']
        checkpoint_b = self.agents[agent_b_id]['checkpoint_path']
        
        # Initialize Ray if not already done
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        try:
            # Load algorithms
            algo_a = Algorithm.from_checkpoint(checkpoint_a)
            algo_b = Algorithm.from_checkpoint(checkpoint_b)
            
            # Create evaluation environment
            env = DogfightMultiAgentEnv()
            
            wins_a = 0
            wins_b = 0
            draws = 0
            episode_rewards_a = []
            episode_rewards_b = []
            
            for episode in range(self.eval_episodes):
                obs, info = env.reset()
                done = False
                total_reward_a = 0
                total_reward_b = 0
                
                while not done:
                    # Get actions from both agents
                    action_a = algo_a.compute_single_action(obs["blue"], policy_id="shared_ppo")
                    action_b = algo_b.compute_single_action(obs["red"], policy_id="shared_ppo")
                    
                    actions = {
                        "blue": action_a,
                        "red": action_b
                    }
                    
                    obs, rewards, terminateds, truncateds, info = env.step(actions)
                    
                    total_reward_a += rewards.get("blue", 0)
                    total_reward_b += rewards.get("red", 0)
                    
                    # Check if episode is done
                    done = any(terminateds.values()) or any(truncateds.values())
                
                episode_rewards_a.append(total_reward_a)
                episode_rewards_b.append(total_reward_b)
                
                # Determine winner
                if total_reward_a > total_reward_b:
                    wins_a += 1
                elif total_reward_b > total_reward_a:
                    wins_b += 1
                else:
                    draws += 1
            
            # Calculate win rate for agent A
            win_rate_a = wins_a / self.eval_episodes
            
            # Convert to Elo score (1.0 = win, 0.5 = draw, 0.0 = loss)
            elo_score_a = (wins_a + 0.5 * draws) / self.eval_episodes
            
            match_result = {
                'agent_a': agent_a_id,
                'agent_b': agent_b_id,
                'wins_a': wins_a,
                'wins_b': wins_b,
                'draws': draws,
                'win_rate_a': win_rate_a,
                'elo_score_a': elo_score_a,
                'avg_reward_a': np.mean(episode_rewards_a),
                'avg_reward_b': np.mean(episode_rewards_b),
                'timestamp': datetime.now().isoformat()
            }
            
            return match_result
            
        finally:
            # Clean up
            try:
                algo_a.stop()
                algo_b.stop()
            except:
                pass
    
    def run_round_robin_tournament(self) -> pd.DataFrame:
        """
        Run a round-robin tournament between all agents in the league
        
        Returns:
            DataFrame with all match results
        """
        agent_ids = list(self.agents.keys())
        
        if len(agent_ids) < 2:
            print("Need at least 2 agents for tournament")
            return pd.DataFrame()
        
        print(f"Running round-robin tournament with {len(agent_ids)} agents")
        
        match_results = []
        
        # Run all pairwise matches
        for i, agent_a in enumerate(agent_ids):
            for j, agent_b in enumerate(agent_ids):
                if i != j:  # Don't play against self
                    result = self.evaluate_matchup(agent_a, agent_b)
                    match_results.append(result)
                    self.match_history.append(result)
        
        # Update Elo ratings based on results
        self._update_elo_ratings(match_results)
        
        # Save updated state
        self._save_league_state()
        
        return pd.DataFrame(match_results)
    
    def _update_elo_ratings(self, match_results: List[Dict]):
        """Update Elo ratings based on match results"""
        for result in match_results:
            agent_a = result['agent_a']
            agent_b = result['agent_b']
            elo_score_a = result['elo_score_a']
            
            current_rating_a = self.elo_ratings[agent_a]
            current_rating_b = self.elo_ratings[agent_b]
            
            new_rating_a, new_rating_b = self.elo_system.update_ratings(
                current_rating_a, current_rating_b, elo_score_a
            )
            
            self.elo_ratings[agent_a] = new_rating_a
            self.elo_ratings[agent_b] = new_rating_b
    
    def generate_elo_matrix_csv(self) -> str:
        """
        Generate CSV file with Elo rating matrix
        
        Returns:
            Path to generated CSV file
        """
        agent_ids = list(self.agents.keys())
        
        if not agent_ids:
            print("No agents in league")
            return ""
        
        # Create matrix of win rates (row vs column)
        matrix = np.zeros((len(agent_ids), len(agent_ids)))
        
        for result in self.match_history:
            try:
                i = agent_ids.index(result['agent_a'])
                j = agent_ids.index(result['agent_b'])
                matrix[i, j] = result['win_rate_a']
            except ValueError:
                # Agent not in current list (might have been removed)
                continue
        
        # Create DataFrame
        df = pd.DataFrame(matrix, index=agent_ids, columns=agent_ids)
        
        # Add Elo ratings as additional row/column
        elo_values = [self.elo_ratings.get(agent_id, self.elo_system.initial_rating) 
                      for agent_id in agent_ids]
        
        df.loc['ELO_RATING'] = elo_values
        df['ELO_RATING'] = elo_values + [0]  # 0 for the corner cell
        
        # Save to CSV
        csv_path = self.league_dir / "artifacts" / "elo_matrix.csv"
        df.to_csv(csv_path)
        
        print(f"Generated Elo matrix CSV: {csv_path}")
        return str(csv_path)
    
    def generate_heatmap(self, save_path: Optional[str] = None) -> str:
        """
        Generate heatmap visualization of the Elo matrix
        
        Returns:
            Path to generated heatmap image
        """
        agent_ids = list(self.agents.keys())
        
        if len(agent_ids) < 2:
            print("Need at least 2 agents for heatmap")
            return ""
        
        # Create win rate matrix
        matrix = np.zeros((len(agent_ids), len(agent_ids)))
        
        for result in self.match_history:
            try:
                i = agent_ids.index(result['agent_a'])
                j = agent_ids.index(result['agent_b'])
                matrix[i, j] = result['win_rate_a']
            except ValueError:
                continue
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(matrix, 
                   xticklabels=agent_ids, 
                   yticklabels=agent_ids,
                   annot=True, 
                   fmt='.2f',
                   cmap='RdYlBu_r',
                   center=0.5,
                   vmin=0, vmax=1,
                   cbar_kws={'label': 'Win Rate'})
        
        plt.title('UAV League Win Rate Matrix\n(Row vs Column)')
        plt.xlabel('Opponent Agent')
        plt.ylabel('Playing Agent')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = self.league_dir / "artifacts" / "elo_heatmap.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Generated heatmap: {save_path}")
        return str(save_path)
    
    def get_leaderboard(self) -> pd.DataFrame:
        """Get current leaderboard sorted by Elo rating"""
        if not self.agents:
            return pd.DataFrame()
        
        leaderboard_data = []
        for agent_id in self.agents:
            agent_info = self.agents[agent_id]
            elo_rating = self.elo_ratings.get(agent_id, self.elo_system.initial_rating)
            
            # Calculate match statistics
            matches_as_a = [m for m in self.match_history if m['agent_a'] == agent_id]
            matches_as_b = [m for m in self.match_history if m['agent_b'] == agent_id]
            
            total_matches = len(matches_as_a) + len(matches_as_b)
            wins = sum(m['wins_a'] for m in matches_as_a) + sum(m['wins_b'] for m in matches_as_b)
            
            win_rate = wins / (total_matches * self.eval_episodes) if total_matches > 0 else 0
            
            leaderboard_data.append({
                'agent_id': agent_id,
                'elo_rating': elo_rating,
                'total_steps': agent_info['total_steps'],
                'total_matches': total_matches,
                'win_rate': win_rate,
                'created_at': agent_info['created_at']
            })
        
        df = pd.DataFrame(leaderboard_data)
        if not df.empty:
            df = df.sort_values('elo_rating', ascending=False)
        
        return df
    
    def print_status(self):
        """Print current league status"""
        print(f"\n=== UAV League Status ===")
        print(f"Total agents: {len(self.agents)}")
        print(f"Total matches: {len(self.match_history)}")
        
        if self.agents:
            leaderboard = self.get_leaderboard()
            print(f"\nTop 5 Agents:")
            print(leaderboard.head()[['agent_id', 'elo_rating', 'total_steps', 'win_rate']].to_string(index=False)) 