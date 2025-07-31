#!/usr/bin/env python3
"""Test script for model visualization with 3v3 fights.

This script loads trained models (MAPPO or IP MARL) and pits them against
scripted red teams in 3v3 combat scenarios, then visualizes the drone paths.

Usage:
    python test_model_visualization.py --model_path models/ip_marl_3v3/ip_marl_3v3_final.zip --model_type ip_marl
    python test_model_visualization.py --model_path models/mappo_3v3/mappo_3v3_final.zip --model_type mappo
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import time
import sys

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from uav_intent_rl.envs.Dogfight3v3Aviary import Dogfight3v3Aviary
from uav_intent_rl.envs.Dogfight3v3MultiAgentEnv import Dogfight3v3MultiAgentEnv
from uav_intent_rl.policies.team_scripted_red import TeamScriptedRedPolicy
from uav_intent_rl.algo.ip_marl import IPMARL
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


class ModelVisualizationTester:
    """Test script for model visualization with 3v3 fights."""
    
    def __init__(
        self,
        model_path: str,
        model_type: str = "ip_marl",
        n_episodes: int = 1,
        render: bool = False,
        save_visualization: bool = True,
        output_dir: str = "visualization_results",
    ):
        """Initialize the visualization tester."""
        self.model_path = model_path
        self.model_type = model_type.lower()
        self.n_episodes = n_episodes
        self.render = render
        self.save_visualization = save_visualization
        self.output_dir = Path(output_dir)
        
        # Create output directory
        if self.save_visualization:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize environment and model
        self._setup_environment()
        self._load_model()
        
        # Data collection for visualization
        self.episode_data = []
        
    def _setup_environment(self):
        """Set up the environment based on model type."""
        if self.model_type == "ip_marl":
            # For IP MARL, we need the multi-agent environment
            self.env = Dogfight3v3MultiAgentEnv(
                env_config={"gui": self.render}
            )
        else:
            # For MAPPO, we need the single-agent environment wrapped
            base_env = Dogfight3v3Aviary(gui=self.render)
            self.env = DummyVecEnv([lambda: base_env])
            
        print(f"✓ Environment set up for {self.model_type} model")
        
    def _load_model(self):
        """Load the trained model."""
        try:
            if self.model_type == "ip_marl":
                self.model = IPMARL.load(self.model_path, env=self.env)
            else:
                self.model = PPO.load(self.model_path)
                
            print(f"✓ Loaded {self.model_type} model from {self.model_path}")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise 
            
    def _collect_episode_data(self, episode: int) -> Dict[str, Any]:
        """Collect data during an episode for visualization."""
        episode_data = {
            "timesteps": [],
            "blue_positions": [],
            "red_positions": [],
            "blue_velocities": [],
            "red_velocities": [],
            "blue_alive": [],
            "red_alive": [],
            "rewards": [],
            "actions": [],
            "episode_length": 0,
            "winner": None,
            "final_reward": 0.0
        }
        
        # Reset environment with random seed for randomized spawns
        random_seed = np.random.randint(0, 10000)
        if self.model_type == "ip_marl":
            obs, info = self.env.reset(seed=random_seed)
        else:
            obs = self.env.reset(seed=random_seed)
            
        step = 0
        total_reward = 0.0
        
        while True:
            # Collect current state
            if self.model_type == "ip_marl":
                # Get positions for all drones
                blue_positions = []
                red_positions = []
                blue_velocities = []
                red_velocities = []
                blue_alive = []
                red_alive = []
                
                # Extract positions from environment
                for i in range(3):  # Blue drones
                    if hasattr(self.env, '_getDroneStateVector'):
                        state = self.env._getDroneStateVector(i)
                        blue_positions.append(state[0:3].tolist())
                        blue_velocities.append(state[3:6].tolist())
                        blue_alive.append(True)  # Simplified
                    else:
                        blue_positions.append([0, 0, 0])
                        blue_velocities.append([0, 0, 0])
                        blue_alive.append(True)
                        
                for i in range(3):  # Red drones
                    if hasattr(self.env, '_getDroneStateVector'):
                        state = self.env._getDroneStateVector(i + 3)
                        red_positions.append(state[0:3].tolist())
                        red_velocities.append(state[3:6].tolist())
                        red_alive.append(True)  # Simplified
                    else:
                        red_positions.append([0, 0, 0])
                        red_velocities.append([0, 0, 0])
                        red_alive.append(True)
                
                # Get actions from model
                actions = {}
                for agent_id in self.env.AGENT_IDS:
                    if agent_id.startswith("blue"):
                        action, _ = self.model.predict(obs[agent_id], deterministic=True)
                        actions[agent_id] = action.tolist()
                    else:
                        # Use scripted red policy
                        actions[agent_id] = [0, 0, 0, 0]  # Simplified
                
                # Step environment
                obs, rewards, terminated, truncated, info = self.env.step(actions)
                
                # Calculate total reward
                episode_reward = sum(rewards.values())
                total_reward += episode_reward
                
                # Check if episode is done
                done = any(terminated.values()) or any(truncated.values())
                
            else:
                # For MAPPO, get positions from the underlying environment
                if hasattr(self.env.envs[0], '_getDroneStateVector'):
                    blue_positions = []
                    red_positions = []
                    blue_velocities = []
                    red_velocities = []
                    blue_alive = []
                    red_alive = []
                    
                    for i in range(3):  # Blue drones
                        state = self.env.envs[0]._getDroneStateVector(i)
                        blue_positions.append(state[0:3].tolist())
                        blue_velocities.append(state[3:6].tolist())
                        blue_alive.append(True)
                        
                    for i in range(3):  # Red drones
                        state = self.env.envs[0]._getDroneStateVector(i + 3)
                        red_positions.append(state[0:3].tolist())
                        red_velocities.append(state[3:6].tolist())
                        red_alive.append(True)
                else:
                    # Fallback if we can't get positions
                    blue_positions = [[0, 0, 0] for _ in range(3)]
                    red_positions = [[0, 0, 0] for _ in range(3)]
                    blue_velocities = [[0, 0, 0] for _ in range(3)]
                    red_velocities = [[0, 0, 0] for _ in range(3)]
                    blue_alive = [True] * 3
                    red_alive = [True] * 3
                
                # Get action from model
                action, _ = self.model.predict(obs, deterministic=True)
                
                # Step environment
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                episode_reward = reward[0] if isinstance(reward, np.ndarray) else reward
                total_reward += episode_reward
                
                done = terminated[0] if isinstance(terminated, np.ndarray) else terminated
                if isinstance(truncated, np.ndarray):
                    done = done or truncated[0]
                else:
                    done = done or truncated
            
            # Store data
            episode_data["timesteps"].append(step)
            episode_data["blue_positions"].append(blue_positions)
            episode_data["red_positions"].append(red_positions)
            episode_data["blue_velocities"].append(blue_velocities)
            episode_data["red_velocities"].append(red_velocities)
            episode_data["blue_alive"].append(blue_alive)
            episode_data["red_alive"].append(red_alive)
            episode_data["rewards"].append(episode_reward)
            
            if self.model_type == "ip_marl":
                episode_data["actions"].append(actions)
            else:
                episode_data["actions"].append(action.tolist() if hasattr(action, 'tolist') else action)
            
            step += 1
            
            if done:
                break
        
        # Finalize episode data
        episode_data["episode_length"] = step
        episode_data["final_reward"] = total_reward
        
        # Determine winner (simplified)
        blue_alive_count = sum(episode_data["blue_alive"][-1])
        red_alive_count = sum(episode_data["red_alive"][-1])
        
        if blue_alive_count > red_alive_count:
            episode_data["winner"] = "blue"
        elif red_alive_count > blue_alive_count:
            episode_data["winner"] = "red"
        else:
            episode_data["winner"] = "draw"
            
        return episode_data 
        
    def run_episodes(self):
        """Run multiple episodes and collect data."""
        print(f"🎯 Running {self.n_episodes} episodes with {self.model_type} model...")
        
        for episode in range(self.n_episodes):
            print(f"📊 Episode {episode + 1}/{self.n_episodes}")
            
            episode_data = self._collect_episode_data(episode)
            self.episode_data.append(episode_data)
            
            print(f"   Episode length: {episode_data['episode_length']} steps")
            print(f"   Final reward: {episode_data['final_reward']:.3f}")
            print(f"   Winner: {episode_data['winner']}")
            
        print("✅ All episodes completed!")
        
    def visualize_paths(self, episode_idx: int = 0):
        """Visualize drone paths for a specific episode."""
        if episode_idx >= len(self.episode_data):
            print(f"❌ Episode {episode_idx} not found. Available episodes: 0-{len(self.episode_data)-1}")
            return
            
        episode_data = self.episode_data[episode_idx]
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract positions
        timesteps = episode_data["timesteps"]
        blue_positions = episode_data["blue_positions"]
        red_positions = episode_data["red_positions"]
        
        # Plot blue team paths
        for drone_idx in range(3):
            x_coords = [pos[drone_idx][0] for pos in blue_positions]
            y_coords = [pos[drone_idx][1] for pos in blue_positions]
            z_coords = [pos[drone_idx][2] for pos in blue_positions]
            
            ax.plot(x_coords, y_coords, z_coords, 
                   color='blue', linewidth=2, alpha=0.7, 
                   label=f'Blue Drone {drone_idx+1}')
            
            # Mark start and end positions
            ax.scatter(x_coords[0], y_coords[0], z_coords[0], 
                      color='blue', s=100, marker='o', alpha=0.8)
            ax.scatter(x_coords[-1], y_coords[-1], z_coords[-1], 
                      color='blue', s=100, marker='s', alpha=0.8)
        
        # Plot red team paths
        for drone_idx in range(3):
            x_coords = [pos[drone_idx][0] for pos in red_positions]
            y_coords = [pos[drone_idx][1] for pos in red_positions]
            z_coords = [pos[drone_idx][2] for pos in red_positions]
            
            ax.plot(x_coords, y_coords, z_coords, 
                   color='red', linewidth=2, alpha=0.7, 
                   label=f'Red Drone {drone_idx+1}')
            
            # Mark start and end positions
            ax.scatter(x_coords[0], y_coords[0], z_coords[0], 
                      color='red', s=100, marker='o', alpha=0.8)
            ax.scatter(x_coords[-1], y_coords[-1], z_coords[-1], 
                      color='red', s=100, marker='s', alpha=0.8)
        
        # Set labels and title
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'{self.model_type.upper()} Model - Episode {episode_idx + 1}\n'
                    f'Winner: {episode_data["winner"].title()}, '
                    f'Length: {episode_data["episode_length"]} steps, '
                    f'Reward: {episode_data["final_reward"]:.3f}')
        
        # Add legend
        ax.legend()
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        # Save plot
        if self.save_visualization:
            plot_path = self.output_dir / f"{self.model_type}_episode_{episode_idx + 1}_paths.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved path visualization to {plot_path}")
        
        plt.show()
        
    def create_animation(self, episode_idx: int = 0):
        """Create an animation of the drone movements."""
        if episode_idx >= len(self.episode_data):
            print(f"❌ Episode {episode_idx} not found. Available episodes: 0-{len(self.episode_data)-1}")
            return
            
        episode_data = self.episode_data[episode_idx]
        
        # Create figure
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract positions
        timesteps = episode_data["timesteps"]
        blue_positions = episode_data["blue_positions"]
        red_positions = episode_data["red_positions"]
        
        # Set axis limits
        all_x = []
        all_y = []
        all_z = []
        
        for pos_list in blue_positions + red_positions:
            for pos in pos_list:
                all_x.extend([pos[0]])
                all_y.extend([pos[1]])
                all_z.extend([pos[2]])
        
        ax.set_xlim(min(all_x) - 1, max(all_x) + 1)
        ax.set_ylim(min(all_y) - 1, max(all_y) + 1)
        ax.set_zlim(min(all_z) - 1, max(all_z) + 1)
        
        # Set labels
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'{self.model_type.upper()} Model - Episode {episode_idx + 1}')
        
        def animate(frame):
            ax.clear()
            ax.set_xlim(min(all_x) - 1, max(all_x) + 1)
            ax.set_ylim(min(all_y) - 1, max(all_y) + 1)
            ax.set_zlim(min(all_z) - 1, max(all_z) + 1)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title(f'{self.model_type.upper()} Model - Episode {episode_idx + 1} - Step {frame}')
            
            # Plot blue drones
            for drone_idx in range(3):
                x_coords = [pos[drone_idx][0] for pos in blue_positions[:frame+1]]
                y_coords = [pos[drone_idx][1] for pos in blue_positions[:frame+1]]
                z_coords = [pos[drone_idx][2] for pos in blue_positions[:frame+1]]
                
                if len(x_coords) > 1:
                    ax.plot(x_coords, y_coords, z_coords, color='blue', linewidth=2, alpha=0.7)
                if x_coords:
                    ax.scatter(x_coords[-1], y_coords[-1], z_coords[-1], 
                              color='blue', s=100, alpha=0.8)
            
            # Plot red drones
            for drone_idx in range(3):
                x_coords = [pos[drone_idx][0] for pos in red_positions[:frame+1]]
                y_coords = [pos[drone_idx][1] for pos in red_positions[:frame+1]]
                z_coords = [pos[drone_idx][2] for pos in red_positions[:frame+1]]
                
                if len(x_coords) > 1:
                    ax.plot(x_coords, y_coords, z_coords, color='red', linewidth=2, alpha=0.7)
                if x_coords:
                    ax.scatter(x_coords[-1], y_coords[-1], z_coords[-1], 
                              color='red', s=100, alpha=0.8)
            
            return ax,
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, frames=len(timesteps), 
            interval=100, blit=False, repeat=True
        )
        
        # Save animation
        if self.save_visualization:
            anim_path = self.output_dir / f"{self.model_type}_episode_{episode_idx + 1}_animation.gif"
            anim.save(anim_path, writer='pillow', fps=10)
            print(f"💾 Saved animation to {anim_path}")
        
        plt.show() 
        
    def visualize_paths_2d(self, episode_idx: int = 0):
        """Visualize drone paths in 2D for a specific episode."""
        if episode_idx >= len(self.episode_data):
            print(f"❌ Episode {episode_idx} not found. Available episodes: 0-{len(self.episode_data)-1}")
            return
            
        episode_data = self.episode_data[episode_idx]
        
        # Create 2D plot
        fig, (ax_xy, ax_xz) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Extract positions
        timesteps = episode_data["timesteps"]
        blue_positions = episode_data["blue_positions"]
        red_positions = episode_data["red_positions"]
        
        # Plot blue team paths - XY view
        for drone_idx in range(3):
            x_coords = [pos[drone_idx][0] for pos in blue_positions]
            y_coords = [pos[drone_idx][1] for pos in blue_positions]
            
            ax_xy.plot(x_coords, y_coords, 
                      color='blue', linewidth=2, alpha=0.7, 
                      label=f'Blue Drone {drone_idx+1}')
            
            # Mark start and end positions
            ax_xy.scatter(x_coords[0], y_coords[0], 
                         color='blue', s=100, marker='o', alpha=0.8)
            ax_xy.scatter(x_coords[-1], y_coords[-1], 
                         color='blue', s=100, marker='s', alpha=0.8)
        
        # Plot red team paths - XY view
        for drone_idx in range(3):
            x_coords = [pos[drone_idx][0] for pos in red_positions]
            y_coords = [pos[drone_idx][1] for pos in red_positions]
            
            ax_xy.plot(x_coords, y_coords, 
                      color='red', linewidth=2, alpha=0.7, 
                      label=f'Red Drone {drone_idx+1}')
            
            # Mark start and end positions
            ax_xy.scatter(x_coords[0], y_coords[0], 
                         color='red', s=100, marker='o', alpha=0.8)
            ax_xy.scatter(x_coords[-1], y_coords[-1], 
                         color='red', s=100, marker='s', alpha=0.8)
        
        # Plot blue team paths - XZ view
        for drone_idx in range(3):
            x_coords = [pos[drone_idx][0] for pos in blue_positions]
            z_coords = [pos[drone_idx][2] for pos in blue_positions]
            
            ax_xz.plot(x_coords, z_coords, 
                      color='blue', linewidth=2, alpha=0.7, 
                      label=f'Blue Drone {drone_idx+1}')
            
            # Mark start and end positions
            ax_xz.scatter(x_coords[0], z_coords[0], 
                         color='blue', s=100, marker='o', alpha=0.8)
            ax_xz.scatter(x_coords[-1], z_coords[-1], 
                         color='blue', s=100, marker='s', alpha=0.8)
        
        # Plot red team paths - XZ view
        for drone_idx in range(3):
            x_coords = [pos[drone_idx][0] for pos in red_positions]
            z_coords = [pos[drone_idx][2] for pos in red_positions]
            
            ax_xz.plot(x_coords, z_coords, 
                      color='red', linewidth=2, alpha=0.7, 
                      label=f'Red Drone {drone_idx+1}')
            
            # Mark start and end positions
            ax_xz.scatter(x_coords[0], z_coords[0], 
                         color='red', s=100, marker='o', alpha=0.8)
            ax_xz.scatter(x_coords[-1], z_coords[-1], 
                         color='red', s=100, marker='s', alpha=0.8)
        
        # Set labels and titles
        ax_xy.set_xlabel('X (m)')
        ax_xy.set_ylabel('Y (m)')
        ax_xy.set_title('Top View (XY)')
        ax_xy.legend()
        ax_xy.grid(True, alpha=0.3)
        ax_xy.set_aspect('equal')
        
        ax_xz.set_xlabel('X (m)')
        ax_xz.set_ylabel('Z (m)')
        ax_xz.set_title('Side View (XZ)')
        ax_xz.legend()
        ax_xz.grid(True, alpha=0.3)
        ax_xz.set_aspect('equal')
        
        # Main title
        fig.suptitle(f'{self.model_type.upper()} Model - Episode {episode_idx + 1}\n'
                    f'Winner: {episode_data["winner"].title()}, '
                    f'Length: {episode_data["episode_length"]} steps, '
                    f'Reward: {episode_data["final_reward"]:.3f}', 
                    fontsize=14)
        
        plt.tight_layout()
        
        # Save plot
        if self.save_visualization:
            plot_path = self.output_dir / f"{self.model_type}_episode_{episode_idx + 1}_paths_2d.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved 2D path visualization to {plot_path}")
        
        plt.show()
        
    def create_animation_2d(self, episode_idx: int = 0):
        """Create a 2D animation of the drone movements."""
        if episode_idx >= len(self.episode_data):
            print(f"❌ Episode {episode_idx} not found. Available episodes: 0-{len(self.episode_data)-1}")
            return
            
        episode_data = self.episode_data[episode_idx]
        
        # Create figure with two subplots
        fig, (ax_xy, ax_xz) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Extract positions
        timesteps = episode_data["timesteps"]
        blue_positions = episode_data["blue_positions"]
        red_positions = episode_data["red_positions"]
        
        # Set axis limits
        all_x = []
        all_y = []
        all_z = []
        
        for pos_list in blue_positions + red_positions:
            for pos in pos_list:
                all_x.extend([pos[0]])
                all_y.extend([pos[1]])
                all_z.extend([pos[2]])
        
        ax_xy.set_xlim(min(all_x) - 1, max(all_x) + 1)
        ax_xy.set_ylim(min(all_y) - 1, max(all_y) + 1)
        ax_xz.set_xlim(min(all_x) - 1, max(all_x) + 1)
        ax_xz.set_ylim(min(all_z) - 1, max(all_z) + 1)
        
        # Set labels
        ax_xy.set_xlabel('X (m)')
        ax_xy.set_ylabel('Y (m)')
        ax_xy.set_title('Top View (XY)')
        ax_xy.grid(True, alpha=0.3)
        ax_xy.set_aspect('equal')
        
        ax_xz.set_xlabel('X (m)')
        ax_xz.set_ylabel('Z (m)')
        ax_xz.set_title('Side View (XZ)')
        ax_xz.grid(True, alpha=0.3)
        ax_xz.set_aspect('equal')
        
        def animate(frame):
            ax_xy.clear()
            ax_xz.clear()
            
            # Set limits
            ax_xy.set_xlim(min(all_x) - 1, max(all_x) + 1)
            ax_xy.set_ylim(min(all_y) - 1, max(all_y) + 1)
            ax_xz.set_xlim(min(all_x) - 1, max(all_x) + 1)
            ax_xz.set_ylim(min(all_z) - 1, max(all_z) + 1)
            
            # Set labels
            ax_xy.set_xlabel('X (m)')
            ax_xy.set_ylabel('Y (m)')
            ax_xy.set_title(f'Top View (XY) - Step {frame}')
            ax_xy.grid(True, alpha=0.3)
            ax_xy.set_aspect('equal')
            
            ax_xz.set_xlabel('X (m)')
            ax_xz.set_ylabel('Z (m)')
            ax_xz.set_title(f'Side View (XZ) - Step {frame}')
            ax_xz.grid(True, alpha=0.3)
            ax_xz.set_aspect('equal')
            
            # Plot blue drones - XY view
            for drone_idx in range(3):
                x_coords = [pos[drone_idx][0] for pos in blue_positions[:frame+1]]
                y_coords = [pos[drone_idx][1] for pos in blue_positions[:frame+1]]
                
                if len(x_coords) > 1:
                    ax_xy.plot(x_coords, y_coords, color='blue', linewidth=2, alpha=0.7)
                if x_coords:
                    ax_xy.scatter(x_coords[-1], y_coords[-1], 
                                color='blue', s=100, alpha=0.8)
            
            # Plot red drones - XY view
            for drone_idx in range(3):
                x_coords = [pos[drone_idx][0] for pos in red_positions[:frame+1]]
                y_coords = [pos[drone_idx][1] for pos in red_positions[:frame+1]]
                
                if len(x_coords) > 1:
                    ax_xy.plot(x_coords, y_coords, color='red', linewidth=2, alpha=0.7)
                if x_coords:
                    ax_xy.scatter(x_coords[-1], y_coords[-1], 
                                color='red', s=100, alpha=0.8)
            
            # Plot blue drones - XZ view
            for drone_idx in range(3):
                x_coords = [pos[drone_idx][0] for pos in blue_positions[:frame+1]]
                z_coords = [pos[drone_idx][2] for pos in blue_positions[:frame+1]]
                
                if len(x_coords) > 1:
                    ax_xz.plot(x_coords, z_coords, color='blue', linewidth=2, alpha=0.7)
                if x_coords:
                    ax_xz.scatter(x_coords[-1], z_coords[-1], 
                                color='blue', s=100, alpha=0.8)
            
            # Plot red drones - XZ view
            for drone_idx in range(3):
                x_coords = [pos[drone_idx][0] for pos in red_positions[:frame+1]]
                z_coords = [pos[drone_idx][2] for pos in red_positions[:frame+1]]
                
                if len(x_coords) > 1:
                    ax_xz.plot(x_coords, z_coords, color='red', linewidth=2, alpha=0.7)
                if x_coords:
                    ax_xz.scatter(x_coords[-1], z_coords[-1], 
                                color='red', s=100, alpha=0.8)
            
            return ax_xy, ax_xz
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, frames=len(timesteps), 
            interval=100, blit=False, repeat=True
        )
        
        # Save animation
        if self.save_visualization:
            anim_path = self.output_dir / f"{self.model_type}_episode_{episode_idx + 1}_animation_2d.gif"
            anim.save(anim_path, writer='pillow', fps=10)
            print(f"💾 Saved 2D animation to {anim_path}")
        
        plt.show()
        
    def save_episode_data(self, episode_idx: int = 0):
        """Save episode data to JSON file."""
        if episode_idx >= len(self.episode_data):
            print(f"❌ Episode {episode_idx} not found. Available episodes: 0-{len(self.episode_data)-1}")
            return
            
        episode_data = self.episode_data[episode_idx]
        
        if self.save_visualization:
            data_path = self.output_dir / f"{self.model_type}_episode_{episode_idx + 1}_data.json"
            
            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj
            
            episode_data_serializable = convert_numpy(episode_data)
            
            with open(data_path, 'w') as f:
                json.dump(episode_data_serializable, f, indent=2)
            
            print(f"💾 Saved episode data to {data_path}")
    
    def print_episode_summary(self, episode_idx: int = 0):
        """Print a summary of the episode."""
        if episode_idx >= len(self.episode_data):
            print(f"❌ Episode {episode_idx} not found. Available episodes: 0-{len(self.episode_data)-1}")
            return
            
        episode_data = self.episode_data[episode_idx]
        
        print(f"\n📊 Episode {episode_idx + 1} Summary:")
        print(f"   Model Type: {self.model_type.upper()}")
        print(f"   Episode Length: {episode_data['episode_length']} steps")
        print(f"   Final Reward: {episode_data['final_reward']:.3f}")
        print(f"   Winner: {episode_data['winner'].title()}")
        
        # Calculate average reward per step
        avg_reward = episode_data['final_reward'] / episode_data['episode_length']
        print(f"   Average Reward per Step: {avg_reward:.3f}")
        
        # Show final positions
        print(f"\n�� Final Positions:")
        for i, pos in enumerate(episode_data['blue_positions'][-1]):
            print(f"   Blue Drone {i+1}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
        for i, pos in enumerate(episode_data['red_positions'][-1]):
            print(f"   Red Drone {i+1}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
    
    def run_full_test(self):
        """Run the complete test with visualization."""
        print(f"🚀 Starting {self.model_type.upper()} Model Visualization Test")
        print("=" * 60)
        
        # Run episodes
        self.run_episodes()
        
        # Process each episode
        for episode_idx in range(len(self.episode_data)):
            print(f"\n🎨 Processing Episode {episode_idx + 1}...")
            
            # Print summary
            self.print_episode_summary(episode_idx)
            
            # Save data
            self.save_episode_data(episode_idx)
            
            # Create visualizations
            self.visualize_paths(episode_idx)
            self.create_animation(episode_idx)
            self.visualize_paths_2d(episode_idx)
            self.create_animation_2d(episode_idx)
        
        print(f"\n✅ {self.model_type.upper()} Model Visualization Test Complete!")
        print(f"📁 Results saved to: {self.output_dir}")


def main():
    """Main function to run the visualization test."""
    parser = argparse.ArgumentParser(
        description="Test model visualization with 3v3 fights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test IP MARL model
  python test_model_visualization.py --model_path models/ip_marl_3v3/ip_marl_3v3_final.zip --model_type ip_marl

  # Test MAPPO model
  python test_model_visualization.py --model_path models/mappo_3v3/mappo_3v3_final.zip --model_type mappo

  # Run with rendering
  python test_model_visualization.py --model_path models/ip_marl_3v3/ip_marl_3v3_final.zip --model_type ip_marl --render

  # Run multiple episodes
  python test_model_visualization.py --model_path models/ip_marl_3v3/ip_marl_3v3_final.zip --model_type ip_marl --n_episodes 3
        """
    )
    
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to the trained model (.zip file)"
    )
    parser.add_argument(
        "--model_type", 
        type=str, 
        choices=["ip_marl", "mappo"],
        default="ip_marl",
        help="Type of model to test"
    )
    parser.add_argument(
        "--n_episodes", 
        type=int, 
        default=1,
        help="Number of episodes to run"
    )
    parser.add_argument(
        "--render", 
        action="store_true",
        help="Render the environment during simulation"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="visualization_results",
        help="Directory to save visualization results"
    )
    parser.add_argument(
        "--no_save", 
        action="store_true",
        help="Don't save visualization results"
    )
    
    args = parser.parse_args()
    
    # Validate model path
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        return
    
    # Create tester
    tester = ModelVisualizationTester(
        model_path=str(model_path),
        model_type=args.model_type,
        n_episodes=args.n_episodes,
        render=args.render,
        save_visualization=not args.no_save,
        output_dir=args.output_dir,
    )
    
    # Run test
    try:
        tester.run_full_test()
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 