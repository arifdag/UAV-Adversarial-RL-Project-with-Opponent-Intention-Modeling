#!/usr/bin/env python3
"""Test script to verify draw detection in 3v3 environment."""

import numpy as np
import time
from pathlib import Path

from uav_intent_rl.envs.Dogfight3v3Aviary import Dogfight3v3Aviary


def test_draw_detection():
    """Test draw detection in the 3v3 environment."""
    print("Testing draw detection in 3v3 environment...")
    
    # Create environment
    env = Dogfight3v3Aviary(gui=False)
    
    # Run multiple episodes to test different scenarios
    n_episodes = 20
    blue_wins = 0
    red_wins = 0
    draws = 0
    
    for episode in range(n_episodes):
        print(f"\nEpisode {episode + 1}/{n_episodes}")
        
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        
        # Use minimal actions to encourage draws (timeout)
        while True:
            # Use very small random actions to avoid elimination
            action = np.random.randn(6, 4) * 0.01
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if terminated or truncated:
                break
        
        # Determine winner based on actual team status
        blue_team_down = info.get('blue_team_down', False)
        red_team_down = info.get('red_team_down', False)
        episode_truncated = info.get('episode_truncated', False)
        
        print(f"  Episode length: {episode_length}")
        print(f"  Episode reward: {episode_reward:.3f}")
        print(f"  Blue team down: {blue_team_down}")
        print(f"  Red team down: {red_team_down}")
        print(f"  Episode truncated: {episode_truncated}")
        
        # Determine winner
        if red_team_down and not blue_team_down:
            blue_wins += 1
            print("  Result: Blue wins")
        elif blue_team_down and not red_team_down:
            red_wins += 1
            print("  Result: Red wins")
        elif episode_truncated and not blue_team_down and not red_team_down:
            draws += 1
            print("  Result: Draw (timeout)")
        else:
            # Fallback
            if episode_reward > 0:
                blue_wins += 1
                print("  Result: Blue wins (fallback)")
            elif episode_reward < 0:
                red_wins += 1
                print("  Result: Red wins (fallback)")
            else:
                draws += 1
                print("  Result: Draw (fallback)")
    
    # Print final statistics
    print(f"\nFinal Statistics:")
    print(f"  Total episodes: {n_episodes}")
    print(f"  Blue wins: {blue_wins} ({blue_wins/n_episodes*100:.1f}%)")
    print(f"  Red wins: {red_wins} ({red_wins/n_episodes*100:.1f}%)")
    print(f"  Draws: {draws} ({draws/n_episodes*100:.1f}%)")
    
    # Check if we have draws
    if draws > 0:
        print("✅ Draw detection is working!")
    else:
        print("⚠️  No draws detected. This might be normal if episodes are too short.")
    
    env.close()


def test_force_draw():
    """Test to force a draw by making drones avoid each other."""
    print("\nTesting forced draw scenario...")
    
    env = Dogfight3v3Aviary(gui=False)
    
    obs, info = env.reset()
    episode_reward = 0
    episode_length = 0
    
    # First, let's check initial conditions
    print("  Initial drone positions:")
    for i in range(6):
        state = env._getDroneStateVector(i)
        pos = state[0:3]
        yaw = state[9]
        team = "Blue" if i < 3 else "Red"
        print(f"    {team} drone {i % 3}: pos={pos}, yaw={np.rad2deg(yaw):.1f}°")
    
    # Calculate initial distances between teams
    blue_positions = []
    red_positions = []
    for i in range(3):
        blue_state = env._getDroneStateVector(i)
        red_state = env._getDroneStateVector(i + 3)
        blue_positions.append(blue_state[0:3])
        red_positions.append(red_state[0:3])
    
    min_distance = float('inf')
    for blue_pos in blue_positions:
        for red_pos in red_positions:
            dist = np.linalg.norm(blue_pos - red_pos)
            min_distance = min(min_distance, dist)
    
    print(f"  Minimum distance between teams: {min_distance:.3f}")
    print(f"  Damage radius: {env.DEF_DMG_RADIUS}")
    
    if min_distance <= env.DEF_DMG_RADIUS:
        print("  ⚠️  Warning: Teams are within damage radius initially!")
    
    # Use actions that make drones move away from each other
    while True:
        # Get current drone positions
        blue_positions = []
        red_positions = []
        
        for i in range(3):  # Blue drones
            state = env._getDroneStateVector(i)
            blue_positions.append(state[0:3])
        
        for i in range(3, 6):  # Red drones
            state = env._getDroneStateVector(i)
            red_positions.append(state[0:3])
        
        # Calculate actions to move drones away from each other
        action = np.zeros((6, 4))
        
        # Blue drones move away from red drones
        for i in range(3):
            if env._blue_drones_alive[i]:
                blue_pos = np.array(blue_positions[i])
                # Find closest red drone
                min_dist = float('inf')
                closest_red_pos = None
                for j in range(3):
                    if env._red_drones_alive[j]:
                        red_pos = np.array(red_positions[j])
                        dist = np.linalg.norm(red_pos - blue_pos)
                        if dist < min_dist:
                            min_dist = dist
                            closest_red_pos = red_pos
                
                if closest_red_pos is not None:
                    # Move away from closest red drone
                    direction = blue_pos - closest_red_pos
                    direction = direction / (np.linalg.norm(direction) + 1e-6)
                    action[i, 0:3] = direction * 0.5  # Increased velocity for faster escape
        
        # Red drones move away from blue drones
        for i in range(3):
            if env._red_drones_alive[i]:
                red_pos = np.array(red_positions[i])
                # Find closest blue drone
                min_dist = float('inf')
                closest_blue_pos = None
                for j in range(3):
                    if env._blue_drones_alive[j]:
                        blue_pos = np.array(blue_positions[j])
                        dist = np.linalg.norm(blue_pos - red_pos)
                        if dist < min_dist:
                            min_dist = dist
                            closest_blue_pos = blue_pos
                
                if closest_blue_pos is not None:
                    # Move away from closest blue drone
                    direction = red_pos - closest_blue_pos
                    direction = direction / (np.linalg.norm(direction) + 1e-6)
                    action[i + 3, 0:3] = direction * 0.5  # Increased velocity for faster escape
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        episode_reward += reward
        episode_length += 1
        
        # Print progress every 100 steps
        if episode_length % 100 == 0:
            print(f"    Step {episode_length}: reward={reward:.3f}, blue_alive={sum(env._blue_drones_alive)}, red_alive={sum(env._red_drones_alive)}")
        
        if terminated or truncated:
            break
    
    # Check result
    blue_team_down = info.get('blue_team_down', False)
    red_team_down = info.get('red_team_down', False)
    episode_truncated = info.get('episode_truncated', False)
    
    print(f"  Episode length: {episode_length}")
    print(f"  Episode reward: {episode_reward:.3f}")
    print(f"  Blue team down: {blue_team_down}")
    print(f"  Red team down: {red_team_down}")
    print(f"  Episode truncated: {episode_truncated}")
    
    if episode_truncated and not blue_team_down and not red_team_down:
        print("✅ Successfully forced a draw!")
    else:
        print("❌ Failed to force a draw")
        if blue_team_down and red_team_down:
            print("  Both teams were eliminated simultaneously")
        elif blue_team_down:
            print("  Blue team was eliminated")
        elif red_team_down:
            print("  Red team was eliminated")
        else:
            print("  Episode ended without timeout")
    
    env.close()


def test_force_draw_with_separation():
    """Test to force a draw by starting with more separation."""
    print("\nTesting forced draw with increased separation...")
    
    # Create environment with custom initial positions
    env = Dogfight3v3Aviary(gui=False)
    
    # Override initial positions to have more separation
    blue_positions = [
        [-5.0, 0.0, 1.0],   # Far left
        [-4.0, 2.0, 1.0],   # Top left
        [-4.0, -2.0, 1.0],  # Bottom left
    ]
    
    red_positions = [
        [5.0, 0.0, 1.0],    # Far right
        [4.0, 2.0, 1.0],    # Top right
        [4.0, -2.0, 1.0],   # Bottom right
    ]
    
    all_positions = blue_positions + red_positions
    env.INIT_XYZS = np.array(all_positions, dtype=float)
    
    # Reset with custom positions
    obs, info = env.reset()
    episode_reward = 0
    episode_length = 0
    
    print("  Initial drone positions (with separation):")
    for i in range(6):
        state = env._getDroneStateVector(i)
        pos = state[0:3]
        team = "Blue" if i < 3 else "Red"
        print(f"    {team} drone {i % 3}: pos={pos}")
    
    # Use minimal actions to encourage timeout
    while True:
        # Use very small random actions to avoid elimination
        action = np.random.randn(6, 4) * 0.01
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        episode_reward += reward
        episode_length += 1
        
        if terminated or truncated:
            break
    
    # Check result
    blue_team_down = info.get('blue_team_down', False)
    red_team_down = info.get('red_team_down', False)
    episode_truncated = info.get('episode_truncated', False)
    
    print(f"  Episode length: {episode_length}")
    print(f"  Episode reward: {episode_reward:.3f}")
    print(f"  Blue team down: {blue_team_down}")
    print(f"  Red team down: {red_team_down}")
    print(f"  Episode truncated: {episode_truncated}")
    
    if episode_truncated and not blue_team_down and not red_team_down:
        print("✅ Successfully forced a draw with separation!")
    else:
        print("❌ Failed to force a draw even with separation")
    
    env.close()


def main():
    """Run all tests."""
    print("🧪 Testing Draw Detection...")
    print("=" * 50)
    
    try:
        test_draw_detection()
        test_force_draw()
        test_force_draw_with_separation()
        
        print("\n" + "=" * 50)
        print("✅ Draw detection tests completed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 