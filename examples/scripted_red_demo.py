#!/usr/bin/env python3

"""
Demonstration script for Scripted Red Policy.

This script demonstrates the scripted "pursue & fire" policy where Red drones
follow a deterministic strategy to chase and eliminate Blue drones.

Example usage:
    python scripted_red_demo.py --episodes 5 --gui
    python scripted_red_demo.py --episodes 100 --headless --verify-dod
"""

import argparse
import os
import sys
from typing import Tuple

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gym_pybullet_drones.envs.DogfightAviary import DogfightAviary
from gym_pybullet_drones.policies.scripted_red import create_scripted_red_policy
from gym_pybullet_drones.utils.enums import ActionType, ObservationType


def run_episode(env: DogfightAviary,
                policy,
                episode_num: int,
                max_steps: int = 400,
                verbose: bool = False) -> Tuple[bool, int, dict]:
    """
    Run a single episode with the scripted red policy.
    
    Parameters
    ----------
    env : DogfightAviary
        The environment instance
    policy
        The scripted red policy
    episode_num : int
        Episode number for seeding
    max_steps : int
        Maximum steps per episode
    verbose : bool
        Whether to print step-by-step information
        
    Returns
    -------
    Tuple[bool, int, dict]
        (red_won, steps_taken, final_info)
    """
    obs, _ = env.reset(seed=42 + episode_num)

    red_team = env.red_team
    blue_team = env.blue_team

    step_count = 0
    red_won = False
    info = {}

    if verbose:
        print(f"\nEpisode {episode_num + 1} starting:")
        print(f"  Red team indices: {red_team}")
        print(f"  Blue team indices: {blue_team}")

    while step_count < max_steps:
        # Get current team status
        blue_alive = env.blue_alive
        red_alive = env.red_alive

        if verbose and step_count % 30 == 0:  # Print every second at 30Hz
            print(f"  Step {step_count:3d}: Blue alive: {sum(blue_alive)}, Red alive: {sum(red_alive)}")

        # Check win conditions
        if not any(blue_alive):
            red_won = True
            if verbose:
                print(f"  Red team wins! All blue drones eliminated at step {step_count}")
            break

        if not any(red_alive):
            if verbose:
                print(f"  Blue team wins! All red drones eliminated at step {step_count}")
            break

        # Generate actions using scripted policy
        actions = policy.get_full_action(obs, red_team, blue_team, blue_alive)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(actions)
        step_count += 1

        if terminated or truncated:
            # Check final status
            final_blue_alive = info.get('blue_alive', sum(blue_alive))
            final_red_alive = info.get('red_alive', sum(red_alive))

            if final_blue_alive == 0 and final_red_alive > 0:
                red_won = True
                if verbose:
                    print(f"  Red team wins! Episode terminated at step {step_count}")
            elif verbose:
                print(f"  Episode ended at step {step_count} - Blue: {final_blue_alive}, Red: {final_red_alive}")
            break

    return red_won, step_count, info


def run_demonstration(episodes: int,
                      gui: bool = False,
                      verbose: bool = False,
                      verify_dod: bool = False) -> dict:
    """
    Run multiple episodes to demonstrate the scripted policy.
    
    Parameters
    ----------
    episodes : int
        Number of episodes to run
    gui : bool
        Whether to show GUI
    verbose : bool
        Whether to print detailed information
    verify_dod : bool
        Whether to verify DoD requirements
        
    Returns
    -------
    dict
        Results summary
    """
    print("=== E2-1 Scripted Red Policy Demo ===")
    print(f"Episodes: {episodes}")
    print(f"GUI: {gui}")
    print(f"DoD Verification: {verify_dod}")
    print()

    # Create environment
    env = DogfightAviary(
        num_drones=4,  # 2 blue, 2 red
        gui=gui,
        record=False,
        act=ActionType.PID,
        obs=ObservationType.KIN,
        pyb_freq=240,
        ctrl_freq=30
    )

    # Create scripted policy with optimized parameters
    policy = create_scripted_red_policy(
        action_type=ActionType.PID,
        kp_xy=2.0,
        kp_z=2.0,
        target_altitude=0.5,
        engagement_range=0.3
    )

    print("Environment and policy created successfully!")

    # Run episodes
    red_wins = 0
    total_steps = 0
    episode_lengths = []

    try:
        for episode in range(episodes):
            red_won, steps, info = run_episode(
                env, policy, episode,
                max_steps=400,
                verbose=verbose and episodes <= 5
            )

            if red_won:
                red_wins += 1

            total_steps += steps
            episode_lengths.append(steps)

            # Print progress
            if episodes <= 10 or (episode + 1) % max(1, episodes // 10) == 0:
                current_rate = red_wins / (episode + 1) * 100
                print(f"Episode {episode + 1:3d}: {'Red wins' if red_won else 'Blue wins'} "
                      f"({steps:3d} steps) - Success rate: {current_rate:.1f}%")

    finally:
        env.close()

    # Calculate results
    success_rate = red_wins / episodes
    avg_episode_length = total_steps / episodes

    results = {
        'episodes': episodes,
        'red_wins': red_wins,
        'success_rate': success_rate,
        'success_percentage': success_rate * 100,
        'avg_episode_length': avg_episode_length,
        'total_steps': total_steps,
        'episode_lengths': episode_lengths
    }

    # Print summary
    print("\n=== Results Summary ===")
    print(f"Total episodes: {episodes}")
    print(f"Red wins: {red_wins}")
    print(f"Success rate: {results['success_percentage']:.1f}%")
    print(f"Average episode length: {avg_episode_length:.1f} steps")
    print(f"Total simulation steps: {total_steps}")

    # DoD verification
    if verify_dod:
        print("\n=== DoD Verification ===")
        print(f"Requirement: Red hits Blue ≥ 65% of runs")
        print(f"Actual: {results['success_percentage']:.1f}%")

        if results['success_rate'] >= 0.65:
            print("DoD REQUIREMENT MET!")
        else:
            print("DoD requirement NOT met")
            print("Consider tuning policy parameters or investigating issues")

    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="E2-1 Scripted Red Policy Demo")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes to run (default: 10)")
    parser.add_argument("--gui", action="store_true",
                        help="Show GUI (default: headless)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed episode information")
    parser.add_argument("--verify-dod", action="store_true",
                        help="Verify DoD requirements (recommended with --episodes 100)")

    args = parser.parse_args()

    # Run demonstration
    results = run_demonstration(
        episodes=args.episodes,
        gui=args.gui,
        verbose=args.verbose,
        verify_dod=args.verify_dod
    )

    # Final message
    print(f"\nDemo completed successfully!")
    if args.verify_dod and results['success_rate'] >= 0.65:
        print("Requirements satisfied!")


if __name__ == "__main__":
    main()
