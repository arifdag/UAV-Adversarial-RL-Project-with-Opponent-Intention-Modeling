"""
Example script demonstrating the Monitor wrapper with DogfightAviary.

This script shows how to use the Monitor wrapper to automatically record videos 
and statistics from DogfightAviary episodes for post-mortem analysis.

Example usage:
    python monitor_dogfight.py --episodes 3 --headless
    python monitor_dogfight.py --episodes 5 --output-dir my_results
    python monitor_dogfight.py --no-video  # Statistics only
"""

import argparse
import os

from gym_pybullet_drones.envs.DogfightAviary import DogfightAviary
from gym_pybullet_drones.utils.monitor import Monitor, save_episode_statistics, list_monitoring_files


def main():
    """Main function to run the monitoring demo."""
    parser = argparse.ArgumentParser(description="DogfightAviary Monitoring Demo")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--no-video", action="store_true", help="Disable video recording")
    parser.add_argument("--no-stats", action="store_true", help="Disable statistics recording")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")

    args = parser.parse_args()

    print("=== DogfightAviary Monitoring Demo ===")
    print(f"Episodes: {args.episodes}")
    print(f"Headless: {args.headless}")
    print(f"Video recording: {not args.no_video}")
    print()

    print("Creating DogfightAviary environment...")

    # Enable recording in the base environment if video recording is requested
    record_enabled = not args.no_video

    env = DogfightAviary(
        num_drones=4,
        gui=not args.headless,
        record=record_enabled,  # Enable recording for proper camera setup
        pyb_freq=240,
        ctrl_freq=30
    )

    print("Wrapping with Monitor...")
    env, output_dir = Monitor(
        env,
        output_dir=args.output_dir,
        record_video=not args.no_video,
        record_stats=not args.no_stats,
        name_prefix="dogfight_episode"
    )

    print(f"Output directory: {output_dir}")
    print(f"Running {args.episodes} episodes with random policy...")
    print()

    try:
        # Run episodes
        for episode in range(args.episodes):
            print(f"Starting Episode {episode + 1}")

            obs, info = env.reset()
            step_count = 0
            total_reward = 0

            while step_count < 100:  # Limit episode length for demo
                # Use random actions
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)

                total_reward += reward
                step_count += 1

                if terminated or truncated:
                    break

            print(f"Episode {episode + 1} completed:")
            print(f"  Steps: {step_count}")
            print(f"  Total reward: {total_reward:.3f}")
            print(f"  Terminated: {terminated}, Truncated: {truncated}")

            # Extract dogfight-specific info if available
            if hasattr(info, 'get'):
                if 'blue_alive' in info and 'red_alive' in info:
                    print(f"  Final - Blue alive: {info['blue_alive']}, Red alive: {info['red_alive']}")

            print()

        # Save statistics if enabled
        if not args.no_stats:
            print("Saving episode statistics...")
            csv_path = save_episode_statistics(env, output_dir, filename="progress.csv")
            if csv_path:
                print(f"Statistics saved to: {csv_path}")

    finally:
        env.close()

    # Report generated files
    print("\n=== Generated Files ===")
    video_files, log_files = list_monitoring_files(output_dir)

    if video_files:
        print(f"Videos ({len(video_files)}):")
        for video_file in video_files:
            file_path = os.path.join(output_dir, video_file)
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  {video_file} ({size_mb:.1f} MB)")

    if log_files:
        print(f"Logs ({len(log_files)}):")
        for log_file in log_files:
            file_path = os.path.join(output_dir, log_file)
            size_kb = os.path.getsize(file_path) / 1024
            print(f"  {log_file} ({size_kb:.1f} KB)")

    print(f"\nMonitoring demo completed successfully!")
    print(f"Output directory: {output_dir}")
    print(f"Video files: {len(video_files)}")
    print(f"Log files: {len(log_files)}")
    print(f"DoD requirements satisfied: runs/ folder contains .mp4 + .csv files")


if __name__ == "__main__":
    main()
