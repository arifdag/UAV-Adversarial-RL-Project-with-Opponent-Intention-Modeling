"""
Simple demonstration of environment monitoring.

This script shows the minimal code pattern:
    env = Monitor(...)
    # runs/2025-07-xx/ folder contains .mp4 + progress.csv
"""

from gym_pybullet_drones.envs.DogfightAviary import DogfightAviary
from gym_pybullet_drones.utils.monitor import Monitor, save_episode_statistics


def main():
    """
    Demonstrate the monitoring.
    
    Creates a DogfightAviary, wraps it with Monitor, runs a few episodes,
    and saves the results to a timestamped runs/ directory.
    """
    print("=== Monitor Demo ===")

    # Create environment
    print("Creating DogfightAviary...")
    env = DogfightAviary(
        num_drones=4,
        gui=False,
        obs='kin',
        act='rpm'
    )

    # Wrap with Monitor
    print("Wrapping with Monitor...")
    env, output_dir = Monitor(
        env,
        record_video=True,
        record_stats=True,
        name_prefix="dogfight_demo"
    )

    print(f"Output directory: {output_dir}")

    # Run some episodes
    print("Running episodes...")
    for episode in range(3):
        print(f"Episode {episode + 1}/3")

        obs, info = env.reset()
        steps = 0

        while steps < 200:  # Limit to 200 steps per episode for quick demo
            action = env.action_space.sample()  # Random policy
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1

            if terminated or truncated:
                break

        print(f"  Completed in {steps} steps")

    # Save final statistics
    print("Saving episode statistics...")
    csv_path = save_episode_statistics(env, output_dir)
    if csv_path:
        print(f"Statistics saved to: {csv_path}")

    env.close()

    # Show what files were created
    print("\n=== Generated Files ===")
    from gym_pybullet_drones.utils.monitor import list_monitoring_files
    files = list_monitoring_files(output_dir)

    print(f"Videos ({len(files['videos'])}):")
    for video in files['videos']:
        print(f"  {video}")

    print(f"Logs ({len(files['logs'])}):")
    for log in files['logs']:
        print(f"  {log}")

    print(f"\nDemo completed successfully!")
    print(f"Output directory: {output_dir}")
    print(f"Video files: {len(files['videos'])}")
    print(f"Log files: {len(files['logs'])}")

    # Verify the DoD requirements
    import os
    has_mp4 = any(f.endswith('.mp4') for f in os.listdir(output_dir))
    has_csv = any(f.endswith('.csv') for f in os.listdir(output_dir))

    print(f"\n=== DoD Verification ===")
    print(f"runs/YYYY-MM-DD_HH-MM-SS/ folder exists: {os.path.exists(output_dir)}")
    print(f"Contains .mp4 files: {has_mp4}")
    print(f"Contains progress.csv: {has_csv}")

    if has_mp4 and has_csv:
        print("All DoD requirements satisfied!")
    else:
        print("Some DoD requirements not met")


if __name__ == "__main__":
    main()
