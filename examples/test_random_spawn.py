"""
Simple demo script to test random spawn functionality in DogfightAviary.

This script creates a DogfightAviary environment and demonstrates the random
spawn positions and orientations across multiple resets.
"""

import numpy as np
from gym_pybullet_drones.envs.DogfightAviary import DogfightAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


def test_random_spawn():
    """Test random spawn functionality with a few resets."""
    print("Testing DogfightAviary random spawn functionality...")
    print("=" * 60)

    # Create environment
    env = DogfightAviary(
        drone_model=DroneModel.CF2X,
        num_drones=4,
        gui=False,
        record=False,
        obs=ObservationType.KIN,
        act=ActionType.PID
    )

    # Test multiple resets
    for reset_num in range(5):
        print(f"\nReset #{reset_num + 1}:")
        print("-" * 30)

        # Reset with different seed
        obs, info = env.reset(seed=reset_num)

        # Display spawn positions
        print("Spawn positions (x, y, z):")
        for i, pos in enumerate(env.INIT_XYZS):
            team = "Blue" if i in env.blue_team else "Red"
            print(f"  Drone {i} ({team}): ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:.3f})")

        # Display spawn orientations
        print("\nSpawn orientations (roll, pitch, yaw):")
        for i, rpy in enumerate(env.INIT_RPYS):
            team = "Blue" if i in env.blue_team else "Red"
            yaw_deg = rpy[2] * 180 / np.pi
            print(f"  Drone {i} ({team}): ({rpy[0]:.3f}, {rpy[1]:.3f}, {rpy[2]:.3f}) [yaw: {yaw_deg:+.1f}°]")

        # Calculate and display distances
        print("\nPairwise distances:")
        distances = []
        for i in range(env.NUM_DRONES):
            for j in range(i + 1, env.NUM_DRONES):
                distance = np.linalg.norm(env.INIT_XYZS[i, :2] - env.INIT_XYZS[j, :2])
                distances.append(distance)
                print(f"  Drone {i} <-> Drone {j}: {distance:.3f}m")

        print(f"\nDistance statistics:")
        print(f"  Min: {np.min(distances):.3f}m")
        print(f"  Max: {np.max(distances):.3f}m")
        print(f"  Mean: {np.mean(distances):.3f}m")
        print(f"  In range [2-4m]: {sum(2.0 <= d <= 4.0 for d in distances)}/{len(distances)}")

    env.close()
    print("\n" + "=" * 60)
    print("Random spawn test completed successfully!")


def quick_distribution_check():
    """Quick check of distance distribution over multiple resets."""
    print("\nRunning quick distribution check...")

    env = DogfightAviary(
        drone_model=DroneModel.CF2X,
        num_drones=4,
        gui=False,
        record=False,
        obs=ObservationType.KIN,
        act=ActionType.PID
    )

    all_distances = []
    num_tests = 20

    for i in range(num_tests):
        env.reset(seed=i)

        # Calculate all pairwise distances
        for j in range(env.NUM_DRONES):
            for k in range(j + 1, env.NUM_DRONES):
                distance = np.linalg.norm(env.INIT_XYZS[j, :2] - env.INIT_XYZS[k, :2])
                all_distances.append(distance)

    env.close()

    # Analyze distribution
    all_distances = np.array(all_distances)
    in_range = (all_distances >= 2.0) & (all_distances <= 4.0)

    print(f"Total distances collected: {len(all_distances)}")
    print(f"Distance range: {np.min(all_distances):.3f}m - {np.max(all_distances):.3f}m")
    print(f"Mean distance: {np.mean(all_distances):.3f}m")
    print(f"Std deviation: {np.std(all_distances):.3f}m")
    print(f"In target range [2-4m]: {np.sum(in_range)}/{len(all_distances)} ({np.mean(in_range) * 100:.1f}%)")

    # Check uniformity (simple bin test)
    bins = np.linspace(2.0, 4.0, 5)  # 4 bins between 2-4m
    target_distances = all_distances[in_range]
    if len(target_distances) > 0:
        hist, _ = np.histogram(target_distances, bins=bins)
        expected_per_bin = len(target_distances) / len(bins - 1)
        print(f"\nUniformity check (target range only):")
        for i, count in enumerate(hist):
            print(f"  Bin {i + 1} [{bins[i]:.1f}-{bins[i + 1]:.1f}m]: {count} (expected: ~{expected_per_bin:.1f})")


if __name__ == "__main__":
    test_random_spawn()
    quick_distribution_check()
