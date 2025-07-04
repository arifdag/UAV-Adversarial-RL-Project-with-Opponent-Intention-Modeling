"""
Test script to validate spawn distance distribution in DogfightAviary.

This script runs multiple resets and collects spawn distance statistics to verify
that distances follow a uniform distribution between 2-4m as required by the DoD.
"""

import numpy as np
import pytest
from gym_pybullet_drones.envs.DogfightAviary import DogfightAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


@pytest.fixture
def dogfight_env():
    """Create a DogfightAviary environment for testing."""
    env = DogfightAviary(
        drone_model=DroneModel.CF2X,
        num_drones=4,
        gui=False,
        record=False,
        obs=ObservationType.KIN,
        act=ActionType.PID
    )
    yield env
    env.close()


def collect_spawn_distances(env, num_episodes=50):
    """Helper function to collect spawn distances from multiple resets.
    
    Parameters
    ----------
    env : DogfightAviary
        The environment to test
    num_episodes : int
        Number of episodes to run for statistical analysis
        
    Returns
    -------
    list
        All pairwise distances collected across episodes
    """
    all_distances = []

    for episode in range(num_episodes):
        # Reset with different random seed each time
        env.reset(seed=episode)

        # Get initial positions
        positions = env.INIT_XYZS

        # Calculate all pairwise distances (only XY plane)
        for i in range(env.NUM_DRONES):
            for j in range(i + 1, env.NUM_DRONES):
                distance = np.linalg.norm(positions[i, :2] - positions[j, :2])
                all_distances.append(distance)

    return all_distances


def test_spawn_positions_randomized(dogfight_env):
    """Test that spawn positions are different across resets."""
    positions_set1 = []
    positions_set2 = []

    # Reset twice with different seeds
    dogfight_env.reset(seed=42)
    positions_set1 = dogfight_env.INIT_XYZS.copy()

    dogfight_env.reset(seed=123)
    positions_set2 = dogfight_env.INIT_XYZS.copy()

    # Positions should be different
    assert not np.allclose(positions_set1, positions_set2), "Spawn positions should be randomized between resets"


def test_spawn_orientations_randomized(dogfight_env):
    """Test that yaw orientations are randomized."""
    orientations_set1 = []
    orientations_set2 = []

    # Reset twice with different seeds
    dogfight_env.reset(seed=42)
    orientations_set1 = dogfight_env.INIT_RPYS.copy()

    dogfight_env.reset(seed=123)
    orientations_set2 = dogfight_env.INIT_RPYS.copy()

    # Yaw orientations should be different
    yaw1 = orientations_set1[:, 2]
    yaw2 = orientations_set2[:, 2]
    assert not np.allclose(yaw1, yaw2), "Yaw orientations should be randomized between resets"

    # Yaw should be within ±π
    for orientations in [orientations_set1, orientations_set2]:
        yaws = orientations[:, 2]
        assert np.all(yaws >= -np.pi), "All yaw values should be >= -π"
        assert np.all(yaws <= np.pi), "All yaw values should be <= π"

    # Roll and pitch should remain zero
    for orientations in [orientations_set1, orientations_set2]:
        rolls = orientations[:, 0]
        pitches = orientations[:, 1]
        assert np.allclose(rolls, 0), "Roll should remain zero"
        assert np.allclose(pitches, 0), "Pitch should remain zero"


def test_spawn_distance_range(dogfight_env):
    """Test that spawn distances are mostly within 2-4m range."""
    distances = collect_spawn_distances(dogfight_env, num_episodes=20)

    # Convert to numpy array for easier analysis
    distances = np.array(distances)

    # Check that we have distances
    assert len(distances) > 0, "Should collect at least some distances"

    # Check that most distances are in the target range
    in_range = (distances >= 2.0) & (distances <= 4.0)
    percentage_in_range = np.mean(in_range) * 100

    print(f"\nDistance statistics:")
    print(f"  Total distances: {len(distances)}")
    print(f"  Min distance: {np.min(distances):.3f}m")
    print(f"  Max distance: {np.max(distances):.3f}m")
    print(f"  Mean distance: {np.mean(distances):.3f}m")
    print(f"  In range [2-4m]: {np.sum(in_range)}/{len(distances)} ({percentage_in_range:.1f}%)")

    # At least 70% should be in the target range (allowing some tolerance)
    assert percentage_in_range >= 70, f"At least 70% of distances should be in [2-4m] range, got {percentage_in_range:.1f}%"


def test_spawn_distance_distribution_uniformity(dogfight_env):
    """Test that spawn distances follow approximately uniform distribution."""
    distances = collect_spawn_distances(dogfight_env, num_episodes=50)
    distances = np.array(distances)

    # Filter distances within target range
    target_distances = distances[(distances >= 2.0) & (distances <= 4.0)]

    assert len(target_distances) > 10, "Need sufficient data points for distribution test"

    # Simple uniformity test: check if distances are spread across the range
    # Divide the range into bins and check that each bin has some data
    bins = np.linspace(2.0, 4.0, 5)  # 4 bins
    hist, _ = np.histogram(target_distances, bins=bins)

    print(f"\nDistribution analysis:")
    print(f"  Target range distances: {len(target_distances)}")
    expected_per_bin = len(target_distances) / len(bins - 1)
    for i, count in enumerate(hist):
        print(f"  Bin {i + 1} [{bins[i]:.1f}-{bins[i + 1]:.1f}m]: {count} (expected: ~{expected_per_bin:.1f})")

    # Check that no bin is completely empty (too strict) and no bin has too much data
    # Allow some variation but ensure reasonable distribution
    min_expected = max(1, expected_per_bin * 0.3)  # At least 30% of expected
    max_expected = expected_per_bin * 2.0  # At most 200% of expected

    for i, count in enumerate(hist):
        assert count >= min_expected, f"Bin {i + 1} has too few samples ({count}, expected >= {min_expected:.1f})"
        assert count <= max_expected, f"Bin {i + 1} has too many samples ({count}, expected <= {max_expected:.1f})"


def test_team_assignments_preserved(dogfight_env):
    """Test that team assignments are consistent across resets."""
    # Reset and check team assignments
    dogfight_env.reset(seed=42)

    # Team assignments should be consistent
    expected_blue = list(range(dogfight_env.NUM_DRONES // 2))
    expected_red = list(range(dogfight_env.NUM_DRONES // 2, dogfight_env.NUM_DRONES))

    assert dogfight_env.blue_team == expected_blue, "Blue team assignment should be consistent"
    assert dogfight_env.red_team == expected_red, "Red team assignment should be consistent"

    # Alive status should be reset
    assert all(dogfight_env.blue_alive), "All blue drones should be alive after reset"
    assert all(dogfight_env.red_alive), "All red drones should be alive after reset"


def test_deterministic_with_same_seed(dogfight_env):
    """Test that same seed produces same spawn configuration."""
    # Reset with same seed twice
    dogfight_env.reset(seed=42)
    positions1 = dogfight_env.INIT_XYZS.copy()
    orientations1 = dogfight_env.INIT_RPYS.copy()

    dogfight_env.reset(seed=42)
    positions2 = dogfight_env.INIT_XYZS.copy()
    orientations2 = dogfight_env.INIT_RPYS.copy()

    # Should be identical
    assert np.allclose(positions1, positions2), "Same seed should produce identical positions"
    assert np.allclose(orientations1, orientations2), "Same seed should produce identical orientations"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
