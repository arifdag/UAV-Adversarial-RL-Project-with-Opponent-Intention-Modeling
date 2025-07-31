"""Unit tests for random spawn logic in `DogfightAviary`.

The test verifies that the environment initialises the two drones with:
1. Horizontal separation uniformly sampled in the range [2, 4] metres.
2. Independent yaw angles within [−π, π].
"""

import numpy as np

from uav_intent_rl.envs import MultiDroneDogfightAviary


def test_random_spawn_distribution():
    """Spawn distance and yaw values lie in the expected ranges and vary."""
    env = MultiDroneDogfightAviary(gui=False)

    distances = []
    yaws = []

    # Use deterministic seeds for reproducibility while ensuring varied samples
    for seed in range(100):
        env.reset(seed=seed)

        # Record horizontal distance between the two drones
        pos = env.INIT_XYZS
        distances.append(np.linalg.norm(pos[0] - pos[1]))

        # Record each drone's yaw
        yaws.extend(env.INIT_RPYS[:, 2])

    env.close()

    distances = np.asarray(distances)
    yaws = np.asarray(yaws)

    # All distances within [2, 4] metres
    assert np.all(distances >= 2.0 - 1e-6)
    assert np.all(distances <= 4.0 + 1e-6)

    # Ensure variation across samples (not a fixed deterministic spawn)
    assert np.ptp(distances) > 1.0  # range (max−min) > 1 m indicates spread

    # Yaw angles within [−π, π] and show significant spread
    assert np.all(yaws >= -np.pi - 1e-6)
    assert np.all(yaws <= np.pi + 1e-6)
    assert np.ptp(yaws) > 3.0  # >180° spread across samples 