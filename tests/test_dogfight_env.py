"""Unit-tests for the custom DogfightAviary environment.

These tests confirm that the observation and action spaces have the expected
shapes and that a single environment step executes without error.
"""

import numpy as np

from uav_intent_rl.envs import DogfightAviary


def test_space_shapes():
    """The environment exposes the correct observation/action space shapes."""
    env = DogfightAviary(gui=False)

    # Expected shapes: (num_drones, action_dim) & (num_drones, obs_dim)
    assert env.action_space.shape == (2, 4)
    assert env.observation_space.shape == (2, 72)

    # reset returns initial observation with correct shape
    obs, _ = env.reset()
    assert obs.shape == (2, 72)

    # One simulation step with zero action should also respect shapes
    zero_action = np.zeros_like(env.action_space.low, dtype=np.float32)
    obs, reward, terminated, truncated, _ = env.step(zero_action)

    assert obs.shape == (2, 72)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)

    env.close()


# -----------------------------------------------------------------------------
# Additional test: hit detection and termination logic
# -----------------------------------------------------------------------------


def test_hit_and_termination():
    """Blue drone should score a hit when red is directly in front within range."""

    positions = np.array(
        [
            [0.0, 0.0, 1.0],  # blue at origin
            [0.2, 0.0, 1.0],  # red 0.2 m in +X — inside 0.3 m radius
        ]
    )

    # Both drones facing +X so only blue sees red in its FOV; red sees blue behind
    rpys = np.array(
        [
            [0.0, 0.0, 0.0],  # blue yaw 0 rad
            [0.0, 0.0, 0.0],  # red yaw 0 rad (won't see blue)
        ]
    )

    env = DogfightAviary(initial_xyzs=positions, initial_rpys=rpys, gui=False)

    env.reset()

    # Zero action (hover) for a single step
    zero_action = np.zeros_like(env.action_space.low, dtype=np.float32)
    _, reward, terminated, _, _ = env.step(zero_action)

    # Blue should hit red: reward > 0, red down, terminated
    assert reward > 0.0
    assert terminated is True
    assert env._red_down() is True
    assert env._blue_down() is False

    env.close() 