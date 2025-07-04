"""
Unit tests for scripted red drone policy.

Tests requirements:
- Red drone follows a scripted "pursue & fire" policy
- DoD: 100 episodes, Red hits Blue ≥ 65% of runs
- Uses deterministic seed for reproducible results
"""

import unittest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gym_pybullet_drones.envs.DogfightAviary import DogfightAviary
from gym_pybullet_drones.policies.scripted_red import ScriptedRedPolicy, create_scripted_red_policy
from gym_pybullet_drones.utils.enums import ActionType, ObservationType


class TestScriptedRedPolicy(unittest.TestCase):
    """Test cases for the scripted red policy."""

    def setUp(self):
        """Set up test environment and policy."""
        self.seed = 42  # Deterministic seed
        np.random.seed(self.seed)

        # Create environment with 4 drones (2 blue, 2 red)
        self.env = DogfightAviary(
            num_drones=4,
            gui=False,
            record=False,
            act=ActionType.PID,
            obs=ObservationType.KIN,
            pyb_freq=240,
            ctrl_freq=30
        )

        # Create scripted policy
        self.policy = create_scripted_red_policy(
            action_type=ActionType.PID,
            kp_xy=2.0,  #
            kp_z=2.0,
            target_altitude=0.5,
            engagement_range=0.3
        )

    def tearDown(self):
        """Clean up after tests."""
        self.env.close()

    def test_policy_initialization(self):
        """Test that the policy initializes correctly."""
        self.assertEqual(self.policy.action_type, ActionType.PID)
        self.assertEqual(self.policy.kp_xy, 2.0)
        self.assertEqual(self.policy.kp_z, 2.0)
        self.assertEqual(self.policy.target_altitude, 0.5)
        self.assertEqual(self.policy.engagement_range, 0.3)

    def test_invalid_action_type(self):
        """Test that invalid action types raise an error."""
        with self.assertRaises(ValueError):
            ScriptedRedPolicy(action_type=ActionType.RPM)

    def test_action_shape(self):
        """Test that actions have the correct shape."""
        obs, _ = self.env.reset(seed=self.seed)

        # Get team indices
        red_team = self.env.red_team
        blue_team = self.env.blue_team
        blue_alive = self.env.blue_alive

        # Test get_action method
        red_states = obs[red_team]
        blue_states = obs[blue_team]
        red_actions = self.policy.get_action(
            red_states, blue_states, red_team, blue_team, blue_alive
        )

        expected_shape = (len(red_team), 3)  # PID action type
        self.assertEqual(red_actions.shape, expected_shape)

        # Test get_full_action method
        full_actions = self.policy.get_full_action(obs, red_team, blue_team, blue_alive)
        expected_full_shape = (self.env.NUM_DRONES, 3)
        self.assertEqual(full_actions.shape, expected_full_shape)

    def test_altitude_control(self):
        """Test that red drones control altitude appropriately."""
        obs, _ = self.env.reset(seed=self.seed)

        red_team = self.env.red_team
        blue_team = self.env.blue_team
        blue_alive = self.env.blue_alive

        actions = self.policy.get_full_action(obs, red_team, blue_team, blue_alive)

        # Check that red drones target reasonable altitudes
        # (gradual climb towards target, not immediate jump)
        for red_idx in red_team:
            target_z = actions[red_idx, 2]
            current_z = obs[red_idx, 2]

            # Should be moving towards target altitude
            self.assertGreater(target_z, current_z)  # Climbing up
            self.assertLessEqual(target_z, self.policy.target_altitude + 0.1)  # Within reasonable range
            self.assertGreaterEqual(target_z, current_z)  # Not going below current

    def test_pursuit_behavior(self):
        """Test that red drones move toward blue drones."""
        obs, _ = self.env.reset(seed=self.seed)

        red_team = self.env.red_team
        blue_team = self.env.blue_team
        blue_alive = self.env.blue_alive

        # Get initial positions
        red_positions = obs[red_team, 0:2]  # XY positions
        blue_positions = obs[blue_team, 0:2]  # XY positions

        # Get actions
        actions = self.policy.get_full_action(obs, red_team, blue_team, blue_alive)
        red_targets = actions[red_team, 0:2]  # XY targets

        # Verify red drones target positions closer to blue drones
        for i, red_idx in enumerate(red_team):
            red_pos = red_positions[i]
            red_target = red_targets[i]

            # Find closest blue
            distances_to_blue = np.linalg.norm(blue_positions - red_pos, axis=1)
            closest_blue_pos = blue_positions[np.argmin(distances_to_blue)]

            # Target should be closer to blue than current position
            dist_current_to_blue = np.linalg.norm(red_pos - closest_blue_pos)
            dist_target_to_blue = np.linalg.norm(red_target - closest_blue_pos)

            # Allow some tolerance due to proportional control
            self.assertLessEqual(dist_target_to_blue, dist_current_to_blue + 0.1)

    def test_single_episode_execution(self):
        """Test that a single episode can run without errors."""
        obs, _ = self.env.reset(seed=self.seed)

        red_team = self.env.red_team
        blue_team = self.env.blue_team

        step_count = 0
        max_steps = 300  # Limit episode length

        while step_count < max_steps:
            # Get blue alive status
            blue_alive = self.env.blue_alive

            # Generate actions using scripted policy
            actions = self.policy.get_full_action(obs, red_team, blue_team, blue_alive)

            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(actions)
            step_count += 1

            # Check that info contains expected keys
            self.assertIn('blue_alive', info)
            self.assertIn('red_alive', info)
            self.assertIn('hits', info)

            if terminated or truncated:
                break

        # Episode should complete without errors
        self.assertLessEqual(step_count, max_steps)

    def test_multiple_episodes_success_rate(self):
        """
        Test the DoD requirement: Red hits Blue ≥ 65% of 100 episodes.
        
        This is the main test for the E2-1 story requirements.
        """
        num_episodes = 100
        red_wins = 0

        print(f"\nRunning {num_episodes} episodes with deterministic seed {self.seed}...")

        for episode in range(num_episodes):
            # Reset with deterministic seed for reproducibility
            obs, _ = self.env.reset(seed=self.seed + episode)

            red_team = self.env.red_team
            blue_team = self.env.blue_team

            step_count = 0
            max_steps = 400  # Limit episode length
            episode_red_won = False

            while step_count < max_steps:
                # Get current team status
                blue_alive = self.env.blue_alive
                red_alive = self.env.red_alive

                # Check if red won (all blue eliminated)
                if not any(blue_alive):
                    episode_red_won = True
                    red_wins += 1
                    break

                # Check if blue won (all red eliminated)
                if not any(red_alive):
                    break

                # Generate actions using scripted policy
                actions = self.policy.get_full_action(obs, red_team, blue_team, blue_alive)

                # Step environment
                obs, reward, terminated, truncated, info = self.env.step(actions)
                step_count += 1

                if terminated or truncated:
                    # Check final status for winner
                    if not any(blue_alive) and any(red_alive):
                        episode_red_won = True
                        red_wins += 1
                    break

            # Print progress every 20 episodes
            if (episode + 1) % 20 == 0:
                current_rate = red_wins / (episode + 1) * 100
                print(f"Episode {episode + 1:3d}: Red wins {red_wins:2d}/{episode + 1:2d} ({current_rate:.1f}%)")

        # Calculate success rate
        success_rate = red_wins / num_episodes
        success_percentage = success_rate * 100

        print(f"\nFinal Results:")
        print(f"Episodes run: {num_episodes}")
        print(f"Red wins: {red_wins}")
        print(f"Success rate: {success_percentage:.1f}%")
        print(f"DoD requirement: ≥65%")

        # Verify DoD requirement
        self.assertGreaterEqual(success_rate, 0.65,
                                f"Red success rate {success_percentage:.1f}% is below the 65% DoD requirement")

        return success_rate


class TestScriptedRedPolicyVelocity(unittest.TestCase):
    """Test the scripted policy with velocity action type."""

    def setUp(self):
        """Set up test environment with velocity control."""
        self.seed = 42
        np.random.seed(self.seed)

        self.env = DogfightAviary(
            num_drones=4,
            gui=False,
            record=False,
            act=ActionType.VEL,
            obs=ObservationType.KIN,
            pyb_freq=240,
            ctrl_freq=30
        )

        self.policy = create_scripted_red_policy(
            action_type=ActionType.VEL,
            kp_xy=2.0,
            kp_z=3.0,
            target_altitude=1.0
        )

    def tearDown(self):
        """Clean up after tests."""
        self.env.close()

    def test_velocity_action_shape(self):
        """Test that velocity actions have the correct shape."""
        obs, _ = self.env.reset(seed=self.seed)

        red_team = self.env.red_team
        blue_team = self.env.blue_team
        blue_alive = self.env.blue_alive

        actions = self.policy.get_full_action(obs, red_team, blue_team, blue_alive)
        expected_shape = (self.env.NUM_DRONES, 4)  # VEL action type
        self.assertEqual(actions.shape, expected_shape)


def run_dod_verification():
    """
    Standalone function to run the DoD verification test.
    
    Returns the success rate for external verification.
    """
    # Create a test instance
    test_instance = TestScriptedRedPolicy()
    test_instance.setUp()

    try:
        success_rate = test_instance.test_multiple_episodes_success_rate()
        return success_rate
    finally:
        test_instance.tearDown()


if __name__ == '__main__':
    unittest.main(verbosity=2)
