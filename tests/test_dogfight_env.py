import pytest
import numpy as np
from gymnasium import spaces

from gym_pybullet_drones.envs.DogfightAviary import DogfightAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class TestDogfightAviary:
    """Test suite for DogfightAviary environment."""

    def setup_method(self):
        """Set up test environment before each test method."""
        self.env = DogfightAviary(
            drone_model=DroneModel.CF2X,
            num_drones=4,
            physics=Physics.PYB,
            gui=False,
            record=False,
            obs=ObservationType.KIN,
            act=ActionType.RPM
        )

    def teardown_method(self):
        """Clean up after each test method."""
        if hasattr(self, 'env'):
            self.env.close()

    def test_action_space_shape(self):
        """Test that action space has correct shape."""
        action_space = self.env.action_space

        # Should be a Box space
        assert isinstance(action_space, spaces.Box)

        # Should have shape (num_drones, 4) for RPM actions
        expected_shape = (self.env.NUM_DRONES, 4)
        assert action_space.shape == expected_shape, f"Expected action space shape {expected_shape}, got {action_space.shape}"

        # Should have proper bounds
        assert np.all(action_space.low == -1.0), "Action space lower bound should be -1.0"
        assert np.all(action_space.high == 1.0), "Action space upper bound should be 1.0"

    def test_observation_space_shape(self):
        """Test that observation space has correct shape."""
        obs_space = self.env.observation_space

        # Should be a Box space
        assert isinstance(obs_space, spaces.Box)

        # For kinematic observations, should have proper dimensionality
        # Each drone has a state vector, exact size depends on implementation
        assert len(obs_space.shape) > 0, "Observation space should have at least one dimension"

        # Verify observation space is not empty
        assert obs_space.shape[0] > 0, "Observation space should not be empty"

        # Should have correct number of drones
        assert obs_space.shape[
                   0] == self.env.NUM_DRONES, f"Observation space should have {self.env.NUM_DRONES} drone observations"

    def test_reset_returns_valid_observation(self):
        """Test that reset returns observation with correct shape."""
        obs, info = self.env.reset()

        # Check observation shape matches observation space
        expected_shape = self.env.observation_space.shape
        assert obs.shape == expected_shape, f"Reset observation shape {obs.shape} doesn't match observation space shape {expected_shape}"

        # Check observation is numeric (not NaN or inf)
        assert np.all(np.isfinite(obs)), "Reset observation contains non-finite values"

        # Check info is a dictionary
        assert isinstance(info, dict), "Info should be a dictionary"

    def test_step_returns_valid_shapes(self):
        """Test that step returns observations with correct shapes."""
        self.env.reset()

        # Create a valid action
        action = self.env.action_space.sample()

        # Take a step
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Check observation shape
        expected_shape = self.env.observation_space.shape
        assert obs.shape == expected_shape, f"Step observation shape {obs.shape} doesn't match observation space shape {expected_shape}"

        # Check observation is numeric (not NaN or inf)
        assert np.all(np.isfinite(obs)), "Step observation contains non-finite values"

        # Check reward is a number
        assert isinstance(reward, (int, float, np.number)), f"Reward should be a number, got {type(reward)}"

        # Check terminated and truncated are booleans
        assert isinstance(terminated, bool), f"Terminated should be boolean, got {type(terminated)}"
        assert isinstance(truncated, bool), f"Truncated should be boolean, got {type(truncated)}"

        # Check info is a dictionary
        assert isinstance(info, dict), "Info should be a dictionary"

    def test_environment_constants(self):
        """Test that environment constants are set correctly."""
        assert self.env.EPISODE_LEN_SEC == 30, f"Episode length should be 30 seconds, got {self.env.EPISODE_LEN_SEC}"
        assert self.env.DEF_DMG_RADIUS == 0.3, f"Damage radius should be 0.3m, got {self.env.DEF_DMG_RADIUS}"

    def test_team_initialization(self):
        """Test that teams are properly initialized."""
        # Should have blue and red teams
        assert hasattr(self.env, 'blue_team'), "Environment should have blue_team attribute"
        assert hasattr(self.env, 'red_team'), "Environment should have red_team attribute"

        # Teams should be non-empty
        assert len(self.env.blue_team) > 0, "Blue team should not be empty"
        assert len(self.env.red_team) > 0, "Red team should not be empty"

        # Teams should split drones evenly
        total_drones = len(self.env.blue_team) + len(self.env.red_team)
        assert total_drones == self.env.NUM_DRONES, f"Total team members {total_drones} should equal number of drones {self.env.NUM_DRONES}"

    def test_different_drone_counts(self):
        """Test environment with different numbers of drones."""
        for num_drones in [2, 4, 6, 8]:
            env = DogfightAviary(
                num_drones=num_drones,
                gui=False,
                record=False
            )

            # Test action space shape
            expected_action_shape = (num_drones, 4)
            assert env.action_space.shape == expected_action_shape, f"Action space shape for {num_drones} drones should be {expected_action_shape}"

            # Test observation space shape
            assert env.observation_space.shape[
                       0] == num_drones, f"Observation space should have {num_drones} drone observations"

            # Test reset works and returns correct shape
            obs, info = env.reset()
            expected_obs_shape = env.observation_space.shape
            assert obs.shape == expected_obs_shape, f"Reset observation shape for {num_drones} drones should be {expected_obs_shape}"

            # Test step works and returns correct shape
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs.shape == expected_obs_shape, f"Step observation shape for {num_drones} drones should be {expected_obs_shape}"

            env.close()

    def test_action_sampling_and_execution(self):
        """Test that we can sample actions and execute them without errors."""
        self.env.reset()

        # Test multiple action samples
        for _ in range(5):
            action = self.env.action_space.sample()

            # Verify action shape
            expected_action_shape = (self.env.NUM_DRONES, 4)
            assert action.shape == expected_action_shape, f"Sampled action shape should be {expected_action_shape}"

            # Verify action bounds
            assert np.all(action >= -1.0) and np.all(action <= 1.0), "Sampled actions should be within [-1, 1]"

            # Execute action
            obs, reward, terminated, truncated, info = self.env.step(action)

            # Basic checks
            assert obs.shape == self.env.observation_space.shape
            assert isinstance(reward, (int, float, np.number))
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)


if __name__ == "__main__":
    pytest.main([__file__])
