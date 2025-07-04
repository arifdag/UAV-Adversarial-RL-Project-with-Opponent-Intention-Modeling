import numpy as np
import time
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType


def run_hover_simulation():
    """
    Spawns two Crazyflie-2X quads in a head-to-head arena
    and steps the Gym environment for 200 steps.
    """
    # Initialize the environment
    env = MultiHoverAviary(
        num_drones=2,
        obs=ObservationType.KIN,
        act=ActionType.VEL,
        gui=True
    )

    # Reset the environment and get initial observations
    obs, info = env.reset()
    print("Initial Observations:\n", obs)

    # Simulation loop
    for i in range(200):
        # Define a zero action (hover) for both drones
        action = np.zeros(env.action_space.shape)

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Print the results for each step
        print(f"\n--- Step {i + 1} ---")
        print("Observations:\n", obs)
        print("Rewards:\n", reward)

        # A short pause to make the simulation easier to watch
        time.sleep(1. / 240.)

    env.close()


if __name__ == "__main__":
    run_hover_simulation()
