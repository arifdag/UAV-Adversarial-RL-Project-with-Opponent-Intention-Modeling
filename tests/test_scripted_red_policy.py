import numpy as np

from uav_intent_rl.envs.DogfightAviary import DogfightAviary
from uav_intent_rl.policies import ScriptedRedPolicy
from gym_pybullet_drones.utils.enums import ActionType, ObservationType


def _build_env():
    """Utility that constructs the DogfightAviary with deterministic poses.

    The red drone is yaw-aligned towards the blue drone so that its forward
    shooting cone immediately covers the target. This avoids needing yaw
    control – the scripted policy only steers using linear velocities.
    """
    # Yaw that points from red's default spawn (≈[0.18,0.18]) towards blue at
    # the origin. Equivalent to -135° (or 225°) w.r.t. the +X axis.
    yaw_red = np.arctan2(-1.0, -1.0)  # −3π/4 ≈ −2.35619 rad

    initial_rpys = np.array(
        [
            [0.0, 0.0, 0.0],   # blue – facing +X (irrelevant)
            [0.0, 0.0, yaw_red],  # red – facing blue
        ],
        dtype=float,
    )

    return DogfightAviary(
        num_drones=2,
        obs=ObservationType.KIN,
        act=ActionType.VEL,
        gui=False,
        initial_rpys=initial_rpys,
    )


def test_scripted_red_policy_hits_blue_majority():
    """Red should down blue in ≥ 65 % of 100 deterministic episodes."""
    env = _build_env()
    policy = ScriptedRedPolicy()

    hits = 0
    episodes = 100

    for ep in range(episodes):
        # Use a reproducible but varied seed each episode
        obs, _ = env.reset(seed=ep)
        terminated = truncated = False
        while not (terminated or truncated):
            action = policy(env)
            obs, _, terminated, truncated, _ = env.step(action)

        if env._blue_down():  # type: ignore[attr-defined] – helper from DogfightAviary
            hits += 1

    env.close()

    assert hits >= int(0.65 * episodes), (
        f"Red hit blue only {hits} times out of {episodes} (expected ≥ 65)"
    )

    # For visibility during test runs, print the achieved hit rate
    print(f"[TEST] Scripted red policy hit rate: {hits / episodes * 100:.1f}% ({hits}/{episodes})") 