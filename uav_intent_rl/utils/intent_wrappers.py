from __future__ import annotations

"""Environment wrappers used by *IntentPPO* training scripts."""

from typing import Any

import gymnasium as gym
import numpy as np

from uav_intent_rl.policies.scripted_red import ScriptedRedPolicy
from uav_intent_rl.algo.intent_ppo import bucketize_red_action

__all__ = ["BlueVsFixedRedWrapper"]


class BlueVsFixedRedWrapper(gym.Wrapper):
    """Same as ppo_nomodel.BlueVsFixedRedWrapper but adds *red_bucket* info."""

    def __init__(self, env: gym.Env, red_policy: ScriptedRedPolicy | None = None):  # noqa: D401
        super().__init__(env)
        self._red_policy = red_policy or ScriptedRedPolicy()

        # Action space unchanged compared to ppo_nomodel wrapper
        assert env.action_space.shape == (2, 4)
        low, high = env.action_space.low[0], env.action_space.high[0]
        self.action_space = gym.spaces.Box(low=low, high=high, shape=(4,), dtype=np.float32)

        # Observation flattened (both drones)
        obs_shape = int(np.prod(env.observation_space.shape))
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32
        )

    # ---------------------------------------------------------------------
    # Overrides
    # ---------------------------------------------------------------------

    def reset(self, **kwargs):  # type: ignore[override]
        obs, info = self.env.reset(**kwargs)
        return obs.flatten().astype(np.float32), info

    def step(self, action):  # type: ignore[override]
        full_action = self._red_policy(self.env)
        full_action[0] = np.asarray(action, dtype=np.float32)

        # Bucketise the red action (predicted target)
        red_bucket = bucketize_red_action(full_action[1])

        obs, reward, terminated, truncated, info = self.env.step(full_action)
        info = dict(info)  # shallow copy
        info["red_bucket"] = red_bucket
        return (
            obs.flatten().astype(np.float32),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        ) 