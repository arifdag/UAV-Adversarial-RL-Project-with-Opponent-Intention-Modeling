"""RLlib-compatible multi-agent wrapper for :class:`Dogfight3v3Aviary`.

This wrapper converts the 3v3 dogfight environment to RLlib's MultiAgentEnv interface,
supporting 6 agents (3 blue + 3 red) with team-based coordination.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym_pybullet_drones.utils.enums import DroneModel, ActionType, ObservationType

from .Dogfight3v3Aviary import Dogfight3v3Aviary

__all__: list[str] = [
    "Dogfight3v3MultiAgentEnv",
    "default_3v3_policy_mapping_fn",
]


class Dogfight3v3MultiAgentEnv(Dogfight3v3Aviary, MultiAgentEnv):
    """Six-drone 3v3 team dog-fight exposed as an RLlib :class:`MultiAgentEnv`."""

    # String identifiers for each drone
    BLUE_AGENT_IDS: Tuple[str, str, str] = ("blue_0", "blue_1", "blue_2")
    RED_AGENT_IDS: Tuple[str, str, str] = ("red_0", "red_1", "red_2")
    AGENT_IDS: Tuple[str, str, str, str, str, str] = BLUE_AGENT_IDS + RED_AGENT_IDS

    def __init__(self, env_config: dict | "EnvContext" | None = None):  # type: ignore[override]
        """Create the underlying :class:`Dogfight3v3Aviary` and derive MA spaces."""

        # Convert env_config to plain dict
        if env_config is None:
            env_config = {}

        cfg_dict = dict(env_config) if hasattr(env_config, "keys") else {}
        
        # Convert string drone_model to DroneModel enum if needed
        if "drone_model" in cfg_dict and isinstance(cfg_dict["drone_model"], str):
            drone_model_str = cfg_dict["drone_model"].lower()
            if drone_model_str == "cf2x":
                cfg_dict["drone_model"] = DroneModel.CF2X
            elif drone_model_str == "cf2p":
                cfg_dict["drone_model"] = DroneModel.CF2P
            elif drone_model_str == "racer":
                cfg_dict["drone_model"] = DroneModel.RACE
            else:
                # Default to CF2X if unknown
                cfg_dict["drone_model"] = DroneModel.CF2X
        
        # Convert string obs to ObservationType enum if needed
        if "obs" in cfg_dict and isinstance(cfg_dict["obs"], str):
            obs_str = cfg_dict["obs"].lower()
            if obs_str == "kin":
                cfg_dict["obs"] = ObservationType.KIN
            elif obs_str == "rgb":
                cfg_dict["obs"] = ObservationType.RGB
            else:
                # Default to KIN if unknown
                cfg_dict["obs"] = ObservationType.KIN
        
        # Convert string act to ActionType enum if needed
        if "act" in cfg_dict and isinstance(cfg_dict["act"], str):
            act_str = cfg_dict["act"].lower()
            if act_str == "vel":
                cfg_dict["act"] = ActionType.VEL
            elif act_str == "rpm":
                cfg_dict["act"] = ActionType.RPM
            elif act_str == "pid":
                cfg_dict["act"] = ActionType.PID
            elif act_str == "one_d_rpm":
                cfg_dict["act"] = ActionType.ONE_D_RPM
            elif act_str == "one_d_pid":
                cfg_dict["act"] = ActionType.ONE_D_PID
            else:
                # Default to VEL if unknown
                cfg_dict["act"] = ActionType.VEL

        # Bootstrap the single-agent base environment
        super().__init__(**cfg_dict)

        # Derive per-agent action/observation spaces
        from gymnasium import spaces

        # Get sample observation to determine shape
        sample_obs = self._computeObs()
        obs_shape = sample_obs[0].shape
        obs_low = -np.inf * np.ones(obs_shape, dtype=np.float32)
        obs_high = np.inf * np.ones(obs_shape, dtype=np.float32)

        self._single_observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )
        act_low, act_high = self.action_space.low[0], self.action_space.high[0]
        self._single_action_space = spaces.Box(
            low=act_low, high=act_high, dtype=np.float32
        )

        # Create dicts keyed by agent_id
        self.observation_space: Dict[str, spaces.Space] = {
            agent_id: self._single_observation_space for agent_id in self.AGENT_IDS
        }
        self.action_space: Dict[str, spaces.Space] = {
            agent_id: self._single_action_space for agent_id in self.AGENT_IDS
        }

        # Inform RLlib about available agent ids
        self._agent_ids = set(self.AGENT_IDS)

        # PettingZoo-style attributes
        self.possible_agents = list(self.AGENT_IDS)
        self.agents = list(self.AGENT_IDS)

    def reset(self, *, seed: int | None = None, options: Dict | None = None):  # type: ignore[override]
        """Reset the underlying environment and convert outputs."""

        obs_arr, info = super().reset(seed=seed, options=options)

        # Convert to agent-specific observations
        obs_dict: Dict[str, np.ndarray] = {}
        for i, agent_id in enumerate(self.AGENT_IDS):
            obs_dict[agent_id] = obs_arr[i].astype(np.float32)

        # Pass same info to all agents
        info_dict: Dict[str, Dict] = {
            agent_id: info.copy() if isinstance(info, dict) else {}
            for agent_id in self.AGENT_IDS
        }

        # Update live-agent list
        self.agents = list(self.AGENT_IDS)
        return obs_dict, info_dict

    def step(
        self, action_dict: Dict[str, np.ndarray]
    ) -> Tuple[
        Dict[str, np.ndarray],  # observations
        Dict[str, float],       # rewards
        Dict[str, bool],        # terminateds
        Dict[str, bool],        # truncateds
        Dict[str, Dict],        # infos
    ]:  # type: ignore[override]
        """Translate action_dict to underlying env, then back again."""

        # Re-assemble actions into (NUM_DRONES, A_DIM) array
        act_dim = self._single_action_space.shape[0]
        batched_actions = np.zeros((6, act_dim), dtype=np.float32)
        
        for idx, agent_id in enumerate(self.AGENT_IDS):
            if agent_id in action_dict:
                batched_actions[idx] = np.asarray(action_dict[agent_id], dtype=np.float32)

        # Step the underlying environment
        obs_arr, reward_scalar, terminated_scalar, truncated_scalar, info = super().step(
            batched_actions
        )

        # Convert back to RLlib dictionaries
        obs_dict: Dict[str, np.ndarray] = {}
        for i, agent_id in enumerate(self.AGENT_IDS):
            obs_dict[agent_id] = obs_arr[i].astype(np.float32)

        # Team-based rewards
        reward_dict: Dict[str, float] = {}
        
        # Blue team gets the original reward
        for agent_id in self.BLUE_AGENT_IDS:
            reward_dict[agent_id] = float(reward_scalar)
        
        # Red team gets negative reward (symmetric)
        for agent_id in self.RED_AGENT_IDS:
            reward_dict[agent_id] = float(-reward_scalar)

        # Termination and truncation
        terminated_dict: Dict[str, bool] = {
            agent_id: bool(terminated_scalar) for agent_id in self.AGENT_IDS
        }
        terminated_dict["__all__"] = bool(terminated_scalar)

        truncated_dict: Dict[str, bool] = {
            agent_id: bool(truncated_scalar) for agent_id in self.AGENT_IDS
        }
        truncated_dict["__all__"] = bool(truncated_scalar)

        # Update live-agent list
        if terminated_scalar or truncated_scalar:
            self.agents = []

        # Info dict
        info_dict: Dict[str, Dict] = {
            agent_id: info.copy() if isinstance(info, dict) else {}
            for agent_id in self.AGENT_IDS
        }

        return obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict

    def get_agent_ids(self):
        """Return the set of agent identifiers used by this environment."""
        return self._agent_ids

    def get_action_space(self, _agent_id: str | None = None):
        """Return the per-agent action space."""
        return self._single_action_space

    def get_observation_space(self, _agent_id: str | None = None):
        """Return the per-agent observation space."""
        return self._single_observation_space

    def get_wrapper_attr(self, name: str):
        """Get an attribute from the underlying environment.
        
        This method is required by Stable-Baselines3's VecEnv interface.
        """
        if hasattr(self, name):
            return getattr(self, name)
        else:
            # Try to get from the underlying environment
            return getattr(super(), name)


# ----------------------------------------------------------------------
# Default policy_mapping_fn for 3v3 RLlib experiments
# ----------------------------------------------------------------------

def default_3v3_policy_mapping_fn(agent_id: str, *_args, **_kwargs) -> str:
    """Return the policy ID for agent_id in 3v3 environment.
    
    This mapping assigns:
    - All blue agents to "blue_policy"
    - All red agents to "red_policy"
    
    For shared policies across teams, you can use:
    - All agents to "shared_policy"
    
    For individual agent policies:
    - Each agent to its own policy name (agent_id)
    """
    
    if agent_id.startswith("blue"):
        return "blue_policy"
    elif agent_id.startswith("red"):
        return "red_policy"
    else:
        return str(agent_id) 