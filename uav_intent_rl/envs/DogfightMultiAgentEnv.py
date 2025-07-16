from __future__ import annotations

"""RLlib‐compatible multi-agent wrapper for :class:`DogfightAviary`.

Ray RLlib expects environments to conform to the :class:`ray.rllib.env.MultiAgentEnv`
interface, i.e. *step* / *reset* must use dictionaries keyed by agent IDs and the
*done* dict must include the special "__all__" key.  This thin adapter converts
our existing single-`gymnasium` environment to that interface so that both UAVs
("blue" & "red") can be trained concurrently.

The wrapper keeps almost all the heavy-lifting in :class:`DogfightAviary`; its sole
job is to translate inputs/outputs between the two APIs.
"""

from typing import Dict, Tuple

import numpy as np

from ray.rllib.env.multi_agent_env import MultiAgentEnv

from .DogfightAviary import DogfightAviary

__all__: list[str] = [
    "DogfightMultiAgentEnv",
    "default_policy_mapping_fn",
]


class DogfightMultiAgentEnv(DogfightAviary, MultiAgentEnv):
    """Two-drone dog-fight exposed as an RLlib :class:`MultiAgentEnv`."""

    #: String identifiers used by RLlib for each drone
    AGENT_IDS: Tuple[str, str] = ("blue", "red")

    # ------------------------------------------------------------------
    # RLlib passes an ``EnvContext`` object to the constructor.  Convert it
    # into keyword arguments understood by :class:`DogfightAviary`.
    # ------------------------------------------------------------------

    def __init__(self, env_config: dict | "EnvContext" | None = None):  # type: ignore[override]
        """Create the underlying :class:`DogfightAviary` and derive MA spaces.

        RLlib passes an :class:`ray.tune.registry.EnvContext` – a dict-like
        wrapper containing both the user-supplied *env_config* entries and
        some bookkeeping fields (worker, vector_idx, etc.).  We only forward
        the user parameters that :class:`DogfightAviary` actually recognises.
        """

        # ---------------------------------------------
        # 1. Convert *env_config* → plain dict of kwargs
        # ---------------------------------------------
        if env_config is None:
            env_config = {}

        cfg_dict = dict(env_config) if hasattr(env_config, "keys") else {}

        # ---------------------------------------------
        # 2. Bootstrap the single-agent base environment
        # ---------------------------------------------
        super().__init__(**cfg_dict)

        # -------------------------------------------------
        # 3. Derive per-agent action/observation spaces and
        #    expose them as dicts → required by RLlib.
        # -------------------------------------------------
        from gymnasium import spaces  # local import to avoid hard dep in docs

        # Get the actual observation shape by computing a sample observation
        # This ensures the observation space matches what the environment actually produces
        sample_obs = self._computeObs()
        obs_shape = sample_obs[0].shape  # Shape of individual agent observation
        obs_low = -np.inf * np.ones(obs_shape, dtype=np.float32)
        obs_high = np.inf * np.ones(obs_shape, dtype=np.float32)

        self._single_observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        act_low, act_high = self.action_space.low[0], self.action_space.high[0]
        self._single_action_space = spaces.Box(low=act_low, high=act_high, dtype=np.float32)

        # Replace Gymnasium spaces with dicts keyed by agent_id so that
        # `env.observation_space["blue"]` works as RLlib expects.
        self.observation_space: Dict[str, spaces.Space] = {
            agent_id: self._single_observation_space for agent_id in self.AGENT_IDS
        }
        self.action_space: Dict[str, spaces.Space] = {
            agent_id: self._single_action_space for agent_id in self.AGENT_IDS
        }

        # Inform RLlib about the available agent ids.
        self._agent_ids = set(self.AGENT_IDS)

        # PettingZoo-style attributes that RLlib now looks for
        self.possible_agents = list(self.AGENT_IDS)
        self.agents = list(self.AGENT_IDS)

    # ------------------------------------------------------------------
    # Gymnasium → MultiAgentEnv translations
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: Dict | None = None):  # type: ignore[override]
        """Reset the underlying :class:`DogfightAviary` and convert outputs.

        Returns
        -------
        Tuple[Dict[str, np.ndarray], Dict[str, Dict]]
            Observation & info dictionaries keyed by "blue" / "red".
        """

        obs_arr, info = super().reset(seed=seed, options=options)

        obs_dict: Dict[str, np.ndarray] = {
            "blue": obs_arr[0].astype(np.float32),
            "red": obs_arr[1].astype(np.float32),
        }
        # Pass the *same* info dict to both agents for now (empty by default)
        info_dict: Dict[str, Dict] = {
            "blue": info.copy() if isinstance(info, dict) else {},
            "red": info.copy() if isinstance(info, dict) else {},
        }

        # Update live-agent list (all agents are alive at reset)
        self.agents = list(self.AGENT_IDS)
        return obs_dict, info_dict

    # ------------------------------------------------------------------
    # MultiAgentEnv interface
    # ------------------------------------------------------------------

    def step(
        self, action_dict: Dict[str, np.ndarray]
    ) -> Tuple[
        Dict[str, np.ndarray],  # observations
        Dict[str, float],       # rewards
        Dict[str, bool],        # terminateds
        Dict[str, bool],        # truncateds
        Dict[str, Dict],        # infos
    ]:  # type: ignore[override]
        """Translate *action_dict* → underlying env, then back again."""

        # ------------------------------------------------------------------
        # 1. Re-assemble actions into the (NUM_DRONES, A_DIM) array expected by
        #    :class:`DogfightAviary`.
        # ------------------------------------------------------------------
        # Determine per-drone action dimensionality from the single-agent space
        act_dim = self._single_action_space.shape[0]
        batched_actions = np.zeros((2, act_dim), dtype=np.float32)
        for idx, agent_id in enumerate(self.AGENT_IDS):
            if agent_id in action_dict:
                batched_actions[idx] = np.asarray(action_dict[agent_id], dtype=np.float32)

        # ------------------------------------------------------------------
        # 2. Step the underlying environment
        # ------------------------------------------------------------------
        obs_arr, reward_scalar, terminated_scalar, truncated_scalar, info = super().step(batched_actions)

        # ------------------------------------------------------------------
        # 3. Convert everything back to RLlib dictionaries
        # ------------------------------------------------------------------
        obs_dict: Dict[str, np.ndarray] = {
            "blue": obs_arr[0].astype(np.float32),
            "red": obs_arr[1].astype(np.float32),
        }

        # The original reward is defined from *blue*'s perspective; taking the
        # negative gives a symmetric reward for the *red* drone.
        reward_dict: Dict[str, float] = {
            "blue": float(reward_scalar),
            "red": float(-reward_scalar),
        }

        terminated_dict: Dict[str, bool] = {
            "blue": bool(terminated_scalar),
            "red": bool(terminated_scalar),
            "__all__": bool(terminated_scalar),
        }

        truncated_dict: Dict[str, bool] = {
            "blue": bool(truncated_scalar),
            "red": bool(truncated_scalar),
            "__all__": bool(truncated_scalar),
        }

        # Update live-agent list: if episode ended remove agents
        if terminated_scalar or truncated_scalar:
            self.agents = []

        info_dict: Dict[str, Dict] = {
            "blue": info.copy() if isinstance(info, dict) else {},
            "red": info.copy() if isinstance(info, dict) else {},
        }

        return obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict

    # ------------------------------------------------------------------
    # RLlib helper
    # ------------------------------------------------------------------

    def get_agent_ids(self):  # noqa: D401
        """Return the set of agent identifiers used by this environment."""

        return self._agent_ids

    # ------------------------------------------------------------------
    # Space helpers expected by RLlib's new API stack
    # ------------------------------------------------------------------

    def get_action_space(self, _agent_id: str | None = None):  # type: ignore[override]
        """Return the *per-agent* action space (identical for all agents)."""

        return self._single_action_space

    def get_observation_space(self, _agent_id: str | None = None):  # type: ignore[override]
        """Return the *per-agent* observation space (identical for all agents)."""

        return self._single_observation_space


# ----------------------------------------------------------------------
# Default *policy_mapping_fn* for RLlib experiments
# ----------------------------------------------------------------------

def default_policy_mapping_fn(agent_id: str, *_args, **_kwargs) -> str:  # noqa: D401
    """Return the policy ID for *agent_id*.

    RLlib will call this function for every agent in the environment to decide
    which *policy* object should control it.  The mapping implemented here is
    trivial – it assigns one distinct policy per drone with the same name as
    the agent ID ("blue" / "red").  When using homogeneous/shared weights, you
    can map both to a single string instead, e.g.::

        return "shared_ppo"

    but keeping the 1-to-1 mapping is convenient for debugging and allows easy
    extension to heterogeneous policies later on.
    """

    return str(agent_id) 