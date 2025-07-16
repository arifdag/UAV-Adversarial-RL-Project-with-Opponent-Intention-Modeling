"""Scripted policy for the *red* drone in `DogfightAviary`.

This deterministic hand-crafted controller steers the red drone towards the
stationary blue drone using simple proportional velocity control in the X–Y
plane while maintaining a target altitude of 1 m. The action space is
`ActionType.VEL`, therefore the 4-element vector per drone corresponds to a
normalised velocity **direction** (first three components) and a speed scale
(last component). When the red drone approaches within 0.3 m of the blue
drone and keeps it inside its forward 60° field-of-view, the environment
registers a *hit* and ends the episode.

The policy is designed for unit-testing purposes and is not meant to be a
competitive baseline.
"""
from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["ScriptedRedPolicy"]


class ScriptedRedPolicy:  # noqa: D101 – simple callable helper
    def __init__(self, target_alt: float = 1.0) -> None:
        self.target_alt = float(target_alt)

    # ---------------------------------------------------------------------
    # API helpers
    # ---------------------------------------------------------------------

    def reset(self) -> None:
        """No internal state to reset – placeholder for compatibility."""
        # Nothing to do – method kept to mirror RL policy interfaces.
        return None

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def _compute_action(self, env: Any) -> np.ndarray:  # noqa: ANN401 – env from gym
        """Compute an (N_DRONES×4) normalised action for the given *env*.

        Notes
        -----
        * Drone **0** (blue) remains stationary – its action is all-zeros.
        * Drone **1** (red) receives a velocity command that:
          1. Drives it horizontally towards the blue drone.
          2. Corrects altitude towards :pyattr:`self.target_alt`.
          3. Uses full speed (scale = 1) in the provided `ActionType.VEL` space.
        """
        # Retrieve complete state vectors
        state_blue = env._getDroneStateVector(0)  # type: ignore[attr-defined]
        state_red = env._getDroneStateVector(1)   # type: ignore[attr-defined]

        # Relative position (blue − red)
        rel_pos = state_blue[0:3] - state_red[0:3]
        # Maintain fixed target altitude
        rel_pos[2] = self.target_alt - state_red[2]

        # Direction – normalise to unit vector (avoid division by zero)
        norm = np.linalg.norm(rel_pos)
        direction = rel_pos / norm if norm > 1e-6 else np.zeros(3)

        # Build normalised action array
        action = np.zeros((env.NUM_DRONES, 4), dtype=np.float32)
        # Blue drone (index 0) – stationary adversary
        action[0, :] = 0.0
        # Red drone (index 1)
        action[1, 0:3] = direction
        action[1, 3] = 1.0  # maximum speed scaling in `ActionType.VEL`

        # Clip to the action bounds [−1, 1] for safety
        np.clip(action, -1.0, 1.0, out=action)
        return action

    # ------------------------------------------------------------------
    # Callable shortcut
    # ------------------------------------------------------------------

    def __call__(self, env: Any) -> np.ndarray:  # noqa: D401
        """Make the policy instance directly callable like a function."""
        return self._compute_action(env) 