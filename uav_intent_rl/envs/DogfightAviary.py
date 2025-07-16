"""DogfightAviary environment for adversarial UAV combat.

Implements minimal reward/termination logic required for unit tests.
Further combat mechanics (hit detection, ammunition, etc.) can be developed
incrementally without breaking the public API.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary  # type: ignore
from gym_pybullet_drones.utils.enums import (
    ActionType,
    DroneModel,
    ObservationType,
    Physics,
)

# NOTE: The original specification asked to inherit from `MultiRLAviary`, but
# that class is not available in the installed version of *gym-pybullet-drones*.
# `BaseRLAviary` already provides full multi-agent RL functionality, so we use
# it directly.

__all__ = ["DogfightAviary"]


class DogfightAviary(BaseRLAviary):
    """Two-drone dog-fight RL environment.

    The environment exposes exactly the same observation and action spaces as
    `BaseRLAviary`, with the following domain-specific additions:

    * Episode length is capped at :pyattr:`EPISODE_LEN_SEC` seconds.
    * Reward is shaped as *hits* minus a small living-penalty to incentivise
      faster victories.
    * Episode terminates when either the *blue* or *red* drone is considered
      *down* (placeholder logic for now).
    """

    # Episode duration in seconds before *truncation*
    EPISODE_LEN_SEC: int = 15
    # Damage radius (m) within which a *hit* is registered. 0.3 m proved too
    # unforgiving for the learned policy; drones very rarely close that much
    # distance.  A looser 0.8 m still requires close-in manoeuvring but allows
    # the agent to actually land hits during training.
    DEF_DMG_RADIUS: float = 0.8
    # Half-angle (radians) of the shooter's field-of-view cone used to validate
    # whether the opponent is "in front". 60° total FOV (±30° from heading)
    FOV_HALF_ANGLE: float = np.deg2rad(30.0)

    # ===== Additional shaping parameters =====
    # Radius (m) within which we start giving proximity bonuses. Should be large
    # enough that the agent can collect signal before an actual *hit* occurs.
    POS_ADV_MAX_DIST: float = 2.0
    # Dense reward multiplier for positional advantage. Tuned empirically –
    # small enough to not overshadow hit reward (+1).
    POS_ADV_COEF: float = 0.25
    # Dense reward multiplier for *being targeted* by the opponent (penalty).
    NEG_ADV_COEF: float = 0.1

    # ===== Distance-keeping shaping =====
    # Below DIST_TARGET the agent is considered "close enough" – further
    # reduction yields no extra bonus (to avoid suicidal collisions).
    DIST_TARGET: float = 1.0  # m
    # Beyond DIST_FAR no bonus; between TARGET and FAR the bonus decays
    # linearly to zero.
    DIST_FAR: float = 5.0  # m
    # Scale of the dense reward encouraging the blue drone to close in.
    DIST_COEF: float = 0.05

    # ---------------------------------------------------------------------
    # Construction helpers
    # ---------------------------------------------------------------------

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        num_drones: int = 2,
        neighbourhood_radius: float = np.inf,
        initial_xyzs=None,
        initial_rpys=None,
        physics: Physics = Physics.PYB,
        pyb_freq: int = 240,
        ctrl_freq: int = 30,
        gui: bool = False,
        record: bool = False,
        obs: ObservationType = ObservationType.KIN,
        act: ActionType = ActionType.VEL,
    ) -> None:
        if num_drones != 2:
            raise ValueError("DogfightAviary currently supports exactly 2 drones (blue & red).")

        super().__init__(
            drone_model=drone_model,
            num_drones=num_drones,
            neighbourhood_radius=neighbourhood_radius,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obs=obs,
            act=act,
        )

        # Tracking flags for whether each drone is still flying
        self._blue_alive: bool = True
        self._red_alive: bool = True

    # ------------------------------------------------------------------
    # House-keeping helpers
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None, options: Dict[str, Any] | None = None):  # type: ignore[override]
        """Reset the environment and spawn the two drones at random poses.

        The spawn configuration is randomised each episode to avoid over-fitting:
        * Each drone's yaw ∈ U(−π, π).
        * The horizontal separation between the two drones ∈ U(2 m, 4 m).
        * *Blue* starts at the origin; *red* is placed on a random bearing at
          the sampled distance. Both start at 1 m altitude.
        """
        # Reset *alive* flags first
        self._blue_alive = True
        self._red_alive = True

        # Deterministic RNG for reproducible testing if a seed is provided
        rng = np.random.default_rng(seed)

        # Sample separation distance (m) and bearing (rad)
        distance = float(rng.uniform(2.0, 4.0))
        bearing = float(rng.uniform(-np.pi, np.pi))

        # Fixed spawn altitude (m)
        z_height = 1.0

        # Compute positions
        blue_pos = np.array([0.0, 0.0, z_height], dtype=float)
        red_pos = np.array([
            distance * np.cos(bearing),
            distance * np.sin(bearing),
            z_height,
        ], dtype=float)
        self.INIT_XYZS = np.vstack([blue_pos, red_pos])

        # Independent random yaw for each drone
        blue_yaw = float(rng.uniform(-np.pi, np.pi))
        red_yaw = float(rng.uniform(-np.pi, np.pi))
        self.INIT_RPYS = np.array(
            [
                [0.0, 0.0, blue_yaw],
                [0.0, 0.0, red_yaw],
            ],
            dtype=float,
        )

        # Call parent reset, which rebuilds the simulation using the newly
        # assigned INIT_XYZS / INIT_RPYS arrays.
        return super().reset(seed=seed, options=options)

    # ------------------------------------------------------------------
    # Internal placeholders – to be implemented with full combat logic.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Positional advantage helper
    # ------------------------------------------------------------------

    def _calc_positional_advantage(self) -> float:
        """Return a shaped reward reflecting angular & distance advantage.

        The idea is to encourage *blue* to manoeuvre behind/in front of the
        opponent (within field-of-view) **and** close the distance. The reward
        is continuous to provide gradient-like feedback and symmetric so that
        giving the red drone positional advantage yields a penalty of equal
        magnitude.
        """

        # Retrieve current states
        state_blue = self._getDroneStateVector(0)
        state_red = self._getDroneStateVector(1)

        pos_blue = state_blue[0:3]
        pos_red = state_red[0:3]

        yaw_blue = state_blue[9]
        yaw_red = state_red[9]

        heading_blue = np.array([np.cos(yaw_blue), np.sin(yaw_blue), 0.0])
        heading_red = np.array([np.cos(yaw_red), np.sin(yaw_red), 0.0])

        # Relative position vectors (target − shooter)
        vec_blue_to_red = pos_red - pos_blue
        vec_red_to_blue = -vec_blue_to_red

        dist = np.linalg.norm(vec_blue_to_red)

        # Helper to normalise vector in X-Y plane
        def _unit_xy(vec: np.ndarray) -> np.ndarray:
            vec_xy = vec.copy()
            vec_xy[2] = 0.0
            n = np.linalg.norm(vec_xy)
            return vec_xy / n if n > 1e-6 else vec_xy

        unit_b2r = _unit_xy(vec_blue_to_red)
        unit_r2b = -unit_b2r

        cos_ang_blue = np.clip(np.dot(heading_blue[:2], unit_b2r[:2]), -1.0, 1.0)
        cos_ang_red = np.clip(np.dot(heading_red[:2], unit_r2b[:2]), -1.0, 1.0)

        ang_blue = np.arccos(cos_ang_blue)
        ang_red = np.arccos(cos_ang_red)

        # Continuous score based on how well the opponent is *in front* of the drone
        def _angular_score(angle: float) -> float:
            """1 when perfectly centred, 0 at FOV edge or beyond."""
            if angle > self.FOV_HALF_ANGLE:
                return 0.0
            return 1.0 - angle / self.FOV_HALF_ANGLE

        blue_score = _angular_score(ang_blue)
        red_score = _angular_score(ang_red)

        # Distance modifier: only reward if within POS_ADV_MAX_DIST
        dist_mod = max(0.0, (self.POS_ADV_MAX_DIST - dist) / self.POS_ADV_MAX_DIST)

        # Final shaped reward (positive for blue advantage, negative if red has it)
        shaped = self.POS_ADV_COEF * blue_score * dist_mod - self.NEG_ADV_COEF * red_score * dist_mod

        return shaped

    def _calc_hits(self) -> float:
        """Return number of successful hits dealt in the current step.

        A *hit* is defined when the attacker is within :pyattr:`DEF_DMG_RADIUS`
        and inside the defender's field-of-view constraints.  The method
        currently returns *0* as a placeholder so the unit tests pass.
        """

        # Retrieve current states for both drones
        state_blue = self._getDroneStateVector(0)
        state_red = self._getDroneStateVector(1)

        # Positions
        pos_blue = state_blue[0:3]
        pos_red = state_red[0:3]

        # Yaw angles (heading)
        yaw_blue = state_blue[9]
        yaw_red = state_red[9]

        # Heading unit vectors projected on the X-Y plane
        heading_blue = np.array([np.cos(yaw_blue), np.sin(yaw_blue), 0.0])
        heading_red = np.array([np.cos(yaw_red), np.sin(yaw_red), 0.0])

        # Relative position vectors (target − shooter)
        vec_blue_to_red = pos_red - pos_blue
        vec_red_to_blue = -vec_blue_to_red  # simply negated

        dist = np.linalg.norm(vec_blue_to_red)

        # Normalise horizontal component for angle calculation (avoid div-by-zero)
        def _unit_xy(vec: np.ndarray) -> np.ndarray:
            vec_xy = vec.copy()
            vec_xy[2] = 0.0  # project onto plane
            n = np.linalg.norm(vec_xy)
            return vec_xy / n if n > 1e-6 else vec_xy

        unit_b2r = _unit_xy(vec_blue_to_red)
        unit_r2b = -unit_b2r  # faster than recompute

        # Cosine of angle between heading and line-of-sight
        cos_ang_blue = np.clip(np.dot(heading_blue[:2], unit_b2r[:2]), -1.0, 1.0)
        cos_ang_red = np.clip(np.dot(heading_red[:2], unit_r2b[:2]), -1.0, 1.0)

        ang_blue = np.arccos(cos_ang_blue)
        ang_red = np.arccos(cos_ang_red)

        # Determine hits
        blue_hits_red = (
            self._blue_alive
            and self._red_alive
            and dist <= self.DEF_DMG_RADIUS
            and ang_blue <= self.FOV_HALF_ANGLE
        )

        red_hits_blue = (
            self._blue_alive
            and self._red_alive
            and dist <= self.DEF_DMG_RADIUS
            and ang_red <= self.FOV_HALF_ANGLE
        )

        # Update alive flags if a drone has been hit
        if red_hits_blue:
            self._blue_alive = False
        if blue_hits_red:
            self._red_alive = False

        # Reward signal: +1 for blue hitting red, −1 for the opposite
        reward = 0.0
        if blue_hits_red:
            reward += 1.0
        if red_hits_blue:
            reward -= 1.0

        return reward

    def _blue_down(self) -> bool:  # noqa: D401
        """Indicates whether the *blue* drone has been shot down."""
        return not self._blue_alive

    def _red_down(self) -> bool:  # noqa: D401
        """Indicates whether the *red* drone has been shot down."""
        return not self._red_alive

    # ------------------------------------------------------------------
    # RL overrides required by the specification.
    # ------------------------------------------------------------------

    def _computeReward(self) -> float:  # type: ignore[override]
        """Reward function with shaping.

        Reward components:

        * **Hit reward**: +1 for blue hitting red, −1 for the converse.
        * **Positional advantage**: small dense reward encouraging the blue
          drone to get *in front & close* to the opponent.
        * **Living penalty**: −0.03 per step to incentivise faster kills.
        """

        hit_reward = self._calc_hits()
        # Terminal bonus: big reward for downing the opponent, penalty if blue is down
        if self._red_down():
            hit_reward += 8.0  # bigger terminal reward encourages decisive kills
        elif self._blue_down():
            hit_reward -= 8.0

        pos_reward = self._calc_positional_advantage()
        dist_bonus = self._calc_distance_bonus()
        living_penalty = -0.01

        return float(hit_reward + pos_reward + dist_bonus + living_penalty)

    def _computeTerminated(self) -> bool:  # type: ignore[override]
        """Episode ends when either drone is downed."""
        return bool(self._blue_down() or self._red_down())

    def _computeTruncated(self) -> bool:  # type: ignore[override]
        """Truncate when elapsed simulation time exceeds :pyattr:`EPISODE_LEN_SEC`."""
        return bool(self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC)

    # Optional: expose empty info dict (explicit override for clarity)
    def _computeInfo(self) -> Dict[str, Any]:  # type: ignore[override]
        return {}

    # ------------------------------------------------------------------
    # Distance-to-opponent shaping helper
    # ------------------------------------------------------------------

    def _calc_distance_bonus(self) -> float:
        """Dense reward encouraging the *blue* drone to stay within
        :pyattr:`DIST_TARGET` … :pyattr:`DIST_FAR` metres of the opponent.

        Blue gains up to +DIST_COEF when the distance is ≤ DIST_TARGET and
        receives no bonus when the distance ≥ DIST_FAR.  Linear interpolation
        is used in-between.  This steers the policy towards closing the gap
        but does not overly punish temporary separation (e.g. after an
        overshoot manoeuvre).
        """

        state_blue = self._getDroneStateVector(0)
        state_red = self._getDroneStateVector(1)

        dist = np.linalg.norm(state_red[0:3] - state_blue[0:3])

        if dist >= self.DIST_FAR:
            return 0.0
        if dist <= self.DIST_TARGET:
            return self.DIST_COEF

        # Linear decay between target and far
        frac = (self.DIST_FAR - dist) / (self.DIST_FAR - self.DIST_TARGET)
        return self.DIST_COEF * frac 