import numpy as np

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class DogfightAviary(BaseRLAviary):
    """Multi-agent RL problem: dogfight scenario with blue vs red teams."""

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 num_drones: int = 4,
                 neighbourhood_radius: float = np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM
                 ):
        """Initialization of a multi-agent dogfight RL environment.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        self.EPISODE_LEN_SEC = 30
        self.DEF_DMG_RADIUS = 0.3  # m, hit if closer and within FOV

        super().__init__(drone_model=drone_model,
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
                         act=act
                         )

        # Initialize team assignments (first half blue, second half red)
        self.blue_team = list(range(num_drones // 2))
        self.red_team = list(range(num_drones // 2, num_drones))

        # Track hits and status
        self.blue_alive = [True] * len(self.blue_team)
        self.red_alive = [True] * len(self.red_team)

    def reset(self, seed=None, options=None):
        """Resets the environment with randomized spawn positions and orientations.
        
        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.
        options : dict, optional
            Additional options (unused).
            
        Returns
        -------
        ndarray
            The initial observation.
        dict
            Additional information.
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Generate random spawn positions with 2-4m separation
        self._randomize_spawn_positions()

        # Generate random yaw orientations (±π)
        self._randomize_spawn_orientations()

        # Reset team alive status
        self.blue_alive = [True] * len(self.blue_team)
        self.red_alive = [True] * len(self.red_team)

        # Call parent reset with randomized positions
        return super().reset(seed=seed, options=options)

    def _randomize_spawn_positions(self):
        """Generate random spawn positions with uniform 2-4m separation."""
        # Target altitude for all drones
        target_altitude = 0.5

        # Generate positions in a circle to ensure proper separation
        positions = []

        for i in range(self.NUM_DRONES):
            if i == 0:
                # First drone at origin
                positions.append([0.0, 0.0, target_altitude])
            else:
                # Generate position with uniform distance distribution between 2-4m from existing drones
                valid_position = False
                attempts = 0
                max_attempts = 100

                while not valid_position and attempts < max_attempts:
                    # Random angle for circular distribution
                    angle = np.random.uniform(0, 2 * np.pi)

                    # Random distance from center with bias towards edges for better separation
                    if i < 2:
                        # First few drones closer to center
                        radius = np.random.uniform(2.0, 3.0)
                    else:
                        # Later drones further out
                        radius = np.random.uniform(2.5, 4.0)

                    # Convert to cartesian coordinates
                    x = radius * np.cos(angle)
                    y = radius * np.sin(angle)
                    candidate_pos = [x, y, target_altitude]

                    # Check distance to all existing drones
                    min_distance = float('inf')
                    for existing_pos in positions:
                        distance = np.linalg.norm(np.array(candidate_pos[:2]) - np.array(existing_pos[:2]))
                        min_distance = min(min_distance, distance)

                    # Accept if distance is in the 2-4m range
                    if 2.0 <= min_distance <= 4.0:
                        valid_position = True
                        positions.append(candidate_pos)

                    attempts += 1

                # Fallback if we can't find a valid position
                if not valid_position:
                    angle = (2 * np.pi * i) / self.NUM_DRONES  # Evenly spaced around circle
                    radius = 2.5 + np.random.uniform(-0.5, 0.5)  # 2-3m radius with small variation
                    x = radius * np.cos(angle)
                    y = radius * np.sin(angle)
                    positions.append([x, y, target_altitude])

        # Convert to numpy array and set as initial positions
        self.INIT_XYZS = np.array(positions)

    def _randomize_spawn_orientations(self):
        """Generate random yaw orientations within ±π."""
        # Roll and pitch remain zero (level flight)
        # Yaw is randomized within ±π
        random_yaws = np.random.uniform(-np.pi, np.pi, self.NUM_DRONES)

        # Create orientation array: [roll, pitch, yaw]
        self.INIT_RPYS = np.zeros((self.NUM_DRONES, 3))
        self.INIT_RPYS[:, 2] = random_yaws  # Set yaw values

    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward based on hits achieved minus shaping penalty.

        """
        return self._calc_hits() - 0.01  # shaping

    def _computeTerminated(self):
        """Computes the current terminated value.

        Returns
        -------
        bool
            Whether the current episode is terminated (one team eliminated).

        """
        return self._blue_down() or self._red_down()

    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out or drones are out of bounds.

        """
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        for i in range(self.NUM_DRONES):
            if (abs(states[i][0]) > 5.0 or abs(states[i][1]) > 5.0 or states[i][2] > 5.0 or states[i][
                2] < 0.1  # Truncate when a drone is too far away
                    or abs(states[i][7]) > .4 or abs(states[i][8]) > .4  # Truncate when a drone is too tilted
            ):
                return True
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False

    def _computeInfo(self):
        """Computes the current info dict(s).

        Returns
        -------
        dict[str, any]
            Information about hits, team status, etc.

        """
        return {
            "blue_alive": sum(self.blue_alive),
            "red_alive": sum(self.red_alive),
            "hits": self._calc_hits(),
            "step": self.step_counter
        }

    def _calc_hits(self):
        """Calculate number of hits between teams.

        Returns
        -------
        int
            Total number of hits achieved this step.

        """
        hits = 0
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])

        # Check red team attacking blue team (red gets priority for E2-1)
        for red_idx_offset, red_idx in enumerate(self.red_team):
            if not self.red_alive[red_idx_offset]:
                continue
            red_state = states[red_idx]
            red_pos = red_state[0:3]

            for blue_idx_offset, blue_idx in enumerate(self.blue_team):
                if not self.blue_alive[blue_idx_offset]:
                    continue
                blue_state = states[blue_idx]
                blue_pos = blue_state[0:3]

                # Check if within damage radius
                distance = np.linalg.norm(red_pos - blue_pos)
                if distance <= self.DEF_DMG_RADIUS:
                    hits += 1
                    self.blue_alive[blue_idx_offset] = False

        # Check blue team attacking red team (only if blue still alive)
        for blue_idx_offset, blue_idx in enumerate(self.blue_team):
            if not self.blue_alive[blue_idx_offset]:
                continue
            blue_state = states[blue_idx]
            blue_pos = blue_state[0:3]

            for red_idx_offset, red_idx in enumerate(self.red_team):
                if not self.red_alive[red_idx_offset]:
                    continue
                red_state = states[red_idx]
                red_pos = red_state[0:3]

                # Check if within damage radius
                distance = np.linalg.norm(blue_pos - red_pos)
                if distance <= self.DEF_DMG_RADIUS:
                    hits += 1
                    self.red_alive[red_idx_offset] = False

        return hits

    def _blue_down(self):
        """Check if all blue team drones are down.

        Returns
        -------
        bool
            True if all blue team drones are eliminated.

        """
        return not any(self.blue_alive)

    def _red_down(self):
        """Check if all red team drones are down.

        Returns
        -------
        bool
            True if all red team drones are eliminated.

        """
        return not any(self.red_alive)
