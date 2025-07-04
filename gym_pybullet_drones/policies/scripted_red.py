"""
Scripted red drone policy for dogfight scenarios.

This module implements a "pursue & fire" policy where red drones follow a scripted
proportional controller to chase blue drones, maintain altitude, and engage when close.
"""

import numpy as np
from typing import List, Tuple, Optional
from gym_pybullet_drones.utils.enums import ActionType


class ScriptedRedPolicy:
    """
    Scripted policy for red drones implementing pursue & fire behavior.
    
    The policy uses a simple proportional controller to:
    - Target Blue drone XY positions
    - Maintain z=1m altitude
    - "Fire" (approach) when distance < 0.3m
    """

    def __init__(self,
                 action_type: ActionType = ActionType.PID,
                 kp_xy: float = 1.5,
                 kp_z: float = 2.0,
                 target_altitude: float = 0.5,
                 engagement_range: float = 0.3):
        """
        Initialize the scripted red policy.
        
        Parameters
        ----------
        action_type : ActionType
            Type of action space (PID recommended for position control)
        kp_xy : float
            Proportional gain for XY position control
        kp_z : float  
            Proportional gain for Z position control
        target_altitude : float
            Desired altitude to maintain (meters)
        engagement_range : float
            Distance threshold for engagement (meters)
        """
        self.action_type = action_type
        self.kp_xy = kp_xy
        self.kp_z = kp_z
        self.target_altitude = target_altitude
        self.engagement_range = engagement_range

        # Verify action type compatibility
        if action_type not in [ActionType.PID, ActionType.VEL]:
            raise ValueError(f"ScriptedRedPolicy requires PID or VEL action type, got {action_type}")

    def get_action(self,
                   red_states: np.ndarray,
                   blue_states: np.ndarray,
                   red_team_indices: List[int],
                   blue_team_indices: List[int],
                   blue_alive: List[bool]) -> np.ndarray:
        """
        Compute actions for red drones based on current states.
        
        Parameters
        ----------
        red_states : np.ndarray
            States of red drones, shape (num_red, state_dim)
        blue_states : np.ndarray
            States of blue drones, shape (num_blue, state_dim)
        red_team_indices : List[int]
            Indices of red team drones in full drone array
        blue_team_indices : List[int]
            Indices of blue team drones in full drone array
        blue_alive : List[bool]
            Alive status of blue drones
        
        Returns
        -------
        np.ndarray
            Actions for red drones, shape (num_red, action_dim)
        """
        num_red = len(red_team_indices)

        if self.action_type == ActionType.PID:
            action_dim = 3  # x, y, z target positions
        elif self.action_type == ActionType.VEL:
            action_dim = 4  # vx, vy, vz, vyaw
        else:
            raise ValueError(f"Unsupported action type: {self.action_type}")

        actions = np.zeros((num_red, action_dim))

        # Find alive blue drones
        alive_blue_positions = []
        for i, blue_idx in enumerate(blue_team_indices):
            if i < len(blue_alive) and blue_alive[i]:
                alive_blue_positions.append(blue_states[i, 0:3])  # xyz position

        if not alive_blue_positions:
            # No blue drones alive, maintain current position
            for i in range(num_red):
                red_pos = red_states[i, 0:3]
                if self.action_type == ActionType.PID:
                    actions[i] = [red_pos[0], red_pos[1], self.target_altitude]
                elif self.action_type == ActionType.VEL:
                    actions[i] = [0, 0, 0, 0]  # No movement
            return actions

        alive_blue_positions = np.array(alive_blue_positions)

        # Compute actions for each red drone
        for i in range(num_red):
            red_pos = red_states[i, 0:3]

            # Find closest blue drone
            distances = np.linalg.norm(alive_blue_positions - red_pos, axis=1)
            closest_blue_idx = np.argmin(distances)
            target_blue_pos = alive_blue_positions[closest_blue_idx]
            closest_distance = distances[closest_blue_idx]

            if self.action_type == ActionType.PID:
                # PID control: set target position directly
                # Move gradually toward blue position for stability
                alpha = min(self.kp_xy * 0.1, 0.8)  # Limit step size
                target_x = red_pos[0] + alpha * (target_blue_pos[0] - red_pos[0])
                target_y = red_pos[1] + alpha * (target_blue_pos[1] - red_pos[1])

                # Set altitude based on current position for smooth transition
                current_z = red_pos[2]
                if current_z < self.target_altitude:
                    target_z = min(current_z + 0.1, self.target_altitude)
                else:
                    target_z = max(current_z - 0.1, self.target_altitude)

                # If very close, move directly to blue position for engagement
                if closest_distance < self.engagement_range * 2:
                    target_x = target_blue_pos[0]
                    target_y = target_blue_pos[1]

                actions[i] = [target_x, target_y, target_z]

            elif self.action_type == ActionType.VEL:
                # Velocity control: set desired velocity
                target_vel_xy = self.kp_xy * (target_blue_pos[0:2] - red_pos[0:2])
                target_vel_z = self.kp_z * (self.target_altitude - red_pos[2])

                # Normalize XY velocity if too large
                xy_speed = np.linalg.norm(target_vel_xy)
                max_xy_speed = 2.0  # m/s
                if xy_speed > max_xy_speed:
                    target_vel_xy = target_vel_xy / xy_speed * max_xy_speed

                # Limit Z velocity
                target_vel_z = np.clip(target_vel_z, -1.0, 1.0)

                actions[i] = [target_vel_xy[0], target_vel_xy[1], target_vel_z, 0]  # no yaw change

        return actions

    def get_full_action(self,
                        obs: np.ndarray,
                        red_team_indices: List[int],
                        blue_team_indices: List[int],
                        blue_alive: List[bool]) -> np.ndarray:
        """
        Get actions for all drones (red uses scripted policy, blue uses zeros).
        
        Parameters
        ----------
        obs : np.ndarray
            Full observation array, shape (num_drones, obs_dim)
        red_team_indices : List[int]
            Indices of red team drones
        blue_team_indices : List[int]
            Indices of blue team drones  
        blue_alive : List[bool]
            Alive status of blue drones
            
        Returns
        -------
        np.ndarray
            Actions for all drones, shape (num_drones, action_dim)
        """
        num_drones = obs.shape[0]

        if self.action_type == ActionType.PID:
            action_dim = 3
        elif self.action_type == ActionType.VEL:
            action_dim = 4
        else:
            raise ValueError(f"Unsupported action type: {self.action_type}")

        actions = np.zeros((num_drones, action_dim))

        # Extract states (obs format: [x,y,z, rpy, vel, angvel, ...])
        # For kinematic obs: [x,y,z, roll,pitch,yaw, vx,vy,vz, wx,wy,wz, ...]
        red_states = obs[red_team_indices]
        blue_states = obs[blue_team_indices]

        # Get red actions from policy
        red_actions = self.get_action(red_states, blue_states,
                                      red_team_indices, blue_team_indices, blue_alive)

        # Assign red actions
        for i, red_idx in enumerate(red_team_indices):
            actions[red_idx] = red_actions[i]

        # Blue drones get zero actions (stationary)
        for blue_idx in blue_team_indices:
            if self.action_type == ActionType.PID:
                # Hover in place at a reasonable altitude
                blue_pos = obs[blue_idx, 0:3]
                hover_altitude = max(blue_pos[2], 0.5)  # At least 0.5m altitude
                actions[blue_idx] = [blue_pos[0], blue_pos[1], hover_altitude]
            elif self.action_type == ActionType.VEL:
                actions[blue_idx] = [0, 0, 0, 0]  # No movement

        return actions


def create_scripted_red_policy(action_type: ActionType = ActionType.PID,
                               **kwargs) -> ScriptedRedPolicy:
    """
    Factory function to create a scripted red policy.
    
    Parameters
    ----------
    action_type : ActionType
        Type of action space
    **kwargs
        Additional arguments for ScriptedRedPolicy
        
    Returns
    -------
    ScriptedRedPolicy
        Configured scripted policy instance
    """
    return ScriptedRedPolicy(action_type=action_type, **kwargs)
