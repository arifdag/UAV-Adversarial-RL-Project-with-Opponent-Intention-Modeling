"""Dogfight3v3Aviary environment for 3v3 adversarial UAV combat.

This environment extends the 1v1 DogfightAviary to support 3v3 team battles
with enhanced reward structures, team coordination mechanics, and multi-agent
combat dynamics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary  # type: ignore
from gym_pybullet_drones.utils.enums import (
    ActionType,
    DroneModel,
    ObservationType,
    Physics,
)

__all__ = ["Dogfight3v3Aviary"]


class Dogfight3v3Aviary(BaseRLAviary):
    """Six-drone 3v3 team dog-fight RL environment.

    The environment supports 3 blue drones vs 3 red drones with team-based
    combat mechanics, enhanced reward structures, and multi-agent coordination.
    """

    # Episode duration in seconds before truncation
    EPISODE_LEN_SEC: int = 40  # Increased from 20 to 40 for longer combat episodes
    
    # Combat parameters
    DEF_DMG_RADIUS: float = 2.5  # Increased from 1.5 to 2.5 for easier hits
    FOV_HALF_ANGLE: float = np.deg2rad(90.0)  # Increased from 45° to 90° for easier hits
    
    # Team-based parameters
    TEAM_COORDINATION_RADIUS: float = 3.0  # Radius for team coordination bonuses
    TEAM_COVERAGE_BONUS: float = 0.2  # Bonus for good team positioning
    TEAM_FOCUS_PENALTY: float = 0.1  # Penalty for all drones targeting same enemy
    
    # Positional advantage parameters
    POS_ADV_MAX_DIST: float = 2.5  # Increased for 3v3
    POS_ADV_COEF: float = 0.25
    
    # Distance-keeping parameters
    DIST_TARGET: float = 1.2
    DIST_FAR: float = 6.0  # Increased for 3v3
    DIST_COEF: float = 0.15
    
    # Team formation parameters
    FORMATION_BONUS_RADIUS: float = 2.0
    FORMATION_BONUS_COEF: float = 0.1
    
    # Survivability parameters
    SURVIVAL_BONUS: float = 0.05  # Bonus per step for staying alive
    TEAM_SURVIVAL_BONUS: float = 0.1  # Bonus for team survival

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        num_drones: int = 6,  # 3 blue + 3 red
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
        if num_drones != 6:
            raise ValueError("Dogfight3v3Aviary supports exactly 6 drones (3 blue & 3 red).")

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

        # Tracking flags for drone status
        self._blue_drones_alive: List[bool] = [True, True, True]
        self._red_drones_alive: List[bool] = [True, True, True]
        
        # Team tracking
        self._blue_team_id = 0
        self._red_team_id = 1
        
        # Combat statistics
        self._blue_hits_dealt = 0
        self._red_hits_dealt = 0
        self._blue_drones_down = 0
        self._red_drones_down = 0

    def reset(self, seed: int | None = None, options: Dict[str, Any] | None = None):
        """Reset the environment and spawn the six drones at strategic positions.

        Blue team starts in a triangular formation, red team in an opposing formation.
        """
        # Reset status flags
        self._blue_drones_alive = [True, True, True]
        self._red_drones_alive = [True, True, True]
        
        # Reset combat statistics
        self._blue_hits_dealt = 0
        self._red_hits_dealt = 0
        self._blue_drones_down = 0
        self._red_drones_down = 0

        # Deterministic RNG for reproducible testing
        rng = np.random.default_rng(seed)

        # Blue team formation (triangle)
        blue_formation_radius = 1.5
        blue_positions = [
            [0.0, 0.0, 1.0],  # Center
            [blue_formation_radius, 0.0, 1.0],  # Right
            [-blue_formation_radius * 0.5, blue_formation_radius * 0.866, 1.0],  # Left
        ]
        
        # Red team formation (opposing triangle)
        red_formation_radius = 1.5
        team_separation = 2.0  # Reduced from 4.0 to 2.0 for closer combat
        red_center = [team_separation, 0.0, 1.0]
        red_positions = [
            [red_center[0], red_center[1], red_center[2]],  # Center
            [red_center[0] + red_formation_radius, red_center[1], red_center[2]],  # Right
            [red_center[0] - red_formation_radius * 0.5, red_center[1] + red_formation_radius * 0.866, red_center[2]],  # Left
        ]

        # Combine positions
        all_positions = blue_positions + red_positions
        self.INIT_XYZS = np.array(all_positions, dtype=float)

        # Random yaw angles for each drone
        yaws = [float(rng.uniform(-np.pi, np.pi)) for _ in range(6)]
        self.INIT_RPYS = np.array([
            [0.0, 0.0, yaw] for yaw in yaws
        ], dtype=float)

        return super().reset(seed=seed, options=options)

    def _get_team_positions(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Get current positions of blue and red team drones."""
        blue_positions = []
        red_positions = []
        
        for i in range(3):
            if self._blue_drones_alive[i]:
                state = self._getDroneStateVector(i)
                blue_positions.append(state[0:3])
            if self._red_drones_alive[i]:
                state = self._getDroneStateVector(i + 3)
                red_positions.append(state[0:3])
        
        return blue_positions, red_positions

    def _calc_team_coordination_bonus(self) -> float:
        """Calculate bonus for good team coordination."""
        blue_positions, red_positions = self._get_team_positions()
        
        if len(blue_positions) < 2:
            return 0.0
            
        # Calculate team spread and coverage
        blue_center = np.mean(blue_positions, axis=0)
        blue_distances = [np.linalg.norm(pos - blue_center) for pos in blue_positions]
        
        # Bonus for good formation (not too close, not too far)
        formation_score = 0.0
        for dist in blue_distances:
            if 0.5 <= dist <= self.FORMATION_BONUS_RADIUS:
                formation_score += self.FORMATION_BONUS_COEF
        
        return formation_score

    def _calc_team_survival_bonus(self) -> float:
        """Calculate bonus for team survival."""
        blue_alive = sum(self._blue_drones_alive)
        red_alive = sum(self._red_drones_alive)
        
        # Bonus for having more drones alive than opponent
        survival_diff = blue_alive - red_alive
        return self.TEAM_SURVIVAL_BONUS * survival_diff

    def _calc_hits(self) -> float:
        """Calculate hits for all drones in 3v3 combat with simultaneous detection."""
        blue_kills, red_kills = set(), set()
        
        # Helper function to check if a hit occurs
        def _is_hit(attacker_idx: int, target_idx: int, is_blue_attacker: bool) -> bool:
            """Check if attacker hits target."""
            if is_blue_attacker:
                attacker_state = self._getDroneStateVector(attacker_idx)
                target_state = self._getDroneStateVector(target_idx + 3)  # Red drones are +3
            else:
                attacker_state = self._getDroneStateVector(attacker_idx + 3)  # Red drones are +3
                target_state = self._getDroneStateVector(target_idx)
            
            attacker_pos = attacker_state[0:3]
            target_pos = target_state[0:3]
            attacker_yaw = attacker_state[9]
            attacker_heading = np.array([np.cos(attacker_yaw), np.sin(attacker_yaw), 0.0])
            
            # Calculate distance and angle
            vec_to_target = target_pos - attacker_pos
            dist = np.linalg.norm(vec_to_target)
            
            if dist <= self.DEF_DMG_RADIUS:
                # Check field of view
                vec_xy = vec_to_target.copy()
                vec_xy[2] = 0.0
                unit_vec = vec_xy / np.linalg.norm(vec_xy) if np.linalg.norm(vec_xy) > 1e-6 else vec_xy
                
                cos_angle = np.clip(np.dot(attacker_heading[:2], unit_vec[:2]), -1.0, 1.0)
                angle = np.arccos(cos_angle)
                
                return angle <= self.FOV_HALF_ANGLE
            return False
        
        # ---------- Detect hits, DON'T mutate yet ----------
        # Check all blue drones hitting red drones
        for blue_idx in range(3):
            if not self._blue_drones_alive[blue_idx]:
                continue
            for red_idx in range(3):
                if not self._red_drones_alive[red_idx]:
                    continue
                if _is_hit(blue_idx, red_idx, is_blue_attacker=True):
                    red_kills.add(red_idx)
        
        # Check all red drones hitting blue drones
        for red_idx in range(3):
            if not self._red_drones_alive[red_idx]:
                continue
            for blue_idx in range(3):
                if not self._blue_drones_alive[blue_idx]:
                    continue
                if _is_hit(red_idx, blue_idx, is_blue_attacker=False):
                    blue_kills.add(blue_idx)
        
        # ---------- Apply hits simultaneously ----------
        for red_idx in red_kills:
            self._red_drones_alive[red_idx] = False
            self._blue_hits_dealt += 1
        
        for blue_idx in blue_kills:
            self._blue_drones_alive[blue_idx] = False
            self._red_hits_dealt += 1
        
        # Calculate reward: +1 for each blue kill, -1 for each red kill
        # Mutual kills cancel out (both teams lose a drone)
        total_reward = len(blue_kills) - len(red_kills)
        
        return total_reward

    def _calc_positional_advantage(self) -> float:
        """Calculate positional advantage for blue team."""
        total_advantage = 0.0
        
        for blue_idx in range(3):
            if not self._blue_drones_alive[blue_idx]:
                continue
                
            blue_state = self._getDroneStateVector(blue_idx)
            blue_pos = blue_state[0:3]
            blue_yaw = blue_state[9]
            blue_heading = np.array([np.cos(blue_yaw), np.sin(blue_yaw), 0.0])
            
            for red_idx in range(3):
                if not self._red_drones_alive[red_idx]:
                    continue
                    
                red_state = self._getDroneStateVector(red_idx + 3)
                red_pos = red_state[0:3]
                
                vec_blue_to_red = red_pos - blue_pos
                dist = np.linalg.norm(vec_blue_to_red)
                
                if dist <= self.POS_ADV_MAX_DIST:
                    # Calculate angular advantage
                    vec_xy = vec_blue_to_red.copy()
                    vec_xy[2] = 0.0
                    unit_vec = vec_xy / np.linalg.norm(vec_xy) if np.linalg.norm(vec_xy) > 1e-6 else vec_xy
                    
                    cos_angle = np.clip(np.dot(blue_heading[:2], unit_vec[:2]), -1.0, 1.0)
                    angle = np.arccos(cos_angle)
                    
                    if angle <= self.FOV_HALF_ANGLE:
                        # Blue has advantage
                        advantage = 1.0 - angle / self.FOV_HALF_ANGLE
                        dist_mod = (self.POS_ADV_MAX_DIST - dist) / self.POS_ADV_MAX_DIST
                        total_advantage += self.POS_ADV_COEF * advantage * dist_mod
        
        return total_advantage

    def _calc_distance_bonus(self) -> float:
        """Calculate distance bonus for blue team."""
        total_bonus = 0.0
        
        for blue_idx in range(3):
            if not self._blue_drones_alive[blue_idx]:
                continue
                
            blue_state = self._getDroneStateVector(blue_idx)
            blue_pos = blue_state[0:3]
            
            # Find closest red drone
            min_dist = float('inf')
            for red_idx in range(3):
                if not self._red_drones_alive[red_idx]:
                    continue
                    
                red_state = self._getDroneStateVector(red_idx + 3)
                red_pos = red_state[0:3]
                dist = np.linalg.norm(red_pos - blue_pos)
                min_dist = min(min_dist, dist)
            
            if min_dist < float('inf'):
                if min_dist >= self.DIST_FAR:
                    bonus = 0.0
                elif min_dist <= self.DIST_TARGET:
                    bonus = self.DIST_COEF
                else:
                    # Linear decay
                    frac = (self.DIST_FAR - min_dist) / (self.DIST_FAR - self.DIST_TARGET)
                    bonus = self.DIST_COEF * frac
                
                total_bonus += bonus
        
        return total_bonus

    def _blue_team_down(self) -> bool:
        """Check if all blue drones are down."""
        return not any(self._blue_drones_alive)

    def _red_team_down(self) -> bool:
        """Check if all red drones are down."""
        return not any(self._red_drones_alive)

    def _computeReward(self) -> float:
        """Compute reward for blue team in 3v3 combat."""
        living_penalty = -0.005  # Reduced penalty
        draw_penalty = -2.0      # Reduced penalty
        
        # Combat rewards
        hit_reward = self._calc_hits()
        
        # Team victory/defeat rewards
        if self._red_team_down():
            hit_reward += 30.0  # Blue team wins
        elif self._blue_team_down():
            hit_reward -= 30.0  # Blue team loses
        
        # Dense shaping rewards (reduced to be less biased)
        pos_reward = self._calc_positional_advantage() * 0.5  # Reduced
        dist_bonus = self._calc_distance_bonus() * 0.5        # Reduced
        team_coord_bonus = self._calc_team_coordination_bonus() * 0.5  # Reduced
        survival_bonus = self._calc_team_survival_bonus() * 0.5        # Reduced
        
        total = (hit_reward + pos_reward + dist_bonus + 
                team_coord_bonus + survival_bonus + living_penalty)
        
        if self._computeTruncated() and not self._computeTerminated():
            total += draw_penalty
            
        return float(total)

    def _computeTerminated(self) -> bool:
        """Episode ends when either team is completely eliminated."""
        return bool(self._blue_team_down() or self._red_team_down())

    def _computeTruncated(self) -> bool:
        """Truncate when elapsed simulation time exceeds EPISODE_LEN_SEC."""
        return bool(self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC)

    def _computeInfo(self) -> Dict[str, Any]:
        """Return additional information about the episode."""
        return {
            "blue_hits_dealt": self._blue_hits_dealt,
            "red_hits_dealt": self._red_hits_dealt,
            "blue_drones_alive": sum(self._blue_drones_alive),
            "red_drones_alive": sum(self._red_drones_alive),
            "blue_team_down": self._blue_team_down(),
            "red_team_down": self._red_team_down(),
            "episode_truncated": self._computeTruncated(),
            "episode_terminated": self._computeTerminated(),
        } 