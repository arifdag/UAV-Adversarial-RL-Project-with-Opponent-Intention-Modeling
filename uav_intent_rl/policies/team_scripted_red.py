"""Team-based scripted policy for the *red* team in 3v3 `Dogfight3v3Aviary`.

This policy implements different tactical behaviors for each red drone:
- Red 0: Aggressive attacker (seeks closest blue drone)
- Red 1: Support/Flanker (positions for team advantage)
- Red 2: Defensive/Guardian (protects team formation)

The policy uses team coordination and formation tactics to create a more
challenging opponent for the learning blue team.
"""

from __future__ import annotations

from typing import Any, List, Tuple
from enum import Enum

import numpy as np

__all__ = ["TeamScriptedRedPolicy", "TeamTactic"]


class TeamTactic(Enum):
    """Different tactical roles for red team drones."""
    AGGRESSIVE = "aggressive"  # Direct attack
    SUPPORT = "support"        # Flanking/support
    DEFENSIVE = "defensive"    # Formation guard


class TeamScriptedRedPolicy:
    """Team-based scripted policy for red team in 3v3 combat."""

    def __init__(
        self, 
        target_alt: float = 1.0,
        formation_radius: float = 1.5,
        coordination_weight: float = 0.7
    ) -> None:
        self.target_alt = float(target_alt)
        self.formation_radius = float(formation_radius)
        self.coordination_weight = float(coordination_weight)
        
        # Assign tactical roles to each red drone
        self.drone_tactics = {
            0: TeamTactic.AGGRESSIVE,  # Red 0: Aggressive
            1: TeamTactic.SUPPORT,     # Red 1: Support
            2: TeamTactic.DEFENSIVE    # Red 2: Defensive
        }

    def reset(self) -> None:
        """No internal state to reset."""
        return None

    def _get_blue_positions(self, env: Any) -> List[np.ndarray]:
        """Get positions of all blue drones."""
        blue_positions = []
        for i in range(3):
            try:
                state = env._getDroneStateVector(i)
                blue_positions.append(state[0:3])
            except:
                # Blue drone might be down
                continue
        return blue_positions

    def _get_red_positions(self, env: Any) -> List[np.ndarray]:
        """Get positions of all red drones."""
        red_positions = []
        for i in range(3, 6):  # Red drones are indices 3, 4, 5
            try:
                state = env._getDroneStateVector(i)
                red_positions.append(state[0:3])
            except:
                # Red drone might be down
                continue
        return red_positions

    def _compute_aggressive_action(self, env: Any, red_idx: int) -> np.ndarray:
        """Compute action for aggressive red drone (seeks closest blue)."""
        red_state = env._getDroneStateVector(red_idx + 3)  # +3 for red drones
        red_pos = red_state[0:3]
        
        blue_positions = self._get_blue_positions(env)
        if not blue_positions:
            return np.zeros(4)
        
        # Find closest blue drone
        distances = [np.linalg.norm(blue_pos - red_pos) for blue_pos in blue_positions]
        closest_idx = np.argmin(distances)
        target_pos = blue_positions[closest_idx]
        
        # Move towards closest blue drone with higher speed
        direction = target_pos - red_pos
        direction[2] = self.target_alt - red_pos[2]  # Maintain altitude
        
        # Normalize direction
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            direction = direction / norm
        
        # Adaptive speed based on direction magnitude
        speed = np.clip(np.linalg.norm(direction) * 1.5, 0.0, 1.0)
        action = np.array([direction[0] * speed, direction[1] * speed, direction[2], speed])
        return np.clip(action, -1.0, 1.0)

    def _compute_support_action(self, env: Any, red_idx: int) -> np.ndarray:
        """Compute action for support red drone (flanking position)."""
        red_state = env._getDroneStateVector(red_idx + 3)
        red_pos = red_state[0:3]
        
        blue_positions = self._get_blue_positions(env)
        red_positions = self._get_red_positions(env)
        
        if not blue_positions:
            return np.zeros(4)
        
        # Calculate blue team center
        blue_center = np.mean(blue_positions, axis=0)
        
        # Find flanking position (perpendicular to blue formation)
        if len(blue_positions) >= 2:
            # Calculate blue formation direction
            blue_dirs = []
            for i in range(len(blue_positions)):
                for j in range(i + 1, len(blue_positions)):
                    dir_vec = blue_positions[j] - blue_positions[i]
                    if np.linalg.norm(dir_vec) > 0.5:
                        blue_dirs.append(dir_vec)
            
            if blue_dirs:
                # Average direction
                avg_dir = np.mean(blue_dirs, axis=0)
                avg_dir[2] = 0  # Keep in XY plane
                norm = np.linalg.norm(avg_dir)
                if norm > 1e-6:
                    avg_dir = avg_dir / norm
                    
                    # Perpendicular direction for flanking
                    flank_dir = np.array([-avg_dir[1], avg_dir[0], 0])
                    
                    # Target position: flanking position relative to blue center
                    target_pos = blue_center + flank_dir * 2.0
                    target_pos[2] = self.target_alt
                    
                    # Move towards flanking position
                    direction = target_pos - red_pos
                    norm = np.linalg.norm(direction)
                    if norm > 1e-6:
                        direction = direction / norm
                    
                    speed = np.clip(np.linalg.norm(direction) * 1.2, 0.0, 1.0)
                    action = np.array([direction[0] * speed, direction[1] * speed, direction[2], speed])
                    return np.clip(action, -1.0, 1.0)
        
        # Fallback: move towards blue center
        direction = blue_center - red_pos
        direction[2] = self.target_alt - red_pos[2]
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            direction = direction / norm
        
        speed = np.clip(np.linalg.norm(direction) * 1.0, 0.0, 1.0)
        action = np.array([direction[0] * speed, direction[1] * speed, direction[2], speed])
        return np.clip(action, -1.0, 1.0)

    def _compute_defensive_action(self, env: Any, red_idx: int) -> np.ndarray:
        """Compute action for defensive red drone (protects formation)."""
        red_state = env._getDroneStateVector(red_idx + 3)
        red_pos = red_state[0:3]
        
        red_positions = self._get_red_positions(env)
        blue_positions = self._get_blue_positions(env)
        
        if len(red_positions) < 2:
            # Fallback to aggressive if not enough teammates
            return self._compute_aggressive_action(env, red_idx)
        
        # Calculate red team center
        red_center = np.mean(red_positions, axis=0)
        
        # Find closest blue drone to red center
        if blue_positions:
            distances = [np.linalg.norm(blue_pos - red_center) for blue_pos in blue_positions]
            closest_blue_idx = np.argmin(distances)
            closest_blue_pos = blue_positions[closest_blue_idx]
            
            # Position between red center and closest blue (defensive position)
            defensive_pos = red_center + (closest_blue_pos - red_center) * 0.3
            defensive_pos[2] = self.target_alt
            
            # Move towards defensive position
            direction = defensive_pos - red_pos
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                direction = direction / norm
            
            speed = np.clip(np.linalg.norm(direction) * 0.8, 0.0, 1.0)
            action = np.array([direction[0] * speed, direction[1] * speed, direction[2], speed])
            return np.clip(action, -1.0, 1.0)
        
        # Fallback: maintain formation
        direction = red_center - red_pos
        direction[2] = self.target_alt - red_pos[2]
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            direction = direction / norm
        
        speed = np.clip(np.linalg.norm(direction) * 0.6, 0.0, 1.0)
        action = np.array([direction[0] * speed, direction[1] * speed, direction[2], speed])
        return np.clip(action, -1.0, 1.0)

    def _compute_action(self, env: Any) -> np.ndarray:
        """Compute actions for all red drones based on their tactical roles."""
        action = np.zeros((env.NUM_DRONES, 4), dtype=np.float32)
        
        # Blue drones (indices 0-2) remain stationary
        action[0:3, :] = 0.0
        
        # Red drones (indices 3-5) get tactical actions
        for red_idx in range(3):
            tactic = self.drone_tactics[red_idx]
            
            if tactic == TeamTactic.AGGRESSIVE:
                drone_action = self._compute_aggressive_action(env, red_idx)
            elif tactic == TeamTactic.SUPPORT:
                drone_action = self._compute_support_action(env, red_idx)
            elif tactic == TeamTactic.DEFENSIVE:
                drone_action = self._compute_defensive_action(env, red_idx)
            else:
                drone_action = np.zeros(4)
            
            action[red_idx + 3] = drone_action
        
        return action

    def __call__(self, env: Any) -> np.ndarray:
        """Make the policy instance directly callable."""
        return self._compute_action(env) 