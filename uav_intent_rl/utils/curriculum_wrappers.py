from __future__ import annotations

"""Curriculum learning wrappers for progressive opponent difficulty.

This module implements curriculum learning for UAV adversarial training,
starting with simple scripted opponents and gradually introducing more
complex adversaries as the agent improves.
"""

from typing import Any, Callable, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from uav_intent_rl.policies.scripted_red import ScriptedRedPolicy
from uav_intent_rl.utils.intent_wrappers import BlueVsFixedRedWrapper

__all__ = ["CurriculumOpponentWrapper", "DifficultyScheduler"]


class DifficultyScheduler:
    """Scheduler for curriculum difficulty progression."""
    
    def __init__(
        self,
        total_timesteps: int,
        difficulty_schedule: str = "linear",
        min_difficulty: float = 0.0,
        max_difficulty: float = 1.0,
        warmup_steps: int = 0,
        **kwargs: Any,
    ) -> None:
        self.total_timesteps = int(total_timesteps)
        self.difficulty_schedule = str(difficulty_schedule)
        self.min_difficulty = float(min_difficulty)
        self.max_difficulty = float(max_difficulty)
        self.warmup_steps = int(warmup_steps)
        
        # Additional parameters for different schedules
        self.step_difficulties = kwargs.get("step_difficulties", [])
        self.curriculum_stages = kwargs.get("curriculum_stages", [])
        
    def get_difficulty(self, timestep: int) -> float:
        """Get current difficulty level based on timestep."""
        progress = min(timestep / max(1, self.total_timesteps), 1.0)
        
        if self.difficulty_schedule == "linear":
            difficulty = self.min_difficulty + (self.max_difficulty - self.min_difficulty) * progress
        elif self.difficulty_schedule == "exponential":
            difficulty = self.min_difficulty + (self.max_difficulty - self.min_difficulty) * (progress ** 2)
        elif self.difficulty_schedule == "cosine":
            import math
            difficulty = self.min_difficulty + (self.max_difficulty - self.min_difficulty) * (0.5 * (1 - math.cos(math.pi * progress)))
        elif self.difficulty_schedule == "step":
            # Step-based difficulty with predefined stages
            if self.curriculum_stages:
                stage_idx = int(progress * len(self.curriculum_stages))
                stage_idx = min(stage_idx, len(self.curriculum_stages) - 1)
                difficulty = self.curriculum_stages[stage_idx]
            else:
                # Default step schedule
                if progress < 0.25:
                    difficulty = self.min_difficulty
                elif progress < 0.5:
                    difficulty = 0.25
                elif progress < 0.75:
                    difficulty = 0.5
                else:
                    difficulty = self.max_difficulty
        else:
            # Default to linear
            difficulty = self.min_difficulty + (self.max_difficulty - self.min_difficulty) * progress
            
        # Apply warmup
        if timestep < self.warmup_steps:
            warmup_progress = timestep / max(1, self.warmup_steps)
            difficulty = self.min_difficulty * warmup_progress
            
        return np.clip(difficulty, self.min_difficulty, self.max_difficulty)


class CurriculumOpponentWrapper(gym.Wrapper):
    """Wrapper that implements curriculum learning for opponent difficulty."""
    
    def __init__(
        self,
        env: gym.Env,
        scheduler: DifficultyScheduler,
        opponent_factory: Callable[[float], Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(env)
        self.scheduler = scheduler
        self.opponent_factory = opponent_factory or self._default_opponent_factory
        self.current_difficulty = 0.0
        self.current_opponent = None
        self.timestep = 0
        
        # Initialize with minimum difficulty
        self._update_opponent(self.scheduler.min_difficulty)
        
    def _default_opponent_factory(self, difficulty: float) -> ScriptedRedPolicy:
        """Default opponent factory that creates scripted policies with varying difficulty."""
        if difficulty < 0.25:
            # Stage 1: Stationary opponent
            return StationaryOpponent()
        elif difficulty < 0.5:
            # Stage 2: Simple pursuit with noise
            return SimplePursuitOpponent(noise_scale=0.1)
        elif difficulty < 0.75:
            # Stage 3: Advanced pursuit with evasion
            return AdvancedPursuitOpponent(evasion_prob=0.3)
        else:
            # Stage 4: Full adversarial opponent
            return AdversarialOpponent()
    
    def _update_opponent(self, difficulty: float, force_update: bool = False) -> None:
        """Update the opponent based on current difficulty."""
        # Use smaller threshold for force updates (from callback) or reduce general threshold
        threshold = 0.001 if force_update else 0.01
        if abs(difficulty - self.current_difficulty) > threshold:
            self.current_difficulty = difficulty
            self.current_opponent = self.opponent_factory(difficulty)
            
            # Update the environment's opponent if it's a BlueVsFixedRedWrapper
            if hasattr(self.env, '_red_policy'):
                self.env._red_policy = self.current_opponent
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and update opponent difficulty."""
        obs, info = super().reset(**kwargs)
        
        # Update difficulty based on current timestep
        difficulty = self.scheduler.get_difficulty(self.timestep)
        self._update_opponent(difficulty)
        
        # Add curriculum info to info dict
        info['curriculum_difficulty'] = self.current_difficulty
        info['curriculum_timestep'] = self.timestep
        
        return obs, info
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step environment and update curriculum."""
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Update timestep and difficulty
        self.timestep += 1
        difficulty = self.scheduler.get_difficulty(self.timestep)
        self._update_opponent(difficulty)
        
        # Add curriculum info
        info['curriculum_difficulty'] = self.current_difficulty
        info['curriculum_timestep'] = self.timestep
        
        return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Curriculum opponent implementations
# ---------------------------------------------------------------------------

class StationaryOpponent(ScriptedRedPolicy):
    """Stage 1: Completely stationary opponent for initial learning."""
    
    def _compute_action(self, env: Any) -> np.ndarray:
        """Return zero action - stationary opponent."""
        action = np.zeros((env.NUM_DRONES, 4), dtype=np.float32)
        return action


class SimplePursuitOpponent(ScriptedRedPolicy):
    """Stage 2: Simple pursuit with configurable noise."""
    
    def __init__(self, noise_scale: float = 0.1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.noise_scale = float(noise_scale)
    
    def _compute_action(self, env: Any) -> np.ndarray:
        """Simple pursuit with reduced noise."""
        state_blue = env._getDroneStateVector(0)
        state_red = env._getDroneStateVector(1)
        
        # Relative position (blue − red)
        rel_pos = state_blue[0:3] - state_red[0:3]
        # Add reduced noise
        rel_pos += np.random.normal(loc=0.0, scale=self.noise_scale, size=3)
        # Maintain fixed target altitude
        rel_pos[2] = self.target_alt - state_red[2] + np.random.normal(loc=0.0, scale=self.noise_scale * 0.5)
        
        # Direction – normalise to unit vector
        norm = np.linalg.norm(rel_pos)
        direction = rel_pos / norm if norm > 1e-6 else np.zeros(3)
        
        # Build action array
        action = np.zeros((env.NUM_DRONES, 4), dtype=np.float32)
        action[0, :] = 0.0  # Blue stationary
        action[1, 0:3] = direction
        action[1, 3] = 0.8  # Reduced speed for easier opponent
        
        np.clip(action, -1.0, 1.0, out=action)
        return action


class AdvancedPursuitOpponent(ScriptedRedPolicy):
    """Stage 3: Advanced pursuit with evasion behavior."""
    
    def __init__(self, evasion_prob: float = 0.3, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.evasion_prob = float(evasion_prob)
        self.evasion_counter = 0
        self.evasion_duration = 0
    
    def _compute_action(self, env: Any) -> np.ndarray:
        """Advanced pursuit with occasional evasion."""
        state_blue = env._getDroneStateVector(0)
        state_red = env._getDroneStateVector(1)
        
        # Check if we should start evasion
        if self.evasion_counter <= 0 and np.random.random() < self.evasion_prob:
            self.evasion_counter = np.random.randint(10, 30)  # Evasion duration
            self.evasion_direction = np.random.uniform(-np.pi, np.pi)  # Random evasion direction
        
        # Update evasion counter
        if self.evasion_counter > 0:
            self.evasion_counter -= 1
        
        # Compute action based on current mode
        if self.evasion_counter > 0:
            # Evasion mode: move perpendicular to blue
            evasion_vec = np.array([
                np.cos(self.evasion_direction),
                np.sin(self.evasion_direction),
                0.0
            ])
            direction = evasion_vec
        else:
            # Pursuit mode: normal pursuit behavior
            rel_pos = state_blue[0:3] - state_red[0:3]
            rel_pos += np.random.normal(loc=0.0, scale=0.15, size=3)
            rel_pos[2] = self.target_alt - state_red[2] + np.random.normal(loc=0.0, scale=0.08)
            
            norm = np.linalg.norm(rel_pos)
            direction = rel_pos / norm if norm > 1e-6 else np.zeros(3)
        
        # Build action array
        action = np.zeros((env.NUM_DRONES, 4), dtype=np.float32)
        action[0, :] = 0.0  # Blue stationary
        action[1, 0:3] = direction
        action[1, 3] = 0.9  # Full speed for advanced opponent
        
        np.clip(action, -1.0, 1.0, out=action)
        return action


class AdversarialOpponent(ScriptedRedPolicy):
    """Stage 4: Full adversarial opponent with complex behavior."""
    
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.behavior_state = "pursuit"  # pursuit, evasion, circling
        self.state_counter = 0
        self.circling_center = None
        self.circling_radius = 2.0
    
    def _compute_action(self, env: Any) -> np.ndarray:
        """Complex adversarial behavior with state machine."""
        state_blue = env._getDroneStateVector(0)
        state_red = env._getDroneStateVector(1)
        
        # Update behavior state
        self._update_behavior_state(state_blue, state_red)
        
        # Compute action based on current behavior
        if self.behavior_state == "pursuit":
            direction = self._pursuit_action(state_blue, state_red)
        elif self.behavior_state == "evasion":
            direction = self._evasion_action(state_blue, state_red)
        elif self.behavior_state == "circling":
            direction = self._circling_action(state_blue, state_red)
        else:
            direction = self._pursuit_action(state_blue, state_red)  # Fallback
        
        # Build action array
        action = np.zeros((env.NUM_DRONES, 4), dtype=np.float32)
        action[0, :] = 0.0  # Blue stationary
        action[1, 0:3] = direction
        action[1, 3] = 1.0  # Full speed
        
        np.clip(action, -1.0, 1.0, out=action)
        return action
    
    def _update_behavior_state(self, state_blue: np.ndarray, state_red: np.ndarray) -> None:
        """Update behavior state based on situation."""
        dist = np.linalg.norm(state_blue[0:3] - state_red[0:3])
        
        # State transitions
        if self.state_counter <= 0:
            if dist < 1.5 and np.random.random() < 0.3:
                # Close range: switch to evasion
                self.behavior_state = "evasion"
                self.state_counter = np.random.randint(20, 40)
            elif dist > 3.0 and np.random.random() < 0.4:
                # Far range: switch to circling
                self.behavior_state = "circling"
                self.circling_center = state_blue[0:3].copy()
                self.state_counter = np.random.randint(30, 60)
            else:
                # Default: pursuit
                self.behavior_state = "pursuit"
                self.state_counter = np.random.randint(15, 30)
        
        self.state_counter -= 1
    
    def _pursuit_action(self, state_blue: np.ndarray, state_red: np.ndarray) -> np.ndarray:
        """Standard pursuit behavior."""
        rel_pos = state_blue[0:3] - state_red[0:3]
        rel_pos += np.random.normal(loc=0.0, scale=0.2, size=3)
        rel_pos[2] = self.target_alt - state_red[2] + np.random.normal(loc=0.0, scale=0.1)
        
        norm = np.linalg.norm(rel_pos)
        return rel_pos / norm if norm > 1e-6 else np.zeros(3)
    
    def _evasion_action(self, state_blue: np.ndarray, state_red: np.ndarray) -> np.ndarray:
        """Evasion behavior: move away from blue."""
        rel_pos = state_red[0:3] - state_blue[0:3]  # Away from blue
        rel_pos += np.random.normal(loc=0.0, scale=0.3, size=3)
        rel_pos[2] = self.target_alt - state_red[2] + np.random.normal(loc=0.0, scale=0.15)
        
        norm = np.linalg.norm(rel_pos)
        return rel_pos / norm if norm > 1e-6 else np.zeros(3)
    
    def _circling_action(self, state_blue: np.ndarray, state_red: np.ndarray) -> np.ndarray:
        """Circling behavior: orbit around blue."""
        if self.circling_center is None:
            self.circling_center = state_blue[0:3].copy()
        
        # Compute tangent direction for circling
        to_center = self.circling_center - state_red[0:3]
        to_center[2] = 0  # Keep in horizontal plane
        
        # Perpendicular direction for circling
        tangent = np.array([-to_center[1], to_center[0], 0])
        tangent += np.random.normal(loc=0.0, scale=0.2, size=3)
        
        # Add some pursuit component
        pursuit_component = state_blue[0:3] - state_red[0:3]
        pursuit_component[2] = 0
        
        # Combine circling and pursuit
        direction = 0.7 * tangent + 0.3 * pursuit_component
        direction[2] = self.target_alt - state_red[2] + np.random.normal(loc=0.0, scale=0.1)
        
        norm = np.linalg.norm(direction)
        return direction / norm if norm > 1e-6 else np.zeros(3) 