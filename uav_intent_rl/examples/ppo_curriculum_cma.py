from __future__ import annotations

"""Training script for PPO-CMA with curriculum learning.

This script demonstrates the integration of:
1. PPO-CMA adaptive variance for exploration
2. Curriculum learning with progressive opponent difficulty
3. Automatic difficulty scheduling based on performance

Usage
-----
python -m uav_intent_rl.examples.ppo_curriculum_cma --total-timesteps 3000000 --curriculum-schedule linear
"""

import argparse
import json
import datetime
from pathlib import Path
from typing import Any, Dict

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from uav_intent_rl.utils.callbacks import WinRateCallback

from uav_intent_rl.algo.ppo_cma import PPOCMA
from uav_intent_rl.envs import DogfightAviary
from uav_intent_rl.utils.curriculum_wrappers import CurriculumOpponentWrapper, DifficultyScheduler
from uav_intent_rl.utils.intent_wrappers import BlueVsFixedRedWrapper


class CurriculumCallback(BaseCallback):
    """Hybrid curriculum: linear ramp during warmup, then performance-based stepping."""
    def __init__(self, verbose=0, win_rate_threshold=0.8, difficulty_step=0.25, warmup_steps=100000, max_warmup_difficulty=0.3):
        super().__init__(verbose)
        self.win_rate_threshold = win_rate_threshold
        self.difficulty_step = difficulty_step
        self.warmup_steps = warmup_steps
        self.max_warmup_difficulty = max_warmup_difficulty
        self.current_difficulty = 0.0
        self._after_warmup = False

    def _on_step(self) -> bool:
        if self.num_timesteps < self.warmup_steps:
            # Linear ramp during warmup
            frac = self.num_timesteps / max(1, self.warmup_steps)
            self.current_difficulty = self.max_warmup_difficulty * frac
            if self.verbose > 0:
                print(f"[CURRICULUM] Warmup phase: t={self.num_timesteps}, frac={frac:.3f}, difficulty={self.current_difficulty:.3f}")
            self._after_warmup = False
        else:
            if not self._after_warmup:
                # Snap to max_warmup_difficulty at end of warmup
                self.current_difficulty = self.max_warmup_difficulty
                self._after_warmup = True
                if self.verbose > 0:
                    print(f"[CURRICULUM] Warmup complete. Starting performance-based at difficulty={self.current_difficulty:.3f}")
            # Performance-based stepping
            if hasattr(self.logger, 'name_to_value'):
                win_rate_key = "eval/win_rate"
                if win_rate_key in self.logger.name_to_value:
                    latest_win_rate = self.logger.name_to_value[win_rate_key]
                    if self.verbose > 0:
                        print(f"[CURRICULUM] Performance phase: t={self.num_timesteps}, win_rate={latest_win_rate:.3f}, current_difficulty={self.current_difficulty:.3f}")
                    if latest_win_rate > self.win_rate_threshold and self.current_difficulty < 1.0:
                        next_diff = min(1.0, self.current_difficulty + self.difficulty_step)
                        if next_diff > self.current_difficulty:
                            self.current_difficulty = next_diff
                            if self.verbose > 0:
                                print(f"[CURRICULUM] Win rate {latest_win_rate:.3f} > {self.win_rate_threshold}, increasing difficulty to {self.current_difficulty:.3f}")
            # else: hold at current value
        # Update all environments with the new difficulty
        if hasattr(self.training_env, 'envs'):
            for e in self.training_env.envs:
                if hasattr(e, '_update_opponent'):
                    e._update_opponent(self.current_difficulty, force_update=True)
        # Log curriculum information
        self.logger.record("curriculum/difficulty", self.current_difficulty)
        self.logger.record("curriculum/performance_based", float(self._after_warmup))
        return True

    def set_training_env(self, env):
        # Robust to SB3 version
        if hasattr(super(), "set_training_env"):
            super().set_training_env(env)
        else:
            self.__dict__["training_env"] = env


class DogfightEvalCallback(BaseCallback):
    """Callback for evaluating dogfight performance metrics."""
    
    def __init__(
        self,
        eval_freq: int = 10000,
        n_eval_episodes: int = 10,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.eval_env = None
    
    def _on_step(self) -> bool:
        """Evaluate model performance periodically."""
        if self.num_timesteps % self.eval_freq == 0:
            self._evaluate_model()
        return True
    
    def _evaluate_model(self):
        """Evaluate model and log metrics."""
        if self.eval_env is None:
            # Create evaluation environment
            from uav_intent_rl.envs import DogfightAviary
            from uav_intent_rl.utils.intent_wrappers import BlueVsFixedRedWrapper
            
            base_env = DogfightAviary(gui=False)
            self.eval_env = BlueVsFixedRedWrapper(base_env)
        
        wins = 0
        losses = 0
        draws = 0
        total_reward = 0.0
        total_episodes = 0
        total_steps = 0
        hit_accuracy = 0.0
        total_hits = 0
        
        for episode in range(self.n_eval_episodes):
            obs_reset = self.eval_env.reset()
            # VecEnv.reset() returns only obs; handle both cases
            if isinstance(obs_reset, tuple):
                obs, info = obs_reset
            else:
                obs = obs_reset
                info = {}
            done = False
            episode_reward = 0.0
            episode_steps = 0
            episode_hits = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_steps += 1
                
                # Track hits (positive rewards indicate successful hits)
                if reward > 0:
                    episode_hits += 1
            
            # Determine episode outcome
            if hasattr(self.eval_env.env, '_red_down') and hasattr(self.eval_env.env, '_blue_down'):
                red_down = self.eval_env.env._red_down()
                blue_down = self.eval_env.env._blue_down()
                
                if red_down and not blue_down:
                    wins += 1
                elif blue_down and not red_down:
                    losses += 1
                else:
                    draws += 1  # Both down or neither down
            
            total_reward += episode_reward
            total_episodes += 1
            total_steps += episode_steps
            total_hits += episode_hits
        
        # Calculate metrics
        win_rate = wins / total_episodes if total_episodes > 0 else 0.0
        loss_rate = losses / total_episodes if total_episodes > 0 else 0.0
        draw_rate = draws / total_episodes if total_episodes > 0 else 0.0
        avg_reward = total_reward / total_episodes if total_episodes > 0 else 0.0
        avg_episode_length = total_steps / total_episodes if total_episodes > 0 else 0.0
        hit_accuracy = total_hits / total_steps if total_steps > 0 else 0.0
        
        # Log metrics
        self.logger.record("eval/win_rate", win_rate)
        self.logger.record("eval/loss_rate", loss_rate)
        self.logger.record("eval/draw_rate", draw_rate)
        self.logger.record("eval/avg_reward", avg_reward)
        self.logger.record("eval/avg_episode_length", avg_episode_length)
        self.logger.record("eval/hit_accuracy", hit_accuracy)
        self.logger.record("eval/total_episodes", total_episodes)
        
        if self.verbose > 0:
            print(f"[EVAL] Timestep {self.num_timesteps}: Win Rate = {win_rate:.3f}, Hit Accuracy = {hit_accuracy:.3f}, Avg Reward = {avg_reward:.3f}")


class IntentAccuracyCallback(BaseCallback):
    """Callback for tracking intent prediction accuracy."""
    
    def __init__(self, eval_freq: int = 10000, verbose: int = 0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.intent_accuracy_history = []
    
    def _on_step(self) -> bool:
        """Track intent accuracy periodically."""
        if self.num_timesteps % self.eval_freq == 0:
            self._track_intent_accuracy()
        return True
    
    def _track_intent_accuracy(self):
        """Track intent prediction accuracy if using intent-based PPO."""
        if hasattr(self.model, 'intent_accuracy'):
            accuracy = self.model.intent_accuracy
            self.intent_accuracy_history.append(accuracy)
            
            # Log intent accuracy
            self.logger.record("intent/accuracy", accuracy)
            self.logger.record("intent/avg_accuracy", np.mean(self.intent_accuracy_history))
            
            if self.verbose > 0:
                print(f"[INTENT] Timestep {self.num_timesteps}: Accuracy = {accuracy:.3f}")


# Add PerformanceCurriculumCallback for true performance-based curriculum
class PerformanceCurriculumCallback(BaseCallback):
    """Hybrid curriculum: linear ramp during warmup, then performance-based stepping."""
    def __init__(self, threshold: float, step_size: float, warmup_steps: int, 
                 max_warmup_difficulty: float = 0.3, 
                 stop_curriculum_at: float = 1.0,
                 min_steps_between_increases: int = 0,
                 verbose: int = 0):
        super().__init__(verbose)
        self.threshold = threshold
        self.step_size = step_size
        self.warmup_steps = warmup_steps
        self.max_warmup_difficulty = max_warmup_difficulty
        self.stop_curriculum_at = stop_curriculum_at  # NEW
        self.min_steps_between_increases = min_steps_between_increases  # NEW
        self.current_difficulty = 0.0
        self._after_warmup = False
        self.performance_based = 0
        self._last_win_rate = None
        self._increased_this_cycle = False
        self._last_increase_timestep = 0  # NEW
        if self.verbose > 0:
            print(f"[PERFORMANCE CURRICULUM] Initialized with step_size={self.step_size:.3f}, threshold={self.threshold:.3f}, warmup_steps={self.warmup_steps}, max_warmup_difficulty={self.max_warmup_difficulty:.3f}, stop_curriculum_at={self.stop_curriculum_at}, min_steps_between_increases={self.min_steps_between_increases}")

    def _on_step(self) -> bool:
        if self.num_timesteps < self.warmup_steps:
            # Linear ramp during warmup
            frac = self.num_timesteps / max(1, self.warmup_steps)
            self.current_difficulty = self.max_warmup_difficulty * frac
            self.performance_based = 0
            if self.verbose > 0:
                print(f"[PERFORMANCE CURRICULUM] Warmup phase: t={self.num_timesteps}, frac={frac:.3f}, difficulty={self.current_difficulty:.3f}")
        else:
            if not self._after_warmup:
                self.current_difficulty = self.max_warmup_difficulty
                self._after_warmup = True
                if self.verbose > 0:
                    print(f"[PERFORMANCE CURRICULUM] Warmup complete. Starting performance-based at difficulty={self.current_difficulty:.3f}")
            # Performance-based stepping
            win_rate = None
            if hasattr(self.logger, 'name_to_value') and "eval/win_rate" in self.logger.name_to_value:
                win_rate = self.logger.name_to_value["eval/win_rate"]
            # Check if win rate has changed (new evaluation cycle)
            if win_rate is not None and win_rate != self._last_win_rate:
                self._increased_this_cycle = False
                self._last_win_rate = win_rate
                if self.verbose > 0:
                    print(f"[PERFORMANCE CURRICULUM] New win rate evaluation: {win_rate:.3f}")
            # Check if enough time has passed since last increase
            steps_since_last = self.num_timesteps - self._last_increase_timestep
            if (win_rate is not None and 
                win_rate >= self.threshold and 
                self.current_difficulty < self.stop_curriculum_at and  # RESPECT CEILING
                not self._increased_this_cycle and
                steps_since_last >= self.min_steps_between_increases):  # MIN WAIT TIME
                prev_diff = self.current_difficulty
                # Only increase if we won't exceed stop_curriculum_at
                next_diff = min(self.current_difficulty + self.step_size, self.stop_curriculum_at)
                if next_diff > self.current_difficulty:
                    self.current_difficulty = next_diff
                    self.performance_based = 1
                    self._increased_this_cycle = True
                    self._last_increase_timestep = self.num_timesteps  # Track when we increased
                    if self.verbose > 0:
                        print(f"[PERFORMANCE CURRICULUM] Win rate {win_rate:.3f} >= {self.threshold}, increasing difficulty from {prev_diff:.3f} to {self.current_difficulty:.3f}")
        # Update all environments
        if hasattr(self.training_env, 'envs'):
            for e in self.training_env.envs:
                if hasattr(e, '_update_opponent'):
                    e._update_opponent(self.current_difficulty, force_update=True)
        # Log curriculum information
        self.logger.record("curriculum/difficulty", self.current_difficulty)
        self.logger.record("curriculum/performance_based", self.performance_based)
        return True

    def set_training_env(self, env):
        # Robust to SB3 version
        if hasattr(super(), "set_training_env"):
            super().set_training_env(env)
        else:
            self.__dict__["training_env"] = env


def make_curriculum_env(
    n_envs: int = 8,
    gui: bool = False,
    curriculum_schedule: str = "exponential",  # Changed default to exponential
    total_timesteps: int = 3_000_000,
    warmup_steps: int = 10_000,  # Reduced from 50_000 to 10_000
    **kwargs: Any,
) -> VecEnv:
    """Create vectorized environment with curriculum learning."""
    
    def make_single_env():
        # Create base environment
        base_env = DogfightAviary(gui=gui)
        
        # Wrap with fixed red opponent
        env = BlueVsFixedRedWrapper(base_env)
        
        # Create difficulty scheduler with faster progression
        scheduler = DifficultyScheduler(
            total_timesteps=total_timesteps,
            difficulty_schedule=curriculum_schedule,
            min_difficulty=0.0,
            max_difficulty=1.0,
            warmup_steps=warmup_steps,
            # For step schedule, define discrete stages
            curriculum_stages=[0.0, 0.25, 0.5, 0.75, 1.0] if curriculum_schedule == "step" else [],
        )
        
        # Wrap with curriculum
        env = CurriculumOpponentWrapper(env, scheduler)
        
        return env
    
    # Create vectorized environment
    env = DummyVecEnv([make_single_env for _ in range(n_envs)])
    
    return env


def train_ppo_curriculum_cma(
    *,
    total_timesteps: int = 3_000_000,
    n_envs: int = 8,
    curriculum_schedule: str = "exponential",  # Changed default
    warmup_steps: int = 10_000,  # Added parameter
    win_rate_threshold: float = 0.8,  # Performance-based curriculum
    difficulty_step: float = 0.25,  # Performance-based curriculum
    cma_learning_rate: float = 0.1,
    cma_memory_size: int = 100,
    cma_min_variance: float = 0.01,
    cma_max_variance: float = 2.0,
    cma_variance_decay: float = 0.95,
    cma_variance_expansion: float = 1.05,
    cma_performance_threshold: float = 0.1,
    eval_freq: int = 10000,
    eval_episodes: int = 10,
    gui: bool = False,
    verbose: int = 1,
    performance_threshold: float = 0.8,
    performance_step_size: float = 0.25,
    performance_warmup_steps: int = 10000,
    checkpoint_freq: int = 100_000,
    max_warmup_difficulty: float = 0.3,
    stop_curriculum_at: float = 1.0,  # NEW
    min_steps_between_increases: int = 0,  # NEW
) -> PPOCMA:
    """Train PPO-CMA with curriculum learning."""

    # Create a unique run_id for this training session
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create environment
    env = make_curriculum_env(
        n_envs=n_envs,
        gui=gui,
        curriculum_schedule=curriculum_schedule,
        total_timesteps=total_timesteps,
        warmup_steps=warmup_steps,
    )

    # Create evaluation environment
    eval_env = make_curriculum_env(
        n_envs=n_envs,
        gui=False,
        curriculum_schedule=curriculum_schedule,
        total_timesteps=total_timesteps,
        warmup_steps=warmup_steps,
    )

    # Create unique directories for this run
    best_model_save_path = Path(f"models/{run_id}/")
    checkpoint_save_path = Path(f"models/checkpoints/{run_id}/")
    best_model_save_path.mkdir(parents=True, exist_ok=True)
    checkpoint_save_path.mkdir(parents=True, exist_ok=True)

    # Create callbacks
    if curriculum_schedule == "performance":
        curriculum_cb = PerformanceCurriculumCallback(
            threshold=performance_threshold,
            step_size=performance_step_size,
            warmup_steps=performance_warmup_steps,
            max_warmup_difficulty=max_warmup_difficulty,  # Added max_warmup_difficulty
            stop_curriculum_at=stop_curriculum_at,  # NEW
            min_steps_between_increases=min_steps_between_increases,  # NEW
            verbose=verbose,
        )
    else:
        curriculum_cb = CurriculumCallback(
            verbose=verbose,
            win_rate_threshold=win_rate_threshold,
            difficulty_step=difficulty_step,
            warmup_steps=warmup_steps,
            max_warmup_difficulty=max_warmup_difficulty, # Added max_warmup_difficulty
        )
    eval_cb = DogfightEvalCallback(eval_freq=eval_freq, n_eval_episodes=eval_episodes, verbose=verbose)
    intent_cb = IntentAccuracyCallback(eval_freq=eval_freq, verbose=verbose)

    # Add WinRateCallback for best winrate model saving (unique name)
    winrate_cb = WinRateCallback(
        eval_env=eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=eval_episodes,
        deterministic=True,
        best_model_save_path=str(best_model_save_path),
        best_model_name=f"amf_best_winrate_{run_id}",
        verbose=verbose,
    )
    # Add CheckpointCallback for periodic checkpoint saving (unique prefix)
    checkpoint_cb = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(checkpoint_save_path),
        name_prefix=f"ppo_cma_step_{run_id}",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=verbose,
    )

    callbacks = CallbackList([
        curriculum_cb,
        eval_cb,
        intent_cb,
        winrate_cb,
        checkpoint_cb,
    ])

    # Create PPO-CMA model
    model = PPOCMA(
        policy="MlpPolicy",
        env=env,
        # PPO parameters
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        # CMA parameters
        cma_learning_rate=cma_learning_rate,
        cma_memory_size=cma_memory_size,
        cma_min_variance=cma_min_variance,
        cma_max_variance=cma_max_variance,
        cma_variance_decay=cma_variance_decay,
        cma_variance_expansion=cma_variance_expansion,
        cma_performance_threshold=cma_performance_threshold,
        # Logging
        tensorboard_log="runs/ppo_curriculum_cma/",
        verbose=verbose,
    )
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )
    
    return model


def evaluate_curriculum_progress(
    model: PPOCMA,
    eval_env: VecEnv,
    n_eval_episodes: int = 100,
) -> Dict[str, float]:
    """Evaluate model performance across different curriculum stages."""
    
    # Test against different difficulty levels
    difficulties = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = {}
    
    for difficulty in difficulties:
        wins = 0
        total_reward = 0.0
        
        for episode in range(n_eval_episodes // len(difficulties)):
            # Set difficulty for evaluation
            for env in eval_env.envs:
                if hasattr(env, 'current_difficulty'):
                    env.current_difficulty = difficulty
                    env._update_opponent(difficulty)
            
            obs_reset = eval_env.reset()
            # VecEnv.reset() returns only obs; handle both cases
            if isinstance(obs_reset, tuple):
                obs, info = obs_reset
            else:
                obs = obs_reset
                info = {}
            done = False
            episode_reward = 0.0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                episode_reward += np.mean(reward)
            
            # Determine if blue won (red is down)
            if hasattr(eval_env.envs[0].env, '_red_down'):
                blue_win = eval_env.envs[0].env._red_down()
                if blue_win:
                    wins += 1
            
            total_reward += episode_reward
        
        # Calculate metrics
        n_episodes = n_eval_episodes // len(difficulties)
        win_rate = wins / n_episodes if n_episodes > 0 else 0.0
        avg_reward = total_reward / n_episodes if n_episodes > 0 else 0.0
        
        results[f'difficulty_{difficulty:.2f}_win_rate'] = win_rate
        results[f'difficulty_{difficulty:.2f}_avg_reward'] = avg_reward
    
    return results


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train PPO-CMA with curriculum learning")
    
    # Training parameters
    parser.add_argument("--total-timesteps", type=int, default=3_000_000,
                       help="Total timesteps for training")
    parser.add_argument("--n-envs", type=int, default=8,
                       help="Number of parallel environments")
    parser.add_argument("--gui", action="store_true",
                       help="Enable GUI visualization")
    parser.add_argument("--verbose", type=int, default=1,
                       help="Verbosity level")
    
    # Curriculum parameters
    parser.add_argument("--curriculum-schedule", type=str, default="exponential",  # Changed default
                       choices=["linear", "exponential", "cosine", "step", "performance"],
                       help="Curriculum difficulty schedule")
    parser.add_argument("--warmup-steps", type=int, default=10_000,  # Added parameter
                       help="Warmup steps before curriculum starts")
    parser.add_argument("--win-rate-threshold", type=float, default=0.8,  # Performance-based
                       help="Win rate threshold for performance-based curriculum")
    parser.add_argument("--difficulty-step", type=float, default=0.25,  # Performance-based
                       help="Difficulty step size for performance-based curriculum")
    
    # Performance-based curriculum flags
    parser.add_argument("--curriculum-threshold", type=float, default=0.8,
                       help="Performance-based curriculum win rate threshold")
    parser.add_argument("--curriculum-step-size", type=float, default=0.25,
                       help="Performance-based curriculum step size")
    parser.add_argument("--curriculum-warmup-steps", type=int, default=10000,
                       help="Performance-based curriculum warmup steps")
    parser.add_argument("--max-warmup-difficulty", type=float, default=0.3,
                       help="Maximum difficulty during warmup phase")
    
    # New curriculum control arguments
    parser.add_argument("--stop-curriculum-at", type=float, default=0.9,
                       help="Maximum curriculum difficulty (stop here)")
    parser.add_argument("--min-steps-between-increases", type=int, default=60000,
                       help="Minimum timesteps between difficulty increases")
    
    # CMA parameters
    parser.add_argument("--cma-learning-rate", type=float, default=0.1,
                       help="CMA learning rate")
    parser.add_argument("--cma-memory-size", type=int, default=100,
                       help="CMA memory size for performance tracking")
    parser.add_argument("--cma-min-variance", type=float, default=0.01,
                       help="Minimum CMA variance")
    parser.add_argument("--cma-max-variance", type=float, default=2.0,
                       help="Maximum CMA variance")
    parser.add_argument("--cma-variance-decay", type=float, default=0.95,
                       help="CMA variance decay factor")
    parser.add_argument("--cma-variance-expansion", type=float, default=1.05,
                       help="CMA variance expansion factor")
    parser.add_argument("--cma-performance-threshold", type=float, default=0.1,
                       help="CMA performance threshold")
    
    # Evaluation parameters
    parser.add_argument("--eval-episodes", type=int, default=100,
                       help="Number of evaluation episodes")
    parser.add_argument("--eval-freq", type=int, default=30000,
                       help="Evaluation frequency (in agent steps)")
    parser.add_argument("--save-model", type=str, default=None,
                       help="Path to save the trained model")
    
    args = parser.parse_args()
    
    print(f"[INFO] Starting PPO-CMA training with curriculum learning")
    print(f"[INFO] Total timesteps: {args.total_timesteps}")
    print(f"[INFO] Curriculum schedule: {args.curriculum_schedule}")
    print(f"[INFO] CMA parameters: lr={args.cma_learning_rate}, memory={args.cma_memory_size}")
    
    # Fix eval_freq so it's in environment steps (SB3 expects env steps, not agent steps)
    eval_freq_env = args.eval_freq // args.n_envs
    if eval_freq_env == 0:
        eval_freq_env = 1

    # Train the model
    model = train_ppo_curriculum_cma(
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        curriculum_schedule=args.curriculum_schedule,
        warmup_steps=args.warmup_steps,
        win_rate_threshold=args.win_rate_threshold,
        difficulty_step=args.difficulty_step,
        cma_learning_rate=args.cma_learning_rate,
        cma_memory_size=args.cma_memory_size,
        cma_min_variance=args.cma_min_variance,
        cma_max_variance=args.cma_max_variance,
        cma_variance_decay=args.cma_variance_decay,
        cma_variance_expansion=args.cma_variance_expansion,
        cma_performance_threshold=args.cma_performance_threshold,
        eval_freq=eval_freq_env,
        eval_episodes=args.eval_episodes,
        verbose=args.verbose,
        performance_threshold=args.curriculum_threshold,
        performance_step_size=args.curriculum_step_size,
        performance_warmup_steps=args.curriculum_warmup_steps,
        max_warmup_difficulty=args.max_warmup_difficulty, # Pass max_warmup_difficulty
        stop_curriculum_at=args.stop_curriculum_at,
        min_steps_between_increases=args.min_steps_between_increases,
    )
    
    # Evaluate the model
    print(f"[INFO] Evaluating model performance across curriculum stages...")
    eval_env = make_curriculum_env(
        n_envs=args.n_envs,
        gui=False,
        curriculum_schedule=args.curriculum_schedule,
        total_timesteps=args.total_timesteps,
        warmup_steps=args.warmup_steps,
    )
    
    results = evaluate_curriculum_progress(
        model=model,
        eval_env=eval_env,
        n_eval_episodes=args.eval_episodes,
    )
    
    # Print results
    print(f"\n[RESULTS] Curriculum evaluation:")
    for key, value in results.items():
        print(f"  {key}: {value:.3f}")
    
    # Save model if requested
    if args.save_model:
        save_path = Path(args.save_model)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(save_path))
        print(f"[INFO] Model saved to {save_path}")
    
    # Save evaluation results
    results_path = Path("results") / "curriculum_evaluation.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"[INFO] Evaluation results saved to {results_path}")
    
    return model, results


if __name__ == "__main__":
    main() 