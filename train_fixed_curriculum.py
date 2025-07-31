#!/usr/bin/env python3
"""
Fixed training script with conservative curriculum and enhanced exploration.

Usage:
    python train_fixed_curriculum.py --config configs/ppo_curriculum_fixed.yaml

Key improvements:
- Conservative curriculum progression with patience and rollback
- Enhanced exploration with higher entropy coefficient
- Symmetric advantage terms (positive/negative)
- Better evaluation metrics and diagnostics
- Delayed AMF auxiliary loss weight
"""

import argparse
import datetime
from pathlib import Path
from typing import Any, Dict

import yaml
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from uav_intent_rl.algo.intent_ppo import IntentPPO
from uav_intent_rl.policies import AMFPolicy
from uav_intent_rl.utils.callbacks import WinRateCallback
from uav_intent_rl.utils.curriculum_wrappers import CurriculumOpponentWrapper, DifficultyScheduler
from uav_intent_rl.utils.intent_wrappers import BlueVsFixedRedWrapper
from uav_intent_rl.examples.ppo_curriculum_cma import ConservativeCurriculumCallback
from uav_intent_rl.envs.DogfightAviary import DogfightAviary

# Use the enhanced DogfightAviary (already implemented)
EnhancedDogfightAviary = DogfightAviary

class FixedDogfightEvalCallback:
    """Enhanced evaluation callback with better diagnostics."""
    def __init__(self, eval_freq: int = 20000, n_eval_episodes: int = 80, verbose: int = 1):
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.verbose = verbose
        self.eval_env = None
        self.logger = None
        self.model = None
        self.num_timesteps = 0

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            self._evaluate_model()
        return True

    def _evaluate_model(self):
        """Enhanced evaluation with detailed metrics."""
        if self.eval_env is None:
            base_env = EnhancedDogfightAviary(gui=False)
            self.eval_env = BlueVsFixedRedWrapper(base_env)
        wins = 0
        losses = 0
        draws = 0
        total_reward = 0.0
        total_steps = 0
        total_blue_shots = 0
        total_blue_hits = 0
        for episode in range(self.n_eval_episodes):
            obs_reset = self.eval_env.reset(seed=episode)
            if isinstance(obs_reset, tuple):
                obs, info = obs_reset
            else:
                obs = obs_reset
                info = {}
            done = False
            episode_reward = 0.0
            episode_steps = 0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_steps += 1
            if hasattr(self.eval_env.env, '_red_down') and hasattr(self.eval_env.env, '_blue_down'):
                red_down = self.eval_env.env._red_down()
                blue_down = self.eval_env.env._blue_down()
                if red_down and not blue_down:
                    wins += 1
                elif blue_down and not red_down:
                    losses += 1
                else:
                    draws += 1
            total_reward += episode_reward
            total_steps += episode_steps
            if hasattr(self.eval_env.env, '_blue_shots_fired'):
                total_blue_shots += self.eval_env.env._blue_shots_fired
                total_blue_hits += self.eval_env.env._blue_hits_landed
        win_rate = wins / self.n_eval_episodes
        loss_rate = losses / self.n_eval_episodes
        draw_rate = draws / self.n_eval_episodes
        avg_reward = total_reward / self.n_eval_episodes
        avg_episode_length = total_steps / self.n_eval_episodes
        hit_accuracy = total_blue_hits / max(1, total_blue_shots)
        shots_per_episode = total_blue_shots / self.n_eval_episodes
        if self.logger:
            self.logger.record("eval/win_rate", win_rate)
            self.logger.record("eval/loss_rate", loss_rate)
            self.logger.record("eval/draw_rate", draw_rate)
            self.logger.record("eval/avg_reward", avg_reward)
            self.logger.record("eval/avg_episode_length", avg_episode_length)
            self.logger.record("eval/hit_accuracy", hit_accuracy)
            self.logger.record("eval/shots_per_episode", shots_per_episode)
            self.logger.record("eval/total_episodes", self.n_eval_episodes)
        if self.verbose > 0:
            print(f"[EVAL] Step {self.num_timesteps:,}: Win={win_rate:.3f}, "
                  f"Hit_Acc={hit_accuracy:.3f}, Avg_Reward={avg_reward:.3f}")


def make_enhanced_curriculum_env(n_envs: int = 8, gui: bool = False, **kwargs) -> VecNormalize:
    """Create vectorized environment with enhanced curriculum and diagnostics."""
    def make_single_env():
        base_env = EnhancedDogfightAviary(gui=gui)
        env = BlueVsFixedRedWrapper(base_env)
        scheduler = DifficultyScheduler(
            total_timesteps=kwargs.get('total_timesteps', 6_000_000),
            difficulty_schedule="performance",
            min_difficulty=0.0,
            max_difficulty=kwargs.get('stop_curriculum_at', 0.85),
            warmup_steps=kwargs.get('warmup_steps', 300_000),
        )
        env = CurriculumOpponentWrapper(env, scheduler)
        return env
    env = DummyVecEnv([make_single_env for _ in range(n_envs)])
    env = VecNormalize(
        env, 
        gamma=0.99, 
        norm_reward=True, 
        clip_reward=10.0,
        norm_obs=False
    )
    return env

def train_fixed_curriculum(cfg: Dict[str, Any], gui: bool = False):
    total_timesteps = int(cfg.get("total_timesteps", 6_000_000))
    n_envs = int(cfg.get("n_envs", 8))
    amf_lambda = float(cfg.get("amf_lambda", 0.30))
    lambda_schedule = str(cfg.get("lambda_schedule", "linear"))
    lambda_warmup = int(cfg.get("lambda_warmup_steps", 2_500_000))
    curriculum_threshold = float(cfg.get("curriculum_threshold", 0.65))
    curriculum_step_size = float(cfg.get("curriculum_step_size", 0.05))
    stop_curriculum_at = float(cfg.get("stop_curriculum_at", 0.85))
    min_steps_between_steps = int(cfg.get("min_steps_between_steps", 5))
    lookback_window = int(cfg.get("curriculum_lookback_window", 3))
    patience_steps = int(cfg.get("curriculum_patience", 200_000))
    rollback_threshold = float(cfg.get("curriculum_rollback_threshold", 0.45))
    max_rollbacks = int(cfg.get("curriculum_max_rollbacks", 2))
    eval_freq = int(cfg.get("eval_freq", 20_000))
    eval_episodes = int(cfg.get("eval_episodes", 80))
    checkpoint_freq = int(cfg.get("checkpoint_freq", 100_000))
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    best_model_save_path = Path(f"models/fixed_curriculum_{run_id}/")
    checkpoint_save_path = Path(f"models/checkpoints/fixed_{run_id}/")
    best_model_save_path.mkdir(parents=True, exist_ok=True)
    checkpoint_save_path.mkdir(parents=True, exist_ok=True)
    print(f"[SETUP] Starting fixed curriculum training:")
    print(f"  - Run ID: {run_id}")
    print(f"  - Total timesteps: {total_timesteps:,}")
    print(f"  - Curriculum threshold: {curriculum_threshold:.2f}")
    print(f"  - Curriculum step size: {curriculum_step_size:.3f}")
    print(f"  - Stop curriculum at: {stop_curriculum_at:.2f}")
    print(f"  - Min consecutive successes: {min_steps_between_steps}")
    print(f"  - AMF lambda: {amf_lambda:.2f} (warmup: {lambda_warmup:,} steps)")
    env = make_enhanced_curriculum_env(
        n_envs=n_envs,
        gui=gui,
        total_timesteps=total_timesteps,
        stop_curriculum_at=stop_curriculum_at,
        warmup_steps=int(cfg.get("curriculum_warmup_steps", 300_000))
    )
    eval_env = make_enhanced_curriculum_env(
        n_envs=n_envs,
        gui=False,
        total_timesteps=total_timesteps,
        stop_curriculum_at=stop_curriculum_at,
        warmup_steps=int(cfg.get("curriculum_warmup_steps", 300_000))
    )
    model = IntentPPO(
        policy=AMFPolicy,
        env=env,
        aux_loss_coef=amf_lambda,
        lambda_schedule=lambda_schedule,
        lambda_warmup_steps=lambda_warmup,
        use_balanced_loss=bool(cfg.get("balanced_loss", True)),
        learning_rate=float(cfg.get("learning_rate", 5e-5)),
        n_steps=int(cfg.get("n_steps", 4096)),
        batch_size=int(cfg.get("batch_size", 512)),
        n_epochs=int(cfg.get("n_epochs", 6)),
        gamma=float(cfg.get("gamma", 0.99)),
        gae_lambda=float(cfg.get("gae_lambda", 0.95)),
        clip_range=float(cfg.get("clip_range", 0.15)),
        vf_coef=float(cfg.get("vf_coef", 1.0)),
        ent_coef=float(cfg.get("ent_coef", 0.025)),
        max_grad_norm=float(cfg.get("max_grad_norm", 0.6)),
        target_kl=float(cfg.get("target_kl", 0.025)),
        policy_kwargs=dict(
            net_arch=[dict(pi=[512, 256, 128], vf=[512, 256, 128])],
            use_lstm=True,
            lstm_hidden_size=128,
            lstm_num_layers=1,
            lstm_type="lstm",
            opp_feature_dim=32,
        ),
        tensorboard_log=f"runs/fixed_curriculum_{run_id}/",
        verbose=1,
    )
    curriculum_cb = ConservativeCurriculumCallback(
        threshold=curriculum_threshold,
        step_size=curriculum_step_size,
        warmup_steps=int(cfg.get("curriculum_warmup_steps", 300_000)),
        max_warmup_difficulty=float(cfg.get("max_warmup_difficulty", 0.4)),
        stop_curriculum_at=stop_curriculum_at,
        min_steps_between_steps=min_steps_between_steps,
        lookback_window=lookback_window,
        patience_steps=patience_steps,
        rollback_threshold=rollback_threshold,
        max_rollbacks=max_rollbacks,
        verbose=1,
    )
    eval_cb = FixedDogfightEvalCallback(
        eval_freq=eval_freq,
        n_eval_episodes=eval_episodes,
        verbose=1
    )
    winrate_cb = WinRateCallback(
        eval_env=eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=eval_episodes,
        deterministic=True,
        best_model_save_path=str(best_model_save_path),
        best_model_name=f"fixed_best_winrate_{run_id}",
        verbose=1,
    )
    checkpoint_cb = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(checkpoint_save_path),
        name_prefix=f"fixed_curriculum_{run_id}",
        save_replay_buffer=False,
        save_vecnormalize=True,
        verbose=1,
    )
    callbacks = CallbackList([
        curriculum_cb,
        eval_cb,
        winrate_cb,
        checkpoint_cb,
    ])
    print(f"[TRAINING] Starting training with conservative curriculum...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )
    final_model_path = best_model_save_path / f"fixed_final_{run_id}.zip"
    model.save(str(final_model_path))
    print(f"[COMPLETE] Training finished!")
    print(f"  - Final model: {final_model_path}")
    print(f"  - Best winrate model: {best_model_save_path}")
    print(f"  - Checkpoints: {checkpoint_save_path}")
    return model

def main():
    parser = argparse.ArgumentParser(description="Train with fixed conservative curriculum")
    parser.add_argument("--config", type=str, default="configs/ppo_curriculum_fixed.yaml",
                       help="YAML config file path")
    parser.add_argument("--gui", action="store_true",
                       help="Enable PyBullet GUI")
    args = parser.parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Error: Config file not found: {cfg_path}")
        print("Using default configuration...")
        cfg = {
            "total_timesteps": 6_000_000,
            "n_envs": 8,
            "curriculum_threshold": 0.65,
            "curriculum_step_size": 0.05,
            "stop_curriculum_at": 0.85,
            "min_steps_between_steps": 5,
            "curriculum_lookback_window": 3,
            "curriculum_patience": 200_000,
            "curriculum_rollback_threshold": 0.45,
            "curriculum_max_rollbacks": 2,
            "amf_lambda": 0.30,
            "lambda_schedule": "linear",
            "lambda_warmup_steps": 2_500_000,
            "learning_rate": 5e-5,
            "clip_range": 0.15,
            "ent_coef": 0.025,
            "max_grad_norm": 0.6,
            "target_kl": 0.025,
            "eval_episodes": 80,
            "eval_freq": 20_000,
            "checkpoint_freq": 100_000,
        }
    else:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    model = train_fixed_curriculum(cfg, gui=args.gui)
    return model

if __name__ == "__main__":
    main() 