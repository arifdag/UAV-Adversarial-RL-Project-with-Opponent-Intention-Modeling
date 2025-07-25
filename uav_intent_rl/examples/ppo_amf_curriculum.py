from __future__ import annotations

"""
Train Blue agent with AMFPolicy (opponent-aware latent fusion) and curriculum learning.

Usage:
    python -m uav_intent_rl.examples.ppo_amf_curriculum --config configs/ppo_amf.yaml

This script adds curriculum learning (performance-based or linear) to AMF training.
Curriculum parameters can be set in the YAML config.
"""

import argparse
import datetime
from pathlib import Path
from typing import Any, Dict

import yaml
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize

from uav_intent_rl.algo.intent_ppo import IntentPPO
from uav_intent_rl.envs import DogfightAviary
from uav_intent_rl.policies import AMFPolicy
from uav_intent_rl.utils.callbacks import AuxAccuracyCallback, WinRateCallback
from uav_intent_rl.utils.curriculum_wrappers import CurriculumOpponentWrapper, DifficultyScheduler
from uav_intent_rl.utils.intent_wrappers import BlueVsFixedRedWrapper
from uav_intent_rl.examples.ppo_curriculum_cma import (
    PerformanceCurriculumCallback, CurriculumCallback, DogfightEvalCallback, IntentAccuracyCallback, make_curriculum_env
)

def train_from_config(cfg: Dict[str, Any], gui: bool = False):
    total_timesteps = int(cfg.get("total_timesteps", 3_000_000))
    n_envs = int(cfg.get("n_envs", 8))

    amf_lambda = float(cfg.get("amf_lambda", 0.5))
    lambda_schedule = str(cfg.get("lambda_schedule", "constant"))
    lambda_warmup = int(cfg.get("lambda_warmup_steps", 0))
    freeze_steps = int(cfg.get("freeze_feature_steps", 0))
    balanced_loss = bool(cfg.get("balanced_loss", False))

    # Curriculum parameters
    curriculum_schedule = str(cfg.get("curriculum_schedule", "performance"))
    warmup_steps = int(cfg.get("warmup_steps", 10000))
    performance_warmup_steps = int(cfg.get("curriculum_warmup_steps", 10000))
    max_warmup_difficulty = float(cfg.get("max_warmup_difficulty", 0.3))
    performance_threshold = float(cfg.get("curriculum_threshold", 0.6))
    performance_step_size = float(cfg.get("curriculum_step_size", 0.1))
    win_rate_threshold = float(cfg.get("win_rate_threshold", 0.8))
    difficulty_step = float(cfg.get("difficulty_step", 0.25))
    eval_freq = int(cfg.get("eval_freq", 20000))
    eval_episodes = int(cfg.get("eval_episodes", 20))
    checkpoint_freq = int(cfg.get("checkpoint_freq", 100000))

    # Unique run_id for saving
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    best_model_save_path = Path(f"models/{run_id}/")
    checkpoint_save_path = Path(f"models/checkpoints/{run_id}/")
    best_model_save_path.mkdir(parents=True, exist_ok=True)
    checkpoint_save_path.mkdir(parents=True, exist_ok=True)

    # Curriculum-enabled environment
    env = make_curriculum_env(
        n_envs=n_envs,
        gui=gui,
        curriculum_schedule=curriculum_schedule,
        total_timesteps=total_timesteps,
        warmup_steps=warmup_steps,
    )
    env = VecNormalize(env, gamma=0.99, norm_reward=True, clip_reward=10.0)
    eval_env = make_curriculum_env(
        n_envs=n_envs,
        gui=False,
        curriculum_schedule=curriculum_schedule,
        total_timesteps=total_timesteps,
        warmup_steps=warmup_steps,
    )
    eval_env = VecNormalize(eval_env, gamma=0.99, norm_reward=True, clip_reward=10.0)

    # Curriculum callback
    if curriculum_schedule == "performance":
        curriculum_cb = PerformanceCurriculumCallback(
            threshold=performance_threshold,
            step_size=performance_step_size,
            warmup_steps=performance_warmup_steps,
            max_warmup_difficulty=max_warmup_difficulty,
            verbose=1,
        )
    else:
        curriculum_cb = CurriculumCallback(
            verbose=1,
            win_rate_threshold=win_rate_threshold,
            difficulty_step=difficulty_step,
            warmup_steps=warmup_steps,
            max_warmup_difficulty=max_warmup_difficulty,
        )
    eval_cb = DogfightEvalCallback(eval_freq=eval_freq, n_eval_episodes=eval_episodes, verbose=1)
    intent_cb = IntentAccuracyCallback(eval_freq=eval_freq, verbose=1)

    winrate_cb = WinRateCallback(
        eval_env=eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=eval_episodes,
        deterministic=True,
        best_model_save_path=str(best_model_save_path),
        best_model_name=f"amf_best_winrate_{run_id}",
        verbose=1,
    )
    checkpoint_cb = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(checkpoint_save_path),
        name_prefix=f"amf_step_{run_id}",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=1,
    )

    callbacks = CallbackList([
        curriculum_cb,
        eval_cb,
        intent_cb,
        winrate_cb,
        checkpoint_cb,
    ])

    model = IntentPPO(
        policy=AMFPolicy,
        env=env,
        aux_loss_coef=amf_lambda,
        lambda_schedule=lambda_schedule,
        lambda_warmup_steps=lambda_warmup,
        freeze_feature_steps=freeze_steps,
        use_balanced_loss=balanced_loss,
        learning_rate=float(cfg.get("learning_rate", 3e-4)),
        n_steps=int(cfg.get("n_steps", 2048)),
        batch_size=int(cfg.get("batch_size", 64)),
        gamma=float(cfg.get("gamma", 0.99)),
        gae_lambda=float(cfg.get("gae_lambda", 0.95)),
        clip_range=float(cfg.get("clip_range", 0.2)),
        vf_coef=float(cfg.get("vf_coef", 0.5)),
        ent_coef=float(cfg.get("ent_coef", 0.01)),
        max_grad_norm=float(cfg.get("max_grad_norm", 0.5)),
        tensorboard_log=f"runs/ppo_amf_curriculum_{run_id}/",
        policy_kwargs=dict(
            net_arch=[dict(pi=[512, 256, 128], vf=[512, 256, 128])],
            use_lstm=True,
            lstm_hidden_size=128,
            lstm_num_layers=1,
            lstm_type="lstm",
        ),
        verbose=1,
    )

    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    model.save(str(best_model_save_path / f"amf_final_{run_id}.zip"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AMFPolicy with IntentPPO and curriculum learning")
    parser.add_argument("--config", type=str, default="configs/ppo_amf.yaml", help="YAML config path")
    parser.add_argument("--gui", action="store_true", help="Enable PyBullet GUI")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_from_config(cfg, gui=args.gui) 