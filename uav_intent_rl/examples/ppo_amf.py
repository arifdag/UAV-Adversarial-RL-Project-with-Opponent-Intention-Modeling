from __future__ import annotations

"""Train Blue agent with *AMFPolicy* (opponent-aware latent fusion).

Example:
    python -m uav_intent_rl.examples.ppo_amf --config configs/ppo_amf.yaml

The script is intentionally minimal: it loads a YAML *config* (same keys
as ``configs/ppo_nomodel.yaml`` **plus** ``amf_lambda``) and kicks off
training with :class:`uav_intent_rl.algo.intent_ppo.IntentPPO` using the
custom :class:`uav_intent_rl.policies.amf_policy.AMFPolicy`.
"""

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

from uav_intent_rl.algo.intent_ppo import IntentPPO
from uav_intent_rl.envs import DogfightAviary
from uav_intent_rl.policies import AMFPolicy
from uav_intent_rl.utils.callbacks import AuxAccuracyCallback, WinRateCallback
from uav_intent_rl.utils.intent_wrappers import BlueVsFixedRedWrapper


# -----------------------------------------------------------------------------
# Helper – build vectorised DogfightAviary wrapped with fixed red opponent
# -----------------------------------------------------------------------------

def _make_env(n_envs: int, *, gui: bool = False):
    def _factory():
        base = DogfightAviary(gui=gui)
        return BlueVsFixedRedWrapper(base)

    return make_vec_env(_factory, n_envs=n_envs)


# -----------------------------------------------------------------------------
# Training routine driven by YAML config
# -----------------------------------------------------------------------------

def train_from_config(cfg: Dict[str, Any], gui: bool = False):
    total_timesteps = int(cfg.get("total_timesteps", 3_000_000))
    n_envs = int(cfg.get("n_envs", 8))

    amf_lambda = float(cfg.get("amf_lambda", 0.5))
    lambda_schedule = str(cfg.get("lambda_schedule", "constant"))
    lambda_warmup = int(cfg.get("lambda_warmup_steps", 0))
    freeze_steps = int(cfg.get("freeze_feature_steps", 0))
    balanced_loss = bool(cfg.get("balanced_loss", False))

    env = _make_env(n_envs=n_envs, gui=gui)
    eval_env = _make_env(n_envs=n_envs, gui=False)

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
        tensorboard_log="runs/ppo_amf/",
        policy_kwargs=dict(
            net_arch=[dict(pi=[512, 256, 128], vf=[512, 256, 128])],
            use_lstm=True,
            lstm_hidden_size=128,
            lstm_num_layers=1,
            lstm_type="lstm",
        ),
        verbose=1,
    )

    callbacks = CallbackList(
        [
            EvalCallback(
                eval_env,
                eval_freq=int(cfg.get("eval_freq", 20_000)),
                n_eval_episodes=10,
                deterministic=True,
            ),
            AuxAccuracyCallback(
                eval_env,
                eval_freq=int(cfg.get("eval_freq", 20_000)),
                n_eval_episodes=10,
                deterministic=True,
                best_model_save_path="models/",
                best_model_name="amf_best",
                verbose=1,
            ),
            WinRateCallback(
                eval_env,
                eval_freq=int(cfg.get("eval_freq", 20_000)),
                n_eval_episodes=20,
                deterministic=True,
                best_model_save_path="models/",
                best_model_name="amf_best_winrate",
                verbose=1,
            ),
            CheckpointCallback(
                save_freq=int(cfg.get("checkpoint_freq", 100_000)),
                save_path="models/",
                name_prefix="blue_amf",
            ),
        ]
    )

    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    model.save("models/blue_amf.zip")


# -----------------------------------------------------------------------------
# CLI entrypoint
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AMFPolicy with IntentPPO")
    parser.add_argument("--config", type=str, default="configs/ppo_amf.yaml", help="YAML config path")
    parser.add_argument("--gui", action="store_true", help="Enable PyBullet GUI")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_from_config(cfg, gui=args.gui) 