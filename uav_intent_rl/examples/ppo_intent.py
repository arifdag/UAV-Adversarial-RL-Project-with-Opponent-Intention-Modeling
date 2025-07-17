from __future__ import annotations

"""Train Blue agent with IntentPPO (auxiliary opponent modelling).

Usage (minimal):
    python -m uav_intent_rl.examples.ppo_intent --total-timesteps 3_000_000

Optional flags:
    --lambda 0.3         weight for auxiliary loss (defaults 0.1)
    --gui                enable PyBullet GUI
    --sweep              run a quick sweep over λ in {0.1,0.3,0.5,0.7,1.0}
"""

import argparse
from pathlib import Path
from typing import List

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

from uav_intent_rl.utils.callbacks import AuxAccuracyCallback
from uav_intent_rl.utils.callbacks import WinRateCallback
from uav_intent_rl.algo.intent_ppo import IntentPPO
from uav_intent_rl.envs import DogfightAviary
from uav_intent_rl.utils.intent_wrappers import BlueVsFixedRedWrapper


# -----------------------------------------------------------------------------
# Helper to build a vectorised environment
# -----------------------------------------------------------------------------

def _make_env(n_envs: int = 8, *, gui: bool = False):
    def factory():
        base_env = DogfightAviary(gui=gui)
        return BlueVsFixedRedWrapper(base_env)

    return make_vec_env(factory, n_envs=n_envs)


# -----------------------------------------------------------------------------
# Training routine
# -----------------------------------------------------------------------------

def train_single(
    total_timesteps: int,
    aux_coef: float,
    gui: bool = False,
    resume: str | None = None,
    learning_rate: float | None = None,
    lambda_schedule: str = "constant",
    lambda_warmup_steps: int = 0,
    lambda_min: float = 0.0,
    freeze_feature_steps: int = 0,
    n_envs: int = 8,
    ent_coef: float = 0.01,
    clip_range: float = 0.2,
    balanced_loss: bool = False,
) -> None:  # noqa: D401
    env = _make_env(n_envs=n_envs, gui=gui)
    eval_env = _make_env(n_envs=n_envs, gui=False)

    if resume:
        print(f"[INFO] Resuming from checkpoint: {resume}")
        model = IntentPPO.load(resume, env=env)
        # Always set aux_loss_coef to current value
        model.aux_loss_coef_max = aux_coef
        model.lambda_schedule = lambda_schedule
        model.lambda_warmup_steps = lambda_warmup_steps
        model.total_timesteps = total_timesteps
        model.aux_loss_coef_min = lambda_min
        model.freeze_feature_steps = freeze_feature_steps
        model.ent_coef = ent_coef  # Ensure entropy bonus floor
        if learning_rate is not None:
            print(f"[INFO] Overriding learning rate to {learning_rate}")
            model.learning_rate = learning_rate
    else:
        model = IntentPPO(
            env=env,
            aux_loss_coef=aux_coef,
            aux_loss_coef_min=lambda_min,
            learning_rate=learning_rate if learning_rate is not None else 3e-4,
            ent_coef=ent_coef,
            use_balanced_loss=balanced_loss,
            n_steps=2048,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=clip_range,
            verbose=1,
            tensorboard_log="runs/ppo_intent/",
            policy_kwargs=dict(
                net_arch=[dict(pi=[256, 256], vf=[256, 256])],
            ),
            lambda_schedule=lambda_schedule,
            lambda_warmup_steps=lambda_warmup_steps,
            freeze_feature_steps=freeze_feature_steps,
            total_timesteps=total_timesteps,
        )

    callbacks: List = []
    # Reward evaluation (unchanged)
    callbacks.append(
        EvalCallback(eval_env, eval_freq=20_000, n_eval_episodes=10, deterministic=True)
    )
    # Validation accuracy & best-model checkpoint
    callbacks.append(
        AuxAccuracyCallback(
            eval_env,
            eval_freq=20_000,
            n_eval_episodes=10,
            deterministic=True,
            best_model_save_path="models/",
            best_model_name=f"intent_best_lambda_{aux_coef:.2f}",
            verbose=1,
        )
    )
    # Win-rate evaluation
    callbacks.append(
        WinRateCallback(
            eval_env,
            eval_freq=20_000,
            n_eval_episodes=20,
            verbose=1,
        )
    )
    callbacks.append(
        CheckpointCallback(
            save_freq=100_000,
            save_path="models/",
            name_prefix=f"intent_{aux_coef:.2f}lambda"
        )
    )

    model.learn(total_timesteps=total_timesteps, callback=CallbackList(callbacks))
    model.save(f"models/intent_final_lambda_{aux_coef:.2f}.zip")


# -----------------------------------------------------------------------------
# Simple λ sweep
# -----------------------------------------------------------------------------

def sweep_lambdas(total_timesteps: int, gui: bool = False):
    for lam in [0.1, 0.3, 0.5, 0.7, 1.0]:
        print(f"\n=== Training with aux_coef = {lam} ===")
        train_single(total_timesteps=total_timesteps, aux_coef=lam, gui=gui)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=3_000_000)
    parser.add_argument("--lambda", dest="aux", type=float, default=0.6)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--sweep", action="store_true", help="Run λ sweep 0.1→1.0")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--balanced-loss", action="store_true", help="Enable balanced class weights for aux loss")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate for resumed or new model")
    parser.add_argument("--lambda-schedule", type=str, default="constant", choices=["constant", "cosine", "linear", "step"], help="Lambda scheduler type")
    parser.add_argument("--lambda-min", type=float, default=0.0, help="Minimum lambda value when using a schedule (allows ramp from >0)")
    parser.add_argument("--lambda-warmup-steps", type=int, default=0, help="Number of steps to ramp lambda from 0 to max")
    parser.add_argument("--freeze-feature-steps", type=int, default=0, help="Freeze shared feature layers for the first N timesteps")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--n-envs", type=int, default=8, help="Number of parallel envs (VecEnv)")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy bonus coefficient (exploration)")
    args = parser.parse_args()

    if args.sweep:
        sweep_lambdas(total_timesteps=args.total_timesteps, gui=args.gui)
    else:
        train_single(
            total_timesteps=args.total_timesteps,
            aux_coef=args.aux,
            gui=args.gui,
            resume=args.resume,
            learning_rate=args.learning_rate,
            lambda_schedule=args.lambda_schedule,
            lambda_warmup_steps=args.lambda_warmup_steps,
            lambda_min=args.lambda_min,
            freeze_feature_steps=args.freeze_feature_steps,
            n_envs=args.n_envs,
            ent_coef=args.ent_coef,
            clip_range=args.clip_range,
            balanced_loss=args.balanced_loss,
        ) 