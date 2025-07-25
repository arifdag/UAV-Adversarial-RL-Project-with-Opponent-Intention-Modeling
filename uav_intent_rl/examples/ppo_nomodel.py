from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import gymnasium as gym
import numpy as np
import re
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
import io
from contextlib import redirect_stdout
import sys
from stable_baselines3.common.vec_env import VecNormalize

from uav_intent_rl.envs import DogfightAviary
from uav_intent_rl.policies import ScriptedRedPolicy

__all__ = ["run"]


class BlueVsFixedRedWrapper(gym.Wrapper):
    """Gym wrapper exposing only the *blue* drone to RL.

    The underlying :class:`~uav_intent_rl.envs.DogfightAviary` contains two
    drones (index 0 = blue, index 1 = red). During training we control only the
    blue drone, while the red drone executes the hand-crafted
    :class:`~uav_intent_rl.policies.ScriptedRedPolicy` on every step. The
    wrapper therefore:

    1. **Action space** – reduces from ``(2, 4)`` to ``(4,)`` (blue only).
    2. **Observation space** – flattens the original ``(2, 72)`` array to a
       1-D vector of length ``144`` so the agent can *see* both drones' states.
    3. **Reward/termination** – passes through unchanged (they are already
       shaped from *blue*'s perspective in the env).
    """

    def __init__(self, env: gym.Env, red_policy: ScriptedRedPolicy | None = None):  # noqa: D401
        super().__init__(env)
        # Lazily create a policy if not supplied so each env has its own copy.
        self._red_policy = red_policy or ScriptedRedPolicy()

        # --- Action space ----------------------------------------------------
        assert env.action_space.shape == (2, 4), "Unexpected base action space"
        low, high = env.action_space.low[0], env.action_space.high[0]
        self.action_space = gym.spaces.Box(low=low, high=high, shape=(4,), dtype=np.float32)

        # --- Observation space ----------------------------------------------
        obs_shape = int(np.prod(env.observation_space.shape))
        self.observation_space = gym.spaces.Box(
            low=-math.inf, high=math.inf, shape=(obs_shape,), dtype=np.float32
        )

    # ---------------------------------------------------------------------
    # Overrides
    # ---------------------------------------------------------------------

    def reset(self, **kwargs):  # type: ignore[override]
        obs, info = self.env.reset(**kwargs)
        return obs.flatten().astype(np.float32), info

    def step(self, action):  # type: ignore[override]
        # Build full 2×4 action array: blue action provided by RL agent, red
        # action from scripted policy.
        full_action = self._red_policy(self.env)  # shape (2,4)
        full_action[0] = np.asarray(action, dtype=np.float32)

        obs, reward, terminated, truncated, info = self.env.step(full_action)
        return obs.flatten().astype(np.float32), float(reward), bool(terminated), bool(truncated), info


# -------------------------------------------------------------------------
# High-level training helper – used by example script & unit-tests
# -------------------------------------------------------------------------

def _load_config() -> dict:
    """Load hyper-parameters from a YAML-style text file.

    Resolution order for the config file path:
    1. Command-line flag ``--config <path>`` or ``--config=<path>``.
    2. Default project file ``configs/ppo_nomodel.yaml``.
    """

    # --------------------------------------------------
    # 1) Resolve path from CLI (if provided)
    # --------------------------------------------------
    cfg_path: Path | None = None
    argv = sys.argv[1:]

    if "--config" in argv:
        idx = argv.index("--config")
        if idx + 1 < len(argv):
            cfg_path = Path(argv[idx + 1]).expanduser().resolve()
    else:
        for arg in argv:
            if arg.startswith("--config="):
                cfg_path = Path(arg.split("=", 1)[1]).expanduser().resolve()
                break

    # 2) Fallback to default path
    if cfg_path is None:
        cfg_path = Path(__file__).resolve().parents[2] / "configs" / "ppo_nomodel.yaml"

    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    # --------------------------------------------------
    # 3) Parse simple ``key: value`` pairs (YAML subset)
    # --------------------------------------------------
    cfg: dict[str, float | int | str] = {}
    pattern = re.compile(r"^(?P<key>\w+):\s*(?P<val>.+)$")

    with cfg_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            match = pattern.match(line.strip())
            if not match:
                continue
            key = match.group("key")
            val_raw = match.group("val")

            # Try int → float → str
            try:
                val: float | int | str = int(val_raw)
            except ValueError:
                try:
                    val = float(val_raw)
                except ValueError:
                    val = val_raw

            cfg[key] = val  # type: ignore[index]

    return cfg


def _make_single_env(gui: bool = False):
    """Factory that builds a *single* wrapped DogfightAviary instance."""

    def _init():  # lazy factory for `make_vec_env`
        # Suppress verbose prints from gym-pybullet-drones during env init
        buff = io.StringIO()
        with redirect_stdout(buff):
            base_env = DogfightAviary(gui=gui)
        return BlueVsFixedRedWrapper(base_env)

    return _init


def _evaluate_win_rate(model: PPO, n_episodes: int = 10) -> float:
    """Run *n_episodes* deterministic roll-outs and return blue's win-rate."""

    wins = 0
    for ep in range(n_episodes):
        env = _make_single_env(gui=False)()
        obs, _ = env.reset(seed=ep)
        done = False
        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, _info = env.step(action)
            done = terminated or truncated
        # Blue wins if red is down (see DogfightAviary helper)
        if env.env._red_down():  # type: ignore[attr-defined]
            wins += 1
        env.close()

    return wins / n_episodes if n_episodes > 0 else 0.0


# -------------------------------------------------------------------------
# Callback for periodic logging
# -------------------------------------------------------------------------

class WinRateCallback(BaseCallback):  # noqa: D101 – internal helper
    def __init__(self, eval_freq: int = 20_000, n_eval_episodes: int = 30, verbose: int = 0):
        super().__init__(verbose)
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)

    def _on_step(self) -> bool:  # type: ignore[override]
        if self.eval_freq > 0 and (self.num_timesteps % self.eval_freq == 0):
            win_rate = _evaluate_win_rate(self.model, n_episodes=self.n_eval_episodes)
            print(
                f"[INFO] Step {self.num_timesteps:,}: win-rate over {self.n_eval_episodes} eval",
                f"episodes = {win_rate * 100:.1f}%",
            )
        return True


class SaveBestWinRateCallback(BaseCallback):  # noqa: D101
    """Save the model whenever the deterministic win-rate improves."""

    def __init__(
        self,
        save_path: str,
        eval_freq: int = 20_000,
        n_eval_episodes: int = 30,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.save_path = save_path
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self._best_win_rate: float = -1.0

    def _on_step(self) -> bool:  # type: ignore[override]
        if self.eval_freq > 0 and (self.num_timesteps % self.eval_freq == 0):
            win_rate = _evaluate_win_rate(self.model, n_episodes=self.n_eval_episodes)
            if win_rate > self._best_win_rate:
                if self.verbose:
                    print(
                        f"[INFO] New best win-rate {win_rate*100:.1f}% → saving model to {self.save_path}"
                    )
                self._best_win_rate = win_rate
                self.model.save(self.save_path)
        return True


def run(
    *,
    total_timesteps: int | None = None,
    gui: bool = False,
    n_eval_episodes: int = 10,
    n_envs: int | None = None,
    verbose: int = 0,
) -> Tuple[PPO, float]:
    """Train a PPO policy for the *blue* drone against the scripted red.

    Parameters
    ----------
    total_timesteps : int | None, optional
        Override the number of training steps defined in the YAML config.
    gui : bool, optional
        Whether to enable PyBullet GUI during *evaluation* roll-outs.
    n_eval_episodes : int, optional
        How many episodes to average the win-rate over after training.
    n_envs : int | None, optional
        Override the number of vectorized environments.
    verbose : int, optional
        Verbosity passed to Stable-Baselines3 (0 = silent, 1 = info).

    Returns
    -------
    Tuple[stable_baselines3.PPO, float]
        The trained *model* and the measured *win-rate* over *n_eval_episodes*.
    """

    cfg = _load_config()
    if total_timesteps is not None:
        cfg["total_timesteps"] = int(total_timesteps)
    if n_envs is not None:
        cfg["n_envs"] = int(n_envs)

    # ------------------------------------------------------------------
    # Vectorised training environment
    # ------------------------------------------------------------------
    n_envs = int(cfg.get("n_envs", 1))
    train_env = make_vec_env(_make_single_env(gui=False), n_envs=n_envs)
    train_env = VecNormalize(train_env, gamma=0.99, norm_reward=True, clip_reward=10.0)

    # Separate single-env for periodic evaluation & best-model tracking
    eval_env = make_vec_env(_make_single_env(gui=False), n_envs=1)
    eval_env = VecNormalize(eval_env, gamma=0.99, norm_reward=True, clip_reward=10.0)

    # ------------------------------------------------------------------
    # Instantiate PPO – either fresh or resumed from checkpoint
    # ------------------------------------------------------------------

    resume_path = cfg.get("resume_model_path")
    if resume_path:
        print(f"[INFO] Resuming training from '{resume_path}'")

        # SB-3 `load()` ignores most extra kwargs, but we can override
        # hyper-parameters through the `custom_objects` mapping.  We
        # replace the stored *learning_rate* schedule by a constant
        # function returning the new scalar; same for *ent_coef* and
        # *target_kl* (added in SB-3 ≥1.8.0).

        lr_const = float(cfg.get("learning_rate", 3e-5))
        ent_coef_const = float(cfg.get("ent_coef", 0.01))
        target_kl_const = float(cfg.get("target_kl", 0.05))

        def _const_fn(_):
            return lr_const

        custom_objects = {
            "learning_rate": lr_const,
            "lr_schedule": _const_fn,
            "ent_coef": ent_coef_const,
            "target_kl": target_kl_const,
        }

        model = PPO.load(
            resume_path,
            env=train_env,
            device="auto",
            custom_objects=custom_objects,
            verbose=verbose,
        )

        # --- Explicitly patch attributes in case the pickled objects were
        # not fully replaced by `custom_objects` (SB-3 quirk) --------------
        model.lr_schedule = lambda _: lr_const  # type: ignore[assignment]
        model.learning_rate = lr_const
        model.ent_coef = ent_coef_const
        # `PPO.target_kl` was introduced in SB-3 1.8. If the attribute does
        # not exist (older version) patching is safe-to-ignore.
        if hasattr(model, "target_kl"):
            model.target_kl = target_kl_const
    else:
        # ------------------------------------------------------------------
        # Instantiate and train PPO
        # ------------------------------------------------------------------
        ppo_kwargs = {
            "gamma": float(cfg.get("gamma", 0.99)),
            "learning_rate": float(cfg.get("learning_rate", 3e-4)),
            "clip_range": float(cfg.get("clip_range", 0.2)),
            # Optional advanced settings
            "n_steps": int(cfg.get("n_steps", 2048)),
            "batch_size": int(cfg.get("batch_size", 64)),
            "gae_lambda": float(cfg.get("gae_lambda", 0.95)),
            # Exploration schedule: starts 0.03, linearly decays to 0.01
            "ent_coef": float(cfg.get("ent_coef", 0.02)),
            "vf_coef": float(cfg.get("vf_coef", 0.5)),
            "max_grad_norm": float(cfg.get("max_grad_norm", 0.5)),
            # Prevent overly large policy updates
            "target_kl": float(cfg.get("target_kl", 0.05)),
        }

        log_dir = Path(__file__).resolve().parents[2] / "runs" / "ppo_nomodel"
        log_dir.mkdir(parents=True, exist_ok=True)

        model = PPO(
            str(cfg.get("policy", "MlpPolicy")),
            train_env,
            verbose=verbose,
            tensorboard_log=str(log_dir),
            **ppo_kwargs,
        )

    # --------------------------------------------------------------
    # Attach periodic callbacks (win-rate logging + model checkpoint)
    # --------------------------------------------------------------

    # 1) Win-rate logger (prints metric)
    winrate_cb = WinRateCallback(verbose=verbose)

    # 1b) Save-by-win-rate checkpoint
    models_dir = Path(__file__).resolve().parents[2] / "models"
    winrate_best_path = str(models_dir / "winrate_best_model.zip")
    winrate_save_cb = SaveBestWinRateCallback(
        save_path=winrate_best_path, verbose=verbose
    )

    # 2) Periodic checkpointing for crash-safe training
    ckpt_dir = models_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = CheckpointCallback(
        save_freq=16_384 * 12,  # ≈ 200k env steps
        save_path=str(ckpt_dir),
        name_prefix="ppo_nomodel_step",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=verbose,
    )

    # 3) SB3 built-in EvalCallback to automatically save *best_model.zip*
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(models_dir),
        log_path=str(models_dir / "eval_logs"),
        eval_freq=int(cfg.get("eval_freq", 10_000)),
        deterministic=True,
        render=False,
        verbose=verbose,
    )

    callback = CallbackList([winrate_cb, winrate_save_cb, checkpoint_cb, eval_cb])

    model.learn(total_timesteps=int(cfg.get("total_timesteps", 1e6)), callback=callback)

    # ------------------------------------------------------------------
    # Quick evaluation (deterministic)
    # ------------------------------------------------------------------
    win_rate = _evaluate_win_rate(model, n_episodes=n_eval_episodes)

    # ------------------------------------------------------------------
    # Persist best/last checkpoint so downstream scripts/tests can load it
    # ------------------------------------------------------------------
    ckpt_path = models_dir / "baseline_no_model.zip"
    # Always save the final model.  For longer training runs users can
    # additionally rely on Stable-Baselines3 callback mechanisms to save
    # intermediate "best_model" checkpoints, but this guarantees that at
    # least the last trained weights are exported and can be re-used.
    model.save(ckpt_path)
    if verbose:
        print(f"[INFO] PPO checkpoint saved to {ckpt_path.relative_to(models_dir.parent)}")

    return model, win_rate


if __name__ == "__main__":  # pragma: no cover
    # Example CLI execution: python -m uav_intent_rl.examples.ppo_nomodel
    # Default quick-start: train for 3M steps if script run directly
    mdl, wr = run(total_timesteps=3_000_000, n_eval_episodes=10, gui=False, n_envs=8, verbose=1)
    print(f"[INFO] Win-rate after quick training: {wr * 100:.1f}%") 