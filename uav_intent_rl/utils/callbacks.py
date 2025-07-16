"""Custom callbacks for evaluation of auxiliary opponent head.

Provides ``AuxAccuracyCallback`` that periodically evaluates the
opponent-prediction accuracy on a hold-out environment and optionally saves
checkpoints whenever a new best accuracy is achieved.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv


class AuxAccuracyCallback(BaseCallback):
    """Evaluate *auxiliary-head* accuracy on a validation environment.

    Parameters
    ----------
    eval_env
        Separate environment that **must** add ``"red_bucket"`` to its ``info``
        dict – e.g. :class:`uav_intent_rl.utils.intent_wrappers.BlueVsFixedRedWrapper`.
    n_eval_episodes
        Number of episodes to average over each evaluation cycle.
    eval_freq
        Evaluate every *eval_freq* calls to :pyfunc:`BaseCallback._on_step`.
    deterministic
        Whether to use deterministic actions for the *blue* policy during
        evaluation.
    best_model_save_path
        Directory in which to save a checkpoint whenever the validation accuracy
        improves. If *None*, checkpoints are not saved.
    best_model_name
        Basename (without extension) for checkpoints; ``.zip`` is appended.
    verbose
        Passed through to :pyclass:`BaseCallback`.
    """

    def __init__(
        self,
        eval_env: VecEnv,
        *,
        n_eval_episodes: int = 10,
        eval_freq: int = 20_000,
        deterministic: bool = True,
        best_model_save_path: str | Path | None = None,
        best_model_name: str = "intent_best_acc",
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.n_eval_episodes = int(n_eval_episodes)
        self.eval_freq = int(eval_freq)
        self.deterministic = bool(deterministic)
        self.best_model_save_path = Path(best_model_save_path) if best_model_save_path else None
        self.best_model_name = str(best_model_name)
        self.best_accuracy: float = -np.inf
        if self.best_model_save_path is not None:
            self.best_model_save_path.mkdir(parents=True, exist_ok=True)

        # Internal counters
        self._episodes_run = 0

    # ------------------------------------------------------------------
    # Core helper – run validation episodes
    # ------------------------------------------------------------------

    def _evaluate_accuracy(self) -> float:
        model = self.model  # type: ignore[attr-defined]
        assert model is not None, "Callback not bound to model yet"

        total_correct = 0
        total_samples = 0
        episodes_completed = 0

        # Per-env episode tracking for vectorised envs
        dones = np.ones(self.eval_env.num_envs, dtype=bool)
        obs = None
        infos: List[dict]

        while episodes_completed < self.n_eval_episodes:
            if np.all(dones):
                obs_reset = self.eval_env.reset()
                # VecEnv.reset() returns only obs; some gym wrappers may return (obs, info)
                if isinstance(obs_reset, tuple):
                    obs = obs_reset[0]
                else:
                    obs = obs_reset
                dones = np.zeros(self.eval_env.num_envs, dtype=bool)

            # Predict blue action
            action, _ = model.predict(obs, deterministic=self.deterministic)

            # Predict red bucket from obs
            obs_tensor = model.policy.obs_to_tensor(obs)[0]
            logits = model.policy.opponent_logits(obs_tensor)
            preds = logits.argmax(dim=-1).cpu().numpy()

            # Step env
            obs, _rews, step_dones, infos = self.eval_env.step(action)

            # Collect labels from info
            labels = np.array([info.get("red_bucket", -1) for info in infos], dtype=np.int64)
            mask = labels >= 0  # some envs might not provide bucket (should not happen)

            total_correct += (preds[mask] == labels[mask]).sum()
            total_samples += mask.sum()

            # Track episode ends
            dones = np.logical_or(dones, step_dones)
            episodes_completed += step_dones.sum()

        # Avoid division by zero
        return float(total_correct) / max(total_samples, 1)

    # ------------------------------------------------------------------
    # Callback overrides
    # ------------------------------------------------------------------

    def _on_step(self) -> bool:  # noqa: D401 – SB3 hook
        # Use *true* timesteps rather than callback calls (which are /env.num_envs)
        if self.eval_freq <= 0 or (self.model.num_timesteps % self.eval_freq) != 0:  # type: ignore[attr-defined]
            return True

        acc = self._evaluate_accuracy()
        self.logger.record("eval/aux_accuracy", acc)
        if self.verbose >= 1:
            print(f"Eval aux-accuracy: {acc:.3f}")

        # Save if best so far
        if acc > self.best_accuracy and self.best_model_save_path is not None:
            self.best_accuracy = acc
            file_path = self.best_model_save_path / f"{self.best_model_name}.zip"
            self.model.save(file_path)  # type: ignore[arg-type]
            if self.verbose >= 1:
                print(f"New best aux-accuracy ⇒ saved model to {file_path}")

        return True 


# ---------------------------------------------------------------------------
# Simple win-rate evaluation callback
# ---------------------------------------------------------------------------


class WinRateCallback(BaseCallback):  # noqa: D101 – mirrors AuxAccuracy structure
    def __init__(
        self,
        eval_env: VecEnv,
        *,
        n_eval_episodes: int = 20,
        eval_freq: int = 20_000,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.n_eval_episodes = int(n_eval_episodes)
        self.eval_freq = int(eval_freq)

    # ------------------------------------------------------------------
    # Core helper – determine win-rate
    # ------------------------------------------------------------------

    def _compute_win_rate(self, model) -> float:  # noqa: D401
        """Run *n_eval_episodes* deterministic games and return blue win-rate.

        We spawn a **fresh** (non-vectorised) evaluation environment for each
        episode.  This avoids the *auto-reset* behaviour of SB3’s
        ``VecEnv`` classes, which otherwise makes it impossible to query the
        terminal state (``_red_down``) after the episode ends.
        """

        from uav_intent_rl.envs import DogfightAviary  # local import to avoid circular
        from uav_intent_rl.utils.intent_wrappers import BlueVsFixedRedWrapper

        wins = 0
        for ep in range(self.n_eval_episodes):
            base_env = DogfightAviary(gui=False)
            env = BlueVsFixedRedWrapper(base_env)

            obs, _info = env.reset(seed=ep)
            obs = obs.astype(np.float32)
            done = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _reward, terminated, truncated, _info = env.step(action)
                done = bool(terminated or truncated)

            # Determine winner *before* env.close() – red down => blue win
            blue_win = bool(env.env._red_down()) if hasattr(env.env, "_red_down") else False  # type: ignore[attr-defined]
            if blue_win:
                wins += 1

            env.close()

        return wins / self.n_eval_episodes if self.n_eval_episodes else 0.0

    # ------------------------------------------------------------------
    # Callback override
    # ------------------------------------------------------------------

    def _on_step(self) -> bool:  # noqa: D401 – SB3 hook
        if self.eval_freq <= 0 or (self.model.num_timesteps % self.eval_freq) != 0:
            return True

        win_rate = self._compute_win_rate(self.model)
        self.logger.record("eval/win_rate", win_rate)
        if self.verbose:
            print(f"Eval win-rate: {win_rate*100:.1f} %")
        return True 