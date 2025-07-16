from __future__ import annotations

"""Utility helpers for common Gymnasium environment wrappers.

This module provides :pyfunc:`make_monitored_env` which sets up a standard
suite of wrappers (``RecordVideo``, ``RecordEpisodeStatistics`` and
Stable-Baselines3's :pyclass:`Monitor`) so that each training run automatically
emits both an ``mp4`` video and a ``progress.csv`` file under a timestamped
sub-folder inside *runs/*.  The helper implements the acceptance criteria of
*Epic E1-4* (Episode video & CSV logs are saved for post-mortem).

Example
-------
>>> from uav_intent_rl.utils.env_wrappers import make_monitored_env
>>> env = make_monitored_env(DogfightAviary, gui=False)
>>> obs, _ = env.reset(seed=0)
>>> ...  # interact with the environment

The returned ``env`` can be used directly with Stable-Baselines3 or any other
Gymnasium-compatible RL training loop.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Type, Union

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from stable_baselines3.common.monitor import Monitor

__all__ = ["make_monitored_env"]


EnvFactory = Union[str, Callable[..., gym.Env], Type[gym.Env]]


def _instantiate(env_factory: EnvFactory, **env_kwargs: Any) -> gym.Env:  # noqa: D401
    """Instantiate an environment given an *id*, constructor or callable."""

    if isinstance(env_factory, str):
        return gym.make(env_factory, **env_kwargs)

    if isinstance(env_factory, type):
        # Class – call to construct instance
        return env_factory(**env_kwargs)  # type: ignore[arg-type]

    # Callable returning an env
    return env_factory(**env_kwargs)  # type: ignore[call-arg]


def make_monitored_env(
    env_factory: EnvFactory,
    runs_root: Union[str, Path] = "runs",
    video_enabled: bool = True,
    **env_kwargs: Any,
) -> gym.Env:
    """Create an environment wrapped for video + CSV logging.

    Parameters
    ----------
    env_factory : Union[str, Callable[..., gym.Env], Type[gym.Env]]
        An environment *id*, constructor, or callable producing a Gymnasium env.
    runs_root : str | Path, optional
        Root directory under which timestamped run folders are created.  The
        default is ``"runs"``.
    video_enabled : bool, optional
        Whether to capture MP4 videos via :pyclass:`RecordVideo`.  Set ``False``
        to disable video recording (CSV logging will remain active).
    **env_kwargs : Any
        Keyword-arguments forwarded to the environment constructor.  If
        ``gui=False`` (or absent), ``render_mode="rgb_array"`` is automatically
        injected so that video recording works in headless mode.

    Returns
    -------
    gymnasium.Env
        The wrapped environment ready for training.
    """

    # ------------------------------------------------------------------
    # Create timestamped run directory – e.g. runs/2025-07-01_12-34-56/
    # ------------------------------------------------------------------
    run_dir = Path(runs_root) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Instantiate the base environment
    # ------------------------------------------------------------------
    env = _instantiate(env_factory, **env_kwargs)

    # ------------------------------------------------------------------
    # Wrap with statistics and optional video recording
    # ------------------------------------------------------------------
    env = RecordEpisodeStatistics(env)

    if video_enabled:
        # Try to ensure the *base* environment advertises a valid render_mode.
        candidate = env
        # Unwrap once if env is a Gymnasium wrapper (RecordEpisodeStatistics)
        if hasattr(candidate, "env"):
            candidate = candidate.env  # type: ignore[assignment]

        if getattr(candidate, "render_mode", None) not in {"rgb_array", "rgb_array_list"}:
            # Direct attribute set on the base env (not on wrappers) avoids
            # the read-only @property defined by some wrappers.
            setattr(candidate, "render_mode", "rgb_array")

        env = RecordVideo(env, video_folder=str(run_dir), name_prefix="episode")

    # Stable-Baselines3 monitor wrapper – writes <name>.monitor.csv by default
    env = Monitor(env, filename=str(run_dir / "progress"))

    return env 