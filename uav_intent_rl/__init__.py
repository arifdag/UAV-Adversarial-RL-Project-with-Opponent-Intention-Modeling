"""Top-level package for custom UAV adversarial RL utilities."""

from importlib import import_module

# Re-export DogfightAviary and helper utilities for convenience
try:
    DogfightAviary = import_module("uav_intent_rl.envs.DogfightAviary").DogfightAviary  # type: ignore
except ModuleNotFoundError:
    DogfightAviary = None  # Will be available once the sub-module is imported elsewhere

from uav_intent_rl.utils.env_wrappers import make_monitored_env  # noqa: E402  # Lazy enough for most cases

__all__ = [
    "DogfightAviary",
    "make_monitored_env",
]