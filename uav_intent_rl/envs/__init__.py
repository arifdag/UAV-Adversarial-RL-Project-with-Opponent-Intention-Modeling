"""Custom environments for UAV adversarial RL project."""

from .DogfightAviary import DogfightAviary  # noqa: F401

# DogfightMultiAgentEnv relies on Ray RLlib; guard against missing dependency
try:
    from .DogfightMultiAgentEnv import DogfightMultiAgentEnv  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    # Gracefully degrade when RLlib is not available – users who need the
    # multi-agent wrapper must `pip install ray[rllib]`.
    DogfightMultiAgentEnv = None  # type: ignore[assignment]

__all__ = [
    "DogfightAviary",
    "DogfightMultiAgentEnv",
] 