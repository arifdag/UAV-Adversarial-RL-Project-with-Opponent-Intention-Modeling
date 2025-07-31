"""Environment modules for UAV adversarial RL."""

from .DogfightAviary import DogfightAviary
from .DogfightMultiAgentEnv import DogfightMultiAgentEnv, default_policy_mapping_fn
from .Dogfight3v3Aviary import Dogfight3v3Aviary
from .Dogfight3v3MultiAgentEnv import Dogfight3v3MultiAgentEnv, default_3v3_policy_mapping_fn

__all__ = [
    "DogfightAviary",
    "DogfightMultiAgentEnv", 
    "default_policy_mapping_fn",
    "Dogfight3v3Aviary",
    "Dogfight3v3MultiAgentEnv",
    "Dogfight3v3MultiAgentVecEnv",
    "make_mappo_vec_env",
    "default_3v3_policy_mapping_fn",
] 