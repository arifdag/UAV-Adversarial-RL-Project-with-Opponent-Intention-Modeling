"""Quick PPO training harness for the multi-agent dog-fight environment.

This script is deliberately **minimal** – it shows how to configure Ray RLlib
so that both *blue* and *red* drones share **one** set of network weights while
still being treated as distinct agents.  Such parameter sharing is often more
sample-efficient in symmetric games and can be enabled by setting the
(non-standard) ``share_observations`` flag in the config dict.

The output directory (`~/ray_results/...`) will contain TensorBoard logs and a
checkpoint that can be evaluated with::

    rllib rollout <checkpoint_dir> --runtime-env 'pip: [gymnasium, pybullet, ray[rllib]]' \
        --steps 1000

which should print samples like ``Sample Batch (blue, red)`` verifying that
RLlib sees **two** policy IDs.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig

from uav_intent_rl.envs import DogfightMultiAgentEnv

# ---------------------------------------------------------------------
# Command-line helpers
# ---------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Train shared-policy PPO on the dog-fight env")
parser.add_argument("--stop-timesteps", type=int, default=250_000, help="Total env timesteps to train")
args = parser.parse_args()

# ---------------------------------------------------------------------
# RLlib configuration
# ---------------------------------------------------------------------

base_config = (
    PPOConfig()
    .environment(env=DogfightMultiAgentEnv)
    .framework("torch")
    .env_runners(num_env_runners=0)  # single-process for quick tests
)

# Convert to dict so we can inject custom keys (RLlib ignores unknown ones)
config_dict = base_config.to_dict()
config_dict["share_observations"] = True  # centralised value function for both drones

# Single shared policy mapped to all agents
config_dict["multiagent"] = {
    # RLlib will automatically infer the spaces from the first
    # environment instance, so we can leave them as *None* here.
    "policies": {
        "shared_ppo": (None, None, None, {})
    },
    "policy_mapping_fn": lambda agent_id, *_: "shared_ppo",
}

# ---------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------

ray.init(ignore_reinit_error=True)

results = tune.Tuner(
    "PPO",
    run_config=air.RunConfig(stop={"num_env_steps_sampled_lifetime": args.stop_timesteps}),
    param_space=config_dict,
).fit()

# ---------------------------------------------------------------------
# Save best checkpoint path to stdout for convenience
# ---------------------------------------------------------------------

best_ckpt: str | None = results.get_best_result().checkpoint.path if results.get_best_result() else None
if best_ckpt:
    print("Best checkpoint saved to", best_ckpt)
else:
    print("Training finished – no checkpoint produced.") 