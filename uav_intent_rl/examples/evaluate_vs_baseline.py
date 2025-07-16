from __future__ import annotations

"""Evaluate Intent-PPO model (blue) against baseline PPO model (red).

Both agents are SB3 checkpoints that output continuous velocity actions for a
single drone.  We manually control each drone in `DogfightAviary` by feeding the
correct observation ordering expected by each policy:

* **Blue** (index 0) expects [blue_state, red_state] → flattened.
* **Red**  (index 1) was trained as *blue*; we therefore *swap* the two state
  vectors so that the red-controlled drone appears first from its own
  perspective.

Usage
-----
python -m uav_intent_rl.examples.evaluate_vs_baseline \
    --blue models/intent_best_lambda_0.30.zip \
    --red  models/best_model_e3.zip \
    --episodes 200
"""

import argparse
import statistics
from pathlib import Path
from typing import Tuple

import numpy as np
from stable_baselines3 import PPO

from uav_intent_rl.envs import DogfightAviary

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _swap_obs(obs: np.ndarray) -> np.ndarray:
    """Swap blue/red slices so the *red* policy sees itself first."""
    assert obs.shape[0] == 2, "Expect obs array with shape (2, N)"
    return np.vstack([obs[1], obs[0]])


def _run_episode(
    blue_model: PPO,
    red_model: PPO,
    *,
    gui: bool = False,
    seed: int | None = None,
) -> Tuple[str, int]:
    """Run one deterministic episode and return (result, steps).
    result: 'blue', 'red', or 'draw'.
    """
    env = DogfightAviary(gui=gui)
    obs, _info = env.reset(seed=seed)
    done = False
    steps = 0

    obs_blue = obs.flatten().astype(np.float32)
    obs_red = _swap_obs(obs).flatten().astype(np.float32)

    while not done:
        act_blue, _ = blue_model.predict(obs_blue, deterministic=True)
        act_red, _ = red_model.predict(obs_red, deterministic=True)

        actions = np.vstack([act_blue, act_red]).astype(np.float32)
        obs, _reward, terminated, truncated, _info = env.step(actions)
        done = terminated or truncated
        steps += 1

        obs_blue = obs.flatten().astype(np.float32)
        obs_red = _swap_obs(obs).flatten().astype(np.float32)

    # Determine winner
    blue_down = bool(env._blue_down()) if hasattr(env, '_blue_down') else False
    red_down = bool(env._red_down()) if hasattr(env, '_red_down') else False
    if red_down and not blue_down:
        result = 'blue'
    elif blue_down and not red_down:
        result = 'red'
    else:
        result = 'draw'
    env.close()
    return result, steps


# -----------------------------------------------------------------------------
# Main evaluation routine
# -----------------------------------------------------------------------------

def evaluate(
    blue_ckpt: Path,
    red_ckpt: Path,
    *,
    episodes: int = 200,
    gui: bool = False,
    device: str = "auto",
) -> None:
    blue_ckpt = blue_ckpt.expanduser().resolve()
    red_ckpt = red_ckpt.expanduser().resolve()

    blue_model = PPO.load(str(blue_ckpt), device=device)
    red_model = PPO.load(str(red_ckpt), device=device)

    print(
        f"[INFO] Loaded blue model '{blue_ckpt.name}', red model '{red_ckpt.name}'.\n"
        f"Evaluating {episodes} episodes…"
    )

    results = {'blue': 0, 'red': 0, 'draw': 0}
    lengths = []
    for ep in range(episodes):
        result, steps = _run_episode(
            blue_model,
            red_model,
            gui=gui,
            seed=ep,  # deterministic but unique across episodes
        )
        results[result] += 1
        lengths.append(steps)

    win_rate_blue = results['blue'] / episodes * 100.0
    win_rate_red = results['red'] / episodes * 100.0
    draw_rate = results['draw'] / episodes * 100.0
    mean_len = statistics.mean(lengths)
    std_len = statistics.stdev(lengths) if episodes > 1 else 0.0

    print("\n=== Results ===")
    print(f"Episodes          : {episodes}")
    print(f"Blue win-rate     : {win_rate_blue:5.1f} %")
    print(f"Red win-rate      : {win_rate_red:5.1f} %")
    print(f"Draw rate         : {draw_rate:5.1f} %")
    print(f"Episode length    : {mean_len:.2f} ± {std_len:.2f} steps (mean ± std)")


# -----------------------------------------------------------------------------
# CLI wrapper
# -----------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description="Evaluate Intent model versus baseline.")
    parser.add_argument("--blue", type=Path, required=True, help="Path to IntentPPO checkpoint (blue)")
    parser.add_argument("--red", type=Path, required=True, help="Path to baseline checkpoint (red)")
    parser.add_argument("--episodes", type=int, default=200, help="Number of evaluation games (default 200)")
    parser.add_argument("--gui", action="store_true", help="Enable PyBullet GUI")
    parser.add_argument("--device", type=str, default="auto", help="Torch device for inference")

    args = parser.parse_args()
    evaluate(args.blue, args.red, episodes=args.episodes, gui=args.gui, device=args.device) 