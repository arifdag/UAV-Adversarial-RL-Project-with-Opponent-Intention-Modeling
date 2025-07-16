from __future__ import annotations

"""Utility to run a *long* evaluation of a saved blue-vs-scripted-red PPO policy.

Example
-------
python -m uav_intent_rl.examples.evaluate_best_model \
    --model  models/best_model.zip \
    --episodes  100 \
    --gui  False
"""

import argparse
import statistics
from pathlib import Path
from typing import List

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from uav_intent_rl.examples.ppo_nomodel import _make_single_env


def _run_episode(model: PPO, env: gym.Env, seed: int | None = None) -> tuple[float, int, bool]:
    """Roll out a *deterministic* episode and collect metrics.

    Parameters
    ----------
    model : stable_baselines3.PPO
        Loaded blue-drone policy.
    env : gym.Env
        Environment created by :func:`_make_single_env` (already wrapped).
    seed : int | None, optional
        Optional deterministic seed for :py:meth:`gym.Env.reset`.

    Returns
    -------
    tuple[float, int, bool]
        *(episode_reward, episode_length, blue_win)*.
    """
    obs, _ = env.reset(seed=seed)
    done = False
    ep_reward = 0.0
    steps = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        ep_reward += float(reward)
        steps += 1
        done = bool(terminated) or bool(truncated)

    # The underlying *DogfightAviary* exposes a helper that tells who crashed.
    blue_win: bool = False
    if hasattr(env.env, "_red_down"):
        blue_win = bool(env.env._red_down())  # type: ignore[attr-defined]

    return ep_reward, steps, blue_win


def evaluate(
    model_path: Path,
    episodes_per_batch: int = 50,
    n_batches: int = 10,
    gui: bool = False,
    device: str = "auto",
    return_results: bool = False,
) -> None:
    """Run multiple evaluation batches for robustness.

    Parameters
    ----------
    model_path : pathlib.Path
        SB3 checkpoint (.zip).
    episodes_per_batch : int, optional
        How many episodes to run *in each batch* (default 50).
    n_batches : int, optional
        Number of independent batches (default 10).  Seeds are offset so each
        episode uses a unique seed.
    gui : bool, optional
        Enable PyBullet GUI.
    device : str, optional
        Torch device string for SB3 `load`.
    return_results : bool, optional
        If True, return a dictionary with evaluation results instead of printing.
        Used by comparison scripts.

    Returns
    -------
    dict or None
        If return_results is True, returns a dictionary with evaluation metrics.
    """

    model_path = Path(model_path).expanduser().resolve()
    if model_path.suffix.lower() == ".zip":
        model_path = model_path.with_suffix("")

    model = PPO.load(str(model_path), device=device)
    total_episodes = episodes_per_batch * n_batches
    print(
        f"[INFO] Loaded model from '{model_path}'.  Evaluating {total_episodes} episodes "
        f"({n_batches} × {episodes_per_batch})…"
    )

    batch_win_rates: List[float] = []
    overall_rewards: List[float] = []
    overall_lengths: List[int] = []

    ep_counter = 0
    for batch in range(n_batches):
        wins = 0
        for ep in range(episodes_per_batch):
            seed = ep_counter  # unique deterministic seed
            ep_counter += 1
            env = _make_single_env(gui=gui)()
            ep_reward, ep_len, blue_win = _run_episode(model, env, seed=seed)
            env.close()

            overall_rewards.append(ep_reward)
            overall_lengths.append(ep_len)
            wins += int(blue_win)

        batch_wr = wins / episodes_per_batch * 100.0
        batch_win_rates.append(batch_wr)
        print(f"Batch {batch + 1}/{n_batches}: win-rate = {batch_wr:5.1f} %")

    # Aggregate statistics across all episodes
    avg_wr = statistics.mean(batch_win_rates)
    mean_rew = statistics.mean(overall_rewards)
    std_rew = statistics.stdev(overall_rewards) if total_episodes > 1 else 0.0
    mean_len = statistics.mean(overall_lengths)
    std_len = statistics.stdev(overall_lengths) if total_episodes > 1 else 0.0

    # Return results as dictionary if requested
    if return_results:
        return {
            "win_rate": avg_wr,
            "reward_mean": mean_rew,
            "reward_std": std_rew,
            "length_mean": mean_len,
            "length_std": std_len,
            "total_episodes": total_episodes,
        }

    print("\n=== Overall summary ===")
    print(f"Total episodes      : {total_episodes} ({n_batches} batches × {episodes_per_batch})")
    print(f"Average win-rate    : {avg_wr:5.1f} % (batch mean)")
    print(f"Reward mean ± std   : {mean_rew:7.2f} ± {std_rew:5.2f}")
    print(f"Ep length mean ± std: {mean_len:7.2f} ± {std_len:5.2f} steps")

    return None


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description="Evaluate a saved PPO policy against the scripted red opponent.")
    parser.add_argument("--model", type=Path, default=Path("models/best_model.zip"), help="Path to SB3 .zip checkpoint")
    parser.add_argument("--episodes_per_batch", type=int, default=50, help="Episodes per batch (default 50)")
    parser.add_argument("--batches", type=int, default=10, help="Number of batches (default 10)")
    parser.add_argument("--gui", action="store_true", help="Enable PyBullet GUI during evaluation")
    parser.add_argument("--device", type=str, default="auto", help="Torch device for inference (auto|cpu|cuda)")

    args = parser.parse_args()
    evaluate(
        args.model,
        episodes_per_batch=args.episodes_per_batch,
        n_batches=args.batches,
        gui=args.gui,
        device=args.device,
    ) 