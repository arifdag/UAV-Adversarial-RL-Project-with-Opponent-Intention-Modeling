from __future__ import annotations
"""Collect a buffer of (observation, red_bucket) pairs with random Blue actions.

Usage
-----
python -m uav_intent_rl.scripts.collect_random_buffer --out data/intent_buffer.npz --steps 100000
"""

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np

from uav_intent_rl.envs import DogfightAviary
from uav_intent_rl.utils.intent_wrappers import BlueVsFixedRedWrapper


def collect_buffer(out_path: Path, steps: int, *, gui: bool = False) -> None:  # noqa: D401
    env: gym.Env = BlueVsFixedRedWrapper(DogfightAviary(gui=gui))
    obs_list: list[np.ndarray] = []
    label_list: list[int] = []

    obs, _ = env.reset()
    for _ in range(steps):
        action = env.action_space.sample()
        obs_next, _rew, term, trunc, info = env.step(action)
        obs_list.append(obs.astype(np.float32))
        label_list.append(int(info.get("red_bucket", -1)))
        done = term or trunc
        obs = obs_next if not done else env.reset()[0]

    env.close()

    obs_arr = np.stack(obs_list)
    labels_arr = np.asarray(label_list, dtype=np.int64)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, obs=obs_arr, labels=labels_arr)
    print(f"[INFO] Saved buffer: {obs_arr.shape[0]} samples → {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Collect random buffer for aux head pre-training")
    p.add_argument("--out", type=Path, required=True, help="Output .npz file")
    p.add_argument("--steps", type=int, default=100000, help="Number of env steps to record")
    p.add_argument("--gui", action="store_true", help="Enable PyBullet GUI")
    args = p.parse_args()

    collect_buffer(args.out, args.steps, gui=args.gui) 