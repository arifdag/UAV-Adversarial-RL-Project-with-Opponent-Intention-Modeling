from __future__ import annotations
"""Supervised pre-training of the auxiliary opponent head.

Loads an IntentPPO checkpoint, trains only the auxiliary head on an offline
buffer of (obs, red_bucket) pairs, then saves the updated checkpoint.

Example
-------
python -m uav_intent_rl.scripts.pretrain_aux \
       --model models/intent_best_lambda_0.30.zip \
       --buffer data/intent_buffer.npz \
       --updates 50000 \
       --lr 3e-4 \
       --save models/intent_aux_warmup.zip
"""

import argparse
from pathlib import Path

import numpy as np
import torch as th
from torch import optim
from torch.nn import functional as F

from uav_intent_rl.algo.intent_ppo import IntentPPO
from uav_intent_rl.examples.ppo_intent import _make_env


def pretrain(model_path: Path, buffer_path: Path, out_path: Path, updates: int, lr: float) -> None:  # noqa: D401
    # Load model
    env = _make_env(gui=False)
    model: IntentPPO = IntentPPO.load(str(model_path), env=env, device="cpu")

    # Load buffer
    data = np.load(buffer_path)
    obs_arr: np.ndarray = data["obs"].astype(np.float32)
    labels_arr: np.ndarray = data["labels"].astype(np.int64)

    device = th.device("cpu")
    obs_tensor = th.from_numpy(obs_arr).to(device)
    labels_tensor = th.from_numpy(labels_arr).to(device)

    dataset = th.utils.data.TensorDataset(obs_tensor, labels_tensor)
    loader = th.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    # Optimise only the aux head parameters
    optimizer = optim.Adam(model.policy.opp_head.parameters(), lr=lr)

    model.policy.train()
    model.policy.set_training_mode(True)

    for i in range(updates):
        for obs_batch, label_batch in loader:
            optimizer.zero_grad()
            logits = model.policy.opponent_logits(obs_batch)
            loss = F.cross_entropy(logits, label_batch)
            loss.backward()
            optimizer.step()
        if (i + 1) % 1000 == 0:
            with th.no_grad():
                preds = model.policy.opponent_logits(obs_tensor).argmax(dim=-1)
                acc = (preds == labels_tensor).float().mean().item()
            print(f"Update {i+1}/{updates} – aux accuracy {acc:.3f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    print(f"[INFO] Saved pre-trained model to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Pre-train aux head on offline buffer")
    p.add_argument("--model", type=Path, required=True, help="Input IntentPPO checkpoint")
    p.add_argument("--buffer", type=Path, required=True, help=".npz file from collect_random_buffer")
    p.add_argument("--updates", type=int, default=50000, help="Number of optimisation steps")
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate for aux head optimiser")
    p.add_argument("--save", type=Path, required=True, help="Output checkpoint path")
    args = p.parse_args()

    pretrain(args.model, args.buffer, args.save, updates=args.updates, lr=args.lr) 