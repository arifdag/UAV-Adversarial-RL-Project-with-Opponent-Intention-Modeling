from __future__ import annotations

"""Compare two models by evaluating both against the scripted red opponent.

This script runs two separate evaluations:
1. Intent model (blue) vs scripted red
2. Baseline model (blue) vs scripted red

It then computes win-rate difference with confidence intervals to determine if
the intent model significantly outperforms the baseline.

Usage
-----
python -m uav_intent_rl.examples.compare_models \
    --intent models/intent_best_lambda_0.30.zip \
    --baseline models/best_model_e3.zip \
    --episodes 200
"""

import argparse
import statistics
import math
from pathlib import Path

from uav_intent_rl.examples.evaluate_best_model import evaluate as run_evaluation


def compare_models(
    intent_ckpt: Path,
    baseline_ckpt: Path,
    *,
    episodes: int = 200,
    gui: bool = False,
    device: str = "auto",
) -> None:
    """Run both models against scripted red and compare win-rates."""
    print(f"[INFO] Comparing models over {episodes} episodes each...")
    print(f"Intent model  : {intent_ckpt.name}")
    print(f"Baseline model: {baseline_ckpt.name}")

    # Run intent model evaluation
    print("\n=== Evaluating Intent Model ===")
    intent_results = run_evaluation(
        intent_ckpt,
        episodes_per_batch=episodes,
        n_batches=1,
        gui=gui,
        device=device,
        return_results=True,
    )

    # Run baseline model evaluation
    print("\n=== Evaluating Baseline Model ===")
    baseline_results = run_evaluation(
        baseline_ckpt,
        episodes_per_batch=episodes,
        n_batches=1,
        gui=gui,
        device=device,
        return_results=True,
    )

    # Extract win rates
    intent_win_rate = intent_results["win_rate"]
    baseline_win_rate = baseline_results["win_rate"]
    
    # Calculate difference and confidence interval
    win_rate_diff = intent_win_rate - baseline_win_rate
    
    # Standard error for difference of proportions
    p1 = intent_win_rate / 100.0  # convert to proportion
    p2 = baseline_win_rate / 100.0
    se_diff = math.sqrt((p1 * (1 - p1) / episodes) + (p2 * (1 - p2) / episodes))
    
    # 95% confidence interval
    margin = 1.96 * se_diff * 100.0  # convert back to percentage
    
    print("\n=== Comparison Results ===")
    print(f"Intent model win-rate   : {intent_win_rate:5.1f} %")
    print(f"Baseline model win-rate : {baseline_win_rate:5.1f} %")
    print(f"Win-rate difference     : {win_rate_diff:5.1f} % ± {margin:.1f} % (95% CI)")
    
    # Check Epic E6-1 acceptance criteria
    print("\n=== Epic E6-1 Acceptance Check ===")
    if intent_win_rate >= 70.0:
        print("✓ Intent model achieves ≥ 70% win-rate")
    else:
        print("✗ Intent model does NOT achieve ≥ 70% win-rate")
        
    if win_rate_diff > 0 and win_rate_diff - margin > 0:
        print("✓ Intent model significantly outperforms baseline (p < 0.05)")
    else:
        print("✗ Intent model does NOT significantly outperform baseline")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare intent model vs baseline against scripted red.")
    parser.add_argument("--intent", type=Path, required=True, help="Path to intent model checkpoint")
    parser.add_argument("--baseline", type=Path, required=True, help="Path to baseline checkpoint")
    parser.add_argument("--episodes", type=int, default=200, help="Episodes per model (default 200)")
    parser.add_argument("--gui", action="store_true", help="Enable PyBullet GUI")
    parser.add_argument("--device", type=str, default="auto", help="Torch device for inference")

    args = parser.parse_args()
    compare_models(
        args.intent,
        args.baseline,
        episodes=args.episodes,
        gui=args.gui,
        device=args.device,
    ) 