#!/usr/bin/env python3
"""Demonstration of performance-based curriculum learning."""

from uav_intent_rl.utils.curriculum_wrappers import DifficultyScheduler


def demonstrate_performance_curriculum():
    """Demonstrate performance-based curriculum behavior."""
    print("Performance-Based Curriculum Learning Demonstration")
    print("=" * 60)
    
    # Create scheduler for baseline difficulty
    scheduler = DifficultyScheduler(
        total_timesteps=10000,
        difficulty_schedule="linear",
        min_difficulty=0.0,
        max_difficulty=1.0,
    )
    
    # Performance-based curriculum parameters
    win_rate_threshold = 0.8
    difficulty_step = 0.25
    current_difficulty = 0.0
    
    # Simulate training with varying win rates
    scenarios = [
        (1000, 0.3, "Poor performance - should stay at baseline"),
        (2000, 0.9, "Good performance - should increase difficulty"),
        (3000, 0.4, "Poor performance again - should not increase"),
        (4000, 0.85, "Good performance - should increase difficulty"),
        (5000, 0.95, "Excellent performance - should increase difficulty"),
        (6000, 0.2, "Poor performance - should not increase"),
        (7000, 0.88, "Good performance - should increase difficulty"),
        (8000, 0.92, "Excellent performance - should increase difficulty"),
    ]
    
    print(f"{'Timestep':<10} | {'Win Rate':<8} | {'Baseline':<8} | {'Current':<8} | {'Action'}")
    print("-" * 70)
    
    for timestep, win_rate, description in scenarios:
        # Get baseline difficulty from scheduler
        baseline = scheduler.get_difficulty(timestep)
        
        # Performance-based difficulty adjustment
        if win_rate > win_rate_threshold and current_difficulty < 1.0:
            # Increase difficulty when performance threshold is met
            next_diff = min(1.0, current_difficulty + difficulty_step)
            if next_diff > current_difficulty:
                current_difficulty = next_diff
                action = "INCREASE"
            else:
                action = "MAX_REACHED"
        else:
            # Use baseline difficulty if performance not ready
            if current_difficulty < baseline:
                current_difficulty = baseline
                action = "BASELINE"
            else:
                action = "HOLD"
        
        print(f"{timestep:<10} | {win_rate:<8.3f} | {baseline:<8.3f} | {current_difficulty:<8.3f} | {action}")
    
    print("\nKey Benefits of Performance-Based Curriculum:")
    print("1. Agent only sees harder opponents when ready")
    print("2. Prevents overwhelming the agent with difficult tasks")
    print("3. Ensures mastery of each difficulty level")
    print("4. Adapts to individual agent learning speed")
    print("5. More efficient training progression")
    
    print(f"\nConfiguration:")
    print(f"- Win rate threshold: {win_rate_threshold}")
    print(f"- Difficulty step: {difficulty_step}")
    print(f"- Final difficulty: {current_difficulty:.3f}")


if __name__ == "__main__":
    demonstrate_performance_curriculum() 