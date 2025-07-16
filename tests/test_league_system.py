"""
Test script for League System

Creates dummy checkpoints and runs a demonstration tournament
to verify the league system works correctly.
"""

import os
import shutil
import tempfile
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from uav_intent_rl.utils.league_manager import LeagueManager


def create_dummy_checkpoint(checkpoint_dir: Path, agent_name: str):
    """Create a dummy checkpoint directory structure"""
    checkpoint_path = checkpoint_dir / agent_name
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    # Create dummy files that look like RLlib checkpoints
    (checkpoint_path / "algorithm_state.pkl").touch()
    (checkpoint_path / "rllib_checkpoint.json").write_text('{"type": "Algorithm"}')
    (checkpoint_path / "policies").mkdir(exist_ok=True)
    (checkpoint_path / "policies" / "shared_ppo").mkdir(exist_ok=True)
    
    return str(checkpoint_path)


def test_league_basic_functionality():
    """Test basic league manager functionality"""
    print("=== Testing League Basic Functionality ===")
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        league_dir = Path(temp_dir) / "test_league"
        
        # Initialize league manager
        league_manager = LeagueManager(
            league_dir=str(league_dir),
            checkpoint_interval=500000,
            eval_episodes=5  # Small number for testing
        )
        
        print(f"Created league in: {league_dir}")
        
        # Create some dummy checkpoints
        checkpoint_dir = Path(temp_dir) / "checkpoints"
        checkpoint_dir.mkdir()
        
        agents_data = [
            ("rookie_agent", 500000),
            ("veteran_agent", 1000000),
            ("expert_agent", 1500000),
        ]
        
        for agent_name, steps in agents_data:
            checkpoint_path = create_dummy_checkpoint(checkpoint_dir, agent_name)
            agent_id = league_manager.add_checkpoint(
                checkpoint_path=checkpoint_path,
                total_steps=steps,
                metadata={
                    'test_agent': True,
                    'performance_metric': steps / 100000  # Dummy metric
                }
            )
            print(f"Added {agent_id} with {steps} steps")
        
        # Test league state persistence
        print(f"\nLeague state before reload:")
        league_manager.print_status()
        
        # Create new league manager instance (simulates restart)
        league_manager2 = LeagueManager(league_dir=str(league_dir))
        print(f"\nLeague state after reload:")
        league_manager2.print_status()
        
        # Verify data persisted
        assert len(league_manager2.agents) == 3
        assert len(league_manager2.elo_ratings) == 3
        
        print("✅ Basic functionality test passed!")
        
        return league_manager2


def test_elo_rating_system():
    """Test Elo rating calculations"""
    print("\n=== Testing Elo Rating System ===")
    
    from uav_intent_rl.utils.league_manager import EloRating
    
    elo = EloRating(k_factor=32, initial_rating=1500)
    
    # Test basic rating update
    rating_a, rating_b = 1500, 1500
    new_a, new_b = elo.update_ratings(rating_a, rating_b, 1.0)  # A wins
    
    print(f"Before: A={rating_a}, B={rating_b}")
    print(f"After A wins: A={new_a:.1f}, B={new_b:.1f}")
    
    assert new_a > rating_a  # Winner should gain rating
    assert new_b < rating_b  # Loser should lose rating
    assert abs((new_a + new_b) - (rating_a + rating_b)) < 0.01  # Total rating conserved
    
    # Test expected score calculation
    expected = elo.expected_score(1600, 1400)  # Higher rated vs lower rated
    print(f"Expected score for 1600 vs 1400: {expected:.3f}")
    assert expected > 0.5  # Higher rated should be favored
    
    print("✅ Elo rating system test passed!")


def test_csv_generation():
    """Test CSV and heatmap generation"""
    print("\n=== Testing CSV and Heatmap Generation ===")
    
    # Use the league from basic functionality test
    with tempfile.TemporaryDirectory() as temp_dir:
        league_dir = Path(temp_dir) / "test_league"
        league_manager = LeagueManager(league_dir=str(league_dir), eval_episodes=3)
        
        # Add some agents
        checkpoint_dir = Path(temp_dir) / "checkpoints"
        checkpoint_dir.mkdir()
        
        agent_ids = []
        for i, (name, steps) in enumerate([("agent_a", 500000), ("agent_b", 1000000)]):
            checkpoint_path = create_dummy_checkpoint(checkpoint_dir, name)
            agent_id = league_manager.add_checkpoint(checkpoint_path, steps)
            agent_ids.append(agent_id)
        
        # Simulate some match results using the actual agent IDs
        league_manager.match_history = [
            {
                'agent_a': agent_ids[0],
                'agent_b': agent_ids[1], 
                'wins_a': 7,
                'wins_b': 3,
                'draws': 0,
                'win_rate_a': 0.7,
                'elo_score_a': 0.7
            },
            {
                'agent_a': agent_ids[1],
                'agent_b': agent_ids[0],
                'wins_a': 3,
                'wins_b': 7, 
                'draws': 0,
                'win_rate_a': 0.3,
                'elo_score_a': 0.3
            }
        ]
        
        # Update Elo ratings based on simulated matches
        league_manager._update_elo_ratings(league_manager.match_history)
        
        # Test CSV generation
        csv_path = league_manager.generate_elo_matrix_csv()
        assert Path(csv_path).exists()
        print(f"Generated CSV: {csv_path}")
        
        # Test heatmap generation (if matplotlib available)
        try:
            heatmap_path = league_manager.generate_heatmap()
            assert Path(heatmap_path).exists()
            print(f"Generated heatmap: {heatmap_path}")
        except ImportError:
            print("Matplotlib not available - skipping heatmap test")
        
        # Test leaderboard
        leaderboard = league_manager.get_leaderboard()
        assert len(leaderboard) == 2
        print("Generated leaderboard:")
        print(leaderboard)
        
        print("✅ CSV and heatmap generation test passed!")


def main():
    """Run all league system tests"""
    print("🧪 Testing UAV League System\n")
    
    try:
        # Test basic functionality
        league_manager = test_league_basic_functionality()
        
        # Test Elo rating system
        test_elo_rating_system()
        
        # Test CSV generation
        test_csv_generation()
        
        print("\n🎉 All league system tests passed!")
        print("\nThe league system is ready for use. To integrate with training:")
        print("1. Use uav_intent_rl/examples/train_with_league.py for training with auto-checkpointing")
        print("2. Use uav_intent_rl/examples/run_tournament.py for manual tournaments")
        print("3. Use scripts/weekly_tournament.py for automated weekly tournaments")
        print("4. Run scripts/setup_cron.sh to setup automatic scheduling")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 