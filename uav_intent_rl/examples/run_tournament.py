"""
Tournament Runner for UAV League

Runs round-robin tournaments between all agents in the league
and generates Elo matrix CSV and heatmap visualizations.
"""

import argparse
import sys
from pathlib import Path

from uav_intent_rl.utils.league_manager import LeagueManager


def main():
    parser = argparse.ArgumentParser(description="Run UAV League Tournament")
    parser.add_argument("--league-dir", type=str, default="league",
                       help="Directory containing league data")
    parser.add_argument("--eval-episodes", type=int, default=50,
                       help="Number of episodes per matchup")
    parser.add_argument("--generate-heatmap", action="store_true",
                       help="Generate heatmap visualization")
    parser.add_argument("--output-csv", type=str, 
                       help="Custom path for CSV output")
    parser.add_argument("--output-heatmap", type=str,
                       help="Custom path for heatmap output")
    
    args = parser.parse_args()
    
    # Initialize league manager
    league_manager = LeagueManager(
        league_dir=args.league_dir,
        eval_episodes=args.eval_episodes
    )
    
    # Check if we have enough agents
    if len(league_manager.agents) < 2:
        print(f"Error: Need at least 2 agents for tournament. Found {len(league_manager.agents)} agents.")
        league_manager.print_status()
        sys.exit(1)
    
    print(f"Starting tournament with {len(league_manager.agents)} agents...")
    league_manager.print_status()
    
    # Run round-robin tournament
    print("\n=== Running Round-Robin Tournament ===")
    results_df = league_manager.run_round_robin_tournament()
    
    if results_df.empty:
        print("Tournament failed - no results generated")
        sys.exit(1)
    
    print(f"Tournament completed! {len(results_df)} matches played.")
    
    # Generate CSV matrix
    print("\n=== Generating Elo Matrix CSV ===")
    csv_path = league_manager.generate_elo_matrix_csv()
    
    if args.output_csv and csv_path:
        # Copy to custom location
        import shutil
        shutil.copy2(csv_path, args.output_csv)
        print(f"CSV also saved to: {args.output_csv}")
    
    # Generate heatmap if requested
    if args.generate_heatmap:
        print("\n=== Generating Heatmap ===")
        heatmap_path = league_manager.generate_heatmap(args.output_heatmap)
        print(f"Heatmap saved to: {heatmap_path}")
    
    # Print final leaderboard
    print("\n=== Final Leaderboard ===")
    leaderboard = league_manager.get_leaderboard()
    if not leaderboard.empty:
        print(leaderboard.to_string(index=False))
    
    # Print match results summary
    print(f"\n=== Tournament Summary ===")
    print(f"Total matches: {len(results_df)}")
    print(f"Average win rate: {results_df['win_rate_a'].mean():.3f}")
    print(f"Win rate std dev: {results_df['win_rate_a'].std():.3f}")
    
    # Show most dominant matchups
    top_matches = results_df.nlargest(5, 'win_rate_a')[['agent_a', 'agent_b', 'win_rate_a']]
    print(f"\nTop 5 Dominant Matchups:")
    for _, match in top_matches.iterrows():
        print(f"  {match['agent_a']} vs {match['agent_b']}: {match['win_rate_a']:.1%} win rate")
    
    print(f"\nArtifacts generated:")
    if csv_path:
        print(f"  - CSV matrix: {csv_path}")
    if args.generate_heatmap and 'heatmap_path' in locals():
        print(f"  - Heatmap: {heatmap_path}")


if __name__ == "__main__":
    main() 