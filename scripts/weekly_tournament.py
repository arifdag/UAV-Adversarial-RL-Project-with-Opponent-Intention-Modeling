#!/usr/bin/env python3
"""
Weekly Tournament Script for UAV League

This script is designed to be run by cron job weekly.
It runs tournaments, generates artifacts, and updates the README.
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from uav_intent_rl.utils.league_manager import LeagueManager


def setup_logging():
    """Setup logging for cron execution"""
    log_dir = project_root / "league" / "logs"
    log_dir.mkdir(exist_ok=True, parents=True)
    
    log_file = log_dir / f"tournament_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def update_readme_with_heatmap(heatmap_path: str, readme_path: str = "README.md"):
    """Update README.md with the latest heatmap"""
    readme_file = project_root / readme_path
    
    if not readme_file.exists():
        logger.warning(f"README.md not found at {readme_file}")
        return
    
    # Read current README
    with open(readme_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Prepare heatmap section
    heatmap_relative_path = Path(heatmap_path).relative_to(project_root)
    heatmap_section = f"""
## League Heatmap

Latest tournament results (updated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}):

![League Heatmap]({heatmap_relative_path})

*Heatmap shows win rates in head-to-head matchups (row vs column)*
"""
    
    # Look for existing heatmap section
    start_marker = "## League Heatmap"
    end_marker = "*Heatmap shows win rates"
    
    start_idx = content.find(start_marker)
    if start_idx != -1:
        # Find end of section
        end_idx = content.find(end_marker, start_idx)
        if end_idx != -1:
            # Find end of line
            end_idx = content.find('\n', end_idx) + 1
            # Replace existing section
            content = content[:start_idx] + heatmap_section.strip() + '\n\n' + content[end_idx:]
        else:
            # Replace from start marker to next ## or end of file
            next_section = content.find('\n## ', start_idx + 1)
            if next_section != -1:
                content = content[:start_idx] + heatmap_section.strip() + '\n\n' + content[next_section:]
            else:
                content = content[:start_idx] + heatmap_section.strip()
    else:
        # Add heatmap section at the end
        content += '\n\n' + heatmap_section.strip()
    
    # Write updated README
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger.info(f"Updated README.md with latest heatmap")


def main():
    global logger
    logger = setup_logging()
    
    logger.info("=== Starting Weekly UAV League Tournament ===")
    
    try:
        # Initialize league manager
        league_dir = project_root / "league"
        league_manager = LeagueManager(
            league_dir=str(league_dir),
            eval_episodes=50  # Balanced between accuracy and execution time
        )
        
        # Check if we have enough agents
        if len(league_manager.agents) < 2:
            logger.warning(f"Insufficient agents for tournament: {len(league_manager.agents)} found")
            league_manager.print_status()
            return
        
        logger.info(f"Starting tournament with {len(league_manager.agents)} agents")
        
        # Run tournament
        results_df = league_manager.run_round_robin_tournament()
        
        if results_df.empty:
            logger.error("Tournament failed - no results generated")
            return
        
        logger.info(f"Tournament completed! {len(results_df)} matches played")
        
        # Generate artifacts
        logger.info("Generating tournament artifacts...")
        
        # Generate CSV (required by DoD)
        csv_path = league_manager.generate_elo_matrix_csv()
        if csv_path:
            # Also copy to artifacts directory for DoD compliance
            artifacts_dir = project_root / "artifacts"
            artifacts_dir.mkdir(exist_ok=True)
            
            import shutil
            dod_csv_path = artifacts_dir / "elo_matrix.csv"
            shutil.copy2(csv_path, dod_csv_path)
            logger.info(f"CSV saved to: {csv_path} and {dod_csv_path}")
        
        # Generate heatmap (required by DoD)
        heatmap_path = league_manager.generate_heatmap()
        if heatmap_path:
            logger.info(f"Heatmap saved to: {heatmap_path}")
            
            # Update README with heatmap (required by DoD)
            update_readme_with_heatmap(heatmap_path)
        
        # Print final statistics
        leaderboard = league_manager.get_leaderboard()
        if not leaderboard.empty:
            logger.info(f"Tournament winner: {leaderboard.iloc[0]['agent_id']} "
                       f"(Elo: {leaderboard.iloc[0]['elo_rating']:.0f})")
        
        logger.info("=== Weekly tournament completed successfully ===")
        
    except Exception as e:
        logger.error(f"Tournament failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main() 