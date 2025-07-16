"""
Simple demonstration of the League System

This script shows how to use the league system with existing checkpoints.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from uav_intent_rl.utils.league_manager import LeagueManager


def demo_league_with_existing_checkpoints():
    """Demonstrate league system using existing model checkpoints"""
    
    print("🎯 UAV League System Demo")
    print("=" * 40)
    
    # Initialize league manager
    league_manager = LeagueManager(
        league_dir="league_demo",
        checkpoint_interval=500000,
        eval_episodes=10  # Small number for demo
    )
    
    print(f"📁 League directory: {league_manager.league_dir}")
    
    # Check for existing checkpoints in models directory
    models_dir = Path("models")
    checkpoints = []
    
    if models_dir.exists():
        for checkpoint_dir in models_dir.iterdir():
            if checkpoint_dir.is_dir() and "checkpoint" in checkpoint_dir.name.lower():
                checkpoints.append(checkpoint_dir)
    
    # Also check ray_results for recent training checkpoints
    ray_results = Path().glob("**/ray_results/**/checkpoint_*")
    for checkpoint_path in ray_results:
        if checkpoint_path.is_dir():
            checkpoints.append(checkpoint_path)
    
    print(f"🔍 Found {len(checkpoints)} potential checkpoint directories")
    
    if not checkpoints:
        print("\n⚠️  No existing checkpoints found.")
        print("   To create checkpoints, run training with:")
        print("   python -m uav_intent_rl.examples.train_with_league --stop-timesteps 1000000")
        return
    
    # Add first few checkpoints to league
    added_agents = 0
    for i, checkpoint_path in enumerate(checkpoints[:3]):  # Limit to 3 for demo
        try:
            # Estimate training steps from directory name or use incremental values
            estimated_steps = (i + 1) * 500000
            
            agent_id = league_manager.add_checkpoint(
                checkpoint_path=str(checkpoint_path),
                total_steps=estimated_steps,
                metadata={
                    'source_path': str(checkpoint_path),
                    'demo_agent': True
                }
            )
            
            print(f"✅ Added {agent_id}")
            added_agents += 1
            
        except Exception as e:
            print(f"❌ Failed to add {checkpoint_path}: {e}")
    
    if added_agents < 2:
        print(f"\n⚠️  Need at least 2 agents for tournament, only have {added_agents}")
        print("   Creating some dummy agents for demonstration...")
        
        # Create minimal dummy checkpoints for demo
        import tempfile
        import os
        
        dummy_dir = Path("league_demo") / "dummy_checkpoints"
        dummy_dir.mkdir(parents=True, exist_ok=True)
        
        for i, name in enumerate(["rookie", "veteran"]):
            checkpoint_dir = dummy_dir / f"{name}_checkpoint"
            checkpoint_dir.mkdir(exist_ok=True)
            
            # Create minimal checkpoint structure
            (checkpoint_dir / "algorithm_state.pkl").touch()
            
            agent_id = league_manager.add_checkpoint(
                checkpoint_path=str(checkpoint_dir),
                total_steps=(i + 1) * 500000,
                metadata={'dummy_agent': True, 'name': name}
            )
            print(f"✅ Added dummy agent: {agent_id}")
    
    # Show league status
    print(f"\n📊 League Status:")
    league_manager.print_status()
    
    # Generate CSV matrix
    print(f"\n📋 Generating Elo Matrix CSV...")
    csv_path = league_manager.generate_elo_matrix_csv()
    
    if csv_path:
        print(f"✅ CSV generated: {csv_path}")
        
        # Copy to artifacts directory for DoD compliance
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(exist_ok=True)
        
        import shutil
        dod_csv = artifacts_dir / "elo_matrix.csv"
        shutil.copy2(csv_path, dod_csv)
        print(f"✅ DoD CSV created: {dod_csv}")
    
    # Try to generate heatmap
    try:
        print(f"\n🎨 Generating Heatmap...")
        heatmap_path = league_manager.generate_heatmap()
        print(f"✅ Heatmap generated: {heatmap_path}")
    except ImportError as e:
        print(f"⚠️  Heatmap requires matplotlib: pip install matplotlib seaborn")
    except Exception as e:
        print(f"❌ Heatmap generation failed: {e}")
    
    print(f"\n🎉 Demo completed!")
    print(f"\nNext steps:")
    print(f"1. Train agents: python -m uav_intent_rl.examples.train_with_league")
    print(f"2. Run tournament: python -m uav_intent_rl.examples.run_tournament")
    print(f"3. Setup automation: bash scripts/setup_cron.sh")


if __name__ == "__main__":
    demo_league_with_existing_checkpoints() 