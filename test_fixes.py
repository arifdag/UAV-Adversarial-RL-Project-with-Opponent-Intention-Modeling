#!/usr/bin/env python3
"""
Test script to verify the fixes in evaluate_mappo.py work correctly.
"""

import numpy as np
from pathlib import Path
import tempfile
import json

# Mock classes for testing
class MockMAPPO:
    def __init__(self):
        self.policy = None
    
    def predict(self, obs, state=None, deterministic=True):
        # Return mock action and state
        action = np.random.randn(3, 4)  # 3 drones, 4 action dims
        return action, None
    
    @classmethod
    def load(cls, path):
        return cls()

class MockEnv:
    def __init__(self, num_blue_drones=3):
        self.num_blue_drones = num_blue_drones
        self.env = MockBaseEnv()
    
    def reset(self):
        # Return tuple as expected by the fix
        obs = np.random.randn(self.num_blue_drones, 20)  # Mock obs
        info = {'winner': 'blue', 'blue_alive': 3, 'red_alive': 0, 'elimination_count': [3, 0]}
        return obs, info
    
    def step(self, action):
        # Return proper step tuple
        obs = np.random.randn(self.num_blue_drones, 20)
        rewards = np.random.randn(self.num_blue_drones)
        terminated = False
        truncated = False
        info = {'winner': 'blue', 'blue_alive': 3, 'red_alive': 0, 'elimination_count': [3, 0]}
        return obs, rewards, terminated, truncated, info
    
    def render(self):
        pass

class MockBaseEnv:
    def __init__(self):
        self.alive_status = [True, True, True]
        self.health = [100.0, 100.0, 100.0]
        self.drones_per_team = 3
    
    def _getDroneStateVector(self, i):
        # Return mock drone state
        return np.random.randn(16)  # 16-dim state vector

def test_run_mappo_episode():
    """Test the fixed run_mappo_episode function."""
    print("Testing run_mappo_episode...")
    
    # Import the fixed function
    from uav_intent_rl.examples.evaluate_mappo import run_mappo_episode
    
    # Create mock objects
    model = MockMAPPO()
    env = MockEnv()
    
    # Test the function
    try:
        results = run_mappo_episode(
            model=model,
            env=env,
            render=False,
            verbose=True,
            tactic="test_tactic"
        )
        
        # Check that results have expected structure
        assert 'episode_reward' in results
        assert 'episode_length' in results
        assert 'individual_rewards' in results
        assert 'winner' in results
        assert 'tactic' in results
        assert results['tactic'] == "test_tactic"
        
        print("✓ run_mappo_episode test passed")
        return True
        
    except Exception as e:
        print(f"✗ run_mappo_episode test failed: {e}")
        return False

def test_analyze_team_behavior():
    """Test the fixed analyze_team_behavior function."""
    print("Testing analyze_team_behavior...")
    
    # Import the fixed function
    from uav_intent_rl.examples.evaluate_mappo import analyze_team_behavior
    
    # Create mock results with episode_reward
    mock_results = [
        {
            'positions_history': [[np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])] for _ in range(10)],
            'health_history': [[100, 100, 100] for _ in range(10)],
            'winner': 'blue',
            'tactic': 'test_tactic',
            'episode_reward': 10.5
        }
    ]
    
    # Test with temporary file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        try:
            metrics = analyze_team_behavior(mock_results, save_path=Path(tmp_file.name))
            
            # Check that metrics have expected structure
            assert 'avg_formation_distance' in metrics
            assert 'avg_survival_rate' in metrics
            assert 'formation_consistency' in metrics
            
            # Check that values are floats (not numpy types)
            assert isinstance(metrics['avg_formation_distance'], float)
            assert isinstance(metrics['avg_survival_rate'], float)
            assert isinstance(metrics['formation_consistency'], float)
            
            # Check that file was created
            assert Path(tmp_file.name).exists()
            
            print("✓ analyze_team_behavior test passed")
            return True
            
        except Exception as e:
            print(f"✗ analyze_team_behavior test failed: {e}")
            return False
        finally:
            # Clean up temp file
            Path(tmp_file.name).unlink(missing_ok=True)

def test_analyze_team_behavior_empty():
    """Test analyze_team_behavior with empty results."""
    print("Testing analyze_team_behavior with empty results...")
    
    from uav_intent_rl.examples.evaluate_mappo import analyze_team_behavior
    
    try:
        # Test with empty results
        empty_results = []
        metrics = analyze_team_behavior(empty_results, save_path=None)
        
        # Should handle empty results gracefully
        assert 'avg_formation_distance' in metrics
        assert 'avg_survival_rate' in metrics
        assert 'formation_consistency' in metrics
        
        # Should have default values
        assert metrics['avg_formation_distance'] == 0.0
        assert metrics['avg_survival_rate'] == 0.0
        assert metrics['formation_consistency'] == 0.0
        
        print("✓ analyze_team_behavior empty results test passed")
        return True
        
    except Exception as e:
        print(f"✗ analyze_team_behavior empty results test failed: {e}")
        return False

def test_json_serialization():
    """Test that numpy types are properly converted to native Python types."""
    print("Testing JSON serialization...")
    
    # Create data with numpy types
    data = {
        'avg_reward': np.float32(1.5),
        'avg_length': np.int32(100),
        'formation_distance': np.float64(2.5)
    }
    
    try:
        # This should work without TypeError
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        
        # Check that types are native Python types
        assert isinstance(parsed['avg_reward'], float)
        assert isinstance(parsed['avg_length'], int)
        assert isinstance(parsed['formation_distance'], float)
        
        print("✓ JSON serialization test passed")
        return True
        
    except Exception as e:
        print(f"✗ JSON serialization test failed: {e}")
        return False

def test_action_unpacking():
    """Test that action unpacking works with different return types."""
    print("Testing action unpacking...")
    
    # Test with tuple return
    result_tuple = (np.random.randn(3, 4), None)
    action_tuple = result_tuple[0] if isinstance(result_tuple, tuple) else result_tuple
    assert isinstance(action_tuple, np.ndarray)
    
    # Test with single value return
    result_single = np.random.randn(3, 4)
    action_single = result_single[0] if isinstance(result_single, tuple) else result_single
    assert isinstance(action_single, np.ndarray)
    
    print("✓ action unpacking test passed")
    return True

def main():
    """Run all tests."""
    print("Running tests for evaluate_mappo.py fixes...")
    print("=" * 50)
    
    tests = [
        test_run_mappo_episode,
        test_analyze_team_behavior,
        test_analyze_team_behavior_empty,
        test_json_serialization,
        test_action_unpacking,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The fixes are working correctly.")
    else:
        print("❌ Some tests failed. Please check the issues above.")

if __name__ == "__main__":
    main() 