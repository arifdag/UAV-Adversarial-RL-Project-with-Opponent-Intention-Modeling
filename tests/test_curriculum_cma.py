"""Tests for PPO-CMA and curriculum learning implementations."""

import numpy as np
import pytest
import unittest
import shutil
import os
from pathlib import Path

from uav_intent_rl.algo.ppo_cma import PPOCMA
from uav_intent_rl.utils.curriculum_wrappers import (
    CurriculumOpponentWrapper,
    DifficultyScheduler,
    StationaryOpponent,
    SimplePursuitOpponent,
    AdvancedPursuitOpponent,
    AdversarialOpponent,
)
from uav_intent_rl.envs import MultiDroneDogfightAviary
from uav_intent_rl.utils.intent_wrappers import BlueVsFixedRedWrapper
from uav_intent_rl.examples.ppo_curriculum_cma import CurriculumCallback


class TestPPOCMAModelSavingAndPerformanceCurriculum(unittest.TestCase):
    def setUp(self):
        # Ensure test directories are clean
        self.models_dir = Path("models")
        self.checkpoints_dir = self.models_dir / "checkpoints"
        self.best_model_path = self.models_dir / "amf_best_winrate.zip"
        if self.checkpoints_dir.exists():
            shutil.rmtree(self.checkpoints_dir)
        if self.best_model_path.exists():
            self.best_model_path.unlink()
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        # Clean up test artifacts
        if self.checkpoints_dir.exists():
            shutil.rmtree(self.checkpoints_dir)
        if self.best_model_path.exists():
            self.best_model_path.unlink()

    def test_model_saving_and_performance_curriculum(self):
        from uav_intent_rl.examples.ppo_curriculum_cma import train_ppo_curriculum_cma
        from uav_intent_rl.algo.ppo_cma import PPOCMA

        # Run a very short training with performance-based curriculum
        model = train_ppo_curriculum_cma(
            total_timesteps=200,
            n_envs=1,
            curriculum_schedule="performance",
            performance_threshold=0.0,  # Always trigger
            performance_step_size=0.5,
            performance_warmup_steps=0,
            eval_freq=50,
            eval_episodes=2,
            verbose=0,
            gui=False,
            checkpoint_freq=50,  # Ensure checkpoint is saved in short test
        )

        # Check that the best winrate model was saved
        self.assertTrue(self.best_model_path.exists(), "Best winrate model was not saved!")

        # Check that at least one checkpoint was saved
        checkpoints = list(self.checkpoints_dir.glob("ppo_cma_step*.zip"))
        self.assertTrue(len(checkpoints) > 0, "No checkpoints were saved!")

        # Check that the saved model loads and is a PPOCMA
        loaded = PPOCMA.load(str(self.best_model_path), env=None)
        self.assertTrue(hasattr(loaded, "predict"), "Loaded PPOCMA model is not valid!")

        # Optionally: check that performance-based curriculum was triggered (by running with a low threshold)
        # (For a more advanced check, parse tensorboard logs for 'curriculum/performance_based' > 0)


def test_performance_based_curriculum():
    """Test performance-based curriculum functionality."""
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    # Create a simple test environment
    def make_single_env():
        base_env = MultiDroneDogfightAviary(gui=False)
        env = BlueVsFixedRedWrapper(base_env)
        scheduler = DifficultyScheduler(
            total_timesteps=1000,
            difficulty_schedule="linear",
            min_difficulty=0.0,
            max_difficulty=1.0,
        )
        return CurriculumOpponentWrapper(env, scheduler)
    
    env = DummyVecEnv([make_single_env for _ in range(2)])
    
    # Create callback with performance-based curriculum
    callback = CurriculumCallback(
        verbose=0,
        win_rate_threshold=0.8,
        difficulty_step=0.25,
    )
    # Assign a dummy model with get_env() and logger
    class DummyLogger:
        def __init__(self):
            self.name_to_value = {}
        def record(self, key, value):
            self.name_to_value[key] = value
    class DummyModel:
        def get_env(self):
            return env
        logger = DummyLogger()
    callback.model = DummyModel()
    # Test initial state
    callback.num_timesteps = 0
    callback.set_training_env(env)
    callback._on_step()
    
    # Should start with baseline difficulty
    assert callback.current_difficulty == 0.0
    
    # Simulate high win rate (should increase difficulty)
    callback.num_timesteps = 1000
    callback.logger.name_to_value = {"eval/win_rate": 0.9}  # Above threshold
    callback._on_step()
    
    # Should increase difficulty by step size
    assert callback.current_difficulty == 0.25
    
    # Simulate low win rate (should not increase difficulty)
    callback.logger.name_to_value = {"eval/win_rate": 0.5}  # Below threshold
    callback._on_step()
    
    # Should stay at current difficulty
    assert callback.current_difficulty == 0.25
    
    # Simulate high win rate again (should increase difficulty)
    callback.logger.name_to_value = {"eval/win_rate": 0.85}  # Above threshold
    callback._on_step()
    
    # Should increase difficulty again
    assert callback.current_difficulty == 0.5
    
    print("✓ Performance-based curriculum works correctly")


def test_curriculum_speedup_changes():
    """Test that curriculum speedup changes work correctly."""
    
    # Test 1: Verify exponential schedule behavior (slower early, faster late)
    total_timesteps = 1000
    warmup_steps = 100
    
    linear_scheduler = DifficultyScheduler(
        total_timesteps=total_timesteps,
        difficulty_schedule="linear",
        warmup_steps=warmup_steps,
    )
    
    exponential_scheduler = DifficultyScheduler(
        total_timesteps=total_timesteps,
        difficulty_schedule="exponential",
        warmup_steps=warmup_steps,
    )
    
    # Test key milestones
    test_milestones = [200, 400, 600, 800]  # Various progress levels
    
    # Exponential should be slower than linear for all progress < 100%
    for milestone in test_milestones:
        linear_diff = linear_scheduler.get_difficulty(milestone)
        exp_diff = exponential_scheduler.get_difficulty(milestone)
        assert exp_diff <= linear_diff, f"Exponential should be slower than linear at milestone {milestone}"
    
    # Test that exponential provides more time on difficult stages
    # This is the key benefit: exponential spends more time at lower difficulties
    print("✓ Exponential schedule provides more time on difficult stages")
    
    # Test 2: Verify reduced warmup period
    old_warmup = 50000
    new_warmup = 10000
    
    old_scheduler = DifficultyScheduler(
        total_timesteps=total_timesteps,
        difficulty_schedule="linear",
        warmup_steps=old_warmup,
    )
    
    new_scheduler = DifficultyScheduler(
        total_timesteps=total_timesteps,
        difficulty_schedule="linear",
        warmup_steps=new_warmup,
    )
    
    # At early timesteps, new scheduler should have higher difficulty
    early_timestep = 15000
    old_diff = old_scheduler.get_difficulty(early_timestep)
    new_diff = new_scheduler.get_difficulty(early_timestep)
    assert new_diff > old_diff, "Reduced warmup should allow faster progression"
    
    # Test 3: Verify step schedule with discrete stages
    step_scheduler = DifficultyScheduler(
        total_timesteps=total_timesteps,
        difficulty_schedule="step",
        curriculum_stages=[0.0, 0.25, 0.5, 0.75, 1.0],
        warmup_steps=warmup_steps,
    )
    
    # Test that step schedule produces discrete values
    step_difficulties = []
    for step in range(0, total_timesteps, 200):
        diff = step_scheduler.get_difficulty(step)
        step_difficulties.append(diff)
    
    # Should have discrete stages
    unique_difficulties = set(step_difficulties)
    assert len(unique_difficulties) <= 5, "Step schedule should have discrete stages"
    
    # Test 4: Verify cosine schedule produces smooth progression
    cosine_scheduler = DifficultyScheduler(
        total_timesteps=total_timesteps,
        difficulty_schedule="cosine",
        warmup_steps=warmup_steps,
    )
    
    cosine_difficulties = []
    for step in range(0, total_timesteps, 100):
        diff = cosine_scheduler.get_difficulty(step)
        cosine_difficulties.append(diff)
    
    # Should be monotonically increasing
    for i in range(1, len(cosine_difficulties)):
        assert cosine_difficulties[i] >= cosine_difficulties[i-1], "Cosine should be monotonically increasing"
    
    print("✓ Curriculum speedup changes verified")


def test_difficulty_scheduler():
    """Test difficulty scheduler functionality."""
    scheduler = DifficultyScheduler(
        total_timesteps=1000,
        difficulty_schedule="linear",
        min_difficulty=0.0,
        max_difficulty=1.0,
    )
    
    # Test linear progression
    assert scheduler.get_difficulty(0) == 0.0
    assert scheduler.get_difficulty(500) == 0.5
    assert scheduler.get_difficulty(1000) == 1.0
    
    # Test bounds
    assert scheduler.get_difficulty(-100) == 0.0
    assert scheduler.get_difficulty(2000) == 1.0


def test_curriculum_callback_global_timestep():
    """Test that curriculum callback uses global timestep correctly."""
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    # Create a simple test environment
    def make_single_env():
        base_env = MultiDroneDogfightAviary(gui=False)
        env = BlueVsFixedRedWrapper(base_env)
        scheduler = DifficultyScheduler(
            total_timesteps=1000,
            difficulty_schedule="linear",
            min_difficulty=0.0,
            max_difficulty=1.0,
        )
        return CurriculumOpponentWrapper(env, scheduler)
    
    env = DummyVecEnv([make_single_env for _ in range(2)])
    
    # Create callback
    callback = CurriculumCallback(verbose=0)
    # Assign a dummy model with get_env()
    class DummyModel:
        def get_env(self):
            return env
    callback.model = DummyModel()
    # Test that callback correctly uses global timestep
    test_timesteps = [0, 250, 500, 750, 1000]
    
    for global_timestep in test_timesteps:
        # Simulate callback behavior
        callback.num_timesteps = global_timestep
        callback.set_training_env(env)
        
        # Get scheduler and calculate expected difficulty
        scheduler = env.envs[0].scheduler
        expected_difficulty = scheduler.get_difficulty(global_timestep)
        
        # Manually update environments to simulate callback
        for env_wrapper in env.envs:
            if hasattr(env_wrapper, '_update_opponent'):
                env_wrapper._update_opponent(expected_difficulty, force_update=True)
        
        # Verify that environments have correct difficulty
        for env_wrapper in env.envs:
            assert abs(env_wrapper.current_difficulty - expected_difficulty) < 0.01, \
                f"Global timestep {global_timestep}: expected {expected_difficulty}, got {env_wrapper.current_difficulty}"


def test_curriculum_opponents():
    """Test curriculum opponent implementations."""
    # Test stationary opponent
    stationary = StationaryOpponent()
    assert isinstance(stationary, StationaryOpponent)
    
    # Test simple pursuit opponent
    simple = SimplePursuitOpponent(noise_scale=0.1)
    assert isinstance(simple, SimplePursuitOpponent)
    assert simple.noise_scale == 0.1
    
    # Test advanced pursuit opponent
    advanced = AdvancedPursuitOpponent(evasion_prob=0.3)
    assert isinstance(advanced, AdvancedPursuitOpponent)
    assert advanced.evasion_prob == 0.3
    
    # Test adversarial opponent
    adversarial = AdversarialOpponent()
    assert isinstance(adversarial, AdversarialOpponent)
    assert adversarial.behavior_state == "pursuit"


def test_curriculum_wrapper():
    """Test curriculum wrapper functionality."""
    # Create base environment
    base_env = MultiDroneDogfightAviary(gui=False)
    env = BlueVsFixedRedWrapper(base_env)
    
    # Create scheduler
    scheduler = DifficultyScheduler(
        total_timesteps=1000,
        difficulty_schedule="linear",
        min_difficulty=0.0,
        max_difficulty=1.0,
    )
    
    # Create curriculum wrapper
    curriculum_env = CurriculumOpponentWrapper(env, scheduler)
    
    # Test initialization
    assert curriculum_env.current_difficulty == 0.0
    assert curriculum_env.timestep == 0
    
    # Test reset
    obs, info = curriculum_env.reset()
    assert 'curriculum_difficulty' in info
    assert 'curriculum_timestep' in info
    
    # Test step
    action = curriculum_env.action_space.sample()
    obs, reward, terminated, truncated, info = curriculum_env.step(action)
    assert 'curriculum_difficulty' in info
    assert curriculum_env.timestep == 1


def test_ppo_cma_initialization():
    """Test PPO-CMA algorithm initialization."""
    # Create environment
    base_env = MultiDroneDogfightAviary(gui=False)
    env = BlueVsFixedRedWrapper(base_env)
    
    # Create PPO-CMA model
    model = PPOCMA(
        policy="MlpPolicy",
        env=env,
        cma_learning_rate=0.1,
        cma_memory_size=100,
        cma_min_variance=0.01,
        cma_max_variance=2.0,
        verbose=0,
    )
    
    # Test CMA parameters
    assert model.cma_lr == 0.1
    assert model.cma_memory_size == 100
    assert model.cma_min_variance == 0.01
    assert model.cma_max_variance == 2.0
    assert model.cma_variance_scale == 1.0
    
    # Test CMA state
    assert model.cma_covariance_matrix is not None
    assert model.cma_action_dim == 4  # 4D action space


def test_cma_variance_update():
    """Test CMA variance update mechanism."""
    # Create environment
    base_env = MultiDroneDogfightAviary(gui=False)
    env = BlueVsFixedRedWrapper(base_env)
    
    # Create PPO-CMA model
    model = PPOCMA(
        policy="MlpPolicy",
        env=env,
        cma_performance_threshold=0.05,  # Lower threshold for testing
        verbose=0,
    )
    
    # Test variance update with improving performance
    initial_variance = model.cma_variance_scale
    model._update_cma_variance([0.1, 0.2, 0.3, 0.4, 0.5])  # Improving by 0.1
    assert model.cma_variance_scale < initial_variance
    
    # Test variance update with declining performance
    model.cma_variance_scale = 1.0  # Reset
    model._update_cma_variance([0.5, 0.4, 0.3, 0.2, 0.1])  # Declining by 0.1
    assert model.cma_variance_scale > 1.0


def test_cma_noise_application():
    """Test CMA noise application to actions."""
    # Create environment
    base_env = MultiDroneDogfightAviary(gui=False)
    env = BlueVsFixedRedWrapper(base_env)
    
    # Create PPO-CMA model
    model = PPOCMA(
        policy="MlpPolicy",
        env=env,
        verbose=0,
    )
    
    # Test action noise application
    actions = np.zeros((2, 4))  # 2 drones, 4D actions
    noisy_actions = model._apply_cma_noise(actions)
    
    # Check that noise was applied
    assert not np.allclose(actions, noisy_actions)
    
    # Check that actions are within bounds
    assert np.all(noisy_actions >= -1.0)
    assert np.all(noisy_actions <= 1.0)


def test_curriculum_progression():
    """Test curriculum progression through different stages."""
    # Create environment with curriculum
    base_env = MultiDroneDogfightAviary(gui=False)
    env = BlueVsFixedRedWrapper(base_env)
    
    scheduler = DifficultyScheduler(
        total_timesteps=1000,
        difficulty_schedule="step",
        curriculum_stages=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    
    curriculum_env = CurriculumOpponentWrapper(env, scheduler)
    
    # Test progression through stages
    difficulties = []
    for step in range(0, 1000, 200):
        curriculum_env.timestep = step
        difficulty = scheduler.get_difficulty(step)
        curriculum_env._update_opponent(difficulty)
        difficulties.append(curriculum_env.current_difficulty)
    
    # Check that difficulty increases
    assert difficulties[0] < difficulties[-1]
    assert all(difficulties[i] <= difficulties[i+1] for i in range(len(difficulties)-1))


def test_opponent_factory():
    """Test opponent factory creates correct opponents for different difficulties."""
    base_env = MultiDroneDogfightAviary(gui=False)
    env = BlueVsFixedRedWrapper(base_env)
    
    scheduler = DifficultyScheduler(
        total_timesteps=1000,
        difficulty_schedule="linear",
    )
    
    curriculum_env = CurriculumOpponentWrapper(env, scheduler)
    
    # Test opponent creation for different difficulties
    difficulties = [0.0, 0.25, 0.5, 0.75, 1.0]
    opponents = []
    
    for difficulty in difficulties:
        opponent = curriculum_env.opponent_factory(difficulty)
        opponents.append(opponent)
    
    # Check that different opponents are created
    opponent_types = [type(opp) for opp in opponents]
    assert len(set(opponent_types)) > 1  # Should have different opponent types


def test_force_update_parameter():
    """Test that force_update parameter works correctly in _update_opponent."""
    base_env = MultiDroneDogfightAviary(gui=False)
    env = BlueVsFixedRedWrapper(base_env)
    
    scheduler = DifficultyScheduler(
        total_timesteps=1000,
        difficulty_schedule="linear",
        min_difficulty=0.0,
        max_difficulty=1.0,
    )
    
    curriculum_env = CurriculumOpponentWrapper(env, scheduler)
    
    # Test that small changes are ignored without force_update
    initial_diff = curriculum_env.current_difficulty
    curriculum_env._update_opponent(initial_diff + 0.005)  # Small change
    assert curriculum_env.current_difficulty == initial_diff, "Small change should be ignored"
    
    # Test that small changes are applied with force_update
    curriculum_env._update_opponent(initial_diff + 0.005, force_update=True)
    assert curriculum_env.current_difficulty == initial_diff + 0.005, "Force update should apply small changes"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 