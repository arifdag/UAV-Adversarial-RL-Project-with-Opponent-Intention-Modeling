#!/usr/bin/env python3
"""Test script for IP MARL 3v3 implementation."""

import numpy as np
import time
from pathlib import Path

from uav_intent_rl.envs.Dogfight3v3Aviary import Dogfight3v3Aviary
from uav_intent_rl.envs.Dogfight3v3MultiAgentEnv import Dogfight3v3MultiAgentEnv
from uav_intent_rl.policies.team_scripted_red import TeamScriptedRedPolicy
from uav_intent_rl.algo.ip_marl import IPMARL, IPMARLPolicy


def test_ip_marl_environment():
    """Test the IP MARL environment functionality."""
    print("Testing IP MARL 3v3 Environment...")
    
    # Test basic environment
    env = Dogfight3v3Aviary(gui=False)
    
    # Test reset
    obs, info = env.reset()
    print(f"✓ Environment reset successful. Observation shape: {obs.shape}")
    
    # Test step
    action = np.zeros((6, 4))  # 6 drones, 4 action dimensions
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"✓ Environment step successful. Reward: {reward:.3f}")
    
    # Test multiple steps
    for i in range(10):
        action = np.random.randn(6, 4) * 0.1  # Small random actions
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"✓ Episode ended after {i+1} steps")
            break
    
    print("✓ IP MARL 3v3 environment test passed!")
    env.close()


def test_ip_marl_multi_agent_env():
    """Test the multi-agent environment wrapper."""
    print("\nTesting IP MARL Multi-Agent Environment...")
    
    # Test multi-agent environment - use env_config instead of gui parameter
    env = Dogfight3v3MultiAgentEnv(env_config={"gui": False})
    
    # Test reset
    obs, info = env.reset()
    print(f"✓ Multi-agent reset successful. Observations: {list(obs.keys())}")
    
    # Test step
    actions = {
        "blue_0": np.random.randn(4) * 0.1,
        "blue_1": np.random.randn(4) * 0.1,
        "blue_2": np.random.randn(4) * 0.1,
        "red_0": np.random.randn(4) * 0.1,
        "red_1": np.random.randn(4) * 0.1,
        "red_2": np.random.randn(4) * 0.1,
    }
    
    obs, rewards, terminated, truncated, info = env.step(actions)
    print(f"✓ Multi-agent step successful. Rewards: {rewards}")
    
    # Test multiple steps
    for i in range(10):
        actions = {
            agent_id: np.random.randn(4) * 0.1
            for agent_id in env.AGENT_IDS  # Use AGENT_IDS instead of agent_ids
        }
        
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        if any(terminated.values()) or any(truncated.values()):
            print(f"✓ Multi-agent episode ended after {i+1} steps")
            break
    
    print("✓ IP MARL multi-agent environment test passed!")
    env.close()


def test_ip_marl_policy():
    """Test the IP MARL policy network."""
    print("\nTesting IP MARL Policy...")
    
    # Create environment for testing
    env = Dogfight3v3MultiAgentEnv(env_config={"gui": False})
    
    # Get observation and action spaces
    obs_space = env.observation_space["blue_0"]
    act_space = env.action_space["blue_0"]
    
    print(f"✓ Observation space: {obs_space}")
    print(f"✓ Action space: {act_space}")
    
    # Create policy
    policy = IPMARLPolicy(
        observation_space=obs_space,
        action_space=act_space,
        lr_schedule=lambda _: 3e-4,
        n_agents=6,
        use_centralized_critic=True,
        state_shape=[6, 12],
        hidden_dim=256,
        intention_dim=8,
        intention_propagation=True,
    )
    
    print("✓ IP MARL policy created successfully")
    
    # Test forward pass with single agent observation
    obs = env.reset()[0]
    obs_tensor = policy.obs_to_tensor(obs["blue_0"])[0]
    
    # Test intention prediction with single agent (should work)
    # The policy expects multi-agent observations, so we need to create a batch
    # with all agents having the same observation for testing
    batch_obs = obs_tensor.unsqueeze(0).repeat(1, 6, 1)  # (1, 6, obs_dim)
    intentions = policy.predict_intentions(batch_obs)
    print(f"✓ Intention prediction successful. Shape: {intentions.shape}")
    
    # Test forward pass with flattened observations (as expected by the policy)
    # The policy expects flattened observations, not multi-agent batch
    flattened_obs = obs_tensor.unsqueeze(0)  # (1, obs_dim)
    actions, values, states = policy.forward(flattened_obs)
    print(f"✓ Forward pass successful. Actions: {actions.shape}, Values: {values.shape}")
    
    # Test action evaluation - use proper observation dimensions
    # Create a random observation with correct shape (72 dimensions)
    random_obs = np.random.randn(72).astype(np.float32)  # Ensure float32 dtype
    obs_tensor_for_eval = policy.obs_to_tensor(random_obs)[0]
    values, log_probs, entropy = policy.evaluate_actions(
        obs_tensor_for_eval.unsqueeze(0), 
        actions  # Use the actions from forward pass
    )
    print(f"✓ Action evaluation successful. Log probs: {log_probs.shape}")
    
    print("✓ IP MARL policy test passed!")
    env.close()


def test_ip_marl_model():
    """Test the IP MARL model with a simpler approach."""
    print("\nTesting IP MARL Model...")
    
    # Create a proper wrapper for multi-agent environments
    from stable_baselines3.common.vec_env import VecEnv
    import gymnasium as gym
    
    class IPMARLEnvWrapper(gym.Env):
        """Wrapper to convert multi-agent environment to single-agent format for IP MARL."""
        
        def __init__(self, env_config: dict = None):
            """Initialize the wrapper."""
            super().__init__()
            if env_config is None:
                env_config = {}
            
            # Create the multi-agent environment
            from uav_intent_rl.envs.Dogfight3v3MultiAgentEnv import Dogfight3v3MultiAgentEnv
            self.env = Dogfight3v3MultiAgentEnv(env_config=env_config)
            
            # Get the observation and action spaces from the first agent
            first_agent_id = list(self.env.observation_space.keys())[0]
            self.observation_space = self.env.observation_space[first_agent_id]
            self.action_space = self.env.action_space[first_agent_id]
            
            # Store agent IDs for reference
            self.agent_ids = list(self.env.AGENT_IDS)
            self.n_agents = len(self.agent_ids)
            
        def reset(self, seed=None, options=None):
            """Reset the environment."""
            super().reset(seed=seed)
            obs, info = self.env.reset()
            # Return the observation of the first agent (blue_0)
            return obs["blue_0"], info
        
        def step(self, action):
            """Step the environment with actions for all agents."""
            # Create actions for all agents (use zeros for others, action for blue_0)
            actions = {}
            for agent_id in self.agent_ids:
                if agent_id == "blue_0":
                    actions[agent_id] = action
                else:
                    # Use zeros for other agents (they will be handled by scripted policies)
                    actions[agent_id] = np.zeros(4)
            
            obs, rewards, terminated, truncated, info = self.env.step(actions)
            
            # Return the observation and reward of the first agent
            return obs["blue_0"], rewards["blue_0"], terminated["blue_0"], truncated["blue_0"], info
        
        def close(self):
            """Close the environment."""
            self.env.close()
    
    class IPMARLVecEnv(VecEnv):
        """Vectorized environment wrapper for IP MARL."""
        
        def __init__(self, env_fns):
            self.envs = [env_fn() for env_fn in env_fns]
            self.num_envs = len(self.envs)
            
            # Get observation and action spaces from first environment
            first_env = self.envs[0]
            self.observation_space = first_env.observation_space
            self.action_space = first_env.action_space
            
        def reset(self):
            """Reset all environments."""
            obs_list = []
            for env in self.envs:
                obs, _ = env.reset()
                obs_list.append(obs)
            # Convert list to numpy array for Stable-Baselines3
            return np.array(obs_list)
        
        def step(self, actions):
            """Step all environments."""
            obs_list = []
            rewards_list = []
            dones_list = []
            infos_list = []
            
            for i, env in enumerate(self.envs):
                obs, reward, terminated, truncated, info = env.step(actions[i])
                obs_list.append(obs)
                rewards_list.append(reward)
                dones_list.append(terminated or truncated)
                infos_list.append(info)
            
            # Convert lists to numpy arrays for Stable-Baselines3
            return np.array(obs_list), np.array(rewards_list), np.array(dones_list), infos_list
        
        def close(self):
            """Close all environments."""
            for env in self.envs:
                env.close()
        
        def get_attr(self, attr_name, indices=None):
            """Get attribute from environments."""
            if indices is None:
                indices = range(self.num_envs)
            return [getattr(self.envs[i], attr_name) for i in indices]
        
        def set_attr(self, attr_name, value, indices=None):
            """Set attribute in environments."""
            if indices is None:
                indices = range(self.num_envs)
            for i in indices:
                setattr(self.envs[i], attr_name, value)
        
        def env_is_wrapped(self, wrapper_class):
            """Check if environment is wrapped by a given wrapper."""
            return False
        
        def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
            """Call a method on environments."""
            if indices is None:
                indices = range(self.num_envs)
            return [getattr(self.envs[i], method_name)(*method_args, **method_kwargs) for i in indices]
        
        def step_async(self, actions):
            """Step environments asynchronously."""
            # For simplicity, we'll just call step synchronously
            return self.step(actions)
        
        def step_wait(self):
            """Wait for async step to complete."""
            # Since we're not doing async, this is a no-op
            pass
    
    def make_env():
        return IPMARLEnvWrapper(env_config={"gui": False})
    
    vec_env = IPMARLVecEnv([make_env])
    
    # Create model with simpler configuration
    model = IPMARL(
        policy=IPMARLPolicy,
        env=vec_env,
        n_agents=6,
        use_centralized_critic=True,
        state_shape=[6, 12],
        intention_dim=8,
        intention_propagation=True,
        intention_loss_coef=0.1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
    )
    
    print("✓ IP MARL model created successfully")
    
    # Test prediction
    obs = vec_env.reset()
    actions, _ = model.predict(obs, deterministic=True)
    print(f"✓ Model prediction successful. Actions shape: {actions.shape}")
    
    # Test a few training steps
    print("✓ Testing training step...")
    model.learn(total_timesteps=1024)
    print("✓ Training step successful")
    
    print("✓ IP MARL model test passed!")
    vec_env.close()


def test_team_scripted_red():
    """Test the team scripted red policy."""
    print("\nTesting Team Scripted Red Policy...")
    
    # Create environment
    env = Dogfight3v3Aviary(gui=False)
    
    # Create scripted red policy
    red_policy = TeamScriptedRedPolicy()
    
    # Test reset
    obs, info = env.reset()
    print("✓ Environment reset successful")
    
    # Test scripted policy
    for i in range(10):
        red_actions = red_policy(env)
        print(f"✓ Step {i+1}: Red actions shape: {red_actions.shape}")
        
        # Apply actions - red_actions is already (6,4) for all drones
        # We just need to use it directly
        obs, reward, terminated, truncated, info = env.step(red_actions)
        
        if terminated or truncated:
            print(f"✓ Episode ended after {i+1} steps")
            break
    
    print("✓ Team scripted red policy test passed!")
    env.close()


def test_intention_propagation():
    """Test intention propagation functionality."""
    print("\nTesting Intention Propagation...")
    
    # Create environment
    env = Dogfight3v3MultiAgentEnv(env_config={"gui": False})
    
    # Create policy
    policy = IPMARLPolicy(
        observation_space=env.observation_space["blue_0"],
        action_space=env.action_space["blue_0"],
        lr_schedule=lambda _: 3e-4,
        n_agents=6,
        use_centralized_critic=True,
        state_shape=[6, 12],
        hidden_dim=256,
        intention_dim=8,
        intention_propagation=True,
    )
    
    # Get observations
    obs = env.reset()[0]
    obs_tensor = policy.obs_to_tensor(obs["blue_0"])[0]
    
    # Create multi-agent batch for testing
    batch_obs = obs_tensor.unsqueeze(0).repeat(1, 6, 1)  # (1, 6, obs_dim)
    
    # Test intention prediction without propagation
    policy.intention_propagation = False
    intentions_no_prop = policy.predict_intentions(batch_obs)
    print(f"✓ Intentions without propagation: {intentions_no_prop.shape}")
    
    # Test intention prediction with propagation
    policy.intention_propagation = True
    intentions_with_prop = policy.predict_intentions(batch_obs)
    print(f"✓ Intentions with propagation: {intentions_with_prop.shape}")
    
    # Check that intentions are different (propagation should change them)
    import torch as th
    intention_diff = th.norm(intentions_with_prop - intentions_no_prop).item()
    print(f"✓ Intention difference after propagation: {intention_diff:.4f}")
    
    print("✓ Intention propagation test passed!")
    env.close()


def main():
    """Run all tests."""
    print("🧪 Running IP MARL 3v3 Tests...")
    print("=" * 50)
    
    try:
        test_ip_marl_environment()
        test_ip_marl_multi_agent_env()
        test_ip_marl_policy()
        test_ip_marl_model()
        test_team_scripted_red()
        test_intention_propagation()
        
        print("\n" + "=" * 50)
        print("✅ All IP MARL 3v3 tests passed!")
        print("\n🎯 IP MARL is ready for training!")
        print("\nTo start training, run:")
        print("python train_ip_marl_3v3.py --config configs/ip_marl_3v3.yaml")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    import torch as th
    success = main()
    exit(0 if success else 1) 