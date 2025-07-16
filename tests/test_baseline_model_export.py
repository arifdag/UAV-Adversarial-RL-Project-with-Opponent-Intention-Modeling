from pathlib import Path


from stable_baselines3 import PPO


# The training helper from the example script
from uav_intent_rl.examples.ppo_nomodel import run as train_baseline




def test_baseline_checkpoint_saved_and_loadable(tmp_path):
    """Ensure training exports a checkpoint that loads without error."""

    # Run a very short training session (few steps to keep test fast)
    train_baseline(total_timesteps=200, n_eval_episodes=2, gui=False, n_envs=1, verbose=0)

    ckpt_path = Path(__file__).resolve().parents[1] / "models" / "baseline_no_model.zip"

    assert ckpt_path.exists(), "Checkpoint file was not created by training script"

    # Attempt to load the checkpoint with Stable-Baselines3 PPO
    loaded = PPO.load(str(ckpt_path))

    # Basic sanity check: loaded model should implement the predict() API
    assert hasattr(loaded, "predict"), "Loaded checkpoint is not a valid SB3 model" 