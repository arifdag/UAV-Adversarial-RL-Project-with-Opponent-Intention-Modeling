from uav_intent_rl.examples.ppo_nomodel import run


def test_ppo_nomodel_pipeline_runs():
    """Smoke-test that the training pipeline executes without error."""

    # Run only a few timesteps to keep CI fast
    model, win_rate = run(total_timesteps=200, n_eval_episodes=2, gui=False, n_envs=1, verbose=0)

    # Basic sanity checks on returned values
    assert hasattr(model, "predict"), "run() did not return a trained SB3 model"
    assert 0.0 <= win_rate <= 1.0, "Win-rate should be a valid probability"

    # Print for visibility in test logs
    print(f"[TEST] Eval win-rate after smoke training: {win_rate * 100:.1f}%") 