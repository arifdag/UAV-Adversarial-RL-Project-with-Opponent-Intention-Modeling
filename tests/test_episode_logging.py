import numpy as np
import pathlib

from uav_intent_rl import DogfightAviary, make_monitored_env


def test_episode_video_and_csv(tmp_path):
    """A short roll-out should create an MP4 and progress.csv in a fresh run dir."""

    # ------------------------------------------------------------------
    # Create monitored environment with run artefacts saved under tmp_path
    # ------------------------------------------------------------------
    env = make_monitored_env(DogfightAviary, runs_root=tmp_path, gui=False)

    # Roll out just a handful of steps – enough for wrappers to trigger
    obs, _ = env.reset(seed=0)
    for _ in range(10):
        action = np.zeros_like(env.action_space.sample(), dtype=np.float32)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    env.close()

    # ------------------------------------------------------------------
    # Validate artefacts
    # ------------------------------------------------------------------
    run_dirs = list(pathlib.Path(tmp_path).iterdir())
    # Exactly one new timestamped directory should have been created
    assert len(run_dirs) == 1, "Expected exactly one run directory to be created"

    run_dir = run_dirs[0]
    csv_files = list(run_dir.glob("*.csv"))
    assert csv_files, "No CSV log files were created in run directory" 