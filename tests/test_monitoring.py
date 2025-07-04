import pytest
import os
import tempfile
import shutil
from pathlib import Path

from gym_pybullet_drones.envs.DogfightAviary import DogfightAviary
from gym_pybullet_drones.utils.monitor import Monitor, save_episode_statistics, list_monitoring_files


class TestMonitoring:
    """Test suite for monitoring functionality."""

    def setup_method(self):
        """Set up test environment before each test method."""
        # Create temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up after each test method."""
        # Remove temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_monitor_wrapper_creation(self):
        """Test that Monitor wrapper can be created successfully."""
        env = DogfightAviary(num_drones=2, gui=False, record=False)

        # Wrap with Monitor
        wrapped_env, output_dir = Monitor(
            env,
            output_dir=self.temp_dir,
            record_video=True,
            record_stats=True
        )

        # Check that output directory was created
        assert os.path.exists(output_dir)
        assert output_dir == self.temp_dir

        # Environment should still be functional
        obs, info = wrapped_env.reset()
        assert obs is not None
        assert isinstance(info, dict)

        wrapped_env.close()

    def test_monitor_no_video_recording(self):
        """Test Monitor with video recording disabled."""
        env = DogfightAviary(num_drones=2, gui=False, record=False)

        wrapped_env, output_dir = Monitor(
            env,
            output_dir=self.temp_dir,
            record_video=False,
            record_stats=True
        )

        # Run a short episode
        obs, info = wrapped_env.reset()
        for _ in range(10):
            action = wrapped_env.action_space.sample()
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            if terminated or truncated:
                break

        wrapped_env.close()

        # Should have no video files
        video_files, log_files = list_monitoring_files(output_dir)
        assert len(video_files) == 0

    def test_monitor_no_stats_recording(self):
        """Test Monitor with statistics recording disabled."""
        env = DogfightAviary(num_drones=2, gui=False, record=False)

        wrapped_env, output_dir = Monitor(
            env,
            output_dir=self.temp_dir,
            record_video=False,
            record_stats=False
        )

        # Run a short episode
        obs, info = wrapped_env.reset()
        for _ in range(10):
            action = wrapped_env.action_space.sample()
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            if terminated or truncated:
                break

        wrapped_env.close()

        # Should not be able to save statistics
        csv_path = save_episode_statistics(wrapped_env, output_dir)
        assert csv_path is None

    def test_episode_recording_integration(self):
        """Test full episode recording with both video and statistics."""
        env = DogfightAviary(num_drones=2, gui=False, record=False)

        wrapped_env, output_dir = Monitor(
            env,
            output_dir=self.temp_dir,
            record_video=True,
            record_stats=True,
            name_prefix="test_episode"
        )

        # Run 2 short episodes
        for episode in range(2):
            obs, info = wrapped_env.reset()
            step_count = 0

            while step_count < 50:  # Limit steps for faster testing
                action = wrapped_env.action_space.sample()
                obs, reward, terminated, truncated, info = wrapped_env.step(action)
                step_count += 1

                if terminated or truncated:
                    break

        # Save statistics
        csv_path = save_episode_statistics(wrapped_env, output_dir)
        assert csv_path is not None
        assert os.path.exists(csv_path)

        wrapped_env.close()

        # Check generated files
        video_files, log_files = list_monitoring_files(output_dir)

        # Should have video files (one per episode)
        assert len(video_files) >= 1, f"Expected at least 1 video file, got {len(video_files)}"

        # Should have CSV log
        assert len(log_files) >= 1, f"Expected at least 1 log file, got {len(log_files)}"

        # Verify file extensions
        for video_file in video_files:
            assert video_file.endswith('.mp4'), f"Video file should be .mp4: {video_file}"

        for log_file in log_files:
            assert log_file.endswith('.csv'), f"Log file should be .csv: {log_file}"

    def test_dod_requirements(self):
        """Test that DoD requirements are satisfied: runs/YYYY-MM-DD/ folder contains .mp4 + progress.csv"""
        env = DogfightAviary(num_drones=2, gui=False, record=False)

        # Use default timestamped directory (similar to runs/2025-07-xx/)
        wrapped_env, output_dir = Monitor(
            env,
            record_video=True,
            record_stats=True
        )

        # Should create a runs/ directory with timestamp (handle both Unix and Windows paths)
        assert "runs" in output_dir
        assert os.path.exists(output_dir)

        # Run one episode
        obs, info = wrapped_env.reset()
        for _ in range(20):  # Short episode for testing
            action = wrapped_env.action_space.sample()
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            if terminated or truncated:
                break

        # Save statistics as progress.csv
        csv_path = save_episode_statistics(wrapped_env, output_dir, filename="progress.csv")
        assert csv_path is not None

        wrapped_env.close()

        # Verify DoD requirements
        output_files = os.listdir(output_dir)
        has_mp4 = any(f.endswith('.mp4') for f in output_files)
        has_progress_csv = 'progress.csv' in output_files

        assert has_mp4, f"Output directory should contain .mp4 files. Files: {output_files}"
        assert has_progress_csv, f"Output directory should contain progress.csv. Files: {output_files}"

        print(f"DoD satisfied: {output_dir} contains .mp4 + progress.csv")

    def test_monitor_with_custom_video_trigger(self):
        """Test Monitor with custom video recording trigger."""
        env = DogfightAviary(num_drones=2, gui=False, record=False)

        # Record only every other episode
        video_trigger = lambda episode_id: episode_id % 2 == 0

        wrapped_env, output_dir = Monitor(
            env,
            output_dir=self.temp_dir,
            record_video=True,
            record_stats=True,
            video_trigger=video_trigger
        )

        # Run 3 episodes (should record episodes 0 and 2)
        for episode in range(3):
            obs, info = wrapped_env.reset()
            for _ in range(10):
                action = wrapped_env.action_space.sample()
                obs, reward, terminated, truncated, info = wrapped_env.step(action)
                if terminated or truncated:
                    break

        wrapped_env.close()

        # Should have fewer videos than episodes
        video_files, log_files = list_monitoring_files(output_dir)
        assert len(video_files) <= 2, f"Should have at most 2 videos for selective recording"

    def test_list_monitoring_files(self):
        """Test the list_monitoring_files utility function."""
        # Test with non-existent directory
        video_files, log_files = list_monitoring_files("/non/existent/path")
        assert video_files == []
        assert log_files == []

        # Test with empty directory
        video_files, log_files = list_monitoring_files(self.temp_dir)
        assert video_files == []
        assert log_files == []

        # Create some test files
        video_path = os.path.join(self.temp_dir, "test_video.mp4")
        log_path = os.path.join(self.temp_dir, "test_log.csv")
        other_path = os.path.join(self.temp_dir, "other_file.txt")

        # Create empty files
        for path in [video_path, log_path, other_path]:
            with open(path, 'w') as f:
                f.write("test content")

        video_files, log_files = list_monitoring_files(self.temp_dir)
        assert len(video_files) == 1
        assert len(log_files) == 1
        assert "test_video.mp4" in video_files
        assert "test_log.csv" in log_files


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
