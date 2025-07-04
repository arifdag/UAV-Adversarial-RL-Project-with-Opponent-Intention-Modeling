"""
Monitoring utilities for gym-pybullet-drones environments.

This module provides easy-to-use functions for wrapping environments with
monitoring capabilities including video recording and episode statistics logging.
"""

import os
import warnings
from datetime import datetime
from pathlib import Path

try:
    import gymnasium as gym
    from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics

    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False
    gym = None
    RecordVideo = None
    RecordEpisodeStatistics = None


class RenderModeWrapper(gym.Wrapper):
    """
    Wrapper to set render_mode for environments that don't have it set.
    
    This wrapper uses PyBullet's camera system to capture actual visual frames
    from the simulation with proper drone framing.
    """

    def __init__(self, env, render_mode="rgb_array"):
        super().__init__(env)
        self._render_mode = render_mode
        self._image_size = (480, 640, 3)  # Default image size (height, width, channels)
        self._camera_setup_done = False
        self._setup_camera()

    def _setup_camera(self):
        """Set up camera parameters for proper drone viewing."""
        # Use environment's existing camera setup if available
        if hasattr(self.env, 'CAM_VIEW') and hasattr(self.env, 'CAM_PRO'):
            self._view_matrix = self.env.CAM_VIEW
            self._proj_matrix = self.env.CAM_PRO
            self._camera_setup_done = True
            return

        # Set up our own camera with better positioning for drone observation
        import pybullet as p

        # Position camera to get a good view of the drone arena
        # Assume drones are operating in a roughly 2x2 meter area around origin
        camera_distance = 3.0  # meters from target
        camera_yaw = 45  # degrees
        camera_pitch = -30  # degrees (looking down)
        target_position = [0.25, 0.25, 0.15]  # Center of typical drone formation

        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=target_position,
            distance=camera_distance,
            yaw=camera_yaw,
            pitch=camera_pitch,
            roll=0,
            upAxisIndex=2
        )

        # Set up projection matrix for good field of view
        self._proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,  # Field of view in degrees
            aspect=self._image_size[1] / self._image_size[0],  # width/height
            nearVal=0.1,
            farVal=100.0
        )

        self._camera_setup_done = True

    @property
    def render_mode(self):
        return self._render_mode

    def render(self):
        """
        Render the environment using PyBullet's camera system.
        
        Returns actual visual frames showing the drones in action.
        """
        if self._render_mode == "rgb_array":
            return self._capture_frame()
        else:
            # Fall back to base environment rendering for other modes
            if hasattr(self.env, 'render'):
                return self.env.render()
            return None

    def _capture_frame(self):
        """Capture a frame using PyBullet's camera system."""
        import pybullet as p
        import numpy as np

        if not self._camera_setup_done:
            self._setup_camera()

        try:
            # Capture image using PyBullet's camera
            width, height = self._image_size[1], self._image_size[0]

            # Get camera image from PyBullet
            (_, _, px, _, _) = p.getCameraImage(
                width=width,
                height=height,
                viewMatrix=self._view_matrix,
                projectionMatrix=self._proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )

            # Convert to RGB array (PyBullet returns RGBA, we want RGB)
            rgb_array = np.array(px, dtype=np.uint8)
            rgb_array = rgb_array[:, :, :3]  # Remove alpha channel
            rgb_array = np.reshape(rgb_array, (height, width, 3))

            return rgb_array

        except Exception as e:
            # Fallback: create a simple visualization showing drone positions
            return self._create_fallback_frame()

    def _create_fallback_frame(self):
        """Create a fallback frame with drone position information."""
        import numpy as np

        # Create a dark blue background
        height, width = self._image_size[0], self._image_size[1]
        frame = np.full((height, width, 3), [30, 50, 100], dtype=np.uint8)

        # Try to get drone positions and draw simple representations
        try:
            if hasattr(self.env, '_getDroneStateVector'):
                # Get positions of all drones
                for i in range(getattr(self.env, 'NUM_DRONES', 4)):
                    try:
                        state = self.env._getDroneStateVector(i)
                        pos = state[0:3]  # x, y, z position

                        # Map 3D position to 2D screen coordinates
                        screen_x = int((pos[0] + 1) * width / 2)  # Map [-1,1] to [0,width]
                        screen_y = int((1 - pos[1]) * height / 2)  # Map [-1,1] to [0,height], flip Y

                        # Clamp to screen bounds
                        screen_x = max(10, min(width - 10, screen_x))
                        screen_y = max(10, min(height - 10, screen_y))

                        # Draw a simple drone representation (small colored square)
                        color = [255, 100, 100] if i < 2 else [100, 100, 255]  # Red for first 2, blue for others
                        frame[screen_y - 5:screen_y + 5, screen_x - 5:screen_x + 5] = color

                    except (IndexError, AttributeError):
                        pass
        except (AttributeError, TypeError):
            pass

        # Add crosshair for reference
        mid_x, mid_y = width // 2, height // 2
        frame[mid_y - 20:mid_y + 20, mid_x - 2:mid_x + 2] = [200, 200, 200]  # Vertical line
        frame[mid_y - 2:mid_y + 2, mid_x - 20:mid_x + 20] = [200, 200, 200]  # Horizontal line

        return frame


def Monitor(env,
            output_dir=None,
            record_video=True,
            record_stats=True,
            video_trigger=None,
            name_prefix="rl-video"):
    """
    Wrap an environment with monitoring capabilities for video recording and statistics.
    
    This function wraps the environment with RecordVideo and RecordEpisodeStatistics
    wrappers to enable automatic logging of episodes for post-mortem analysis.
    
    Parameters
    ----------
    env : gym.Env
        The environment to wrap
    output_dir : str, optional
        Directory to save outputs. If None, creates timestamped directory in 'runs/'
    record_video : bool, optional
        Whether to record videos (default: True)
    record_stats : bool, optional  
        Whether to record episode statistics (default: True)
    video_trigger : callable, optional
        Function that takes episode number and returns True if video should be recorded
        Default: record every episode
    name_prefix : str, optional
        Prefix for video filenames (default: "rl-video")
        
    Returns
    -------
    tuple
        (wrapped_environment, output_directory_path)
        
    Example
    -------
    >>> env = DogfightAviary(num_drones=2, gui=False)
    >>> env, output_dir = Monitor(env)
    >>> # Run episodes... automatically saves videos and statistics
    """
    # Set up output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = os.path.join("runs", timestamp)

    os.makedirs(output_dir, exist_ok=True)

    # Ensure the base environment has camera setup for video recording
    if record_video and hasattr(env, 'RECORD'):
        if not env.RECORD:
            warnings.warn(
                "Video recording requested but environment was not initialized with record=True. "
                "For best video quality, initialize your environment with record=True. "
                "Falling back to wrapper-based camera setup."
            )

    # Set up video trigger
    if video_trigger is None:
        video_trigger = lambda episode_id: True  # Record every episode

    wrapped_env = env

    # Add episode statistics recording
    if record_stats:
        wrapped_env = RecordEpisodeStatistics(wrapped_env)

    # Add render mode wrapper if needed for video recording
    if record_video:
        # Check if environment has proper render mode
        if not hasattr(wrapped_env, 'render_mode') or wrapped_env.render_mode != "rgb_array":
            wrapped_env = RenderModeWrapper(wrapped_env, render_mode="rgb_array")

        # Add video recording
        try:
            wrapped_env = RecordVideo(
                wrapped_env,
                video_folder=output_dir,
                episode_trigger=video_trigger,
                name_prefix=name_prefix
            )
        except Exception as e:
            warnings.warn(f"Failed to set up video recording: {e}")

    return wrapped_env, output_dir


def save_episode_statistics(env, output_dir, filename="progress.csv"):
    """
    Save episode statistics from a monitored environment to CSV.
    
    Parameters
    ----------
    env : gym.Env
        Environment wrapped with RecordEpisodeStatistics
    output_dir : str
        Directory to save the CSV file
    filename : str
        Name of the CSV file (default: "progress.csv")
        
    Returns
    -------
    str or None
        Path to saved CSV file if successful, None if no statistics available
    """
    # Find the RecordEpisodeStatistics wrapper
    current_env = env
    stats_wrapper = None

    while hasattr(current_env, 'env'):
        if isinstance(current_env, RecordEpisodeStatistics):
            stats_wrapper = current_env
            break
        current_env = current_env.env

    if stats_wrapper is None:
        warnings.warn(
            "No RecordEpisodeStatistics wrapper found. Make sure environment is wrapped with RecordEpisodeStatistics.")
        return None

    # Handle different formats of episode_returns (might be float, list, or deque)
    try:
        episode_returns = stats_wrapper.episode_returns
        episode_lengths = stats_wrapper.episode_lengths

        # Convert to list if needed
        if hasattr(episode_returns, '__len__'):
            if len(episode_returns) == 0:
                warnings.warn("No episodes completed yet. Make sure to run some episodes before saving statistics.")
                return None
            episode_returns_list = list(episode_returns)
            episode_lengths_list = list(episode_lengths)
        else:
            # Single value (current episode in progress)
            episode_returns_list = [float(episode_returns)]
            episode_lengths_list = [int(episode_lengths)]

    except AttributeError:
        warnings.warn("Episode statistics not available. Make sure episodes have completed.")
        return None

    try:
        import pandas as pd
    except ImportError:
        warnings.warn("pandas is required to save statistics. Please install with: pip install pandas")
        return None

    # Create DataFrame with episode statistics
    stats_data = {
        'episode': list(range(len(episode_returns_list))),
        'episode_return': episode_returns_list,
        'episode_length': episode_lengths_list,
    }

    # Add episode times if available
    if hasattr(stats_wrapper, 'episode_times') and stats_wrapper.episode_times:
        try:
            episode_times = list(stats_wrapper.episode_times)
            if len(episode_times) == len(episode_returns_list):
                stats_data['episode_time'] = episode_times
        except (AttributeError, TypeError):
            pass  # Episode times not available or incompatible

    df = pd.DataFrame(stats_data)

    # Save to CSV
    csv_path = os.path.join(output_dir, filename)
    df.to_csv(csv_path, index=False)

    return csv_path


def list_monitoring_files(output_dir):
    """
    List all monitoring files in the output directory.
    
    Parameters
    ----------
    output_dir : str
        Directory to scan for monitoring files
        
    Returns
    -------
    tuple
        Tuple of (video_files, log_files) where each is a list of filenames
    """
    output_path = Path(output_dir)

    if not output_path.exists():
        return [], []

    video_files = []
    log_files = []

    for file_path in output_path.iterdir():
        if file_path.is_file():
            if file_path.suffix == '.mp4':
                video_files.append(file_path.name)
            elif file_path.suffix == '.csv':
                log_files.append(file_path.name)

    return video_files, log_files


# Compatibility alias
def create_monitored_env(*args, **kwargs):
    """
    Legacy alias for Monitor function.
    
    Deprecated: Use Monitor() instead.
    """
    warnings.warn(
        "create_monitored_env is deprecated. Use Monitor() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return Monitor(*args, **kwargs)
