"""
Utilities for gym-pybullet-drones.

This module provides utility functions and classes for working with
gym-pybullet-drones environments.
"""

# Make Monitor function available at package level for easy import
try:
    from .monitor import Monitor, save_episode_statistics, list_monitoring_files
    __all__ = ['Monitor', 'save_episode_statistics', 'list_monitoring_files']
except ImportError:
    # gymnasium not available - monitoring functions won't work
    __all__ = []
