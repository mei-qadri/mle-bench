"""
Plugin Library

Pre-built plugins for common ML engineering tasks.

Note: Plugins are imported lazily to avoid dependency issues.
Import them directly when needed:
    from agents.planning_system.plugins.data_exploration import create_data_exploration_plugin
"""

def create_data_exploration_plugin():
    """Lazy import of data exploration plugin"""
    from agents.planning_system.plugins.data_exploration import create_data_exploration_plugin as _create
    return _create()

def create_ml_training_plugin():
    """Lazy import of ML training plugin"""
    from agents.planning_system.plugins.ml_training import create_ml_training_plugin as _create
    return _create()

def create_submission_validation_plugin():
    """Lazy import of submission validation plugin"""
    from agents.planning_system.plugins.submission_validation import create_submission_validation_plugin as _create
    return _create()

__all__ = [
    "create_data_exploration_plugin",
    "create_ml_training_plugin",
    "create_submission_validation_plugin",
]
