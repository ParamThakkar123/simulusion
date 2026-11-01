"""
Utils sub-module - Utility functions and classes.

Exported Components:
    - Logger: Experiment logger for scalars and GIF rollouts
    - FreezeParameters: Context manager to disable gradients
    - get_parameters: Extract parameters from modules
    - compute_return: Compute returns for value estimation
    - preprocess_obs: Preprocess observations for Dreamer
"""

__all__ = [
    "Logger",
    "FreezeParameters",
    "get_parameters",
    "compute_return",
    "preprocess_obs",
    "MetricsLogger",
    "assert_finite",
    "assert_finite_values",
    "collect_system_stats",
    "get_package_logger",
    "setup_logging",
]


from typing import Any


def __getattr__(name: str) -> Any:
    if name in ("Logger", "FreezeParameters", "get_parameters", "compute_return"):
        from . import dreamer_utils

        return getattr(dreamer_utils, name)
    if name in (
        "MetricsLogger",
        "assert_finite",
        "assert_finite_values",
        "collect_system_stats",
        "get_package_logger",
        "setup_logging",
    ):
        from . import logging_utils

        return getattr(logging_utils, name)
    if name == "preprocess_obs":
        from simulusion.models.dreamer import preprocess_obs

        return preprocess_obs

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
