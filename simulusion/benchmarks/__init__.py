"""
Benchmarks sub-module - Benchmark runners and adapters for world models.

This package defines the benchmark suites every model is evaluated on (task
lists, step budgets, environment options and Atari reference scores, in
``suites.py``) and provides tools for running standardized evaluations of
world models (Dreamer, IRIS, DIAMOND) across multiple seeds and computing
aggregate metrics.

Usage:
    from simulusion.benchmarks import get_benchmark, human_normalized_score
    fields = get_benchmark("atari100k").env_fields("pong")

    from simulusion.benchmarks import BenchmarkRunner, DiamondAdapter
    runner = BenchmarkRunner(adapter_cls=DiamondAdapter, ...)
"""

from importlib import import_module
from typing import Any


def __getattr__(name: str) -> Any:
    _lazy = {
        "ATARI100K_GAMES": ".suites",
        "ATARI57_GAMES": ".suites",
        "ATARI_HUMAN_SCORES": ".suites",
        "ATARI_RANDOM_SCORES": ".suites",
        "BENCHMARKS": ".suites",
        "BSUITE_ENVIRONMENTS": ".suites",
        "Benchmark": ".suites",
        "DMC_PROPRIO_TASKS": ".suites",
        "DMC_VISUAL_TASKS": ".suites",
        "DMLAB30_LEVELS": ".suites",
        "MINECRAFT_OBS_KEYS": ".suites",
        "PROCGEN_GAMES": ".suites",
        "atari_env_id": ".suites",
        "atari_game": ".suites",
        "bsuite_settings": ".suites",
        "get_benchmark": ".suites",
        "human_normalized_score": ".suites",
        "BenchmarkRunner": ".runner",
        "MultiAgentBenchmarkRunner": ".runner",
        "BaseAdapter": ".adapters",
        "DreamerAdapter": ".adapters",
        "IRISAdapter": ".adapters",
        "DiamondAdapter": ".adapters",
        "compute_aggregate_metrics": ".metrics",
        "bootstrap_ci": ".metrics",
        "iqm_of_array": ".metrics",
    }
    if name not in _lazy:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_lazy[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = [
    "ATARI100K_GAMES",
    "ATARI57_GAMES",
    "ATARI_HUMAN_SCORES",
    "ATARI_RANDOM_SCORES",
    "BENCHMARKS",
    "BSUITE_ENVIRONMENTS",
    "Benchmark",
    "DMC_PROPRIO_TASKS",
    "DMC_VISUAL_TASKS",
    "DMLAB30_LEVELS",
    "MINECRAFT_OBS_KEYS",
    "PROCGEN_GAMES",
    "atari_env_id",
    "atari_game",
    "bsuite_settings",
    "get_benchmark",
    "human_normalized_score",
    "BenchmarkRunner",
    "MultiAgentBenchmarkRunner",
    "BaseAdapter",
    "DreamerAdapter",
    "IRISAdapter",
    "DiamondAdapter",
    "compute_aggregate_metrics",
    "bootstrap_ci",
    "iqm_of_array",
]
