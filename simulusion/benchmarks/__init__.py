"""
Benchmarks sub-module - Benchmark runners and adapters for world models.

This package provides tools for running standardized evaluations of world
models (Dreamer, IRIS, DIAMOND) across multiple seeds and computing aggregate
metrics.

Usage:
    from simulusion.benchmarks import BenchmarkRunner, DiamondAdapter
    runner = BenchmarkRunner(adapter_cls=DiamondAdapter, ...)
"""

from importlib import import_module
from typing import Any


def __getattr__(name: str) -> Any:
    _lazy = {
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
