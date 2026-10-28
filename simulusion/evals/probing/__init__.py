"""Probe world-model latents for the true state of the simulator that made them.

See :mod:`simulusion.evals.probing.data` for the labelled datasets.
"""

from simulusion.evals.probing.data import (
    ProbeDataset,
    StateTargets,
    collect_probe_episodes,
    unobservable_dims,
)

__all__ = [
    "ProbeDataset",
    "StateTargets",
    "collect_probe_episodes",
    "unobservable_dims",
]
