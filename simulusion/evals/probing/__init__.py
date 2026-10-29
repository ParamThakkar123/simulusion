"""Probe world-model latents for the true state of the simulator that made them.

:mod:`~simulusion.evals.probing.data` records labelled datasets and
:mod:`~simulusion.evals.probing.latents` turns a world model's state for every
frame into features to probe.
"""

from simulusion.evals.probing.data import (
    ProbeDataset,
    StateTargets,
    collect_probe_episodes,
    unobservable_dims,
)
from simulusion.evals.probing.latents import (
    Latents,
    extract_latents,
    latent_features,
    pixel_features,
    reinitialized,
)

__all__ = [
    "Latents",
    "ProbeDataset",
    "StateTargets",
    "collect_probe_episodes",
    "extract_latents",
    "latent_features",
    "pixel_features",
    "reinitialized",
    "unobservable_dims",
]
