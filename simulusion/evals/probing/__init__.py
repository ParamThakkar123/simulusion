"""Probe world-model latents for the true state of the simulator that made them.

:mod:`~simulusion.evals.probing.data` records labelled datasets,
:mod:`~simulusion.evals.probing.latents` turns a world model's state for every
frame into features, and :mod:`~simulusion.evals.probing.metrics` scores them
with probes, DCI and MIG.
"""

from simulusion.evals.probing.data import (
    ProbeDataset,
    StateTargets,
    collect_probe_episodes,
    unobservable_dims,
)
from simulusion.evals.probing.metrics import (
    DCIScores,
    LinearProbe,
    MIGScores,
    MLPProbe,
    ProbeScores,
    dci,
    factor_groups,
    fit_linear_probe,
    fit_mlp_probe,
    mig,
    r2_score,
    score_latents,
)
from simulusion.evals.probing.latents import (
    Latents,
    extract_latents,
    latent_features,
    pixel_features,
    reinitialized,
)

__all__ = [
    "DCIScores",
    "Latents",
    "LinearProbe",
    "MIGScores",
    "MLPProbe",
    "ProbeDataset",
    "ProbeScores",
    "StateTargets",
    "collect_probe_episodes",
    "dci",
    "extract_latents",
    "factor_groups",
    "fit_linear_probe",
    "fit_mlp_probe",
    "latent_features",
    "mig",
    "pixel_features",
    "r2_score",
    "reinitialized",
    "score_latents",
    "unobservable_dims",
]
