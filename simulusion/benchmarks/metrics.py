from __future__ import annotations

from typing import Iterable, List, Dict

import numpy as np


def compute_aggregate_metrics(per_seed_means: Iterable[float]) -> Dict[str, float]:
    arr = np.array(list(per_seed_means), dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "median": 0.0, "iqm": 0.0, "num_seeds": 0}

    mean = float(np.mean(arr))
    median = float(np.median(arr))
    # IQM: interquartile mean (mean of values between 25th and 75th percentiles)
    iqm = float(iqm_of_array(arr))

    return {
        "mean": mean,
        "median": median,
        "iqm": iqm,
        "num_seeds": int(arr.size),
    }


def bootstrap_ci(
    values: List[float], num_samples: int = 1000, alpha: float = 0.05
) -> tuple[float, float]:
    """Compute simple bootstrap 1-alpha CI on the mean."""
    if not values:
        return (0.0, 0.0)
    vals = np.array(values)
    n = vals.size
    means = []
    for _ in range(num_samples):
        sample = np.random.choice(vals, size=n, replace=True)
        means.append(sample.mean())
    lower = float(np.percentile(means, 100 * (alpha / 2)))
    upper = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lower, upper


def iqm_of_array(values: Iterable[float]) -> float:
    """Compute the Interquartile Mean (IQM) of an array of values.

    IQM is the mean of values that lie between the 25th and 75th percentiles
    (inclusive). This is a robust central tendency measure used in RL
    benchmark reporting.
    """
    arr = np.array(list(values), dtype=float)
    if arr.size == 0:
        return 0.0
    lo = float(np.percentile(arr, 25))
    hi = float(np.percentile(arr, 75))
    # Keep values within [lo, hi]
    mask = (arr >= lo) & (arr <= hi)
    if not mask.any():
        # Fallback to simple mean
        return float(arr.mean())
    return float(arr[mask].mean())


def bootstrap_iqm_ci(
    values: List[float], num_samples: int = 1000, alpha: float = 0.05
) -> tuple[float, float]:
    """Bootstrap a confidence interval for the IQM.

    Returns (lower, upper) percentiles of the bootstrap IQM distribution.
    """
    if not values:
        return (0.0, 0.0)
    vals = np.array(values)
    n = vals.size
    iqms = np.empty(num_samples, dtype=float)
    for i in range(num_samples):
        sample = np.random.choice(vals, size=n, replace=True)
        iqms[i] = iqm_of_array(sample)
    lower = float(np.percentile(iqms, 100 * (alpha / 2)))
    upper = float(np.percentile(iqms, 100 * (1 - alpha / 2)))
    return lower, upper
