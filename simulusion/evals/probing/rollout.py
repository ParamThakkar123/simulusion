"""Whether the true state survives a world model's imagination.

An encoder can map frames to a latent that holds the state while the dynamics
model drifts away from it within a few steps. :func:`rollout_probe` checks the
latter: it encodes real frames up to ``t0``, steps the model forward on the
recorded actions, and reads each imagined state with a probe fitted on encoded
latents. The R² at each horizon, next to the R² of the encoded latents on the
same frames, shows how far ahead the physical variables stay intact.

.. code-block:: python

    probe = fit_linear_probe(z_train, y_train, z_val, y_val)   # on "deter"
    scores = rollout_probe(wm, test, probe, key="deter", targets=test.targets())
    scores.r2           # (len(horizons), K)
    scores.encoded_r2   # the same frames, read from real observations
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol, Sequence

import numpy as np
import torch

from simulusion.evals.probing.data import ProbeDataset, StateTargets
from simulusion.evals.probing.latents import (
    FeatureFn,
    _batched,
    _encode_steps,
    latent_features,
)
from simulusion.evals.probing.metrics import nanmean, r2_score, scorable


@dataclass(frozen=True)
class RolloutScores:
    """Probe R² on imagined states, per horizon.

    Attributes:
        horizons: Steps imagined past the last real frame.
        names: The ``K`` target names.
        r2: ``(len(horizons), K)`` R² of the probe on imagined states.
        encoded_r2: ``(len(horizons), K)`` R² on the encoded latents of the
            same frames: the ceiling the imagined states are measured against.
        rollouts: How many imagined trajectories each horizon averages over.
    """

    horizons: tuple[int, ...]
    names: tuple[str, ...]
    r2: np.ndarray
    encoded_r2: np.ndarray
    rollouts: int

    def mean(self) -> np.ndarray:
        """``(len(horizons),)`` R² averaged over the scorable targets."""
        return nanmean(self.r2, axis=1)


class _Probe(Protocol):
    """Anything with ``predict(z) -> y``: a fitted linear or MLP probe."""

    def predict(self, z: np.ndarray) -> np.ndarray: ...


def rollout_probe(
    model: Any,
    data: ProbeDataset,
    probe: _Probe,
    *,
    key: str,
    targets: StateTargets,
    horizons: Sequence[int] = (1, 5, 15),
    starts_every: int = 10,
    context: Optional[int] = None,
    action_fn: Optional[Callable[[np.ndarray], Any]] = None,
    batch_size: int = 64,
    seed: int = 0,
    features: FeatureFn = latent_features,
) -> RolloutScores:
    """Read imagined states with a probe fitted on encoded latents.

    Args:
        model: A :class:`~simulusion.world_model.WorldModel`.
        data: Held-out episodes; rollouts start every ``starts_every`` frames.
        probe: Fitted on the ``key`` feature of encoded latents.
        key: The feature to read, as named by :func:`extract_latents`.
        targets: ``data.targets(...)`` with the columns the probe predicts.
        horizons: Imagined steps at which to score.
        context: Frames of history before each start, as for ``extract_latents``.
        action_fn: Map DMC actions to the model's; needed for discrete-action
            models, which cannot imagine without actions.
        batch_size: Rollouts imagined together.
        seed: Seeds the model's sampling.
    """
    horizons = tuple(sorted({int(h) for h in horizons}))
    if not horizons or horizons[0] < 1:
        raise ValueError("horizons must be positive step counts.")
    spec = model.spec
    if action_fn is None and spec.action.kind == "discrete":
        raise ValueError(
            f"{spec.family} takes discrete actions; pass action_fn to map DMC's "
            "continuous actions, since imagination needs them."
        )
    H = horizons[-1]
    window = context or spec.max_context or 1
    first = max(spec.context_frames, window) - 1
    lookup = _RowLookup(targets)
    torch.manual_seed(seed)

    imagined: Dict[int, list[np.ndarray]] = {h: [] for h in horizons}
    encoded: Dict[int, list[np.ndarray]] = {h: [] for h in horizons}
    truth: Dict[int, list[np.ndarray]] = {h: [] for h in horizons}
    rollouts = 0
    for index, episode in enumerate(data):
        T = len(episode)
        starts = list(range(first, T - H, starts_every))
        if not starts or episode.actions is None:
            continue
        raw = action_fn(episode.actions) if action_fn else episode.actions
        acts = _batched(raw)
        states = _encode_steps(model, episode.frames[None], acts, context, 256)
        for i in range(0, len(starts), batch_size):
            batch = starts[i : i + batch_size]
            state = {
                k: torch.cat([states[t][k] for t in batch]) for k in states[batch[0]]
            }
            plan = torch.stack([acts[0, t : t + H] for t in batch])
            for h in range(1, H + 1):
                state = model.step(state, plan[:, h - 1]).state
                if h not in imagined:
                    continue
                steps = [t + h for t in batch]
                imagined[h].append(probe.predict(_feature(model, state, key, features)))
                real = {
                    k: torch.cat([states[t][k] for t in steps])
                    for k in states[steps[0]]
                }
                encoded[h].append(probe.predict(_feature(model, real, key, features)))
                truth[h].append(targets.values[lookup(index, steps)])
            rollouts += len(batch)
    if not rollouts:
        raise ValueError(
            f"No episode is long enough for a {H}-step rollout after "
            f"{first + 1} context frames."
        )
    # Targets that barely move over the scored frames get no R² (as in
    # score_latents), measured against the spread the probe was fitted on.
    reference = getattr(getattr(probe, "y_scale", None), "std", None)
    if reference is None:
        reference = targets.values.std(axis=0)

    def scores(predictions: Dict[int, list[np.ndarray]]) -> np.ndarray:
        rows = []
        for h in horizons:
            y = np.concatenate(truth[h])
            r2 = r2_score(y, np.concatenate(predictions[h]))
            rows.append(np.where(scorable(y, reference), r2, np.nan))
        return np.stack(rows)

    return RolloutScores(
        horizons=horizons,
        names=targets.names,
        r2=scores(imagined),
        encoded_r2=scores(encoded),
        rollouts=rollouts,
    )


def _feature(
    model: Any, state: Dict[str, torch.Tensor], key: str, features: FeatureFn
) -> np.ndarray:
    with torch.no_grad():
        values = features(model, state)
    if key not in values:
        raise ValueError(f"No feature {key!r}; this model gives {sorted(values)}.")
    return values[key].detach().cpu().numpy()


class _RowLookup:
    """Row of ``targets`` for an (episode, step) pair."""

    def __init__(self, targets: StateTargets) -> None:
        self.width = int(targets.step.max(initial=0)) + 1
        keys = targets.episode * self.width + targets.step
        self.order = np.argsort(keys, kind="stable")
        self.keys = keys[self.order]

    def __call__(self, episode: int, steps: Sequence[int]) -> np.ndarray:
        wanted = episode * self.width + np.asarray(steps)
        found = np.searchsorted(self.keys, wanted)
        if np.any(found >= len(self.keys)) or not np.array_equal(
            self.keys[np.minimum(found, len(self.keys) - 1)], wanted
        ):
            raise ValueError("targets do not cover these frames; build them from data.")
        return self.order[found]
