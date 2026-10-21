"""Convert RLDS episodes into Simulusion episodes.

`RLDS`_ (the format of Open X-Embodiment and many other robot datasets)
stores each episode as a sequence of *steps*, every step a dict with an
``observation`` (itself a dict of cameras and state), an ``action``, a
``reward`` and the flags ``is_first`` / ``is_last`` / ``is_terminal``.

:func:`episode_from_rlds` turns one such episode, given as plain Python and
NumPy values, into an :class:`~simulusion.data.episode.Episode`; reading the
dataset files themselves is left to whatever produced the steps.

RLDS puts the action *in the step it is taken from*, so a ``T``-step episode
has ``T`` actions; the last one leads past the final observation and is
dropped, as are its reward and flags. ``is_terminal`` on the final step marks
the last transition terminated; an episode that ends without it is
``truncated``.

.. _RLDS: https://github.com/google-research/rlds
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from simulusion.data.episode import Episode


@dataclass(frozen=True)
class RLDSKeys:
    """Where the parts of an episode live inside an RLDS step.

    Attributes:
        image: Observation key of the main camera.
        cameras: Further observation keys to keep as cameras.
        state: Observation keys of low-dimensional state to keep.
        action: Action keys, in order, when ``action`` is a dict (the RT-1
            format); they are flattened and concatenated.
            Empty when ``action`` is already an array.
        language: Step or observation key of the language instruction.
    """

    image: str = "image"
    cameras: Sequence[str] = ()
    state: Sequence[str] = ()
    action: Sequence[str] = ()
    language: str = "language_instruction"


def episode_from_rlds(
    episode: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    keys: RLDSKeys = RLDSKeys(),
) -> Episode:
    """One RLDS episode (``{"steps": [...]}`` or the steps themselves) as an Episode."""
    steps: List[Mapping[str, Any]] = list(
        episode["steps"] if isinstance(episode, Mapping) else episode
    )
    if not steps:
        raise ValueError("The RLDS episode has no steps.")
    T = len(steps)
    observations = [s["observation"] for s in steps]
    frames = np.stack([_array(o[keys.image]) for o in observations])
    cameras = {c: np.stack([_array(o[c]) for o in observations]) for c in keys.cameras}
    state = {
        k: np.stack([_array(o[k]).astype(np.float32).reshape(-1) for o in observations])
        for k in keys.state
    }
    actions = None
    if T > 1 and "action" in steps[0]:
        actions = np.stack([_action(s["action"], keys.action) for s in steps[: T - 1]])
    rewards = None
    if T > 1 and "reward" in steps[0]:
        rewards = np.array(
            [float(_array(s["reward"])) for s in steps[: T - 1]], np.float32
        )
    terminated = truncated = None
    if T > 1:
        terminated = np.zeros(T - 1, dtype=bool)
        last_terminal = bool(_array(steps[-1].get("is_terminal", False)))
        terminated[-1] = last_terminal
        truncated = np.zeros(T - 1, dtype=bool)
        truncated[-1] = not last_terminal
    return Episode(
        frames=frames,
        actions=actions,
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
        cameras=cameras,
        observations=state,
        language=_language(steps[0], keys.language),
    )


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _action(action: Any, keys: Sequence[str]) -> np.ndarray:
    if isinstance(action, Mapping):
        names = list(keys) or sorted(action)
        return np.concatenate(
            [_array(action[k]).astype(np.float32).reshape(-1) for k in names]
        )
    return _array(action).astype(np.float32).reshape(-1)


def _language(step: Mapping[str, Any], key: str) -> Optional[str]:
    for source in (step, step.get("observation", {})):
        if isinstance(source, Mapping) and key in source:
            value = _array(source[key])
            if value.ndim != 0:
                return None
            item = value.item()
            if isinstance(item, bytes):
                return item.decode("utf-8", "replace")
            return str(item)
    return None


__all__ = ["RLDSKeys", "episode_from_rlds"]
