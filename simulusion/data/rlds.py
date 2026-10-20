"""Read RLDS datasets, including Open X-Embodiment, as Simulusion episodes.

`Open X-Embodiment`_ publishes 60+ robot datasets in the `RLDS`_ format:
each episode is a sequence of *steps*, every step a dict with an
``observation`` (itself a dict of cameras and state), an ``action``, a
``reward`` and the flags ``is_first`` / ``is_last`` / ``is_terminal``.

:func:`episode_from_rlds` turns one such episode into an
:class:`~simulusion.data.episode.Episode`, from plain Python / NumPy values,
so it works on whatever produced the steps. :func:`load_oxe` reads an Open
X-Embodiment dataset with ``tensorflow_datasets`` (installed separately:
``pip install tensorflow tensorflow-datasets``; Simulusion does not depend on
TensorFlow) and yields episodes.

.. code-block:: python

    from simulusion.data import load_oxe, EpisodeWriter

    with EpisodeWriter("data/fractal") as writer:
        for episode in load_oxe("fractal20220817_data", split="train[:50]"):
            writer.write(episode)

RLDS puts the action *in the step it is taken from*, so a ``T``-step episode
has ``T`` actions; the last one leads past the final observation and is
dropped, as are its reward and flags. ``is_terminal`` on the final step marks
the last transition terminated; an episode that ends without it is
``truncated``.

.. _Open X-Embodiment: https://robotics-transformer-x.github.io/
.. _RLDS: https://github.com/google-research/rlds
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

import numpy as np

from simulusion.data.episode import Episode

OXE_GCS_ROOT = "gs://gresearch/robotics"


@dataclass(frozen=True)
class RLDSKeys:
    """Where the parts of an episode live inside an RLDS step.

    Attributes:
        image: Observation key of the main camera.
        cameras: Further observation keys to keep as cameras.
        state: Observation keys of low-dimensional state to keep.
        action: Action keys, in order, when ``action`` is a dict (Open
            X-Embodiment's RT-1 format); they are flattened and concatenated.
            Empty when ``action`` is already an array.
        language: Step or observation key of the language instruction.
    """

    image: str = "image"
    cameras: Sequence[str] = ()
    state: Sequence[str] = ()
    action: Sequence[str] = ()
    language: str = "language_instruction"


#: Key layouts for commonly used Open X-Embodiment datasets.
OXE_PRESETS: Dict[str, RLDSKeys] = {
    "fractal20220817_data": RLDSKeys(
        image="image",
        state=("base_pose_tool_reached",),
        action=("world_vector", "rotation_delta", "gripper_closedness_action"),
        language="natural_language_instruction",
    ),
    "bridge": RLDSKeys(
        image="image", state=("state",), language="language_instruction"
    ),
    "kuka": RLDSKeys(
        image="image",
        action=("world_vector", "rotation_delta", "gripper_closedness_action"),
        language="natural_language_instruction",
    ),
    "taco_play": RLDSKeys(
        image="rgb_static",
        cameras=("rgb_gripper",),
        action=("rel_actions_world",),
        language="natural_language_instruction",
    ),
    "berkeley_autolab_ur5": RLDSKeys(
        image="image",
        cameras=("hand_image",),
        state=("robot_state",),
        action=("world_vector", "rotation_delta", "gripper_closedness_action"),
        language="natural_language_instruction",
    ),
    "language_table": RLDSKeys(
        image="rgb", state=("effector_translation",), language="instruction"
    ),
}

#: Dataset versions on the Open X-Embodiment bucket.
OXE_VERSIONS: Dict[str, str] = {
    "fractal20220817_data": "0.1.0",
    "bridge": "0.1.0",
    "kuka": "0.1.0",
    "taco_play": "0.1.0",
    "berkeley_autolab_ur5": "0.1.0",
    "language_table": "0.0.1",
}


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


def load_oxe(
    name: str,
    *,
    split: str = "train",
    data_dir: str = OXE_GCS_ROOT,
    version: Optional[str] = None,
    keys: Optional[RLDSKeys] = None,
) -> Iterator[Episode]:
    """Yield the episodes of an Open X-Embodiment (or any RLDS) dataset.

    Args:
        name: Dataset name (``"fractal20220817_data"``, ``"bridge"``, ...).
        split: A TFDS split, which may slice (``"train[:100]"``).
        data_dir: Root holding ``<name>/<version>``; the public bucket by
            default, or a local download.
        version: Dataset version; the known one for presets, else required.
        keys: Where things live in a step; :data:`OXE_PRESETS` for known
            datasets, else the RLDS defaults.
    """
    try:
        import tensorflow_datasets as tfds
    except ImportError as exc:
        raise ImportError(
            "Reading RLDS / Open X-Embodiment needs TensorFlow Datasets: "
            "pip install tensorflow tensorflow-datasets"
        ) from exc
    version = version or OXE_VERSIONS.get(name)
    if version is None:
        raise ValueError(f"No known version for '{name}'; pass version=...")
    builder = tfds.builder_from_directory(f"{data_dir.rstrip('/')}/{name}/{version}")
    layout = keys or OXE_PRESETS.get(name, RLDSKeys())
    for episode in tfds.as_numpy(builder.as_dataset(split=split)):
        yield episode_from_rlds(episode, layout)


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


__all__ = [
    "OXE_GCS_ROOT",
    "OXE_PRESETS",
    "OXE_VERSIONS",
    "RLDSKeys",
    "episode_from_rlds",
    "load_oxe",
]
