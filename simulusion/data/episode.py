"""The episode: Simulusion's one trajectory format, for video and for RL.

An :class:`Episode` with ``T`` frames has ``T - 1`` transitions, the same
convention as the :class:`~simulusion.world_model.WorldModel` protocol:
``actions[t]`` is the action taken at ``frames[t]``, which led to
``frames[t + 1]`` and earned ``rewards[t]``; ``terminated[t]`` says
``frames[t + 1]`` ended the episode, ``truncated[t]`` that it was cut off
(a time limit) without ending. Everything but ``frames`` is optional, so a
plain video is an episode too.

On disk an episode is one ``.npz`` file:

=====================  ==========================================  ===========
Key                    Contents                                    Required
=====================  ==========================================  ===========
``frames``             ``(T, H, W, C)`` uint8, the main camera     yes
``actions``            ``(T-1,)`` int (discrete) or ``(T-1, A)``   no
``rewards``            ``(T-1,)`` float                            no
``terminated``         ``(T-1,)`` bool                             no
``truncated``          ``(T-1,)`` bool                             no
``frames.<camera>``    ``(T, H, W, C)`` uint8, further cameras     no
``obs.<name>``         ``(T, ...)`` low-dimensional observations   no
                       (robot joint state, proprioception)
``language``           0-d string, a task instruction              no
=====================  ==========================================  ===========

This is the layout :class:`~simulusion.envs.recording.RecordingWrapper`
writes, extended for robot data (several cameras, proprioception, language).
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

CAMERA_PREFIX = "frames."
OBS_PREFIX = "obs."
_TRANSITION_KEYS = ("actions", "rewards", "terminated", "truncated")


@dataclass
class Episode:
    """One trajectory; see the module docstring for the conventions.

    Attributes:
        frames: ``(T, H, W, C)`` uint8 frames from the main camera.
        actions: ``(T - 1, ...)`` actions, or ``None`` for video.
        rewards: ``(T - 1,)`` rewards, or ``None``.
        terminated: ``(T - 1,)`` bool episode ends, or ``None``.
        truncated: ``(T - 1,)`` bool time-limit cuts, or ``None``.
        cameras: Further views, each ``(T, H, W, C)`` uint8.
        observations: Low-dimensional observations, each ``(T, ...)``.
        language: A natural-language task instruction, if any.
    """

    frames: np.ndarray
    actions: Optional[np.ndarray] = None
    rewards: Optional[np.ndarray] = None
    terminated: Optional[np.ndarray] = None
    truncated: Optional[np.ndarray] = None
    cameras: Dict[str, np.ndarray] = field(default_factory=dict)
    observations: Dict[str, np.ndarray] = field(default_factory=dict)
    language: Optional[str] = None

    def __post_init__(self) -> None:
        self.frames = as_uint8_frames(self.frames)
        self.cameras = {k: as_uint8_frames(v) for k, v in self.cameras.items()}
        self.observations = {k: np.asarray(v) for k, v in self.observations.items()}
        if self.rewards is not None:
            self.rewards = np.asarray(self.rewards, dtype=np.float32).reshape(-1)
        for name in ("terminated", "truncated"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, np.asarray(value, dtype=bool).reshape(-1))
        if self.actions is not None:
            self.actions = np.asarray(self.actions)
        self.validate()

    def __len__(self) -> int:
        return int(self.frames.shape[0])

    @property
    def num_transitions(self) -> int:
        return max(len(self) - 1, 0)

    @property
    def has_actions(self) -> bool:
        return self.actions is not None

    def validate(self) -> None:
        """Raise ``ValueError`` if the arrays disagree on the episode's length."""
        T = len(self)
        if T == 0:
            raise ValueError("An episode needs at least one frame.")
        for name in _TRANSITION_KEYS:
            value = getattr(self, name)
            if value is not None and value.shape[0] != T - 1:
                raise ValueError(
                    f"{name} has {value.shape[0]} entries for {T} frames; "
                    f"expected T - 1 = {T - 1} (one per transition)."
                )
        for kind, mapping in (
            ("camera", self.cameras),
            ("observation", self.observations),
        ):
            for key, value in mapping.items():
                if value.shape[0] != T:
                    raise ValueError(
                        f"{kind} '{key}' has {value.shape[0]} entries for {T} frames."
                    )

    def slice(self, start: int, stop: int) -> "Episode":
        """Frames ``start:stop`` and the transitions between them."""

        def tr(value: Optional[np.ndarray]) -> Optional[np.ndarray]:
            return None if value is None else value[start : max(stop - 1, start)]

        return Episode(
            frames=self.frames[start:stop],
            actions=tr(self.actions),
            rewards=tr(self.rewards),
            terminated=tr(self.terminated),
            truncated=tr(self.truncated),
            cameras={k: v[start:stop] for k, v in self.cameras.items()},
            observations={k: v[start:stop] for k, v in self.observations.items()},
            language=self.language,
        )

    def to_arrays(self) -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {"frames": self.frames}
        for name in _TRANSITION_KEYS:
            value = getattr(self, name)
            if value is not None:
                arrays[name] = value
        for key, value in self.cameras.items():
            arrays[CAMERA_PREFIX + key] = value
        for key, value in self.observations.items():
            arrays[OBS_PREFIX + key] = value
        if self.language is not None:
            arrays["language"] = np.array(self.language)
        return arrays

    def save(self, path: str | Path, *, compress: bool = True) -> Path:
        """Write the episode as ``.npz``; returns the path written."""
        out = Path(path)
        if out.suffix != ".npz":
            out = out.with_suffix(".npz")
        out.parent.mkdir(parents=True, exist_ok=True)
        writer = np.savez_compressed if compress else np.savez
        # Write through a temporary name so a reader never sees half a file.
        tmp = out.with_name(out.name + ".tmp")
        with open(tmp, "wb") as handle:
            writer(handle, **self.to_arrays())  # type: ignore[arg-type]
        tmp.replace(out)
        return out

    @classmethod
    def from_arrays(cls, arrays: Any) -> "Episode":
        keys = list(arrays.keys())
        frame_key = next(
            (k for k in ("frames", "obs", "observations") if k in keys), None
        )
        if frame_key is None:
            raise ValueError(f"No 'frames' array among {keys}.")
        language = arrays["language"] if "language" in keys else None
        frames = arrays[frame_key]
        T = len(frames)

        def transitions(*names: str) -> Optional[np.ndarray]:
            # Per-step layouts (the World Models rollouts, RLDS) store one
            # entry per frame: the last leads past the final frame and is
            # dropped, so every array has one entry per transition.
            name = next((n for n in names if n in keys), None)
            if name is None:
                return None
            value = np.asarray(arrays[name])
            return value[: T - 1] if len(value) == T else value

        return cls(
            frames=frames,
            actions=transitions("actions"),
            rewards=transitions("rewards"),
            # World Models rollouts call the episode ends `terminals`.
            terminated=_ends(arrays, keys, T, ("terminated", "terminals")),
            truncated=transitions("truncated"),
            cameras={
                k[len(CAMERA_PREFIX) :]: arrays[k]
                for k in keys
                if k.startswith(CAMERA_PREFIX)
            },
            observations={
                k[len(OBS_PREFIX) :]: arrays[k]
                for k in keys
                if k.startswith(OBS_PREFIX)
            },
            language=None if language is None else str(np.asarray(language)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Episode":
        with np.load(path, allow_pickle=False) as arrays:
            return cls.from_arrays(arrays)


def _ends(arrays: Any, keys: list, T: int, names: tuple) -> Optional[np.ndarray]:
    """Episode-end flags per transition; a per-step array's final flag (the
    step that ended the episode) moves onto the last transition kept."""
    name = next((n for n in names if n in keys), None)
    if name is None:
        return None
    value = np.asarray(arrays[name]).astype(bool).reshape(-1)
    if len(value) == T and T > 1:
        ends = value[: T - 1].copy()
        ends[-1] = ends[-1] or bool(value[-1])
        return ends
    return value


def npz_length(path: str | Path) -> int:
    """The number of frames in an episode file, read from its header only."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        name = next(
            (
                f"{k}.npy"
                for k in ("frames", "obs", "observations")
                if f"{k}.npy" in names
            ),
            None,
        )
        if name is None:
            raise ValueError(f"{path} holds no 'frames' array.")
        with archive.open(name) as handle:
            version = np.lib.format.read_magic(handle)  # type: ignore[no-untyped-call]
            if version == (1, 0):
                shape, _, _ = np.lib.format.read_array_header_1_0(handle)  # type: ignore[no-untyped-call]
            else:
                shape, _, _ = np.lib.format.read_array_header_2_0(handle)  # type: ignore[no-untyped-call]
    return int(shape[0])


def as_uint8_frames(frames: Any) -> np.ndarray:
    """Frames as ``(T, H, W, C)`` uint8, from uint8 or ``[0, 1]`` floats, HWC or CHW."""
    array = np.asarray(frames)
    if array.ndim == 3:  # (T, H, W) grayscale
        array = array[..., None]
    if array.ndim != 4:
        raise ValueError(f"Expected frames shaped (T, H, W, C); got {array.shape}.")
    if array.shape[-1] not in (1, 3, 4) and array.shape[1] in (1, 3, 4):
        array = np.transpose(array, (0, 2, 3, 1))
    if array.dtype != np.uint8:
        array = array.astype(np.float32)
        if array.size and float(array.max()) <= 1.0:
            array = array * 255.0
        array = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    if array.shape[-1] == 4:
        array = array[..., :3]
    return np.ascontiguousarray(array)


__all__ = ["CAMERA_PREFIX", "OBS_PREFIX", "Episode", "as_uint8_frames", "npz_length"]
