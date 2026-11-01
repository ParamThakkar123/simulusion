"""Frame preprocessing for game data: crop, resize, grayscale, frame skip.

Game footage arrives at its native resolution and frame rate (Genesis 320 x
224 at 60 fps, a Terraria capture 1920 x 1080), while world models train on
small frames a few steps apart. A :class:`FramePipeline` describes that
reduction once, so recordings, replays and videos of the same game are
prepared the same way, and is saved in the dataset's ``meta.json``.

.. code-block:: python

    from simulusion.data import FramePipeline

    pipeline = FramePipeline(crop=(0, 16, 0, 0), size=(64, 64), frame_skip=4)
    small = pipeline.apply(episode)          # an Episode, 1/4 of the frames

Frame skip keeps every ``frame_skip``-th frame. The transitions between two
kept frames merge into one: rewards add up, an episode end anywhere in them
ends the merged transition, and actions reduce as ``action_reduce`` says:
``"first"`` keeps the action taken at the kept frame (exact for data recorded
with action repeat), ``"any"`` keeps a button pressed on any of the merged
frames (for human play, where a short press would otherwise vanish), and
``"mean"`` averages them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from simulusion.data.episode import Episode

ACTION_REDUCTIONS = ("first", "any", "mean")


@dataclass
class FramePipeline:
    """How raw game frames become training frames.

    Attributes:
        crop: Pixels to cut from the ``(top, bottom, left, right)`` edges
            before resizing (a score bar, letterboxing); ``None`` keeps all.
        size: Resize to ``(height, width)``; ``None`` keeps the cropped size.
        grayscale: Store one luminance channel instead of RGB.
        frame_skip: Keep every ``frame_skip``-th frame.
        action_reduce: How the actions of merged transitions combine:
            ``"first"``, ``"any"`` or ``"mean"`` (see the module docstring).
    """

    crop: Optional[Tuple[int, int, int, int]] = None
    size: Optional[Tuple[int, int]] = None
    grayscale: bool = False
    frame_skip: int = 1
    action_reduce: str = "first"

    def __post_init__(self) -> None:
        if self.crop is not None:
            edges = tuple(int(v) for v in self.crop)
            if len(edges) != 4 or min(edges) < 0:
                raise ValueError(
                    "crop takes four non-negative edges: (top, bottom, left, right)."
                )
            self.crop = edges
        if self.size is not None:
            self.size = (int(self.size[0]), int(self.size[1]))
            if min(self.size) < 1:
                raise ValueError(f"size must be positive; got {self.size}.")
        if self.frame_skip < 1:
            raise ValueError("frame_skip must be at least 1.")
        if self.action_reduce not in ACTION_REDUCTIONS:
            raise ValueError(
                f"action_reduce must be one of {ACTION_REDUCTIONS}; "
                f"got '{self.action_reduce}'."
            )

    @property
    def is_identity(self) -> bool:
        return (
            self.crop is None
            and self.size is None
            and not self.grayscale
            and self.frame_skip == 1
        )

    def frame(self, frame: np.ndarray) -> np.ndarray:
        """One ``(H, W, C)`` uint8 frame, cropped, resized and converted."""
        return self.frames(np.asarray(frame)[None])[0]

    def frames(self, frames: np.ndarray) -> np.ndarray:
        """``(T, H, W, C)`` uint8 frames, cropped, resized and converted
        (frame skip is not applied here; see :meth:`apply`)."""
        frames = np.asarray(frames)
        if self.crop is not None:
            top, bottom, left, right = self.crop
            H, W = frames.shape[1:3]
            if top + bottom >= H or left + right >= W:
                raise ValueError(
                    f"crop {self.crop} leaves nothing of a {H} x {W} frame."
                )
            frames = frames[:, top : H - bottom, left : W - right]
        if self.size is not None and tuple(frames.shape[1:3]) != self.size:
            frames = np.stack([_resize(f, self.size) for f in frames])
        if self.grayscale and frames.shape[-1] == 3:
            luma = frames.astype(np.float32) @ np.array(
                [0.299, 0.587, 0.114], np.float32
            )
            frames = np.clip(np.rint(luma), 0, 255).astype(np.uint8)[..., None]
        return np.ascontiguousarray(frames)

    def apply(self, episode: Episode) -> Episode:
        """The episode with every frame processed and ``frame_skip`` applied."""
        if self.is_identity:
            return episode
        keep = np.arange(0, len(episode), self.frame_skip)
        return Episode(
            frames=self.frames(episode.frames[keep]),
            actions=self._merge(episode.actions, keep, self.action_reduce),
            rewards=self._merge(episode.rewards, keep, "sum"),
            terminated=self._merge(episode.terminated, keep, "any"),
            truncated=self._merge(episode.truncated, keep, "any"),
            cameras={k: self.frames(v[keep]) for k, v in episode.cameras.items()},
            observations={k: v[keep] for k, v in episode.observations.items()},
            language=episode.language,
        )

    def to_json(self) -> Dict[str, Any]:
        values = asdict(self)
        for key in ("crop", "size"):
            if values[key] is not None:
                values[key] = list(values[key])
        return values

    @classmethod
    def from_json(cls, values: Dict[str, Any]) -> "FramePipeline":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in values.items() if k in known})

    def _merge(
        self, values: Optional[np.ndarray], keep: np.ndarray, how: str
    ) -> Optional[np.ndarray]:
        """Transitions ``keep[i] .. keep[i + 1]`` combined into one each."""
        if values is None:
            return None
        if self.frame_skip == 1:
            return values
        groups = [values[a:b] for a, b in zip(keep[:-1], keep[1:])]
        if not groups:
            return values[:0]
        if how == "first":
            return np.stack([g[0] for g in groups])
        if how == "sum":
            return np.array([g.sum() for g in groups], dtype=values.dtype)
        if how == "any":
            return np.stack([np.any(g, axis=0) for g in groups]).astype(values.dtype)
        return np.stack([g.mean(axis=0) for g in groups]).astype(values.dtype)


def _resize(frame: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize one uint8 frame to ``(height, width)``: box filter down (keeps
    thin sprites visible), bilinear up."""
    from PIL import Image

    H, W = size
    squeeze = frame.shape[-1] == 1
    image = Image.fromarray(frame[..., 0] if squeeze else frame)
    shrinking = H <= frame.shape[0] and W <= frame.shape[1]
    resample = Image.Resampling.BOX if shrinking else Image.Resampling.BILINEAR
    out = np.asarray(image.resize((W, H), resample=resample))
    return out[..., None] if squeeze else out


__all__ = ["ACTION_REDUCTIONS", "FramePipeline"]
