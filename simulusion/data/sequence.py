"""Fixed-length windows over episodes, as a PyTorch dataset.

World models train on short clips cut from longer episodes. A
:class:`SequenceDataset` indexes every window of ``sequence_length`` frames
that fits inside an episode and returns it as a dict of tensors:

==================  ==============================  ========================
Key                 Shape                           When
==================  ==============================  ========================
``frames``          ``(L, C, H, W)`` uint8          always
``frames.<cam>``    ``(L, C, H, W)`` uint8          a camera was requested
``actions``         ``(L-1,)`` int64 / ``(L-1, A)``  the episodes have them
``rewards``         ``(L-1,)`` float32              the episodes have them
``terminated``      ``(L-1,)`` bool                 the episodes have them
``truncated``       ``(L-1,)`` bool                 the episodes have them
``obs.<name>``      ``(L, ...)`` float32            an observation was requested
==================  ==============================  ========================

Frames stay uint8 so batches are a quarter of the size on the way to the
GPU; tasks convert them with :func:`frames_to_float`.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from simulusion.data.episode import Episode

logger = logging.getLogger(__name__)


class SequenceDataset(Dataset):
    """Every ``sequence_length``-frame window of a collection of episodes.

    Args:
        episodes: An :class:`~simulusion.data.folder.EpisodeFolder`, or any
            sequence of :class:`~simulusion.data.episode.Episode`.
        sequence_length: Frames per window ``L``.
        stride: Frames between consecutive window starts.
        image_size: Resize frames to ``(size, size)`` or ``(H, W)``; ``None``
            keeps them as stored.
        cameras: Extra cameras to return as ``frames.<cam>``.
        main_camera: Return this extra camera as ``frames`` instead of the
            episode's main view.
        observation_keys: Low-dimensional observations to return.
        require_actions: Raise if an episode has no actions, rather than
            returning windows without them.
        cache_episodes: Decoded episodes kept per worker. Raise it (or
            ``preload``) when episodes are re-read often.
        preload: Load every episode up front.
    """

    def __init__(
        self,
        episodes: Sequence[Episode],
        sequence_length: int,
        *,
        stride: int = 1,
        image_size: Optional[int | Tuple[int, int]] = None,
        cameras: Sequence[str] = (),
        main_camera: Optional[str] = None,
        observation_keys: Sequence[str] = (),
        require_actions: bool = False,
        cache_episodes: int = 16,
        preload: bool = False,
    ) -> None:
        if sequence_length < 1:
            raise ValueError("sequence_length must be at least 1.")
        if stride < 1:
            raise ValueError("stride must be at least 1.")
        self.episodes = episodes
        self.sequence_length = int(sequence_length)
        self.stride = int(stride)
        self.image_size = _size_tuple(image_size)
        self.cameras = list(cameras)
        self.main_camera = main_camera
        self.observation_keys = list(observation_keys)
        self.require_actions = require_actions
        self._cache: "OrderedDict[int, Episode]" = OrderedDict()
        self._cache_size = max(1, int(cache_episodes))
        lengths = getattr(episodes, "lengths", None)
        if lengths is None:
            lengths = [len(ep) for ep in episodes]
        self.index: List[Tuple[int, int]] = []
        short = 0
        for ep, length in enumerate(lengths):
            if length < self.sequence_length:
                short += 1
                continue
            for start in range(0, length - self.sequence_length + 1, self.stride):
                self.index.append((ep, start))
        if short:
            logger.warning(
                "%d of %d episodes are shorter than %d frames and were skipped.",
                short,
                len(lengths),
                self.sequence_length,
            )
        if not self.index:
            raise ValueError(
                f"No episode has {self.sequence_length} frames; lower sequence_length."
            )
        if preload:
            self._cache_size = len(lengths)
            for ep in range(len(lengths)):
                self._episode(ep)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_index, start = self.index[idx]
        episode = self._episode(ep_index)
        stop = start + self.sequence_length
        window = episode.slice(start, stop)
        main = window.frames
        if self.main_camera is not None:
            if self.main_camera not in window.cameras:
                raise KeyError(
                    f"Camera '{self.main_camera}' not in episode {ep_index} "
                    f"(has: {', '.join(sorted(window.cameras)) or 'none'})."
                )
            main = window.cameras[self.main_camera]
        item: Dict[str, torch.Tensor] = {"frames": self._frames(main)}
        for cam in self.cameras:
            item[f"frames.{cam}"] = self._frames(window.cameras[cam])
        if window.actions is not None:
            actions = torch.from_numpy(np.ascontiguousarray(window.actions))
            item["actions"] = (
                actions.long()
                if not torch.is_floating_point(actions) and actions.dim() == 1
                else actions.float()
            )
        elif self.require_actions:
            raise ValueError(f"Episode {ep_index} has no actions.")
        if window.rewards is not None:
            item["rewards"] = torch.from_numpy(window.rewards.astype(np.float32))
        if window.terminated is not None:
            item["terminated"] = torch.from_numpy(window.terminated.copy())
        if window.truncated is not None:
            item["truncated"] = torch.from_numpy(window.truncated.copy())
        for key in self.observation_keys:
            item[f"obs.{key}"] = torch.from_numpy(
                np.asarray(window.observations[key], dtype=np.float32)
            )
        return item

    def _episode(self, index: int) -> Episode:
        cached = self._cache.get(index)
        if cached is not None:
            self._cache.move_to_end(index)
            return cached
        episode = self.episodes[index]
        self._cache[index] = episode
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return episode

    def _frames(self, frames: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2)
        if self.image_size is not None and tuple(tensor.shape[-2:]) != self.image_size:
            resized = F.interpolate(
                tensor.float(),
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            tensor = resized.round().clamp(0, 255).to(torch.uint8)
        return tensor.contiguous()


def frames_to_float(frames: torch.Tensor) -> torch.Tensor:
    """uint8 frames to float in ``[0, 1]`` (floats pass through)."""
    if frames.dtype == torch.uint8:
        return frames.float().div_(255.0)
    return frames.float()


def one_hot_actions(actions: torch.Tensor, n: int) -> torch.Tensor:
    """Integer actions ``(..., )`` to float one-hot ``(..., n)``; floats pass through."""
    if torch.is_floating_point(actions):
        return actions
    return F.one_hot(actions.long(), n).float()


def _size_tuple(size: Optional[int | Sequence[int]]) -> Optional[Tuple[int, int]]:
    if size is None:
        return None
    if isinstance(size, int):
        return (size, size)
    values = tuple(int(s) for s in size)
    if len(values) != 2:
        raise ValueError(f"image_size must be an int or (H, W); got {size!r}.")
    return values[0], values[1]


def collate_sequences(items: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Stack windows into a batch, keeping only the keys every window has."""
    keys = set(items[0])
    for item in items[1:]:
        keys &= set(item)
    return {key: torch.stack([item[key] for item in items]) for key in sorted(keys)}


__all__ = [
    "SequenceDataset",
    "collate_sequences",
    "frames_to_float",
    "one_hot_actions",
]
